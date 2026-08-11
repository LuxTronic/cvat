from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

app = FastAPI(debug=True)
log = logging.getLogger("sam-service")
logging.basicConfig(level=logging.INFO)

MODEL_TYPE = os.getenv("SAM_MODEL_TYPE", "vit_b")
CHECKPOINT = Path(os.getenv("SAM_CHECKPOINT", "/sam-service/weights/sam_vit_b_01ec64.pth"))
CHECKPOINT_URL = os.getenv(
    "SAM_CHECKPOINT_URL",
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_LOCK = threading.Lock()
INFER_LOCK = threading.Lock()
PREDICTOR: Optional[SamPredictor] = None


def _gpu_stats() -> str:
    if DEVICE.type != "cuda":
        return "device=cpu"

    try:
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        allocated_mb = torch.cuda.memory_allocated(idx) / (1024 * 1024)
        reserved_mb = torch.cuda.memory_reserved(idx) / (1024 * 1024)
        return (
            f"device=cuda:{idx}({name}) "
            f"alloc_mb={allocated_mb:.1f} "
            f"reserved_mb={reserved_mb:.1f}"
        )
    except Exception:
        return "device=cuda(unavailable stats)"


def _ensure_checkpoint() -> None:
    if CHECKPOINT.exists():
        return

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    log.info("[SAM] Downloading checkpoint from %s", CHECKPOINT_URL)
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)  # nosec B310
    log.info("[SAM] Checkpoint downloaded to %s", CHECKPOINT)


def _get_predictor() -> SamPredictor:
    global PREDICTOR
    with MODEL_LOCK:
        if PREDICTOR is not None:
            return PREDICTOR

        _ensure_checkpoint()
        log.info(
            "[SAM] Loading model | type=%s | device=%s | checkpoint=%s",
            MODEL_TYPE,
            DEVICE,
            CHECKPOINT,
        )
        sam_model = sam_model_registry[MODEL_TYPE](checkpoint=str(CHECKPOINT))
        sam_model.to(device=DEVICE)
        PREDICTOR = SamPredictor(sam_model)
        return PREDICTOR


def _parse_json_field(raw: Optional[str], *, fallback: Any) -> Any:
    if raw is None:
        return fallback

    value = raw.strip()
    if not value:
        return fallback

    return json.loads(value)


def _normalize_bbox(raw_bbox: Any) -> Optional[np.ndarray]:
    if raw_bbox is None:
        return None

    if (
        isinstance(raw_bbox, list)
        and len(raw_bbox) == 2
        and all(isinstance(p, list) and len(p) == 2 for p in raw_bbox)
    ):
        (x1, y1), (x2, y2) = raw_bbox
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    if isinstance(raw_bbox, list) and len(raw_bbox) == 4:
        return np.array(raw_bbox, dtype=np.float32)

    raise ValueError("bbox must be [[x1, y1], [x2, y2]] or [x1, y1, x2, y2]")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_exists": CHECKPOINT.exists(),
    }


@app.post("/infer")
async def infer(
    image: UploadFile = File(...),
    task_id: int = 0,
    pos_points: str = Form("[]"),
    neg_points: str = Form("[]"),
    bbox: Optional[str] = Form(None),
    multimask_output: bool = Form(False),
) -> JSONResponse:
    try:
        t_start = time.perf_counter()
        t0 = t_start
        predictor = _get_predictor()
        t_model = time.perf_counter()
        pil_image = Image.open(image.file).convert("RGB")
        image_np = np.array(pil_image)
        t_decode = time.perf_counter()

        pos = _parse_json_field(pos_points, fallback=[])
        neg = _parse_json_field(neg_points, fallback=[])
        raw_bbox = _parse_json_field(bbox, fallback=None)
        box = _normalize_bbox(raw_bbox)

        if not pos and not neg and box is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Provide at least one foreground/background point or bbox"},
            )

        points = (pos or []) + (neg or [])
        labels = ([1] * len(pos or [])) + ([0] * len(neg or []))

        point_coords = np.array(points, dtype=np.float32) if points else None
        point_labels = np.array(labels, dtype=np.int32) if labels else None

        with INFER_LOCK:
            predictor.set_image(image_np)
            t_set_image = time.perf_counter()
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=bool(multimask_output),
            )
            t_predict = time.perf_counter()

        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(np.uint8)
        t_postprocess = time.perf_counter()
        best_score = float(scores[best_idx])

        log.info(
            (
                "[SAM] infer ok | task_id=%s | pos=%d | neg=%d | has_box=%s | score=%.4f | "
                "timings_ms model_ready=%.1f decode=%.1f set_image=%.1f predict=%.1f post=%.1f total=%.1f | %s"
            ),
            task_id,
            len(pos),
            len(neg),
            box is not None,
            best_score,
            (t_model - t0) * 1000,
            (t_decode - t_model) * 1000,
            (t_set_image - t_decode) * 1000,
            (t_predict - t_set_image) * 1000,
            (t_postprocess - t_predict) * 1000,
            (t_postprocess - t_start) * 1000,
            _gpu_stats(),
        )

        t_serialize_start = time.perf_counter()
        response_body = {
            "mask": best_mask.tolist(),
            "score": best_score,
            "width": int(image_np.shape[1]),
            "height": int(image_np.shape[0]),
        }
        t_serialize_end = time.perf_counter()
        log.info(
            "[SAM] serialize ok | task_id=%s | timings_ms mask_to_list=%.1f",
            task_id,
            (t_serialize_end - t_serialize_start) * 1000,
        )

        return JSONResponse(content=response_body)
    except Exception as ex:
        log.exception("[SAM] infer failed")
        return JSONResponse(status_code=500, content={"error": str(ex)})
