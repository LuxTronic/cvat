from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from ultralytics import SAM

SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


@dataclass
class JsonLabel:
    class_name: str
    bbox: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert JSON bbox labels to YOLO segmentation polygons using SAM/SAM2."
    )
    parser.add_argument("--images-dir", type=Path, default=HASTY_ROOT / "imgs")
    parser.add_argument("--json-path", type=Path, default=HASTY_ROOT / "annotations.json")
    parser.add_argument("--classes-file", type=Path, default=HASTY_ROOT / "classes.txt")
    parser.add_argument("--output-dir", type=Path, default=HASTY_ROOT / "seg_annotations")
    parser.add_argument("--sam-model", type=str, default="sam2_b.pt")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--fallback-to-box",
        action="store_true",
        help="If SAM fails for an object, write bbox rectangle as a 4-point polygon.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output label files.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sample SAM exceptions for debugging.",
    )
    return parser.parse_args()


def load_json_export(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_to_index(classes_file: Path, export_data: dict) -> dict[str, int]:
    class_names: list[str] = []
    class_names = [c.get("class_name", "").strip() for c in export_data.get("label_classes", []) if c.get("class_name")]
    if not class_names and classes_file.exists():
        class_names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {name: idx for idx, name in enumerate(class_names)}


def sanitize_bbox_xyxy(bbox: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x1_raw, y1_raw, x2_raw, y2_raw = bbox
    left = min(x1_raw, x2_raw)
    right = max(x1_raw, x2_raw)
    top = min(y1_raw, y2_raw)
    bottom = max(y1_raw, y2_raw)

    x1 = int(np.clip(np.floor(left), 0, img_w - 1))
    y1 = int(np.clip(np.floor(top), 0, img_h - 1))
    x2 = int(np.clip(np.ceil(right), 1, img_w))
    y2 = int(np.clip(np.ceil(bottom), 1, img_h))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
    return x1, y1, x2, y2


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def extract_best_polygon(result) -> np.ndarray | None:
    if result.masks is None or not hasattr(result.masks, "xy") or not result.masks.xy:
        return None
    polys = [np.asarray(p, dtype=np.float32) for p in result.masks.xy if len(p) >= 3]
    if not polys:
        return None
    return max(polys, key=polygon_area)


def box_to_polygon(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def generate_ultralytics_seg_row(class_id: int, polygon_xy: np.ndarray, img_w: int, img_h: int) -> str | None:
    """Build one Ultralytics YOLO-seg import row: '<class> x1 y1 x2 y2 ...'."""
    if polygon_xy.shape[0] < 3:
        return None
    norm = polygon_xy.copy()
    norm[:, 0] = np.clip(norm[:, 0] / img_w, 0.0, 1.0)
    norm[:, 1] = np.clip(norm[:, 1] / img_h, 0.0, 1.0)
    coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
    return f"{class_id} {coords}"


def write_ultralytics_seg_file(output_path: Path, rows: list[str]) -> None:
    """Write one YOLO-seg label file (one row per object)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows), encoding="utf-8")


def process_image(
    sam_model: SAM,
    image_path: Path,
    labels: list[JsonLabel],
    output_path: Path,
    class_to_index: dict[str, int],
    fallback_to_box: bool,
    device: str,
    debug: bool,
) -> tuple[int, int, int]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return 0, 0, 0
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = image_rgb.shape[:2]

    lines_out: list[str] = []
    ok = 0
    failed = 0
    skipped_unknown_class = 0
    debug_errors_shown = 0

    for lab in labels:
        class_id = class_to_index.get(lab.class_name)
        if class_id is None:
            skipped_unknown_class += 1
            continue

        bbox = sanitize_bbox_xyxy(lab.bbox, img_w, img_h)
        if bbox is None:
            failed += 1
            continue
        x1, y1, x2, y2 = bbox
        poly_global: np.ndarray | None = None

        try:
            kwargs = {"bboxes": [[x1, y1, x2, y2]], "verbose": False}
            if device:
                kwargs["device"] = device
            result_list = sam_model(image_rgb, **kwargs)
            if result_list:
                best = extract_best_polygon(result_list[0])
                if best is not None:
                    poly_global = best
        except Exception as exc:
            if debug and debug_errors_shown < 10:
                print(
                    f"[DEBUG] SAM failed image={image_path.name} class={lab.class_name} "
                    f"bbox={[x1, y1, x2, y2]} err={exc}"
                )
                debug_errors_shown += 1
            poly_global = None

        if poly_global is None and fallback_to_box:
            poly_global = box_to_polygon(x1, y1, x2, y2)
        if poly_global is None:
            failed += 1
            continue

        row = generate_ultralytics_seg_row(class_id, poly_global, img_w, img_h)
        if row is None:
            failed += 1
            continue
        lines_out.append(row)
        ok += 1

    write_ultralytics_seg_file(output_path, lines_out)
    return ok, failed, skipped_unknown_class


def parse_json_labels_for_image(image_entry: dict) -> list[JsonLabel]:
    out: list[JsonLabel] = []
    for raw in image_entry.get("labels", []):
        class_name = str(raw.get("class_name", "")).strip()
        bbox = raw.get("bbox")
        if not class_name or not isinstance(bbox, list):
            continue
        out.append(JsonLabel(class_name=class_name, bbox=bbox))
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    export_data = load_json_export(args.json_path)
    class_to_index = load_class_to_index(args.classes_file, export_data)
    sam_model = SAM(args.sam_model)

    total_images = 0
    total_objects = 0
    total_ok = 0
    total_failed = 0
    total_unknown_class = 0
    skipped_existing = 0
    missing_images = 0

    for image_entry in export_data.get("images", []):
        image_name = image_entry.get("image_name")
        if not image_name:
            continue
        image_path = args.images_dir / image_name
        if not image_path.exists():
            missing_images += 1
            continue

        labels = parse_json_labels_for_image(image_entry)
        if not labels:
            continue

        out_path = args.output_dir / f"{Path(image_name).stem}.txt"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        total_images += 1
        total_objects += len(labels)

        ok, failed, unknown = process_image(
            sam_model=sam_model,
            image_path=image_path,
            labels=labels,
            output_path=out_path,
            class_to_index=class_to_index,
            fallback_to_box=args.fallback_to_box,
            device=args.device,
            debug=args.debug,
        )
        total_ok += ok
        total_failed += failed
        total_unknown_class += unknown
        print(f"[{Path(image_name).stem}] objects={len(labels)} ok={ok} failed={failed} unknown_class={unknown}")

    print("\nDone")
    print(f"Processed images: {total_images}")
    print(f"Total objects: {total_objects}")
    print(f"Segmented objects: {total_ok}")
    print(f"Failed objects: {total_failed}")
    print(f"Skipped unknown class: {total_unknown_class}")
    print(f"Missing images: {missing_images}")
    print(f"Skipped existing outputs: {skipped_existing}")
    print(f"Output labels: {args.output_dir}")


if __name__ == "__main__":
    main()
