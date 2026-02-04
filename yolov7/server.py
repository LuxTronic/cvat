from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
from pathlib import Path
import subprocess
import torch
import json

app = FastAPI()

WEIGHTS = "yolov7.pt"
DEVICE = "0" if torch.cuda.is_available() else "cpu"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    }


@app.post("/infer")
async def infer(image: UploadFile = File(...)):
    """
    Run YOLOv7 inference on a single image.
    Returns detections in YOLO normalized format.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save uploaded image
        image_path = tmpdir / image.filename
        image_path.write_bytes(await image.read())

        # Output directory
        out_dir = tmpdir / "runs"

        # Run YOLOv7 detect.py
        cmd = [
            "python", "detect.py",
            "--weights", WEIGHTS,
            "--source", str(image_path),
            "--device", DEVICE,
            "--save-txt",
            "--save-conf",
            "--conf", "0.25",
            "--project", str(out_dir),
            "--name", "pred",
            "--exist-ok",
            "--nosave"
        ]

        subprocess.run(cmd, check=True)

        # Parse YOLO output
        labels_dir = out_dir / "pred" / "labels"
        detections = []

        if labels_dir.exists():
            for label_file in labels_dir.glob("*.txt"):
                for line in label_file.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) > 5 else None

                    detections.append({
                        "class_id": class_id,
                        "xc": xc,
                        "yc": yc,
                        "w": w,
                        "h": h,
                        "confidence": conf,
                    })

        return JSONResponse(content=detections)

