"""In-image check: the process boots on CPU and /infer returns a list.

A gray frame may produce no boxes. That is still a successful call.
"""

import hashlib
import io
import os
import shutil
import sys
from pathlib import Path

from PIL import Image

WEIGHTS = Path("/yolov7/weights/yolov7.pt")
ACTIVE = Path("/models/yolov7/task_1/active/weights/best.pt")


def main() -> int:
    if not WEIGHTS.is_file():
        print(f"missing baked weights: {WEIGHTS}", file=sys.stderr)
        return 1
    if WEIGHTS.stat().st_size != 12639769:
        print(f"unexpected baked weight size {WEIGHTS.stat().st_size}", file=sys.stderr)
        return 1
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    expected = "0fa1423606133c06362ec716029dade1169d5631e83195b75d8efa66d2ee0b77"
    if digest != expected:
        print(f"unexpected baked weight sha256 {digest}", file=sys.stderr)
        return 1
    if os.geteuid() != 1000:
        print(f"sidecar is uid {os.geteuid()}, expected 1000", file=sys.stderr)
        return 1

    ACTIVE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(WEIGHTS, ACTIVE)

    from fastapi.testclient import TestClient

    from server import app

    client = TestClient(app)
    health = client.get("/health")
    if health.status_code != 200:
        print(health.text, file=sys.stderr)
        return 1
    body = health.json()
    if body.get("status") != "ok" or body.get("cuda") is not False or body.get("use_half") is not False:
        print(f"unexpected health: {body}", file=sys.stderr)
        return 1

    frame = io.BytesIO()
    Image.new("RGB", (128, 128), (120, 120, 120)).save(frame, format="JPEG")
    frame.seek(0)
    inferred = client.post(
        "/infer",
        files={"image": ("frame.jpg", frame, "image/jpeg")},
        params={"task_id": 1},
    )
    if inferred.status_code != 200:
        print(inferred.text, file=sys.stderr)
        return 1
    detections = inferred.json()
    if not isinstance(detections, list):
        print(f"/infer did not return a list: {detections!r}", file=sys.stderr)
        return 1
    for det in detections:
        missing = {"class_id", "confidence", "xc", "yc", "w", "h"} - set(det)
        if missing:
            print(f"detection missing {sorted(missing)}: {det}", file=sys.stderr)
            return 1

    refused = client.post("/train", json={"task_id": 1})
    if refused.status_code != 400:
        print(f"/train without a dataset returned {refused.status_code}: {refused.text}", file=sys.stderr)
        return 1

    import train  # noqa: F401  # the trainer must still import on this stack

    print(
        f"smoke ok: health cpu, /infer returned {len(detections)} boxes, "
        "/train refused a missing dataset, train.py imported"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
