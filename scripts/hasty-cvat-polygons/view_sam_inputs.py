from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the exact bbox prompts passed to SAM from JSON labels."
    )
    parser.add_argument("--images-dir", type=Path, default=HASTY_ROOT / "imgs")
    parser.add_argument("--json-path", type=Path, default=HASTY_ROOT / "annotations.json")
    parser.add_argument("--output-dir", type=Path, default=HASTY_ROOT / "vis_sam_inputs")
    parser.add_argument("--line-thickness", type=int, default=2)
    return parser.parse_args()


def load_export(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def color_for_class_name(class_name: str) -> tuple[int, int, int]:
    seed = abs(hash(class_name)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    color = rng.integers(40, 256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x0 = max(0, x)
    y0 = max(th + 4, y)
    cv2.rectangle(image, (x0, y0 - th - 4), (x0 + tw + 4, y0), color, -1)
    cv2.putText(image, text, (x0 + 2, y0 - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_one(image_path: Path, image_entry: dict, output_path: Path, line_thickness: int) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0
    h, w = image.shape[:2]
    drawn = 0
    for lab in image_entry.get("labels", []):
        class_name = str(lab.get("class_name", "")).strip()
        bbox = lab.get("bbox")
        if not class_name or not isinstance(bbox, list):
            continue
        xyxy = sanitize_bbox_xyxy(bbox, w, h)
        if xyxy is None:
            continue
        x1, y1, x2, y2 = xyxy
        color = color_for_class_name(class_name)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, line_thickness))
        draw_label(image, class_name, x1, y1 - 2, color)
        draw_label(image, f"{x1},{y1},{x2},{y2}", x1, min(h - 2, y1 + 18), color)
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return drawn


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export = load_export(args.json_path)

    total_images = 0
    total_boxes = 0
    missing_images = 0

    for image_entry in export.get("images", []):
        image_name = image_entry.get("image_name")
        if not image_name:
            continue
        if not image_entry.get("labels"):
            continue
        image_path = args.images_dir / image_name
        if not image_path.exists():
            missing_images += 1
            continue
        out_path = args.output_dir / image_name
        drawn = process_one(
            image_path=image_path,
            image_entry=image_entry,
            output_path=out_path,
            line_thickness=args.line_thickness,
        )
        total_images += 1
        total_boxes += drawn
        print(f"[{Path(image_name).stem}] boxes_drawn={drawn} -> {out_path.name}")

    print("\nDone")
    print(f"Images visualized: {total_images}")
    print(f"Boxes drawn: {total_boxes}")
    print(f"Missing source images: {missing_images}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
