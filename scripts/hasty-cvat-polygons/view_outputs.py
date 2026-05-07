from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw YOLO segmentation annotations on original images."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("hasty_exports/imgs"))
    parser.add_argument("--labels-dir", type=Path, default=Path("hasty_exports/seg_annotations"))
    parser.add_argument("--json-path", type=Path, default=Path("hasty_exports/annotations.json"))
    parser.add_argument("--classes-file", type=Path, default=Path("hasty_exports/classes.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("hasty_exports/vis_outputs"))
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Polygon border thickness in pixels.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Fill alpha for polygon overlays in [0,1].",
    )
    parser.add_argument(
        "--show-vertices",
        action="store_true",
        help="Draw a dot at each polygon vertex.",
    )
    parser.add_argument(
        "--vertex-radius",
        type=int,
        default=3,
        help="Vertex dot radius in pixels when --show-vertices is enabled.",
    )
    return parser.parse_args()


def load_class_names(json_path: Path, classes_file: Path) -> list[str]:
    if json_path.exists():
        try:
            export = json.loads(json_path.read_text(encoding="utf-8"))
            names = [c.get("class_name", "").strip() for c in export.get("label_classes", [])]
            names = [name for name in names if name]
            if names:
                return names
        except Exception:
            pass
    if classes_file.exists():
        return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def color_for_class(class_id: int) -> tuple[int, int, int]:
    # deterministic pseudo-random color per class_id (BGR for OpenCV)
    rng = np.random.default_rng(seed=class_id + 1337)
    color = rng.integers(50, 256, size=3, dtype=np.int32)
    return int(color[0]), int(color[1]), int(color[2])


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def build_image_lookup(images_dir: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    valid_exts = {ext.lower() for ext in IMAGE_EXTENSIONS}
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in valid_exts:
            continue
        lookup.setdefault(p.stem, []).append(p)

    for stem in lookup:
        lookup[stem].sort(key=lambda x: x.as_posix())
    return lookup


def resolve_image_for_stem(images_dir: Path, stem: str, image_lookup: dict[str, list[Path]]) -> Path | None:
    direct = find_image_for_stem(images_dir, stem)
    if direct is not None:
        return direct

    candidates = image_lookup.get(stem, [])
    if not candidates:
        return None
    return candidates[0]


def parse_label_line(line: str) -> tuple[int, np.ndarray] | None:
    parts = line.strip().split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        return None
    try:
        class_id = int(float(parts[0]))
        values = np.array([float(v) for v in parts[1:]], dtype=np.float32)
    except ValueError:
        return None
    if values.size < 6:
        return None
    points = values.reshape(-1, 2)
    return class_id, points


def normalized_poly_to_pixels(poly_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    poly = poly_norm.copy()
    poly[:, 0] = np.clip(poly[:, 0] * width, 0, width - 1)
    poly[:, 1] = np.clip(poly[:, 1] * height, 0, height - 1)
    return poly.astype(np.int32)


def draw_polygon_with_label(
    image: np.ndarray,
    poly_px: np.ndarray,
    class_id: int,
    class_name: str,
    alpha: float,
    line_thickness: int,
    show_vertices: bool,
    vertex_radius: int,
) -> None:
    color = color_for_class(class_id)
    overlay = image.copy()

    cv2.fillPoly(overlay, [poly_px], color)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0, image)
    cv2.polylines(image, [poly_px], isClosed=True, color=color, thickness=line_thickness)
    if show_vertices:
        for x, y in poly_px:
            cv2.circle(image, (int(x), int(y)), max(1, vertex_radius), (255, 255, 255), -1)
            cv2.circle(image, (int(x), int(y)), max(1, vertex_radius), color, 1)

    x, y = poly_px[np.argmin(poly_px[:, 1])]
    label = f"{class_id}:{class_name}" if class_name else str(class_id)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    x0 = int(max(0, x))
    y0 = int(max(th + 4, y))
    cv2.rectangle(image, (x0, y0 - th - 4), (x0 + tw + 4, y0), color, -1)
    cv2.putText(image, label, (x0 + 2, y0 - 2), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def process_one(
    image_path: Path,
    label_path: Path,
    output_path: Path,
    class_names: list[str],
    alpha: float,
    line_thickness: int,
    show_vertices: bool,
    vertex_radius: int,
) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0, 0
    height, width = image.shape[:2]

    drawn = 0
    skipped = 0
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(line)
        if parsed is None:
            skipped += 1
            continue
        class_id, poly_norm = parsed
        if poly_norm.shape[0] < 3:
            skipped += 1
            continue
        poly_px = normalized_poly_to_pixels(poly_norm, width, height)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else ""
        draw_polygon_with_label(
            image=image,
            poly_px=poly_px,
            class_id=class_id,
            class_name=class_name,
            alpha=alpha,
            line_thickness=line_thickness,
            show_vertices=show_vertices,
            vertex_radius=vertex_radius,
        )
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return drawn, skipped


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(args.json_path, args.classes_file)
    image_lookup = build_image_lookup(args.images_dir)

    total_images = 0
    missing_images = 0
    total_drawn = 0
    total_skipped = 0

    for label_path in sorted(args.labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = resolve_image_for_stem(args.images_dir, stem, image_lookup)
        if image_path is None:
            missing_images += 1
            continue

        output_path = args.output_dir / image_path.name
        drawn, skipped = process_one(
            image_path=image_path,
            label_path=label_path,
            output_path=output_path,
            class_names=class_names,
            alpha=float(np.clip(args.alpha, 0.0, 1.0)),
            line_thickness=max(1, args.line_thickness),
            show_vertices=args.show_vertices,
            vertex_radius=max(1, args.vertex_radius),
        )
        total_images += 1
        total_drawn += drawn
        total_skipped += skipped
        print(f"[{stem}] polygons_drawn={drawn} skipped={skipped} -> {output_path.name}")

    print("\nDone")
    print(f"Images visualized: {total_images}")
    print(f"Polygons drawn: {total_drawn}")
    print(f"Skipped invalid polygons: {total_skipped}")
    print(f"Missing source images: {missing_images}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
