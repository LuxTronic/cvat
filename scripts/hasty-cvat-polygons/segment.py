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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


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
        "--epsilon-ratio",
        type=float,
        default=0.01,
        help="Contour simplification ratio: epsilon = epsilon_ratio * arcLength.",
    )
    parser.add_argument(
        "--contour-mode",
        type=str,
        choices=("external", "tree"),
        default="external",
        help="Contour extraction mode for mask->polygon conversion.",
    )
    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=100.0,
        help="Skip contours with area below this threshold in pixels.",
    )
    parser.add_argument(
        "--max-polygons-per-object",
        type=int,
        default=1,
        help="Keep up to N largest polygons per object after filtering.",
    )
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
    class_names = [
        c.get("class_name", "").strip()
        for c in export_data.get("label_classes", [])
        if c.get("class_name")
    ]
    if not class_names and classes_file.exists():
        class_names = [
            line.strip()
            for line in classes_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return {name: idx for idx, name in enumerate(class_names)}


def sanitize_bbox_xyxy(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[int, int, int, int] | None:
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


def extract_first_mask(result) -> np.ndarray | None:
    if result.masks is None or not hasattr(result.masks, "data"):
        return None
    data = result.masks.data
    if data is None or len(data) == 0:
        return None
    mask = data[0].cpu().numpy()
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    return mask_u8


def extract_masks_batch(result, expected_count: int) -> list[np.ndarray | None]:
    if result.masks is None or not hasattr(result.masks, "data"):
        return [None] * expected_count
    data = result.masks.data
    if data is None:
        return [None] * expected_count

    masks: list[np.ndarray | None] = []
    count = min(len(data), expected_count)
    for idx in range(count):
        mask = data[idx].cpu().numpy()
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        masks.append(mask_u8)

    if len(masks) < expected_count:
        masks.extend([None] * (expected_count - len(masks)))
    return masks[:expected_count]


def contour_mode_from_name(name: str) -> int:
    if name == "tree":
        return cv2.RETR_TREE
    return cv2.RETR_EXTERNAL


def mask_to_polygons(
    mask_u8: np.ndarray,
    epsilon_ratio: float,
    contour_mode: int,
    min_contour_area: float,
    max_polygons_per_object: int,
) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, contour_mode, cv2.CHAIN_APPROX_NONE)
    polys: list[np.ndarray] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        epsilon = max(0.0, epsilon_ratio) * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        approx_xy = approx.reshape(-1, 2).astype(np.float32)
        if approx_xy.shape[0] < 3:
            continue
        polys.append(approx_xy)

    polys.sort(key=polygon_area, reverse=True)
    max_keep = max(0, int(max_polygons_per_object))
    if max_keep == 0:
        return []
    return polys[:max_keep]


def box_to_polygon(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def generate_ultralytics_seg_row(
    class_id: int, polygon_xy: np.ndarray, img_w: int, img_h: int
) -> str | None:
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


def build_image_lookup(images_dir: Path) -> dict[str, list[Path]]:
    lookup: dict[str, list[Path]] = {}
    for p in images_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        lookup.setdefault(p.name, []).append(p)

    for key in lookup:
        lookup[key].sort(key=lambda x: x.as_posix())
    return lookup


def resolve_image_path(
    images_dir: Path, image_entry: dict, image_lookup: dict[str, list[Path]]
) -> Path | None:
    image_name = image_entry.get("image_name")
    if not image_name:
        return None

    direct = images_dir / image_name
    if direct.exists():
        return direct

    basename = Path(str(image_name)).name
    direct_basename = images_dir / basename
    if direct_basename.exists():
        return direct_basename

    candidates = image_lookup.get(basename, [])
    if not candidates:
        return None

    dataset_name = str(image_entry.get("dataset_name", "")).strip().lower()
    if dataset_name:
        dataset_matched = [p for p in candidates if dataset_name in p.as_posix().lower()]
        if dataset_matched:
            return dataset_matched[0]

    return candidates[0]


def process_image(
    sam_model: SAM,
    image_path: Path,
    labels: list[JsonLabel],
    output_path: Path,
    class_to_index: dict[str, int],
    fallback_to_box: bool,
    device: str,
    debug: bool,
    epsilon_ratio: float,
    contour_mode: int,
    min_contour_area: float,
    max_polygons_per_object: int,
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

    valid_items: list[tuple[int, tuple[int, int, int, int], str]] = []
    for lab in labels:
        class_id = class_to_index.get(lab.class_name)
        if class_id is None:
            skipped_unknown_class += 1
            continue

        bbox = sanitize_bbox_xyxy(lab.bbox, img_w, img_h)
        if bbox is None:
            failed += 1
            continue
        valid_items.append((class_id, bbox, lab.class_name))

    masks_by_item: list[np.ndarray | None] = [None] * len(valid_items)
    if valid_items:
        try:
            kwargs = {"bboxes": [list(item[1]) for item in valid_items], "verbose": False}
            if device:
                kwargs["device"] = device
            result_list = sam_model(image_rgb, **kwargs)
            if result_list:
                masks_by_item = extract_masks_batch(result_list[0], len(valid_items))
        except Exception as exc:
            if debug and debug_errors_shown < 10:
                print(f"[DEBUG] Batch SAM failed image={image_path.name} err={exc}")
                debug_errors_shown += 1
            masks_by_item = [None] * len(valid_items)

    for idx, (class_id, bbox, class_name) in enumerate(valid_items):
        x1, y1, x2, y2 = bbox
        poly_globals: list[np.ndarray] = []

        try:
            mask_u8 = masks_by_item[idx]
            if mask_u8 is not None:
                poly_globals = mask_to_polygons(
                    mask_u8=mask_u8,
                    epsilon_ratio=epsilon_ratio,
                    contour_mode=contour_mode,
                    min_contour_area=min_contour_area,
                    max_polygons_per_object=max_polygons_per_object,
                )
        except Exception as exc:
            if debug and debug_errors_shown < 10:
                print(
                    f"[DEBUG] SAM failed image={image_path.name} class={class_name} "
                    f"bbox={[x1, y1, x2, y2]} err={exc}"
                )
                debug_errors_shown += 1
            poly_globals = []

        if not poly_globals and fallback_to_box:
            poly_globals = [box_to_polygon(x1, y1, x2, y2)]
        if not poly_globals:
            failed += 1
            continue

        wrote_any = False
        for poly_global in poly_globals:
            row = generate_ultralytics_seg_row(class_id, poly_global, img_w, img_h)
            if row is None:
                continue
            lines_out.append(row)
            wrote_any = True
        if wrote_any:
            ok += 1
        else:
            failed += 1

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
    image_lookup = build_image_lookup(args.images_dir)

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
        image_path = resolve_image_path(args.images_dir, image_entry, image_lookup)
        if image_path is None or not image_path.exists():
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
            epsilon_ratio=float(args.epsilon_ratio),
            contour_mode=contour_mode_from_name(args.contour_mode),
            min_contour_area=float(args.min_contour_area),
            max_polygons_per_object=int(args.max_polygons_per_object),
        )
        total_ok += ok
        total_failed += failed
        total_unknown_class += unknown
        print(
            f"[{Path(image_name).stem}] objects={len(labels)} ok={ok} failed={failed} unknown_class={unknown}"
        )

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
