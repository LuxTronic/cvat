from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from segment import load_class_to_index, load_json_export


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package existing YOLO-seg labels for CVAT (Ultralytics) or Hasty import."
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["cvat", "hasty"],
        default="cvat",
        help="Output target format. Defaults to cvat.",
    )
    parser.add_argument("--images-dir", type=Path, default=HASTY_ROOT / "imgs")
    parser.add_argument("--labels-dir", type=Path, default=HASTY_ROOT / "seg_annotations")
    parser.add_argument("--json-path", type=Path, default=HASTY_ROOT / "annotations.json")
    parser.add_argument("--classes-file", type=Path, default=HASTY_ROOT / "classes.txt")
    parser.add_argument("--output-root", type=Path, default=HASTY_ROOT / "import_dataset")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Include val split in dataset.yaml (defaults to single-split for CVAT import).",
    )
    parser.add_argument(
        "--train-prefix",
        type=str,
        default="hasty_exports/images/train",
        help="Prefix used in train.txt entries, e.g. 'hasty_exports/images/train'.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Hasty output JSON path. Defaults to <output-root>/import_dataset_hasty_like.json.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project name override for Hasty output.",
    )
    parser.add_argument(
        "--create-date",
        type=str,
        default=None,
        help="Optional fixed create date for Hasty output, e.g. 2026-03-02T22:10:35Z.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Disable zip archive creation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate output directories and overwrite existing files.",
    )
    return parser.parse_args()


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
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


def build_dataset_hint_by_stem(export_data: dict) -> dict[str, str]:
    hint: dict[str, str] = {}
    for image in export_data.get("images", []):
        image_name = image.get("image_name")
        if not image_name:
            continue
        stem = Path(str(image_name)).stem
        dataset_name = str(image.get("dataset_name", "")).strip()
        if dataset_name and stem not in hint:
            hint[stem] = dataset_name
    return hint


def resolve_image_for_stem(
    images_dir: Path,
    stem: str,
    image_lookup: dict[str, list[Path]],
    dataset_hint: str = "",
) -> Path | None:
    direct = find_image_for_stem(images_dir, stem)
    if direct is not None:
        return direct

    candidates = image_lookup.get(stem, [])
    if not candidates:
        return None

    if dataset_hint:
        hint = dataset_hint.lower()
        dataset_matched = [p for p in candidates if hint in p.as_posix().lower()]
        if dataset_matched:
            return dataset_matched[0]
    return candidates[0]


def write_yaml(path: Path, dataset_root: Path, split: str, names: dict[int, str], include_val: bool) -> None:
    lines = [
        f"path: {dataset_root.as_posix()}",
        f"train: images/{split}",
        "names:",
    ]
    if include_val:
        lines.insert(2, f"val: images/{split}")
    for idx in sorted(names):
        lines.append(f"  {idx}: {names[idx]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_train_txt(path: Path, image_names: list[str], train_prefix: str) -> None:
    prefix = train_prefix.strip().rstrip("/")
    lines = [f"{prefix}/{name}" for name in sorted(image_names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now_utc_string() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_uuid(seed: str) -> str:
    return str(uuid5(NAMESPACE_URL, seed))


def default_color_from_index(index: int) -> str:
    # Keep alpha suffix so values resemble Hasty export colors.
    palette = [
        "#1F78B44D",
        "#A603034D",
        "#33A02C4D",
        "#FB9A994D",
        "#FF7F004D",
        "#6A3D9A4D",
        "#B159284D",
    ]
    return palette[index % len(palette)]


def build_label_classes(index_to_class: dict[int, str], export_data: dict) -> list[dict]:
    existing = export_data.get("label_classes", [])
    if isinstance(existing, list) and existing:
        return existing

    label_classes: list[dict] = []
    for class_index, idx in enumerate(sorted(index_to_class)):
        class_name = index_to_class[idx]
        label_classes.append(
            {
                "class_id": stable_uuid(f"class:{class_name}"),
                "parent_class_id": None,
                "class_name": class_name,
                "class_type": "object",
                "color": default_color_from_index(class_index),
                "norder": float(10 + class_index),
                "icon_url": None,
                "attributes": [],
                "description": None,
                "use_description_as_prompt": False,
            }
        )
    return label_classes


def parse_yolo_seg_row(row: str, img_w: int, img_h: int) -> tuple[int, list[list[int]], list[int]] | None:
    parts = row.strip().split()
    if len(parts) < 7:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    polygon: list[list[int]] = []
    xs: list[int] = []
    ys: list[int] = []
    for i in range(0, len(coords), 2):
        x = int(round(coords[i] * img_w))
        y = int(round(coords[i + 1] * img_h))
        x = max(0, min(img_w - 1, x))
        y = max(0, min(img_h - 1, y))
        polygon.append([x, y])
        xs.append(x)
        ys.append(y)

    if len(polygon) < 3:
        return None

    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return class_id, polygon, bbox


def package_for_cvat(args: argparse.Namespace, export_data: dict, index_to_class: dict[int, str]) -> None:
    images_out = args.output_root / "images" / args.split
    labels_out = args.output_root / "labels" / args.split
    yaml_out = args.output_root / "dataset.yaml"
    train_txt_out = args.output_root / "train.txt"
    image_lookup = build_image_lookup(args.images_dir)
    dataset_hint_by_stem = build_dataset_hint_by_stem(export_data)

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_images = 0
    empty_labels = 0
    copied_image_names: list[str] = []

    for label_path in sorted(args.labels_dir.glob("*.txt")):
        stem = label_path.stem
        image_path = resolve_image_for_stem(
            images_dir=args.images_dir,
            stem=stem,
            image_lookup=image_lookup,
            dataset_hint=dataset_hint_by_stem.get(stem, ""),
        )
        if image_path is None:
            missing_images += 1
            continue

        label_text = label_path.read_text(encoding="utf-8")
        if not label_text.strip():
            empty_labels += 1

        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / f"{stem}.txt")
        copied_image_names.append(image_path.name)
        copied += 1

    write_yaml(yaml_out, args.output_root, args.split, index_to_class, args.include_val)
    write_train_txt(train_txt_out, copied_image_names, args.train_prefix)

    zip_out: Path | None = None
    if not args.no_zip:
        zip_base = args.output_root.parent / args.output_root.name
        zip_out_str = shutil.make_archive(str(zip_base), "zip", root_dir=args.output_root.parent, base_dir=args.output_root.name)
        zip_out = Path(zip_out_str)

    print("Done")
    print("Target: CVAT")
    print(f"Copied pairs: {copied}")
    print(f"Missing images for labels: {missing_images}")
    print(f"Empty label files copied: {empty_labels}")
    print(f"Dataset root: {args.output_root}")
    print(f"YAML: {yaml_out}")
    print(f"train.txt: {train_txt_out}")
    if zip_out is not None:
        print(f"ZIP: {zip_out}")


def package_for_hasty(args: argparse.Namespace, export_data: dict, index_to_class: dict[int, str]) -> None:
    images_by_stem = {
        Path(image.get("image_name", "")).stem: image
        for image in export_data.get("images", [])
        if image.get("image_name")
    }
    image_lookup = build_image_lookup(args.images_dir)

    output_images: list[dict] = []
    missing_images = 0
    unknown_class_rows = 0
    invalid_rows = 0
    skipped_no_metadata = 0

    for label_path in sorted(args.labels_dir.glob("*.txt")):
        stem = label_path.stem
        source_image = images_by_stem.get(stem, {})
        image_path = resolve_image_for_stem(
            images_dir=args.images_dir,
            stem=stem,
            image_lookup=image_lookup,
            dataset_hint=str(source_image.get("dataset_name", "")).strip(),
        )
        if image_path is None:
            missing_images += 1
            continue

        width = source_image.get("width")
        height = source_image.get("height")
        if not isinstance(width, int) or not isinstance(height, int):
            skipped_no_metadata += 1
            continue

        rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines() if row.strip()]
        labels = []
        for label_index, row in enumerate(rows):
            parsed = parse_yolo_seg_row(row, width, height)
            if parsed is None:
                invalid_rows += 1
                continue

            class_id, polygon, bbox = parsed
            class_name = index_to_class.get(class_id)
            if class_name is None:
                unknown_class_rows += 1
                continue

            labels.append(
                {
                    "id": stable_uuid(f"label:{stem}:{label_index}:{class_name}"),
                    "class_name": class_name,
                    "bbox": bbox,
                    "polygon": polygon,
                    "mask": None,
                    "z_index": label_index,
                    "keypoints": [],
                    "attributes": {},
                }
            )

        output_images.append(
            {
                "image_id": source_image.get("image_id") or stable_uuid(f"image:{image_path.name}"),
                "width": width,
                "height": height,
                "image_name": image_path.name,
                "image_mode": source_image.get("image_mode", "RGB"),
                "dataset_name": source_image.get("dataset_name", "default"),
                "image_status": source_image.get("image_status", "TO REVIEW"),
                "labels": labels,
                "tags": source_image.get("tags", []),
                "tag_groups": source_image.get("tag_groups", {}),
                "attributes": source_image.get("attributes", {}),
            }
        )

    now_export = now_utc_string()
    create_date = args.create_date or export_data.get("create_date") or now_export
    project_name = args.project_name or export_data.get("project_name") or "IMPORT_DATASET"
    label_classes = build_label_classes(index_to_class, export_data)

    payload = {
        "project_name": project_name,
        "create_date": create_date,
        "export_format_version": export_data.get("export_format_version", "1.1"),
        "export_date": now_export,
        "label_classes": label_classes,
        "keypoint_schemas": export_data.get("keypoint_schemas", []),
        "tag_groups": export_data.get(
            "tag_groups",
            [
                {
                    "group_id": stable_uuid("tag_group:Default"),
                    "group_name": "Default",
                    "group_type": "MULTIPLE-SELECTION",
                    "tags": [],
                }
            ],
        ),
        "images": output_images,
        "attributes": export_data.get("attributes", []),
    }

    output_json = args.output_json or (args.output_root / "import_dataset_hasty_like.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Done")
    print("Target: Hasty")
    print(f"Images written: {len(output_images)}")
    print(f"Missing images for labels: {missing_images}")
    print(f"Rows with unknown class index: {unknown_class_rows}")
    print(f"Invalid YOLO rows skipped: {invalid_rows}")
    print(f"Skipped images without width/height metadata: {skipped_no_metadata}")
    print(f"Output JSON: {output_json}")


def main() -> None:
    args = parse_args()

    export_data = load_json_export(args.json_path) if args.json_path.exists() else {}
    class_to_index = load_class_to_index(args.classes_file, export_data)
    index_to_class = {idx: name for name, idx in class_to_index.items()}
    if args.target == "hasty":
        package_for_hasty(args, export_data, index_to_class)
    else:
        package_for_cvat(args, export_data, index_to_class)


if __name__ == "__main__":
    main()
