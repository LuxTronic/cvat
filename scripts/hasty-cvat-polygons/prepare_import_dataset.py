from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from segment import load_class_to_index, load_json_export


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SCRIPT_DIR = Path(__file__).resolve().parent
HASTY_ROOT = SCRIPT_DIR / "hasty_exports"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package existing YOLO-seg labels into an Ultralytics import-ready dataset layout."
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


def main() -> None:
    args = parse_args()

    export_data = load_json_export(args.json_path) if args.json_path.exists() else {}
    class_to_index = load_class_to_index(args.classes_file, export_data)
    index_to_class = {idx: name for name, idx in class_to_index.items()}

    images_out = args.output_root / "images" / args.split
    labels_out = args.output_root / "labels" / args.split
    yaml_out = args.output_root / "dataset.yaml"
    train_txt_out = args.output_root / "train.txt"

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
        image_path = find_image_for_stem(args.images_dir, stem)
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
    print(f"Copied pairs: {copied}")
    print(f"Missing images for labels: {missing_images}")
    print(f"Empty label files copied: {empty_labels}")
    print(f"Dataset root: {args.output_root}")
    print(f"YAML: {yaml_out}")
    print(f"train.txt: {train_txt_out}")
    if zip_out is not None:
        print(f"ZIP: {zip_out}")


if __name__ == "__main__":
    main()
