import yaml
from .model_paths import task_data_dir
import zipfile
from pathlib import Path
import logging


log = logging.getLogger(__name__)
def ensure_data_yaml(dataset_dir: Path, class_names: list[str]):
    """
    Write a minimal YOLOv7-compatible data.yaml
    EXACTLY matching:
      train: /models/task_X/data/images
      val:   /models/task_X/data/images
    """
    data_yaml = dataset_dir / "data.yaml"

    content = {
        "train": str(dataset_dir / "images"),
        "val": str(dataset_dir / "images"),
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    log.info("[DATASET] Writing YOLO data.yaml → %s", data_yaml)
    log.info("[DATASET] YOLO classes: %s", class_names)

    with open(data_yaml, "w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def unpack_yolo_dataset(zip_path: Path, data_dir: Path):
    """
    Unpack CVAT YOLO export ZIP into data_dir.
    """
    log.info("[DATASET] Unpacking %s into %s", zip_path, data_dir)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # CVAT exports directly into images/, labels/, train.txt
    images = data_dir / "images" / "train"
    labels = data_dir / "labels" / "train"

    if not images.exists() or not labels.exists():
        raise RuntimeError(
            f"Invalid YOLO dataset after unzip. "
            f"images={images.exists()}, labels={labels.exists()}, "
            f"contents={list(data_dir.iterdir())}"
        )

    log.info("[DATASET] YOLO dataset ready: %s", data_dir)

import random
import shutil
from pathlib import Path

def split_train_val(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    """
    Split YOLO dataset into train/val sets.
    Only images WITH labels are considered.
    Unlabeled images are ignored.
    """

    images_train = data_dir / "images" / "train"
    labels_train = data_dir / "labels" / "train"

    images_val = data_dir / "images" / "val"
    labels_val = data_dir / "labels" / "val"

    images_val.mkdir(parents=True, exist_ok=True)
    labels_val.mkdir(parents=True, exist_ok=True)

    # Collect ONLY labeled image/label pairs
    labeled_pairs = []
    skipped = 0

    for img_path in images_train.iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        label_path = labels_train / f"{img_path.stem}.txt"
        if label_path.exists():
            labeled_pairs.append((img_path, label_path))
        else:
            skipped += 1

    if not labeled_pairs:
        raise RuntimeError("No labeled images found to split")

    random.seed(seed)
    random.shuffle(labeled_pairs)

    val_count = max(1, int(len(labeled_pairs) * val_ratio))
    val_pairs = labeled_pairs[:val_count]

    for img_path, label_path in val_pairs:
        shutil.move(img_path, images_val / img_path.name)
        shutil.move(label_path, labels_val / label_path.name)

    log.info(
        "[DATASET] Split dataset: %d train / %d val (skipped %d unlabeled images)",
        len(labeled_pairs) - val_count,
        val_count,
        skipped,
    )


def clean_orphan_labels(data_dir):
    img_dir = data_dir / "images" / "train"
    lbl_dir = data_dir / "labels" / "train"

    for lbl in lbl_dir.glob("*.txt"):
        img = img_dir / (lbl.stem + ".png")
        if not img.exists():
            log.warning("[DATASET] Removing orphan label %s", lbl.name)
            lbl.unlink()

def clear_yolo_cache(data_dir):
    cache_dir = data_dir / "labels"
    for cache in cache_dir.glob("*.cache"):
        log.info("[DATASET] Removing YOLO cache %s", cache)
        cache.unlink()