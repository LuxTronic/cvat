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
