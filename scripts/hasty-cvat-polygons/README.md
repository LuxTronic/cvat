# Hasty -> SAM2 -> CVAT Pipeline

Downloads annotations/images from Hasty, converts bbox labels to YOLO segmentation
polygons with SAM2, packages an Ultralytics dataset zip, and imports it into CVAT.

## Run (full pipeline)

```bash
python run_pipeline.py \
  --project-id <HASTY_PROJECT_ID> \
  --dataset-id <HASTY_DATASET_ID> \
  --project-name <CVAT_PROJECT_NAME>
```

## Required `.env`

```env
HASTY_API_KEY=...
CVAT_HOST=http://localhost:8080
CVAT_TOKEN=...            # or CVAT_USERNAME/CVAT_PASSWORD
CVAT_USERNAME=...
CVAT_PASSWORD=...
```

## Scripts

- `run_pipeline.py`: orchestrates all steps end-to-end.
  Key args: `--project-id`, `--dataset-id` (repeatable), `--project-name` (required),
  `--device {gpu|cpu}`, `--skip-*`.
- `fetch_hasty_exports.py`: starts Hasty export jobs, polls completion, downloads
  `annotations.json` + images.
  Key args: `--project-id`, `--dataset-id`, `--overwrite`.
- `segment.py`: runs SAM2 on each bbox from `hasty_exports/annotations.json` and writes
  YOLO-seg labels to `hasty_exports/seg_annotations`.
  Key args: `--device`, `--sam-model`, `--overwrite`.
- `prepare_import_dataset.py`: builds the CVAT/Ultralytics import dataset (`images/`,
  `labels/`, `dataset.yaml`, `train.txt`) and zips it.
  Key args: `--overwrite`, `--include-val`, `--no-zip`.
- `push_to_cvat.py`: finds or creates the CVAT project and imports
  `hasty_exports/import_dataset.zip`.
  Key args: `--project-name` (required), `--format-name`, `--dataset-zip`.
- `view_sam_inputs.py`: visualizes bbox prompts from JSON before SAM. (FOR DEBUGGING)
- `view_outputs.py`: visualizes generated segmentation polygons on images. (FOR DEBUGGING)
- `env_utils.py`: minimal `.env` loader used by scripts.
