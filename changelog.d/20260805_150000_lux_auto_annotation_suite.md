### Added

- \[Lux\] Auto-annotation from a YOLOv7 sidecar service, per-frame classification
  from a YOLOv8-cls sidecar service, interactive SAM mask generation, and
  Gemini-backed prompt annotation, with automatic model retraining once a task
  accumulates enough newly supervised frames
  (<https://github.com/LuxTronic/cvat/pull/7>)

- \[Lux\] CVAT can be served under a configurable base path (`CVAT_UI_BASE_PATH`,
  `CVAT_UI_URL`) so it can be embedded in the annotation workbench
  (<https://github.com/LuxTronic/cvat/pull/7>)

- \[Lux\] Standalone `scripts/hasty-cvat-polygons` pipeline that converts Hasty
  bounding boxes into segmentation polygons with SAM2 and imports them into CVAT
  (<https://github.com/LuxTronic/cvat/pull/7>)

- Draw control tooltips now name the shortcut that repeats the last drawing
  action, and Backspace deletes the active object alongside Del
  (<https://github.com/LuxTronic/cvat/pull/7>)
