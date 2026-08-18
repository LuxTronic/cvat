### Changed

- \[Lux\] Annotation autosave is enabled by default, so unsaved work is bounded by
  the autosave interval rather than by whenever the annotator last pressed Save.
  Browsers with settings stored before the default changed are migrated once; an
  explicit opt-out is preserved
  (<https://github.com/LuxTronic/cvat/pull/10>)

### Added

- `Job.frames.hasUnsavedChanges()` and `Task.frames.hasUnsavedChanges()` report
  whether frame meta (deleted or restored frames) has local changes pending
  (<https://github.com/LuxTronic/cvat/pull/10>)

### Fixed

- \[Lux\] Autosave, the unsaved-changes prompt shown when leaving the annotation
  page, and the saved-state indicator now account for deleted and restored
  frames. Previously they only inspected the annotation collection, so deleting a
  frame was not autosaved and navigating away discarded it without warning
  (<https://github.com/LuxTronic/cvat/pull/10>)
