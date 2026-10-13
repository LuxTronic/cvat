.PHONY: help lint test build

help:
	@echo "Targets for the YOLOv7 sidecar. The CVAT application"
	@echo "linters stay in .github/workflows/linters.yml."
	@$(MAKE) -C yolov7 help

lint test build:
	$(MAKE) -C yolov7 $@
