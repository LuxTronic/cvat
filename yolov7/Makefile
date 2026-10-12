IMAGE ?= yolov7:dev
PLATFORM ?= linux/amd64

.PHONY: help lint test build smoke

help:
	@echo "lint   hadolint the Dockerfile and byte-compile the sidecar"
	@echo "test   source contract tests, then the in-image smoke"
	@echo "build  build $(IMAGE) for $(PLATFORM)"
	@echo "smoke  /health, /infer, and a no-dataset /train inside $(IMAGE)"

lint:
	docker run --rm \
	  -v "$(CURDIR)/Dockerfile:/Dockerfile:ro" \
	  hadolint/hadolint:2.12.0 \
	  hadolint -t error /Dockerfile
	python3 -m py_compile server.py tests/test_upgrade_contract.py tests/smoke_infer.py

test: build
	python3 -m unittest discover -s tests -p 'test_*.py' -v
	$(MAKE) smoke

build:
	docker buildx build --platform $(PLATFORM) --load -t $(IMAGE) "$(CURDIR)"

smoke:
	docker run --rm --platform $(PLATFORM) \
	  -e PYTHONPATH=/yolov7 \
	  -w /yolov7 \
	  --entrypoint python \
	  -v "$(CURDIR)/tests/smoke_infer.py:/tmp/smoke_infer.py:ro" \
	  $(IMAGE) /tmp/smoke_infer.py
