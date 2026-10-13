# Plan: rebuild the YOLOv7 sidecar on Ubuntu 24.04 and Python 3.12

This round changes the image build in this repository. It does not publish
or deploy an image.

## Outcome

`yolov7/Dockerfile` produces a `linux/amd64` image on Ubuntu 24.04 and
Python 3.12 with a CPU torch wheel. `GET /health` reports CPU.
`POST /infer` still returns a JSON list of
`{class_id, confidence, xc, yc, w, h}`. `POST /train` still accepts the
same JSON body and still refuses a task that has no `data.yaml`.

## Why a base-image swap is not enough

The previous base is `nvcr.io/nvidia/pytorch:21.08-py3` (Ubuntu 20.04,
Python 3.8, torch 1.10). This process does not use a GPU. A CUDA base
image is the wrong target. Torch 2.x also rejects three patterns in this
tree:

- `torch.meshgrid([a, b])` without `indexing`
- `torch.load` of a pickled checkpoint, because torch 2.6 defaults
  `weights_only=True`
- `np.int`, removed in NumPy 1.24
- `utils/loss.py` empty-target tensors hardcoded to `cuda:0`, which fails
  CPU training

`opencv-python-headless==4.5.5.64` has no Python 3.12 wheel. The old
`numpy<1.24` cap exists only to keep that NumPy alias.

## Pins for this round

| Piece | Pin | Why |
| --- | --- | --- |
| OS | `ubuntu:24.04` @ `534baea6a22c` | Distro Python 3.12. The digest is pinned in the Dockerfile. |
| torch | `2.14.1+cpu` | CPU wheel from the PyTorch CPU index. The PyPI Linux wheel is CUDA. |
| torchvision | `0.29.1+cpu` | Matches torch 2.14. |
| OpenCV | `opencv-python-headless==4.14.0.94` | Newest 4.x wheel with a Python 3.12 build. |
| pip | `26.2.1` | Current pip, pinned in the image. |
| baked weights | `/yolov7/weights/yolov7.pt` | Public yolov7-tiny file. The build checks its size and SHA-256. |

The server runs as uid 1000, the same identity `yolov8-cls` uses.
The entrypoint starts as root, gives `/models` to that user, then drops privileges.
That updates a volume left by a root sidecar, and a new empty volume is writable.
`USE_HALF` stays tied to CUDA availability. This image does not turn it on.

The HTTP contract and checkpoint layout stay as they are in `server.py`.
Do not change the detector architecture.

## This round

- [x] Replace the Dockerfile base and pin the CPU stack above.
- [x] Pass `indexing="ij"` at every `torch.meshgrid` call.
- [x] Pass `weights_only=False` at every `torch.load` of a checkpoint or cache.
- [x] Replace `np.int` with `np.int64`.
- [x] Build empty training targets on `targets.device` instead of `cuda:0`.
- [x] Add `make help`, `make lint`, `make test`, and `make build`.
- [x] Contract tests, plus an in-image smoke of `/health`, `/infer`, and `/train`.
- [x] `make lint`, `make test`, and `make build` pass on this branch.

## Not this round

- [ ] A parity check that loaded checkpoints still return the same class
  and nearly the same boxes.
- [ ] A tiny `/train` that writes a `best.pt` `/infer` can load.
- [ ] Publishing or deploying the image.

## Validation

```bash
make lint
make test
make build
```

`make test` runs the source contract tests and, after `make build`, the
in-image smoke. The smoke copies the baked tiny weights onto a fake
`task_1/active` path and posts one JPEG. A gray image may return an empty
list. An empty list is a successful detection call. A 500 is not.

Local result on 2026-10-11: `make lint` passed, `make build` produced a
local image of about 1.9 GB, and the smoke reported Python 3.12.3, torch
`2.14.1+cpu`, CUDA false, `/infer` HTTP 200 with 0 boxes, and `/train`
HTTP 400.

## Risks

- Boxes will not be bit-identical to torch 1.10.
- OpenCV 4.14 can decode a byte differently from 4.5.5.
- `pip install -r requirements.txt` must keep the `+cpu` local version.
  Dropping that suffix installs the CUDA wheel from PyPI.
