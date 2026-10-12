# syntax=docker/dockerfile:1

# CPU sidecar. Do not switch this base back to an NVIDIA PyTorch image:
# that image is Ubuntu 20.04 / Python 3.8 and it pulls CUDA libraries this
# process does not use.
# Digest is the ubuntu:24.04 tag resolved on 2026-10-11. Bump it on purpose.
FROM ubuntu:24.04@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PATH=/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Apt package versions stay unpinned. Ubuntu drops superseded versions from
# the archive, so a pin here fails the next security rebuild. Hadolint
# reports that choice as DL3008, which is below the error threshold.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        ca-certificates \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && pip install --no-cache-dir "pip==26.2.1"

WORKDIR /yolov7

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Public yolov7-tiny weights, stored under the name the trainer already uses.
# The size check rejects a truncated download.
RUN python -c "import urllib.request; from pathlib import Path; url='https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt'; dest=Path('/yolov7/weights/yolov7.pt'); dest.parent.mkdir(parents=True, exist_ok=True); urllib.request.urlretrieve(url, dest); size=dest.stat().st_size; expected=12639769; assert size == expected, 'yolov7-tiny.pt size %s != %s' % (size, expected)"

EXPOSE 8000

# Root on purpose. The models directory is a mounted volume, and a non-root
# user cannot write a cached model there when that volume is root-owned.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
