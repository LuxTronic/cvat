FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /sam-service

# Remove conflicting OpenCV packages (same pattern as existing services)
RUN apt-get update && apt-get remove -y \
    python3-opencv \
    libopencv* \
    && apt-get clean

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Segment Anything
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/segment-anything.git

# Copy service code
COPY . .

# Pre-download SAM checkpoint (can be overridden via env at runtime)
ARG SAM_CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
RUN mkdir -p /sam-service/weights && \
    if [ ! -f /sam-service/weights/sam_vit_b_01ec64.pth ]; then \
        echo "[SAM] Downloading checkpoint..." && \
        wget -q ${SAM_CHECKPOINT_URL} -O /sam-service/weights/sam_vit_b_01ec64.pth ; \
    else \
        echo "[SAM] Checkpoint already present, skipping download"; \
    fi

ENV SAM_MODEL_TYPE=vit_b
ENV SAM_CHECKPOINT=/sam-service/weights/sam_vit_b_01ec64.pth
ENV SAM_CHECKPOINT_URL=${SAM_CHECKPOINT_URL}

EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
