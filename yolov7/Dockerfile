FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /yolov7



# Remove conflicting OpenCV packages
RUN apt-get update && apt-get remove -y \
    python3-opencv \
    libopencv* \
    && apt-get clean

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn \
    requests

# Copy YOLOv7 code
COPY . .

# ---- INSTALL YOLOv7 BASE WEIGHTS (CRITICAL FIX) ----
RUN mkdir -p /yolov7/weights && \
    if [ ! -f /yolov7/weights/yolov7.pt ]; then \
        echo "[YOLO] Downloading yolov7.pt weights..." && \
        wget -q https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt \
            -O /yolov7/weights/yolov7.pt ; \
    else \
        echo "[YOLO] yolov7.pt already present, skipping download"; \
    fi

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
