FROM nvcr.io/nvidia/pytorch:26.02-py3

WORKDIR /yolov8-cls

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-deps ultralytics

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

# Copy service code
COPY . .

# Download base classification weights
RUN mkdir -p /yolov8-cls/weights && \
    if [ ! -f /yolov8-cls/weights/yolov8n-cls.pt ]; then \
        echo "[YOLOv8-CLS] Downloading yolov8n-cls.pt weights..." && \
        wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-cls.pt \
            -O /yolov8-cls/weights/yolov8n-cls.pt ; \
    else \
        echo "[YOLOv8-CLS] yolov8n-cls.pt already present, skipping download"; \
    fi

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
