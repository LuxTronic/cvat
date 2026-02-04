FROM nvcr.io/nvidia/pytorch:21.08-py3

WORKDIR /yolov7

RUN apt-get update && apt-get remove -y \
    python3-opencv \
    libopencv* \
    && apt-get clean

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy YOLOv7 code
COPY . .

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
