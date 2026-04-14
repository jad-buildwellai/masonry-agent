FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    ffmpeg git build-essential \
    libsndfile1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Symlink python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install numpy first (critical ordering per RunPod expert)
RUN pip3 install --no-cache-dir numpy==1.26.4

# Install PyTorch (CUDA 12.1) + ultralytics + runpod
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    runpod

WORKDIR /app

# Bake trained model into image
COPY masonry.pt /masonry.pt

# Copy handler
COPY runpod_handler.py /app/handler.py

CMD ["python3", "/app/handler.py"]
