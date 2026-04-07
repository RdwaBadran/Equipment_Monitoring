# cv_service/Dockerfile
FROM python:3.11-slim

# System dependencies:
#   libgl1-mesa-glx  — OpenCV headless rendering
#   libglib2.0-0     — required by OpenCV on Linux
#   ffmpeg           — H.264 re-encoding of output video
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source code
COPY . .

# ── Environment defaults (override in docker-compose or with -e) ──
ENV VIDEO_PATH=/videos/video1.mp4
ENV MODEL_PATH=/app/yolov8m.pt
ENV CONFIDENCE=0.35
ENV PROCESS_EVERY=5

# The videos/ folder is expected to be mounted as a volume:
#   volumes:
#     - ./videos:/videos
# This lets the CV service read input videos and write output.mp4 / latest_frame.jpg
# without baking large video files into the image.

CMD ["python", "main.py"]