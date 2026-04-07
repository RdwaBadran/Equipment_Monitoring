# ui_service/Dockerfile
FROM python:3.11-slim

# OpenCV needs these even in headless mode (for cv2.imread / imencode)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Environment defaults ──────────────────────────────────────────
ENV DATABASE_URL=postgresql://admin:admin123@timescaledb:5432/equipment_db

EXPOSE 8000

CMD ["python", "app.py"]