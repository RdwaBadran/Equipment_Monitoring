# consumer_service/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source code
COPY . .

# ── Environment defaults ──────────────────────────────────────────
# Override the DB host to use the docker-compose service name instead
# of localhost (which would refer to the container itself).
ENV DATABASE_URL=postgresql://admin:admin123@timescaledb:5432/equipment_db

CMD ["python", "db_consumer.py"]