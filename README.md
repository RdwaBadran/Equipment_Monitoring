# Equipment Utilization & Activity Classification Prototype

A real-time, microservices-based pipeline that processes video clips of
construction equipment, tracks their utilization states, classifies work
activities, and streams results through Apache Kafka to a live dashboard.

---

## Architecture Overview

```
Video File
│
▼
┌─────────────────────────────────────────┐
│           CV Service (cv_service/)      │
│  YOLOv8m Detection                      │
│  → BoT-SORT Tracker + Appearance Re-ID  │
│  → Optical Flow (3-region motion)       │
│  → Activity Classifier (rule-based)     │
│  → State Machine (frame-accurate time)  │
└─────────────┬───────────────────────────┘
              │ JSON payload (Kafka topic: equipment.events)
              ▼
┌─────────────────────────────────────────┐
│         Apache Kafka                    │
│         Topic: equipment.events         │
└──────┬──────────────────────┬───────────┘
       │                      │
       ▼                      ▼
┌─────────────┐      ┌─────────────────────┐
│ DB Consumer │      │    UI Service       │
│ (consumer_  │      │  FastAPI + HTML/JS  │
│  service/)  │      │  Live MJPEG stream  │
│             │      │  REST API           │
│ TimescaleDB │◄─────│  Polls DB every 2s  │
└─────────────┘      └─────────────────────┘
```

---

## Requirements

- Python 3.11+
- Docker Desktop (for Kafka + Zookeeper + TimescaleDB)
- ffmpeg (for H.264 re-encoding of output video)

---

## Setup & Run

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd equipment-monitor
python -m venv venv

# Windows
venv\Scripts\Activate

# Linux / macOS
source venv/bin/activate

# Install all dependencies (single command)
pip install -r requirements.txt
```

### 2. Start infrastructure (Kafka + TimescaleDB)

```bash
docker-compose up -d
```

Verify all 3 containers are running:

```bash
docker ps
# You should see: kafka, zookeeper, timescaledb
```

Wait ~15 seconds for Kafka to be fully ready before proceeding.

### 3. Add a video

```bash
pip install yt-dlp          # recommended downloader
python download_videos.py   # downloads a sample construction video
```

Or place your own `.mp4` file in the `videos/` folder and name it `video1.mp4`.

### 4. Run the full pipeline

```bash
python run.py
```

The browser opens automatically at **http://localhost:8000**

Options:

```bash
python run.py --video videos/video2.mp4   # use a different video
python run.py --every 3                   # process every 3rd frame (more accurate)
python run.py --confidence 0.25           # lower detection threshold
python run.py --no-browser                # skip auto-opening browser
```

### 5. Verify the pipeline is healthy

```bash
python health_check.py
```

This checks all 6 components end-to-end: Docker, Kafka, TimescaleDB (including schema),
UI API, live frame, and output video.

---

## Dashboard

Open **http://localhost:8000** to see:

- **Live Video Feed** — annotated MJPEG stream with bounding boxes, state labels, and utilization %
- **Machine Status Cards** — per-machine ACTIVE/INACTIVE state, activity, working time, idle time, and full idle stop history
- **Utilization Timeline Chart** — utilization % over video time
- **Live Kafka Feed** — the raw JSON payload streaming from the CV service through Kafka

---

## Component Details

### CV Service (`cv_service/`)

| File | Purpose |
|---|---|
| `tracker.py` | YOLOv8m BoT-SORT tracking + appearance-based Re-ID |
| `motion_analyzer.py` | Farneback optical flow, 3-region analysis (arm/cab/tracks) |
| `activity_classifier.py` | Rule-based activity classification |
| `state_machine.py` | Frame-accurate dwell time + idle session recording |
| `kafka_producer.py` | Sends JSON payloads to Kafka |
| `main.py` | Orchestrates the full pipeline |

### Consumer Service (`consumer_service/`)

Reads from Kafka topic `equipment.events` and writes to three TimescaleDB tables:

- `equipment_events` — full time-series event log (TimescaleDB hypertable)
- `equipment_summary` — latest state per machine (upserted on each event)
- `equipment_idle_sessions` — one row per completed idle stop (توقف), with duration and start time

### UI Service (`ui_service/`)

FastAPI backend serving:

| Endpoint | Description |
|---|---|
| `GET /` | HTML dashboard |
| `GET /video` | MJPEG live stream of annotated frames |
| `GET /api/summary` | Latest status for all machines |
| `GET /api/machine/{id}` | Full data card: summary + idle stop history + activity breakdown |
| `GET /api/timeline/{id}` | Time-series utilization for one machine |
| `GET /api/kafka/latest` | Last raw Kafka payload (pipeline proof) |
| `GET /api/video/status` | Whether video processing is complete |
| `GET /health` | End-to-end pipeline health check |

---

## Kafka Payload Format

The CV service sends this JSON structure to the `equipment.events` topic
for every processed frame, per machine detected:

```json
{
  "frame_id": 450,
  "equipment_id": "EX-001",
  "equipment_class": "excavator",
  "timestamp": "00:00:15.000",
  "utilization": {
    "current_state": "ACTIVE",
    "current_activity": "DIGGING",
    "motion_source": "arm_only"
  },
  "time_analytics": {
    "total_tracked_seconds": 15.0,
    "total_active_seconds": 12.5,
    "total_idle_seconds": 2.5,
    "current_idle_seconds": 0.0,
    "utilization_percent": 83.3,
    "idle_sessions_count": 1,
    "longest_idle_seconds": 2.5,
    "avg_idle_seconds": 2.5
  },
  "completed_idle_session": {
    "session_number": 1,
    "started_at_secs": 10.0,
    "started_at_ts": "00:00:10.000",
    "duration_secs": 2.5,
    "duration_ts": "00:02"
  }
}
```

> **Note:** `completed_idle_session` is only non-empty on the exact frame an idle stop ends
> (INACTIVE → ACTIVE transition). On all other frames it is an empty object `{}`.
> The DB consumer uses this field to insert exactly one row into `equipment_idle_sessions`
> per stop, preventing duplicate records.

---

## Technical Design Decisions

### Articulated Motion Problem

Standard whole-bbox motion analysis fails for excavators because the arm moves
while the tracks are stationary. Our solution splits each bounding box into
**3 vertical regions** and applies Farneback optical flow independently to each:

| Region | Maps to | Detection |
|---|---|---|
| Top 1/3 | Arm / boom | `arm_only` |
| Middle 1/3 | Cab / upper body | `cab_only` |
| Bottom 1/3 | Tracks / undercarriage | `tracks` |

If **any** region exceeds the motion threshold, the machine is classified as
`ACTIVE`. The `motion_source` field tells us which part triggered it, enabling
accurate activity classification even when the machine body is stationary.

**Trade-off:** Vertical splitting assumes the machine is roughly upright in the
frame. If the camera angle is oblique (> ~45°), region boundaries may not align
with actual arm/cab/tracks. A future improvement would be keypoint-based segmentation
to draw boundaries from the actual joint positions.

### Activity Classification

Rule-based on smoothed optical flow scores over an 8-frame history window:

| Activity | Trigger Condition |
|---|---|
| `DUMPING` | Top region score > 4.5 (fast downward arm release) |
| `DIGGING` | `motion_source == arm_only` AND top score > 1.5 |
| `SWINGING/LOADING` | `cab_only` motion OR (top > 1.2 AND middle > 1.0) |
| `WAITING` | All regions below threshold, or unclassified active state |

**Trade-off:** Rule-based logic is transparent and fast, but requires manual tuning
of thresholds per camera/equipment type. An ML classifier (e.g., a small CNN or
LSTM over the region score time-series) would generalize better across different
equipment models and distances, at the cost of needing labelled training data.

### Re-ID for Stable Dwell Time

When a machine disappears behind an obstacle and reappears, standard trackers
assign a new ID — resetting the dwell time counter. We solve this with a
**lost-track buffer**: every track stores an HSV color histogram (its visual
fingerprint). When a new detection appears, we compare its histogram to all
recently-lost tracks using cosine similarity. If similarity ≥ 0.60, the
original ID is restored and dwell time continues accumulating.

**Trade-off:** HSV histograms are fast to compute but can fail when two machines
of similar color are nearby. A stronger approach would use a learned re-id
embedding (e.g., OSNet), which is the direction production systems take.

### Frame-Based Time Tracking

All time calculations use `dt = process_every / fps` instead of wall-clock
time. This means the reported active/idle seconds reflect **actual video time**
regardless of how fast or slow the machine processes frames.

### Idle Session Tracking (Per-Stop Detail)

Each idle stop (توقف) is tracked individually with its start time and duration.
This is stored in `equipment_idle_sessions` and served by `/api/machine/{id}`.
The use case: since most construction equipment is rented, operators need to know
not just the total idle time but *when* each stop occurred and *how long* it
lasted — enabling cost attribution and scheduling analysis.

---

## Stopping

```bash
Ctrl+C
```

All services shut down gracefully: Kafka producer flushes, VideoWriter closes,
DB transactions commit, output is re-encoded to H.264.