# ui_service/app.py

import os
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from sqlalchemy import create_engine, text
import uvicorn
from decimal import Decimal

_HERE             = Path(__file__).parent
PROJECT_ROOT      = _HERE.parent
VIDEO_PATH        = PROJECT_ROOT / "videos" / "output.mp4"
DONE_FLAG         = PROJECT_ROOT / "videos" / ".processing_done"
HTML_PATH         = _HERE / "index.html"
LIVE_FRAME_PATH   = PROJECT_ROOT / "videos" / "latest_frame.jpg"
LAST_PAYLOAD_PATH = PROJECT_ROOT / "videos" / "last_kafka_payload.json"

DB_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost:5432/equipment_db")
engine = create_engine(DB_URL, pool_pre_ping=True)
app    = FastAPI(title="Equipment Monitor", version="1.0")


def _sanitize(value):
    """Recursively convert Decimal → float so JSONResponse can serialize it."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def run_query(sql: str, params: dict = None) -> list:
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            cols   = list(result.keys())
            rows   = [dict(zip(cols, row)) for row in result.fetchall()]
            return _sanitize(rows)          # ← sanitize before returning
    except Exception as e:
        print(f"[DB Error] {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# API — Summary & Timeline
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/summary")
def get_summary():
    rows = run_query("""
        SELECT
            equipment_id,
            equipment_class,
            current_state,
            current_activity,
            motion_source,
            total_active_secs,
            total_idle_secs,
            utilization_percent,
            idle_sessions_count,
            longest_idle_seconds,
            avg_idle_seconds,
            last_updated::text
        FROM equipment_summary
        ORDER BY equipment_id
    """)
    return JSONResponse(rows)


@app.get("/api/timeline/{equipment_id}")
def get_timeline(equipment_id: str):
    rows = run_query("""
        SELECT
            time::text,
            current_state,
            utilization_percent,
            current_idle_secs
        FROM equipment_events
        WHERE equipment_id = :eid
        ORDER BY time DESC
        LIMIT 200
    """, {"eid": equipment_id})
    return JSONResponse(rows)


# ─────────────────────────────────────────────────────────────────────────────
# API — Full Machine Data Card
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/machine/{equipment_id}")
def get_machine_full(equipment_id: str):
    """
    Complete data card for one machine — everything we know about it.
    Powers the expanded machine card in the dashboard.

    Returns:
        summary        → latest state, utilization, aggregate idle stats
        idle_sessions  → every individual idle stop (توقف) with duration
        activity_breakdown → % time spent in each activity
        recent_events  → last 20 state events for the timeline
    """
    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = run_query("""
        SELECT
            equipment_id, equipment_class,
            current_state, current_activity, motion_source,
            total_active_secs, total_idle_secs, utilization_percent,
            idle_sessions_count, longest_idle_seconds, avg_idle_seconds,
            last_updated::text
        FROM equipment_summary
        WHERE equipment_id = :eid
    """, {"eid": equipment_id})

    if not summary_rows:
        return JSONResponse({"error": f"Machine '{equipment_id}' not found"}, status_code=404)

    summary = summary_rows[0]

    # ── Individual idle sessions ───────────────────────────────────────────────
    idle_sessions = run_query("""
        SELECT
            session_number,
            started_at_ts,
            started_at_secs,
            duration_secs,
            duration_ts,
            recorded_at::text
        FROM equipment_idle_sessions
        WHERE equipment_id = :eid
        ORDER BY session_number ASC
    """, {"eid": equipment_id})

    # ── Activity breakdown ─────────────────────────────────────────────────────
    activity_rows = run_query("""
        SELECT
            current_activity,
            COUNT(*) AS event_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
        FROM equipment_events
        WHERE equipment_id = :eid
        GROUP BY current_activity
        ORDER BY event_count DESC
    """, {"eid": equipment_id})

    # ── Recent 20 state-change events ─────────────────────────────────────────
    recent_events = run_query("""
        SELECT
            time::text,
            current_state,
            current_activity,
            current_idle_secs,
            utilization_percent
        FROM equipment_events
        WHERE equipment_id = :eid
        ORDER BY time DESC
        LIMIT 20
    """, {"eid": equipment_id})

    # ── Derived totals ─────────────────────────────────────────────────────────
    total_tracked = (summary.get("total_active_secs") or 0) + (summary.get("total_idle_secs") or 0)

    return JSONResponse({
        "summary":           summary,
        "total_tracked_secs": round(total_tracked, 1),
        "idle_sessions":     idle_sessions,
        "activity_breakdown": activity_rows,
        "recent_events":     recent_events,
    })


# ─────────────────────────────────────────────────────────────────────────────
# API — Kafka + Health + Video Status
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/kafka/latest")
def get_last_kafka_payload():
    if not LAST_PAYLOAD_PATH.exists():
        return JSONResponse(
            {"error": "No payload yet — CV service may still be starting."},
            status_code=202
        )
    try:
        return JSONResponse(json.loads(LAST_PAYLOAD_PATH.read_text(encoding="utf-8")))
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/video/status")
def video_status():
    return JSONResponse({
        "processing_done": DONE_FLAG.exists(),
        "output_ready":    VIDEO_PATH.exists(),
    })


@app.get("/api/stats")
def get_stats():
    rows = run_query("""
        SELECT
            COUNT(*)                                        AS total_machines,
            SUM(CASE WHEN current_state='ACTIVE' THEN 1 ELSE 0 END) AS active_count,
            ROUND(AVG(utilization_percent)::numeric, 1)    AS avg_utilization,
            SUM(total_active_secs)                         AS total_active_secs,
            SUM(total_idle_secs)                           AS total_idle_secs,
            SUM(idle_sessions_count)                       AS total_idle_sessions
        FROM equipment_summary
    """)
    return JSONResponse(rows[0] if rows else {})


@app.get("/health")
def health_check():
    db_ok = False
    event_count = machine_count = idle_sessions_count = 0
    try:
        r1 = run_query("SELECT COUNT(*) AS n FROM equipment_events")
        event_count = r1[0]["n"] if r1 else 0
        r2 = run_query("SELECT COUNT(*) AS n FROM equipment_summary")
        machine_count = r2[0]["n"] if r2 else 0
        r3 = run_query("SELECT COUNT(*) AS n FROM equipment_idle_sessions")
        idle_sessions_count = r3[0]["n"] if r3 else 0
        db_ok = True
    except Exception as e:
        print(f"[Health] DB error: {e}")

    live_frame_age = None
    if LIVE_FRAME_PATH.exists():
        live_frame_age = round(time.time() - LIVE_FRAME_PATH.stat().st_mtime, 1)

    return JSONResponse({
        "status":               "ok" if db_ok else "degraded",
        "db_connected":         db_ok,
        "events_in_db":         event_count,
        "machines_tracked":     machine_count,
        "idle_sessions_saved":  idle_sessions_count,
        "live_frame_age_s":     live_frame_age,
        "kafka_payload_seen":   LAST_PAYLOAD_PATH.exists(),
        "processing_done":      DONE_FLAG.exists(),
    })


# ─────────────────────────────────────────────────────────────────────────────
# MJPEG Live Stream
# ─────────────────────────────────────────────────────────────────────────────

def make_placeholder(message: str) -> bytes:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, message, (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def generate_frames():
    target = 1.0 / 25
    while True:
        t0 = time.time()
        if not LIVE_FRAME_PATH.exists():
            jpeg = make_placeholder("Waiting for CV service...")
        else:
            frame = cv2.imread(str(LIVE_FRAME_PATH))
            if frame is None:
                time.sleep(0.02)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            jpeg = buf.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        sleep = target - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)


@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────────────────────────────────────────────────────
# HTML Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(f"<h1>index.html not found</h1><p>{HTML_PATH}</p>")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Dashboard:    http://localhost:8000")
    print("  Health:       http://localhost:8000/health")
    print("  Kafka feed:   http://localhost:8000/api/kafka/latest")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", use_colors=False)