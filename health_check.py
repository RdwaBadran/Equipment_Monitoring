#!/usr/bin/env python3
"""
health_check.py
End-to-end pipeline validation for Equipment Monitor.

Checks every component in the pipeline:
  Docker → Kafka → TimescaleDB → UI API → Live Frame

Usage:
    python health_check.py
    python health_check.py --verbose
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Try to import optional libs — we check each component independently
try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    from kafka import KafkaConsumer, KafkaAdminClient
    KAFKA_OK = True
except ImportError:
    KAFKA_OK = False

try:
    from sqlalchemy import create_engine, text
    SQL_OK = True
except ImportError:
    SQL_OK = False


# ── Config (matches .env defaults) ────────────────────────────────────────────
KAFKA_SERVERS = os.getenv("KAFKA_SERVERS", "localhost:9092").split(",")
KAFKA_TOPIC   = os.getenv("KAFKA_TOPIC",   "equipment.events")
DB_URL        = os.getenv("DATABASE_URL",   "postgresql://admin:admin123@localhost:5432/equipment_db")
UI_URL        = "http://localhost:8000"
VIDEOS_DIR    = Path(__file__).parent / "videos"

PASS  = "  ✓"
FAIL  = "  ✗"
WARN  = "  ⚠"


def check_docker():
    """Verify all required Docker containers are running."""
    print("\n[1/6] Docker Containers")
    required = ["kafka", "zookeeper", "timescaledb"]
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        running = result.stdout.strip().split("\n")
        all_ok  = True
        for c in required:
            if c in running:
                print(f"{PASS} {c} is running")
            else:
                print(f"{FAIL} {c} is NOT running  → run: docker-compose up -d")
                all_ok = False
        return all_ok
    except FileNotFoundError:
        print(f"{FAIL} Docker not found in PATH")
        return False
    except subprocess.CalledProcessError:
        print(f"{FAIL} Docker daemon is not running")
        return False


def check_kafka():
    """Verify Kafka is reachable and the topic exists."""
    print("\n[2/6] Kafka Broker")
    if not KAFKA_OK:
        print(f"{WARN} kafka-python not installed — skipping Kafka check")
        return None

    try:
        admin = KafkaAdminClient(bootstrap_servers=KAFKA_SERVERS, request_timeout_ms=5000)
        topics = admin.list_topics()
        admin.close()

        print(f"{PASS} Kafka reachable at {KAFKA_SERVERS}")

        if KAFKA_TOPIC in topics:
            print(f"{PASS} Topic '{KAFKA_TOPIC}' exists")
            return True
        else:
            print(f"{WARN} Topic '{KAFKA_TOPIC}' does not exist yet")
            print(f"      It will be auto-created when CV service starts producing.")
            return True  # Not a hard failure — topic auto-creates

    except Exception as e:
        print(f"{FAIL} Cannot connect to Kafka: {e}")
        print(f"      Expected: {KAFKA_SERVERS}")
        return False


def check_database():
    """Verify TimescaleDB is reachable, all 3 tables exist, and schema is correct."""
    print("\n[3/6] TimescaleDB")
    if not SQL_OK:
        print(f"{WARN} sqlalchemy not installed — skipping DB check")
        return None

    try:
        engine = create_engine(DB_URL, connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            # ── Check all 3 tables exist ─────────────────────────────────────
            tables_result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN (
                    'equipment_events',
                    'equipment_summary',
                    'equipment_idle_sessions'
                  )
            """))
            found_tables = [row[0] for row in tables_result]

            print(f"{PASS} Database reachable")

            all_tables_ok = True
            for t in ["equipment_events", "equipment_summary", "equipment_idle_sessions"]:
                if t in found_tables:
                    print(f"{PASS} Table '{t}' exists")
                else:
                    print(f"{WARN} Table '{t}' missing — will be created when consumer starts")
                    all_tables_ok = False

            # ── Check motion_source column exists (the schema migration) ──────
            if "equipment_summary" in found_tables:
                col_result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'equipment_summary'
                      AND column_name = 'motion_source'
                """))
                if col_result.fetchone():
                    print(f"{PASS} Column 'motion_source' exists in equipment_summary")
                else:
                    print(
                        f"{FAIL} Column 'motion_source' is MISSING from equipment_summary\n"
                        f"      FIX: restart db_consumer.py — it will auto-add the column."
                    )
                    all_tables_ok = False

            # ── Row counts ────────────────────────────────────────────────────
            event_count = conn.execute(
                text("SELECT COUNT(*) FROM equipment_events")
            ).scalar() if "equipment_events" in found_tables else 0
            print(f"{PASS} Events in DB: {event_count:,}")
            if event_count == 0:
                print(f"      DB is empty — start run.py to populate it")

            machine_count = conn.execute(
                text("SELECT COUNT(*) FROM equipment_summary")
            ).scalar() if "equipment_summary" in found_tables else 0
            print(f"{PASS} Machines tracked: {machine_count}")

            idle_session_count = conn.execute(
                text("SELECT COUNT(*) FROM equipment_idle_sessions")
            ).scalar() if "equipment_idle_sessions" in found_tables else 0
            print(f"{PASS} Idle sessions saved: {idle_session_count}")

            return all_tables_ok

    except Exception as e:
        print(f"{FAIL} Database error: {e}")
        print(f"      URL: {DB_URL}")
        return False


def check_ui_api():
    """Verify the UI FastAPI service is responding."""
    print("\n[4/6] UI Service (FastAPI)")
    if not REQUESTS_OK:
        print(f"{WARN} requests not installed — skipping UI check")
        return None

    try:
        # Health endpoint
        r = requests.get(f"{UI_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"{PASS} UI service reachable at {UI_URL}")
            status = data.get("status", "unknown")
            icon   = PASS if status == "ok" else WARN
            print(f"{icon} Pipeline status: {status}")
            print(f"{PASS} Events in DB (via API): {data.get('events_in_db', 0):,}")
            print(f"{PASS} Machines tracked: {data.get('machines_tracked', 0)}")
            print(f"{PASS} Idle sessions saved: {data.get('idle_sessions_saved', 0)}")
            return data.get("db_connected", False)
        else:
            print(f"{FAIL} UI returned HTTP {r.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"{FAIL} Cannot reach UI at {UI_URL}")
        print(f"      Is ui_service/app.py running?")
        return False
    except Exception as e:
        print(f"{FAIL} UI check error: {e}")
        return False


def check_live_frame():
    """Verify the CV service is writing annotated frames."""
    print("\n[5/6] Live Frame (CV Service output)")
    frame_path = VIDEOS_DIR / "latest_frame.jpg"

    if not frame_path.exists():
        print(f"{WARN} No live frame yet: {frame_path}")
        print(f"      CV service may still be starting up.")
        return None

    age = time.time() - frame_path.stat().st_mtime
    if age < 5:
        print(f"{PASS} Live frame exists (age: {age:.1f}s — CV service is active)")
        return True
    elif age < 60:
        print(f"{WARN} Live frame exists but is {age:.0f}s old — CV service may have paused")
        return True
    else:
        print(f"{FAIL} Live frame is {age:.0f}s old — CV service may not be running")
        return False


def check_output_video():
    """Verify the output annotated video exists."""
    print("\n[6/6] Output Video")
    output_path = VIDEOS_DIR / "output.mp4"
    done_flag   = VIDEOS_DIR / ".processing_done"

    if done_flag.exists():
        print(f"{PASS} Processing complete (.processing_done flag set)")
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"{PASS} Output video exists: {output_path} ({size_mb:.1f} MB)")
        else:
            print(f"{WARN} Processing done flag set but output.mp4 not found")
        return True
    else:
        print(f"{WARN} Processing still in progress (no .processing_done flag yet)")
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"{PASS} Partial output video: {size_mb:.1f} MB written so far")
        return None


def main():
    parser = argparse.ArgumentParser(description="Equipment Monitor — Health Check")
    parser.add_argument("--verbose", action="store_true", help="Show extra detail")
    args = parser.parse_args()

    print("=" * 55)
    print("  Equipment Monitor — Pipeline Health Check")
    print("=" * 55)

    results = {
        "Docker":       check_docker(),
        "Kafka":        check_kafka(),
        "TimescaleDB":  check_database(),
        "UI API":       check_ui_api(),
        "Live Frame":   check_live_frame(),
        "Output Video": check_output_video(),
    }

    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)

    all_critical_ok = True
    for name, status in results.items():
        if status is True:
            icon = "✓"
        elif status is False:
            icon = "✗"
            all_critical_ok = False
        else:
            icon = "–"   # skipped / None
        print(f"  {icon}  {name}")

    print()
    if all_critical_ok:
        print("All checks passed. Pipeline is healthy.")
    else:
        print("Some checks failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()