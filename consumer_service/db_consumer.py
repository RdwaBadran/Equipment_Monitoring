# consumer_service/db_consumer.py
# Listens to Kafka topic and saves equipment events + idle sessions to TimescaleDB.

import json
import time
import os
from pathlib import Path
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DB_URL        = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost:5432/equipment_db")
KAFKA_TOPIC   = "equipment.events"
KAFKA_SERVERS = os.getenv("KAFKA_SERVERS", "localhost:9092").split(",")
GROUP_ID      = "db-consumer-group"

_HERE             = Path(__file__).parent
PROJECT_ROOT      = _HERE.parent
LAST_PAYLOAD_PATH = PROJECT_ROOT / "videos" / "last_kafka_payload.json"


def create_tables(engine):
    with engine.connect() as conn:

        # ── Time-series events (one row per processed frame per machine) ────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS equipment_events (
                time                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                frame_id            INTEGER,
                equipment_id        TEXT,
                equipment_class     TEXT,
                current_state       TEXT,
                current_activity    TEXT,
                motion_source       TEXT,
                total_tracked_secs  FLOAT,
                total_active_secs   FLOAT,
                total_idle_secs     FLOAT,
                current_idle_secs   FLOAT,
                utilization_percent FLOAT
            );
        """))

        try:
            conn.execute(text("""
                SELECT create_hypertable('equipment_events','time',if_not_exists=>TRUE);
            """))
        except Exception:
            pass

        # ── Latest state per machine (upserted) ──────────────────────────────────
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS equipment_summary (
                equipment_id          TEXT PRIMARY KEY,
                equipment_class       TEXT,
                current_state         TEXT,
                current_activity      TEXT,
                motion_source         TEXT,
                total_active_secs     FLOAT DEFAULT 0,
                total_idle_secs       FLOAT DEFAULT 0,
                utilization_percent   FLOAT DEFAULT 0,
                idle_sessions_count   INTEGER DEFAULT 0,
                longest_idle_seconds  FLOAT DEFAULT 0,
                avg_idle_seconds      FLOAT DEFAULT 0,
                last_updated          TIMESTAMPTZ DEFAULT NOW()
            );
        """))

        # ── MIGRATIONS: Idempotent — safe to run every startup ───────────────────
        # These ALTER TABLE statements add columns that may be missing on existing
        # databases created before the schema was updated.
        # ADD COLUMN IF NOT EXISTS does nothing if the column already exists.
        migrations = [
            # Added in v1.1 — motion source tracking
            "ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS motion_source TEXT",
            # Added in v1.2 — idle session statistics (THE FIX)
            "ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS idle_sessions_count  INTEGER DEFAULT 0",
            "ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS longest_idle_seconds FLOAT   DEFAULT 0.0",
            "ALTER TABLE equipment_summary ADD COLUMN IF NOT EXISTS avg_idle_seconds     FLOAT   DEFAULT 0.0",
        ]
        for sql in migrations:
            conn.execute(text(sql))

        # ── Individual idle sessions — one row per completed stop ─────────────────
        # Each row = one idle stop for one machine.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS equipment_idle_sessions (
                id               SERIAL PRIMARY KEY,
                recorded_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                equipment_id     TEXT        NOT NULL,
                equipment_class  TEXT,
                session_number   INTEGER,
                started_at_secs  FLOAT,
                started_at_ts    TEXT,
                duration_secs    FLOAT,
                duration_ts      TEXT
            );
        """))

        # Index for fast lookups by machine
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_idle_sessions_equip
            ON equipment_idle_sessions (equipment_id, session_number);
        """))

        conn.commit()
        print("[DB] Tables ready.")
        print("[DB]   equipment_events        → time-series event log")
        print("[DB]   equipment_summary       → latest state per machine")
        print("[DB]   equipment_idle_sessions → individual idle stop records")


def save_event(session, payload: dict):
    """Save one Kafka message to the DB. Handles events + idle sessions."""
    util  = payload["time_analytics"]
    state = payload["utilization"]
    ta    = payload["time_analytics"]

    # ── 1. Insert into time-series events ────────────────────────────────────
    session.execute(text("""
        INSERT INTO equipment_events (
            frame_id, equipment_id, equipment_class,
            current_state, current_activity, motion_source,
            total_tracked_secs, total_active_secs, total_idle_secs,
            current_idle_secs, utilization_percent
        ) VALUES (
            :frame_id, :equipment_id, :equipment_class,
            :current_state, :current_activity, :motion_source,
            :total_tracked, :total_active, :total_idle,
            :current_idle, :utilization
        )
    """), {
        "frame_id":         payload["frame_id"],
        "equipment_id":     payload["equipment_id"],
        "equipment_class":  payload["equipment_class"],
        "current_state":    state["current_state"],
        "current_activity": state["current_activity"],
        "motion_source":    state["motion_source"],
        "total_tracked":    util["total_tracked_seconds"],
        "total_active":     util["total_active_seconds"],
        "total_idle":       util["total_idle_seconds"],
        "current_idle":     util["current_idle_seconds"],
        "utilization":      util["utilization_percent"],
    })

    # ── 2. Upsert summary (includes idle session stats + motion_source) ───────
    session.execute(text("""
        INSERT INTO equipment_summary (
            equipment_id, equipment_class,
            current_state, current_activity, motion_source,
            total_active_secs, total_idle_secs,
            utilization_percent,
            idle_sessions_count, longest_idle_seconds, avg_idle_seconds,
            last_updated
        ) VALUES (
            :equipment_id, :equipment_class,
            :current_state, :current_activity, :motion_source,
            :total_active, :total_idle,
            :utilization,
            :idle_count, :longest_idle, :avg_idle,
            NOW()
        )
        ON CONFLICT (equipment_id) DO UPDATE SET
            current_state         = EXCLUDED.current_state,
            current_activity      = EXCLUDED.current_activity,
            motion_source         = EXCLUDED.motion_source,
            total_active_secs     = EXCLUDED.total_active_secs,
            total_idle_secs       = EXCLUDED.total_idle_secs,
            utilization_percent   = EXCLUDED.utilization_percent,
            idle_sessions_count   = EXCLUDED.idle_sessions_count,
            longest_idle_seconds  = EXCLUDED.longest_idle_seconds,
            avg_idle_seconds      = EXCLUDED.avg_idle_seconds,
            last_updated          = NOW()
    """), {
        "equipment_id":     payload["equipment_id"],
        "equipment_class":  payload["equipment_class"],
        "current_state":    state["current_state"],
        "current_activity": state["current_activity"],
        "motion_source":    state["motion_source"],
        "total_active":     util["total_active_seconds"],
        "total_idle":       util["total_idle_seconds"],
        "utilization":      util["utilization_percent"],
        "idle_count":       ta.get("idle_sessions_count", 0),
        "longest_idle":     ta.get("longest_idle_seconds", 0),
        "avg_idle":         ta.get("avg_idle_seconds", 0),
    })

    # ── 3. Save completed idle session (only when one just finished) ──────────
    # The state machine only populates completed_idle_session on the exact
    # frame an idle stop ends (INACTIVE → ACTIVE transition).
    idle_session = payload.get("completed_idle_session", {})
    if idle_session:
        session.execute(text("""
            INSERT INTO equipment_idle_sessions (
                equipment_id, equipment_class,
                session_number,
                started_at_secs, started_at_ts,
                duration_secs,   duration_ts
            ) VALUES (
                :equipment_id, :equipment_class,
                :session_number,
                :started_at_secs, :started_at_ts,
                :duration_secs,   :duration_ts
            )
        """), {
            "equipment_id":    payload["equipment_id"],
            "equipment_class": payload["equipment_class"],
            "session_number":  idle_session["session_number"],
            "started_at_secs": idle_session["started_at_secs"],
            "started_at_ts":   idle_session["started_at_ts"],
            "duration_secs":   idle_session["duration_secs"],
            "duration_ts":     idle_session["duration_ts"],
        })
        print(
            f"[Consumer] Idle session saved: {payload['equipment_id']} "
            f"stop #{idle_session['session_number']} — "
            f"{idle_session['duration_ts']} at {idle_session['started_at_ts']}"
        )

    session.commit()


def write_last_payload(payload: dict):
    try:
        enriched = dict(payload)
        enriched["_received_at"] = datetime.utcnow().strftime("%H:%M:%S UTC")
        LAST_PAYLOAD_PATH.write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[Consumer] Warning: could not write last payload: {e}")


def connect_kafka(retries=10, delay=3.0):
    print(f"[Kafka] Connecting to {KAFKA_SERVERS}...")
    for attempt in range(1, retries + 1):
        try:
            print(f"[Kafka] Connecting... attempt {attempt}/{retries}")
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_SERVERS,
                group_id=GROUP_ID,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True
            )
            print("[Kafka] Connected successfully.")
            return consumer
        except NoBrokersAvailable:
            if attempt < retries:
                print(f"[Kafka] Not ready, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise RuntimeError("[Kafka] Could not connect after all retries.")


def main():
    print("=" * 50)
    print("DB Consumer — Starting")
    print("=" * 50)

    engine  = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    create_tables(engine)
    consumer = connect_kafka()

    print(f"[Consumer] Listening to topic: {KAFKA_TOPIC}")
    print("[Consumer] Waiting for messages... (Ctrl+C to stop)")

    count      = 0
    idle_saved = 0

    try:
        for message in consumer:
            payload = message.value
            try:
                save_event(session, payload)
                write_last_payload(payload)
                count += 1
                if payload.get("completed_idle_session"):
                    idle_saved += 1
            except Exception as e:
                session.rollback()
                print(f"[Consumer] DB error on message {count}: {e}")
                continue

            if count % 100 == 0:
                print(
                    f"[Consumer] Events: {count:,} | "
                    f"Idle sessions saved: {idle_saved} | "
                    f"Latest: {payload['equipment_id']} "
                    f"→ {payload['utilization']['current_state']}"
                )

    except KeyboardInterrupt:
        print(f"\n[Consumer] Stopped.")
        print(f"[Consumer] Events saved:        {count:,}")
        print(f"[Consumer] Idle sessions saved: {idle_saved}")
    finally:
        session.close()
        consumer.close()


if __name__ == "__main__":
    main()