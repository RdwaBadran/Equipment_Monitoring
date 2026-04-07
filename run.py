# run.py  (project root)
# Single entry point for the entire equipment monitor project.
# Starts all services in the correct order:
#   1. DB Consumer  (listens to Kafka → saves to TimescaleDB)
#   2. CV Service   (processes video → sends to Kafka)
#   3. UI Service   (reads from TimescaleDB → shows dashboard)
#
# Usage:
#   python run.py
#   python run.py --video videos/video2.mp4
#   python run.py --video videos/video1.mp4 --every 3

import subprocess
import sys
import time
import argparse
import os
import signal
import webbrowser
from pathlib import Path


def check_docker():
    """Verify all required containers are running before starting services."""
    required = ["kafka", "zookeeper", "timescaledb"]
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        running = result.stdout.strip().split("\n")
        missing = [c for c in required if c not in running]

        if missing:
            print(f"[ERROR] These Docker containers are not running: {missing}")
            print("[ERROR] Please run:  docker-compose up -d")
            print("[ERROR] Then wait 15 seconds and try again.")
            sys.exit(1)

        print(f"[Launcher] Docker containers OK: {required}")

    except FileNotFoundError:
        print("[ERROR] Docker is not installed or not in PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError:
        print("[ERROR] Docker is not running. Please open Docker Desktop first.")
        sys.exit(1)


# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Equipment Monitor — Run All Services")
parser.add_argument(
    "--video",
    default="videos/video1.mp4",
    help="Path to video file (default: videos/video1.mp4)"
)
parser.add_argument(
    "--every",
    default="5",
    help="Process every Nth frame (default: 5, lower = slower but more accurate)"
)
parser.add_argument(
    "--confidence",
    default="0.35",
    help="YOLOv8 detection confidence threshold (default: 0.35)"
)
parser.add_argument(
    "--no-browser",
    action="store_true",
    help="Skip auto-opening the browser"
)
args = parser.parse_args()

check_docker()

# ── Validate video path ───────────────────────────────────────────────────────
if not Path(args.video).exists():
    print(f"[ERROR] Video not found: {args.video}")
    print("[ERROR] Run:  python download_videos.py")
    print("\nAvailable videos:")
    for v in Path("videos").glob("*.mp4"):
        print(f"  {v}")
    sys.exit(1)

# ── Find the venv Python automatically ───────────────────────────────────────
_win_python   = Path("venv/Scripts/python.exe")
_linux_python = Path("venv/bin/python")

if _win_python.exists():
    PYTHON = str(_win_python.resolve())
elif _linux_python.exists():
    PYTHON = str(_linux_python.resolve())
else:
    PYTHON = sys.executable

print(f"[Launcher] Using Python: {PYTHON}")

# ── Service definitions ───────────────────────────────────────────────────────
SERVICES = [
    {
        "name":  "DB Consumer",
        "cmd":   [PYTHON, "consumer_service/db_consumer.py"],
        "env":   {},
        "delay": 0,
    },
    {
        "name":  "CV Service",
        "cmd":   [PYTHON, "cv_service/main.py"],
        "env":   {
            "VIDEO_PATH":    args.video,
            "PROCESS_EVERY": args.every,
            "CONFIDENCE":    args.confidence,
        },
        "delay": 3,
    },
    {
        "name":  "UI Service",
        "cmd":   [PYTHON, "ui_service/app.py"],
        "env":   {},
        "delay": 5,
    },
]


def start_service(service: dict) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(service["env"])
    print(f"[Launcher] Starting {service['name']}...")
    proc = subprocess.Popen(
        service["cmd"],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    print(f"[Launcher] {service['name']} started (PID: {proc.pid})")
    return proc


def stop_service(name: str, proc: subprocess.Popen):
    print(f"[Launcher] Stopping {name} (PID: {proc.pid})...")
    try:
        if name == "UI Service":
            proc.kill()
        elif sys.platform == "win32":
            proc.send_signal(signal.CTRL_C_EVENT)
        else:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt, OSError):
        proc.kill()
    print(f"[Launcher] {name} stopped.")


def main():
    print("=" * 55)
    print("Equipment Monitor — Starting All Services")
    print("=" * 55)
    print(f"  Video:      {args.video}")
    print(f"  Every Nth:  {args.every} frames")
    print(f"  Confidence: {args.confidence}")
    print("=" * 55)
    print()

    processes = []

    try:
        for service in SERVICES:
            if service["delay"] > 0:
                print(f"[Launcher] Waiting {service['delay']}s before starting {service['name']}...")
                time.sleep(service["delay"])

            proc = start_service(service)
            processes.append((service["name"], proc))

        print()
        print("=" * 55)
        print("All services running.")
        print()
        # FIX: Always show localhost:8000 — 0.0.0.0:8000 is NOT a valid browser URL
        print("  Dashboard URL:  http://localhost:8000")
        print("  Health check:   http://localhost:8000/health")
        print("  Video stream:   http://localhost:8000/video")
        print()
        print("Press Ctrl+C to stop everything.")
        print("=" * 55)
        print()

        # Auto-open browser after a short delay (UI service needs time to start)
        if not args.no_browser:
            time.sleep(3)
            try:
                webbrowser.open("http://localhost:8000")
                print("[Launcher] Browser opened at http://localhost:8000")
            except Exception:
                pass  # Non-critical — user can open manually

        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    if ret == 0:
                        print(f"\n[Launcher] {name} finished successfully.")
                    else:
                        print(f"\n[Launcher] WARNING: {name} stopped unexpectedly (exit code {ret})")
                    print("[Launcher] Shutting down all services...")
                    raise KeyboardInterrupt
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n[Launcher] Stopping all services...")
        for name, proc in reversed(processes):
            stop_service(name, proc)
        print("[Launcher] All services stopped. Goodbye.")


if __name__ == "__main__":
    main()