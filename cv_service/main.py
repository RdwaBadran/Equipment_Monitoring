# cv_service/main.py
# Entry point for the CV microservice.
# Reads video frame by frame and runs the full pipeline:
#   1. Track + Re-ID   (YOLOv8 BoT-SORT — detection is internal to the tracker)
#   2. Analyze motion  (Optical Flow, 3-region including cab_only)
#   3. Classify activity (Rule-based)
#   4. Update state machine (Frame-accurate dwell time)
#   5. Send to Kafka
#   6. Write annotated output video (mp4v → ffmpeg re-encode to H264)

import cv2
import time
import os
import sys
import shutil
import subprocess

from pathlib import Path
from ultralytics import YOLO

from tracker             import Tracker
from motion_analyzer     import MotionAnalyzer
from activity_classifier import ActivityClassifier
from state_machine       import StateMachine
from kafka_producer      import EquipmentProducer

# ── Resolve paths relative to THIS file ─────────────────────────────────────
_CV_DIR      = Path(__file__).parent          # .../cv_service/
PROJECT_ROOT = _CV_DIR.parent                 # .../equipment-monitor/
VIDEOS_DIR   = PROJECT_ROOT / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# ── Configuration (overridable via environment variables) ────────────────────
VIDEO_PATH    = os.getenv("VIDEO_PATH",    str(PROJECT_ROOT / "videos" / "video1.mp4"))
MODEL_PATH    = os.getenv("MODEL_PATH",    str(_CV_DIR / "yolov8m.pt"))
CONFIDENCE    = float(os.getenv("CONFIDENCE",    "0.35"))
PROCESS_EVERY = int(os.getenv("PROCESS_EVERY",   "5"))

RESIZE_WIDTH  = 960
RESIZE_HEIGHT = 540

# Path where CV service writes the latest annotated frame.
# The UI service reads this file to serve the live MJPEG stream.
LIVE_FRAME_PATH = VIDEOS_DIR / "latest_frame.jpg"

# ── Drawing helpers ──────────────────────────────────────────────────────────
STATE_COLORS = {
    "ACTIVE":   (0, 255, 0),
    "INACTIVE": (0, 0, 255),
}

ACTIVITY_COLORS = {
    "DIGGING":          (255, 165, 0),
    "SWINGING/LOADING": (255, 255, 0),
    "DUMPING":          (0, 165, 255),
    "WAITING":          (128, 128, 128),
}


def draw_overlay(frame, tracked_item, state, payload):
    x1, y1, x2, y2  = tracked_item["bbox"]
    current_state    = state.current_state
    current_activity = state.current_activity
    eq_id            = state.equipment_id

    box_color = STATE_COLORS.get(current_state, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Cyan dot = Re-ID match restored this track
    if tracked_item.get("is_reid"):
        cv2.circle(frame, (x1 + 10, y1 + 10), 6, (0, 255, 255), -1)

    label      = f"{eq_id} | {current_state} | {current_activity}"
    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
    label_y    = max(y1 - 8, 20)
    cv2.rectangle(
        frame,
        (x1, label_y - label_size[1] - 4),
        (x1 + label_size[0] + 4, label_y + 4),
        box_color, -1
    )
    cv2.putText(
        frame, label,
        (x1 + 2, label_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (0, 0, 0), 1, cv2.LINE_AA
    )

    time_info = payload["time_analytics"]
    idle_secs = time_info["current_idle_seconds"]
    if current_state == "INACTIVE" and idle_secs > 0:
        cv2.putText(
            frame, f"IDLE: {idle_secs:.0f}s",
            (x1 + 2, y2 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 0, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        frame, f"UTIL: {time_info['utilization_percent']}%",
        (x1 + 2, y2 + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (255, 255, 255), 1, cv2.LINE_AA
    )
    return frame


def draw_dashboard(frame, all_states: dict):
    panel_x, panel_y = 10, 10
    line_h  = 22
    padding = 8

    lines = ["=== EQUIPMENT DASHBOARD ==="]
    for state in all_states.values():
        lines.append(
            f"{state.equipment_id}: {state.current_state} | "
            f"Active:{state.total_active:.0f}s | "
            f"Idle:{state.total_idle:.0f}s | "
            f"Util:{state.utilization_percent}%"
        )

    panel_h = len(lines) * line_h + padding * 2
    panel_w = 520

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 0, 0), -1
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(lines):
        y     = panel_y + padding + (i + 1) * line_h
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        cv2.putText(
            frame, line,
            (panel_x + padding, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            color, 1, cv2.LINE_AA
        )
    return frame


def _make_writer(out_path: str, fps: float) -> cv2.VideoWriter:
    """
    Create a VideoWriter using mp4v codec.

    FIX: Previously tried 'avc1' (H.264/OpenH264) first, which produced
    noisy error messages about missing DLLs on Windows even when it worked.
    We now use 'mp4v' (MPEG-4) directly — it initializes silently on all
    platforms. The output is then re-encoded to proper H.264 by ffmpeg at
    the end of processing, so browser/VSCode compatibility is unchanged.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (RESIZE_WIDTH, RESIZE_HEIGHT))
    if writer.isOpened():
        print(f"[Video] Writer opened with codec: mp4v (will re-encode to H.264 at end)")
        return writer

    raise RuntimeError(
        "[Video] Could not open VideoWriter with mp4v. "
        "Check that OpenCV was built with video write support."
    )


def _reencode_h264(src: str, dst: str) -> bool:
    """
    Re-encode src to dst using ffmpeg with H.264 + yuv420p.
    yuv420p is required for compatibility with QuickTime, VSCode, and
    most web browsers. Returns True if ffmpeg succeeded.
    """
    if not shutil.which("ffmpeg"):
        print("[Video] ffmpeg not found — output stays as mp4v.")
        print("[Video] Install ffmpeg for H.264 output: https://ffmpeg.org/download.html")
        return False

    print("[Video] Re-encoding to H.264 for browser/VSCode compatibility...")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", src,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            dst
        ],
        capture_output=True, text=True
    )

    if result.returncode == 0:
        print("[Video] Re-encode complete.")
        return True
    else:
        print(f"[Video] ffmpeg error:\n{result.stderr[-500:]}")
        return False


def main():
    print("=" * 50)
    print("Equipment Monitor — CV Service Starting")
    print("=" * 50)

    if not Path(VIDEO_PATH).exists():
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        print("[ERROR] Run: python download_videos.py")
        return

    # ── Initialize components ─────────────────────────────────────────
    model      = YOLO(MODEL_PATH)
    tracker    = Tracker(model=model)
    motion     = MotionAnalyzer()
    classifier = ActivityClassifier()
    sm         = StateMachine()
    producer   = EquipmentProducer()

    # ── Open video ────────────────────────────────────────────────────
    cap          = cv2.VideoCapture(VIDEO_PATH)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[Video] {VIDEO_PATH}")
    print(f"[Video] FPS={fps:.1f} | Total frames={total_frames}")
    print(f"[Video] Processing every {PROCESS_EVERY}th frame at {RESIZE_WIDTH}x{RESIZE_HEIGHT}")

    # ── Output writer ─────────────────────────────────────────────────
    raw_out_path   = str(VIDEOS_DIR / "output_raw.mp4")
    final_out_path = str(VIDEOS_DIR / "output.mp4")
    writer = _make_writer(raw_out_path, fps)

    # Remove stale done-flag from previous run
    done_flag = VIDEOS_DIR / ".processing_done"
    done_flag.unlink(missing_ok=True)

    frame_id             = 0
    start_time           = time.time()
    last_annotated_frame = None

    print("[Main] Starting processing loop...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] Video ended.")
                break

            frame_id += 1
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # ── Skip non-processed frames ─────────────────────────────
            if frame_id % PROCESS_EVERY != 0:
                if last_annotated_frame is not None:
                    writer.write(last_annotated_frame)
                else:
                    writer.write(frame)
                continue

            # ── Full pipeline for processed frames ────────────────────
            current_gray  = motion.update(frame)
            tracked_items = tracker.update(frame)

            for item in tracked_items:
                track_id   = item["track_id"]
                bbox       = item["bbox"]
                class_name = item["class_name"]

                motion_result = motion.analyze_bbox(current_gray, bbox)
                activity      = classifier.classify(track_id, motion_result)

                state = sm.update(
                    track_id        = track_id,
                    equipment_class = class_name,
                    is_active       = motion_result["is_active"],
                    activity        = activity,
                    motion_source   = motion_result["motion_source"],
                    fps             = fps,
                    process_every   = PROCESS_EVERY,
                )

                payload = sm.get_payload(state, frame_id)
                producer.send(payload)

                # Debug log every 300 processed frames
                if frame_id % 300 == 0:
                    print(
                        f"[CV] {state.equipment_id}: {state.current_state} | "
                        f"{state.current_activity} | src={motion_result['motion_source']} | "
                        f"util={state.utilization_percent}%"
                    )

                frame = draw_overlay(frame, item, state, payload)

            frame = draw_dashboard(frame, sm.states)

            # Save annotated frame for MJPEG live stream
            cv2.imwrite(str(LIVE_FRAME_PATH), frame)

            last_annotated_frame = frame.copy()
            writer.write(frame)

            if frame_id % 300 == 0:
                elapsed = time.time() - start_time
                pct     = (frame_id / total_frames) * 100
                print(
                    f"[Progress] Frame {frame_id}/{total_frames} "
                    f"({pct:.1f}%) | {elapsed:.1f}s elapsed"
                )

    except KeyboardInterrupt:
        print("[Main] Stopped by user.")

    finally:
        producer.flush()
        producer.close()
        cap.release()
        writer.release()

        success = _reencode_h264(raw_out_path, final_out_path)

        if success:
            try:
                os.remove(raw_out_path)
            except OSError:
                pass
            print(f"[Main] Done. Output: {final_out_path}")
        else:
            try:
                if os.path.exists(final_out_path):
                    os.remove(final_out_path)
                os.rename(raw_out_path, final_out_path)
            except OSError:
                pass
            print(f"[Main] Done (no re-encode). Output: {final_out_path}")

        done_flag.touch()


if __name__ == "__main__":
    main()