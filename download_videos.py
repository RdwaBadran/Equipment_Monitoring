"""
download_videos.py
------------------
Downloads sample construction equipment videos into the videos/ folder.

Usage:
    python download_videos.py              # downloads all default videos
    python download_videos.py --list       # shows which videos will be downloaded
    python download_videos.py --video 1    # downloads only video1

Requirements:
    pip install yt-dlp     (run once if not already installed)
    ffmpeg must be in PATH (for post-processing by yt-dlp)

The script auto-installs yt-dlp if it is not present.
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

VIDEOS_DIR = Path(__file__).parent / "videos"

# ── Video catalogue ───────────────────────────────────────────────────────────
# Public YouTube videos showing construction equipment in action.
# Feel free to replace these URLs with any other construction site footage.
VIDEOS = [
    {
        "id":   1,
        "name": "video1.mp4",
        "url":  "https://www.youtube.com/watch?v=cRfEgCPTQSI",
        "desc": "Excavator at construction site (wide-angle, good for tracking)",
    },
    {
        "id":   2,
        "name": "video2.mp4",
        "url":  "https://www.youtube.com/watch?v=TaJEXvn0m_4",
        "desc": "Multiple construction machines on site",
    },
    {
        "id":   3,
        "name": "video3.mp4",
        "url":  "https://www.youtube.com/watch?v=Y5Xne6PQvBk",
        "desc": "Dump trucks and excavators — mixed equipment",
    },
]


def ensure_ytdlp() -> bool:
    """Check if yt-dlp is available; offer to install it if not."""
    try:
        import yt_dlp  # noqa: F401
        return True
    except ImportError:
        pass

    print("[Setup] yt-dlp is not installed.")
    answer = input("Install it now? (y/n): ").strip().lower()
    if answer == "y":
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp"],
            check=True
        )
        return True

    print("[Setup] Skipping download — install yt-dlp manually with:")
    print("        pip install yt-dlp")
    return False


def download_video(video: dict, output_dir: Path) -> bool:
    """Download one video using yt-dlp. Returns True on success."""
    import yt_dlp

    out_path = output_dir / video["name"]

    if out_path.exists():
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[Skip] {video['name']} already exists ({size_mb:.1f} MB)")
        return True

    print(f"\n[Download] {video['name']}")
    print(f"  Source : {video['url']}")
    print(f"  Desc   : {video['desc']}")

    ydl_opts = {
        # Download best quality MP4, max 1080p to keep file sizes manageable
        "format":    "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl":   str(output_dir / video["name"]),
        "quiet":     False,
        "no_warnings": False,
        # Merge into a single mp4 file
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video["url"]])

        if out_path.exists():
            size_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Saved: {out_path} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"[ERROR] File not found after download: {out_path}")
            return False

    except Exception as e:
        print(f"[ERROR] Download failed for {video['name']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download sample construction videos")
    parser.add_argument(
        "--video", type=int, default=None,
        help="Download only a specific video by ID (1, 2, or 3)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available videos without downloading"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("Equipment Monitor — Video Downloader")
    print("=" * 55)

    if args.list:
        print("\nAvailable videos:")
        for v in VIDEOS:
            exists = "✓" if (VIDEOS_DIR / v["name"]).exists() else " "
            print(f"  [{exists}] {v['id']}. {v['name']:15s} — {v['desc']}")
        return

    VIDEOS_DIR.mkdir(exist_ok=True)

    if not ensure_ytdlp():
        sys.exit(1)

    targets = VIDEOS
    if args.video is not None:
        targets = [v for v in VIDEOS if v["id"] == args.video]
        if not targets:
            print(f"[ERROR] No video with ID {args.video}. Use --list to see options.")
            sys.exit(1)

    success = 0
    for video in targets:
        if download_video(video, VIDEOS_DIR):
            success += 1

    print(f"\n[Done] {success}/{len(targets)} video(s) downloaded to {VIDEOS_DIR}/")

    if success > 0:
        print("\nRun the pipeline with:")
        print("    python run.py                          # uses video1.mp4")
        print("    python run.py --video videos/video2.mp4")


if __name__ == "__main__":
    main()