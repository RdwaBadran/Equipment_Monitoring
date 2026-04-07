# export_project.py
# Bundles all important project files into one text file
# so you can share the entire project in one paste.

from pathlib import Path

INCLUDE_EXTENSIONS = {".py", ".yml", ".txt", ".md", ".html"}
INCLUDE_NAMES      = {"Dockerfile", "docker-compose.yml", ".env.example"}

EXCLUDE_DIRS = {
    "venv", "__pycache__", ".git", ".pytest_cache",
    "site-packages", "node_modules", ".idea"
}

EXCLUDE_FILES = {
    "export_project.py",
    "tree_view.py",
    "download_videos.py",
}

OUTPUT_FILE = "project_export.txt"


def collect_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.name in EXCLUDE_FILES:
            continue
        if path.suffix in INCLUDE_EXTENSIONS or path.name in INCLUDE_NAMES:
            files.append(path)
    return files


def main():
    root  = Path(".")
    files = collect_files(root)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=" * 60 + "\n")
        out.write("PROJECT EXPORT — equipment-monitor\n")
        out.write("=" * 60 + "\n\n")

        out.write("FILES INCLUDED:\n")
        for f in files:
            out.write(f"  {f}\n")
        out.write("\n" + "=" * 60 + "\n\n")

        for f in files:
            out.write(f"\n{'='*60}\n")
            out.write(f"FILE: {f}\n")
            out.write(f"{'='*60}\n")
            try:
                out.write(f.read_text(encoding="utf-8"))
            except Exception as e:
                out.write(f"[Could not read: {e}]\n")
            out.write("\n")

    print(f"Done — {len(files)} files exported to {OUTPUT_FILE}")
    size_kb = Path(OUTPUT_FILE).stat().st_size / 1024
    print(f"Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()