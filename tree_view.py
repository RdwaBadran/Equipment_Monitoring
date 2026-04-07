from pathlib import Path

EXCLUDE_DIRS = {
    "venv", "__pycache__", ".git", ".pytest_cache", "site-packages"
}

INCLUDE_FILES = {
    ".py", ".md"
}

INCLUDE_NAMES = {
    "Dockerfile", "requirements.txt", ".env.example"
}

def print_tree(path: Path, prefix=""):
    entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

    entries = [
        e for e in entries
        if e.name not in EXCLUDE_DIRS
        and (e.is_dir() or e.suffix in INCLUDE_FILES or e.name in INCLUDE_NAMES)
    ]

    for i, entry in enumerate(entries):
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry.name)

        if entry.is_dir():
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(entry, prefix + extension)

if __name__ == "__main__":
    root = Path(".")
    print(root.name)
    print_tree(root)

