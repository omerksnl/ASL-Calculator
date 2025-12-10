from pathlib import Path
import argparse


def count_images(root: Path, exts=None):
    if exts is None:
        exts = {".jpg", ".jpeg", ".png"}
    if not root.exists():
        return {}, 0
    counts = {}
    total = 0
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        c = sum(1 for f in entry.iterdir() if f.is_file() and f.suffix.lower() in exts)
        counts[entry.name] = c
        total += c
    return counts, total


def main():
    parser = argparse.ArgumentParser(description="Count images per class folder.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/master_dataset/master_data"),
        help="Root folder containing class subfolders.",
    )
    args = parser.parse_args()

    counts, total = count_images(args.root)
    if not counts:
        print(f"No class folders found under {args.root}")
        return

    print(f"Root: {args.root}")
    width = max(len(name) for name in counts) if counts else 0
    for name, c in counts.items():
        print(f"{name.ljust(width)} : {c}")
    print(f"Total images: {total}")


if __name__ == "__main__":
    main()

