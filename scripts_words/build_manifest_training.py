#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manifest CSV from dataset/words/training/<label>/* videos."
    )
    parser.add_argument(
        "--training_dir",
        type=Path,
        default=Path("dataset/words/training"),
        help="Root directory containing label subfolders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/words_manifest.csv"),
        help="Output manifest CSV path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split value to write for every row (default: train).",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="mp4,mov,avi,mkv",
        help="Comma-separated video extensions to include.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.training_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {args.training_dir}")

    exts = {("." + e.strip().lower()).replace("..", ".") for e in args.exts.split(",") if e.strip()}
    if not exts:
        raise ValueError("--exts produced an empty extension set.")

    label_dirs = sorted([p for p in args.training_dir.iterdir() if p.is_dir()])
    label_to_index = {label_dir.name.lower(): i for i, label_dir in enumerate(label_dirs)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "video_id",
        "video_path",
        "label_gloss",
        "class_index",
        "split",
    ]

    rows_written = 0
    per_label = Counter()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label_dir in label_dirs:
            label = label_dir.name.lower()
            class_index = label_to_index[label]

            files = sorted(
                [
                    p
                    for p in label_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                ]
            )

            for video_path in files:
                video_id = video_path.stem
                writer.writerow(
                    [
                        video_id,
                        str(video_path),
                        label,
                        class_index,
                        args.split,
                    ]
                )
                rows_written += 1
                per_label[label] += 1

    print(f"Saved: {args.out}")
    print(f"Total labels found: {len(label_dirs)}")
    print(f"Rows written: {rows_written}")
    print("Per-label counts:")
    for label in sorted(per_label):
        print(f"  {label}: {per_label[label]}")


if __name__ == "__main__":
    main()

# Example:
# python scripts_words/build_manifest_training.py --training_dir dataset/words/training --out data/words_manifest.csv
