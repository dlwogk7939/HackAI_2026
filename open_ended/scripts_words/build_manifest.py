#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_CLASS_INDICES = "402,566,164,1789,1779,516,749,715,1840,1917,1980"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a minimal manifest CSV for selected WLASL class indices."
    )
    parser.add_argument(
        "--wlasl_json",
        type=Path,
        default=Path("dataset/words/WLASL_v0.3.json"),
        help="Path to WLASL_v0.3.json",
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("dataset/words/videos"),
        help="Directory with downloaded .mp4 files",
    )
    parser.add_argument(
        "--class_indices",
        type=str,
        default=DEFAULT_CLASS_INDICES,
        help='Comma-separated class indices, e.g. "402,566,164"',
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/words_manifest.csv"),
        help="Output manifest CSV path",
    )
    return parser.parse_args()


def parse_class_indices(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("No class indices provided.")
    return out


def main() -> None:
    args = parse_args()

    if not args.wlasl_json.exists():
        raise FileNotFoundError(f"WLASL json not found: {args.wlasl_json}")
    if not args.video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    class_indices = parse_class_indices(args.class_indices)

    with args.wlasl_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("WLASL json must be a list of dict entries.")

    for idx in class_indices:
        if idx < 0 or idx >= len(data):
            raise IndexError(f"class index out of range: {idx} (len={len(data)})")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "video_id",
        "video_path",
        "label_gloss",
        "class_index",
        "split",
        "fps",
        "frame_start",
        "frame_end",
        "bbox",
        "signer_id",
        "source",
    ]

    total_instances = 0
    kept_instances = 0
    per_label_kept: Counter[str] = Counter()
    split_kept: Counter[str] = Counter()

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for class_index in class_indices:
            entry = data[class_index]
            gloss = str(entry.get("gloss", f"idx_{class_index}"))
            instances = entry.get("instances", [])
            if not isinstance(instances, list):
                continue

            for instance in instances:
                total_instances += 1
                video_id = str(instance.get("video_id", "")).strip()
                if not video_id:
                    continue

                video_path = args.video_dir / f"{video_id}.mp4"
                if not video_path.exists():
                    continue

                split = str(instance.get("split", ""))
                fps = instance.get("fps", "")
                frame_start = instance.get("frame_start", "")
                frame_end = instance.get("frame_end", "")
                bbox = instance.get("bbox", "")
                signer_id = instance.get("signer_id", "")
                source = instance.get("source", "")

                writer.writerow(
                    [
                        video_id,
                        str(video_path),
                        gloss,
                        class_index,
                        split,
                        fps,
                        frame_start,
                        frame_end,
                        json.dumps(bbox, ensure_ascii=False),
                        signer_id,
                        source,
                    ]
                )
                kept_instances += 1
                per_label_kept[gloss] += 1
                split_kept[split] += 1

    missing_instances = total_instances - kept_instances

    print(f"Saved: {args.out}")
    print(f"Total instances considered: {total_instances}")
    print(f"Kept instances (available videos): {kept_instances}")
    print(f"Missing instances: {missing_instances}")
    print("Per label_gloss kept counts:")
    for label in sorted(per_label_kept):
        print(f"  {label}: {per_label_kept[label]}")
    print("Split distribution among kept rows:")
    for split in sorted(split_kept):
        print(f"  {split}: {split_kept[split]}")


if __name__ == "__main__":
    main()
