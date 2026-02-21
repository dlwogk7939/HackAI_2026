#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract normalized MediaPipe hand landmarks from image folders into CSV."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/alphabet"),
        help="Dataset root: dataset/alphabet/<label>/*.jpg|jpeg|png",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/asl_landmarks.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum samples per class.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order in each class before sampling.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize (e.g. --resize 256).",
    )
    return parser.parse_args()


def get_image_paths(label_dir: Path) -> list[Path]:
    files = [p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def normalize_landmarks(hand_landmarks) -> np.ndarray | None:
    coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = float(np.linalg.norm(coords[12]))
    if scale <= 1e-6:
        return None
    coords /= scale
    return coords.reshape(-1)


def extract_feature(image_bgr: np.ndarray, hands) -> np.ndarray | None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    return normalize_landmarks(result.multi_hand_landmarks[0])


def main() -> None:
    args = parse_args()

    if args.max_per_class is not None and args.max_per_class <= 0:
        raise ValueError("--max_per_class must be > 0.")
    if args.resize is not None and args.resize <= 0:
        raise ValueError("--resize must be > 0.")
    if not args.root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.root}")

    label_dirs = sorted([p for p in args.root.iterdir() if p.is_dir()])
    if not label_dirs:
        raise RuntimeError(f"No label directories found under: {args.root}")

    rng = random.Random(42)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_images = 0
    kept_rows = 0
    dropped = 0
    kept_per_label: dict[str, int] = defaultdict(int)

    header = [f"f{i}" for i in range(42)] + ["label"]

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    ) as hands, args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for label_dir in label_dirs:
            label = label_dir.name
            img_paths = get_image_paths(label_dir)
            if args.shuffle:
                rng.shuffle(img_paths)
            if args.max_per_class is not None:
                img_paths = img_paths[: args.max_per_class]

            for img_path in img_paths:
                total_images += 1
                image = cv2.imread(str(img_path))
                if image is None:
                    dropped += 1
                    continue
                if args.resize is not None:
                    image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

                feature = extract_feature(image, hands)
                if feature is None:
                    dropped += 1
                    continue

                writer.writerow([*feature.tolist(), label])
                kept_rows += 1
                kept_per_label[label] += 1

    print(f"Saved CSV: {args.out}")
    print(f"Total images: {total_images}")
    print(f"Kept rows: {kept_rows}")
    print(f"Dropped count: {dropped}")
    print("Kept per label:")
    for label in sorted([p.name for p in label_dirs]):
        print(f"  {label}: {kept_per_label.get(label, 0)}")


if __name__ == "__main__":
    main()
