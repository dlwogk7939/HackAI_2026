#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import cv2

# Improve compatibility on macOS/headless execution.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import mediapipe as mp


@dataclass
class VideoStats:
    label: str
    total_frames: int
    detected_frames: int
    dropped_frames: int
    segments_written: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split training_videos into multiple clips using no-hand frames as boundaries. "
            "Only leading/intermediate/trailing no-hand gaps are removed; detected-hand "
            "sequences are kept as separate clips."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dataset/words/training_videos"),
        help="Input directory containing source label videos (e.g., hello.mp4).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/words/training"),
        help="Output training root. Clips are saved under output_dir/<label>/.",
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="mp4,mov,avi,mkv",
        help="Comma-separated input video extensions.",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.35,
        help="MediaPipe minimum detection confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.35,
        help="MediaPipe minimum tracking confidence.",
    )
    parser.add_argument(
        "--max_num_hands",
        type=int,
        default=2,
        help="Maximum number of hands to detect.",
    )
    parser.add_argument(
        "--min_segment_frames",
        type=int,
        default=1,
        help="Minimum detected-hand frames required to keep a segment.",
    )
    parser.add_argument(
        "--clear_existing",
        action="store_true",
        default=True,
        help="Clear existing .mp4 files in each target label folder before writing.",
    )
    parser.add_argument(
        "--no_clear_existing",
        dest="clear_existing",
        action="store_false",
        help="Do not clear existing .mp4 files in target label folders.",
    )
    return parser.parse_args()


def list_input_videos(input_dir: Path, exts: set[str]) -> list[Path]:
    files = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()
    return files


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    return writer


def process_video(video_path: Path, output_root: Path, hands, args: argparse.Namespace) -> VideoStats:
    label = video_path.stem.lower()
    label_dir = output_root / label
    label_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_existing:
        for old in label_dir.glob("*.mp4"):
            old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[SKIP] {video_path.name}: cannot open video.")
        return VideoStats(label, 0, 0, 0, 0)

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        print(f"[SKIP] {video_path.name}: invalid video size.")
        cap.release()
        return VideoStats(label, 0, 0, 0, 0)

    total_frames = 0
    detected_frames = 0
    dropped_frames = 0
    segments_written = 0
    segment_frames = 0
    writer: cv2.VideoWriter | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        has_hand = bool(result.multi_hand_landmarks)

        if has_hand:
            detected_frames += 1
            if writer is None:
                segments_written += 1
                out_path = label_dir / f"{label}_{segments_written:03d}.mp4"
                writer = open_writer(out_path, fps, width, height)
                if not writer.isOpened():
                    print(f"[SKIP] {video_path.name}: failed to open writer {out_path.name}")
                    segments_written -= 1
                    writer = None
                    dropped_frames += 1
                    continue
                segment_frames = 0
            writer.write(frame)
            segment_frames += 1
        else:
            dropped_frames += 1
            if writer is not None:
                writer.release()
                writer = None
                if segment_frames < args.min_segment_frames:
                    bad_path = label_dir / f"{label}_{segments_written:03d}.mp4"
                    if bad_path.exists():
                        bad_path.unlink()
                    segments_written -= 1
                segment_frames = 0

    if writer is not None:
        writer.release()
        if segment_frames < args.min_segment_frames:
            bad_path = label_dir / f"{label}_{segments_written:03d}.mp4"
            if bad_path.exists():
                bad_path.unlink()
            segments_written -= 1

    cap.release()
    return VideoStats(label, total_frames, detected_frames, dropped_frames, segments_written)


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.max_num_hands <= 0:
        raise ValueError("--max_num_hands must be > 0.")
    if not (0.0 <= args.min_detection_confidence <= 1.0):
        raise ValueError("--min_detection_confidence must be in [0, 1].")
    if not (0.0 <= args.min_tracking_confidence <= 1.0):
        raise ValueError("--min_tracking_confidence must be in [0, 1].")
    if args.min_segment_frames <= 0:
        raise ValueError("--min_segment_frames must be > 0.")

    exts = {
        ("." + e.strip().lower()).replace("..", ".")
        for e in args.exts.split(",")
        if e.strip()
    }
    if not exts:
        raise ValueError("--exts produced an empty extension set.")

    videos = list_input_videos(args.input_dir, exts)
    print(f"Input videos found: {len(videos)}")
    if not videos:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_stats: list[VideoStats] = []
    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as hands:
        for video in videos:
            stats = process_video(video, args.output_dir, hands, args)
            all_stats.append(stats)
            print(
                f"[OK] {video.name} -> label={stats.label}, "
                f"segments={stats.segments_written}, detected_frames={stats.detected_frames}, "
                f"dropped_frames={stats.dropped_frames}, total_frames={stats.total_frames}"
            )

    total_videos = len(all_stats)
    total_segments = sum(s.segments_written for s in all_stats)
    total_frames = sum(s.total_frames for s in all_stats)
    total_detected = sum(s.detected_frames for s in all_stats)
    total_dropped = sum(s.dropped_frames for s in all_stats)

    print("\nSummary")
    print(f"  total_videos: {total_videos}")
    print(f"  total_segments_written: {total_segments}")
    print(f"  total_frames: {total_frames}")
    print(f"  detected_frames: {total_detected}")
    print(f"  dropped_no_hand_frames: {total_dropped}")
    print(f"  output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()

# Example:
# python scripts_words/split_training_videos_by_hand.py --input_dir dataset/words/training_videos --output_dir dataset/words/training
