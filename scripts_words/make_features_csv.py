#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Force CPU path for broader compatibility in headless/macOS environments.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pooled MediaPipe hand-landmark features from WLASL manifest videos."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/words_manifest.csv"),
        help="Input manifest CSV path.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/words_features.csv"),
        help="Output feature CSV path.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=80,
        help="Maximum valid detected frames to use per sample.",
    )
    parser.add_argument(
        "--sample_fps",
        type=float,
        default=10.0,
        help="Target sampling FPS from video.",
    )
    parser.add_argument(
        "--min_detected_ratio",
        type=float,
        default=0.8,
        help="Minimum detected_any/total_frames ratio required to keep a sample.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional output width for resize while preserving aspect ratio.",
    )
    parser.add_argument(
        "--two_hands",
        action="store_true",
        help="Use two-hand mode (84 dims = left42 + right42).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle manifest rows before processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --shuffle.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["mean", "seg3"],
        default="seg3",
        help="Temporal pooling mode.",
    )
    parser.add_argument(
        "--debug_detect",
        action="store_true",
        help="Print per-video detection summary and write detect report CSV.",
    )
    return parser.parse_args()


def parse_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_landmarks(hand_landmarks) -> np.ndarray | None:
    coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = float(np.linalg.norm(coords[12]))
    if scale <= 1e-6:
        return None
    coords /= scale
    return coords.reshape(-1)


def frame_feature_one_hand(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None
    return normalize_landmarks(results.multi_hand_landmarks[0])


def frame_feature_two_hands(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None

    left = np.zeros(42, dtype=np.float32)
    right = np.zeros(42, dtype=np.float32)
    left_set = False
    right_set = False

    handedness = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        feat = normalize_landmarks(hand_landmarks)
        if feat is None:
            continue

        side = None
        if idx < len(handedness) and handedness[idx].classification:
            side = handedness[idx].classification[0].label

        if side == "Left" and not left_set:
            left = feat
            left_set = True
        elif side == "Right" and not right_set:
            right = feat
            right_set = True
        elif not left_set:
            left = feat
            left_set = True
        elif not right_set:
            right = feat
            right_set = True

    if not left_set and not right_set:
        return None
    return np.concatenate([left, right], axis=0)


def resolve_video_path(video_path_raw: str, manifest_path: Path) -> Path:
    path = Path(video_path_raw)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path

    candidate = manifest_path.parent / path
    if candidate.exists():
        return candidate

    return path


def longest_true_run(flags: list[bool]) -> int:
    best = 0
    cur = 0
    for flag in flags:
        if flag:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def split_three_counts(flags: list[bool]) -> list[int]:
    n = len(flags)
    if n == 0:
        return [0, 0, 0]
    if n == 1:
        return [int(flags[0]), 0, 0]
    if n == 2:
        return [int(flags[0]), int(flags[1]), 0]

    b1 = n // 3
    b2 = (2 * n) // 3
    b1 = max(1, min(b1, n - 2))
    b2 = max(b1 + 1, min(b2, n - 1))

    return [
        int(sum(flags[:b1])),
        int(sum(flags[b1:b2])),
        int(sum(flags[b2:])),
    ]


def process_video_row(
    row: dict[str, str],
    manifest_path: Path,
    hands,
    expected_dim: int,
    max_frames: int,
    sample_fps: float,
    min_detected_ratio: float,
    pool_mode: str,
    resize: int | None,
    two_hands: bool,
) -> tuple[str, list[np.ndarray] | None, int, dict[str, object]]:
    video_path = resolve_video_path(str(row.get("video_path", "")).strip(), manifest_path)
    debug_row: dict[str, object] = {
        "label": str(row.get("label_gloss", "")),
        "video_path": str(video_path),
        "total_frames": 0,
        "detected_any": 0,
        "detected_both": 0,
        "seg1_any": 0,
        "seg2_any": 0,
        "seg3_any": 0,
        "seg1_both": 0,
        "seg2_both": 0,
        "seg3_both": 0,
        "longest_run_any": 0,
        "longest_run_both": 0,
        "keep_or_skip": "skip",
        "skip_reason": "other:decode_failure",
        "threshold_total": float(min_detected_ratio),
        "threshold_seg": "NA",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return "decode_fail", None, 0, debug_row

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_start = parse_int(row.get("frame_start"), 0)
    frame_end_manifest = parse_int(row.get("frame_end"), -1)

    if frame_end_manifest == -1:
        frame_end = total_frames - 1 if total_frames > 0 else frame_start
    else:
        frame_end = frame_end_manifest

    if total_frames > 0:
        frame_start = max(0, min(frame_start, total_frames - 1))
        frame_end = max(0, min(frame_end, total_frames - 1))

    if frame_end < frame_start:
        cap.release()
        return "decode_fail", None, 0, debug_row

    fps_manifest = parse_float(row.get("fps"), 0.0)
    fps_cap = float(cap.get(cv2.CAP_PROP_FPS))
    video_fps = fps_manifest if fps_manifest > 0 else fps_cap
    if video_fps <= 0:
        video_fps = sample_fps
    step = max(1, int(round(video_fps / sample_fps)))

    feats: list[np.ndarray] = []
    decoded_any = False
    any_flags: list[bool] = []
    both_flags: list[bool] = []

    for frame_idx in range(frame_start, frame_end + 1, step):
        if len(feats) >= max_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        decoded_any = True

        if resize is not None:
            h, w = frame.shape[:2]
            if w > 0 and h > 0:
                new_w = resize
                new_h = max(1, int(round(h * (new_w / float(w)))))
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        detected_hands = len(result.multi_hand_landmarks or [])
        any_flags.append(detected_hands >= 1)
        both_flags.append(detected_hands >= 2)

        feat = frame_feature_two_hands(result) if two_hands else frame_feature_one_hand(result)
        if feat is None:
            continue
        if feat.shape[0] != expected_dim:
            cap.release()
            raise RuntimeError(
                f"Inconsistent feature dimension: got {feat.shape[0]}, expected {expected_dim}."
            )
        feats.append(feat)

    cap.release()

    seg_any = split_three_counts(any_flags)
    seg_both = split_three_counts(both_flags)
    debug_row["total_frames"] = int(len(any_flags))
    debug_row["detected_any"] = int(sum(any_flags))
    debug_row["detected_both"] = int(sum(both_flags))
    debug_row["seg1_any"] = seg_any[0]
    debug_row["seg2_any"] = seg_any[1]
    debug_row["seg3_any"] = seg_any[2]
    debug_row["seg1_both"] = seg_both[0]
    debug_row["seg2_both"] = seg_both[1]
    debug_row["seg3_both"] = seg_both[2]
    debug_row["longest_run_any"] = longest_true_run(any_flags)
    debug_row["longest_run_both"] = longest_true_run(both_flags)

    if not decoded_any:
        return "decode_fail", None, 0, debug_row
    detected_any = int(sum(any_flags))
    total_sampled = int(len(any_flags))
    detected_ratio = (detected_any / total_sampled) if total_sampled > 0 else 0.0

    if detected_ratio < min_detected_ratio:
        debug_row["skip_reason"] = "total_detected_below_threshold"
        return "low_detected", None, len(feats), debug_row
    if pool_mode == "seg3" and len(feats) < 3:
        debug_row["skip_reason"] = "segment_detected_below_threshold"
        return "low_detected", None, len(feats), debug_row
    if len(feats) == 0:
        debug_row["skip_reason"] = "other:no_valid_features_after_normalization"
        return "low_detected", None, 0, debug_row

    debug_row["keep_or_skip"] = "keep"
    debug_row["skip_reason"] = ""
    return "ok", feats, len(feats), debug_row


def pool_feature_sequence(
    frame_feats: list[np.ndarray],
    base_dim: int,
    pool_mode: str,
    warn_state: dict[str, bool],
) -> np.ndarray:
    if not frame_feats:
        raise RuntimeError("Cannot pool empty frame feature list.")

    seq = np.stack(frame_feats, axis=0)
    if seq.shape[1] != base_dim:
        raise RuntimeError(
            f"Inconsistent frame feature dimension: got {seq.shape[1]}, expected {base_dim}."
        )

    k = seq.shape[0]
    if pool_mode == "mean":
        return np.mean(seq, axis=0)

    # seg3 pooling
    if k < 3:
        if not warn_state.get("seg3_small_warned", False):
            print(
                "WARNING: seg3 requested but some samples have K < 3 valid frames. "
                "Falling back to mean pooling for those samples."
            )
            warn_state["seg3_small_warned"] = True
        m = np.mean(seq, axis=0)
        return np.concatenate([m, m, m], axis=0)

    b1 = k // 3
    b2 = (2 * k) // 3

    # Ensure each segment is non-empty when K >= 3.
    b1 = max(1, min(b1, k - 2))
    b2 = max(b1 + 1, min(b2, k - 1))

    s1 = seq[:b1]
    s2 = seq[b1:b2]
    s3 = seq[b2:]

    if len(s1) == 0 or len(s2) == 0 or len(s3) == 0:
        # Final safety net.
        m = np.mean(seq, axis=0)
        return np.concatenate([m, m, m], axis=0)

    m1 = np.mean(s1, axis=0)
    m2 = np.mean(s2, axis=0)
    m3 = np.mean(s3, axis=0)
    return np.concatenate([m1, m2, m3], axis=0)


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    if args.max_frames <= 0:
        raise ValueError("--max_frames must be > 0.")
    if args.sample_fps <= 0:
        raise ValueError("--sample_fps must be > 0.")
    if not (0.0 <= args.min_detected_ratio <= 1.0):
        raise ValueError("--min_detected_ratio must be in [0.0, 1.0].")
    if args.resize is not None and args.resize <= 0:
        raise ValueError("--resize must be > 0.")

    with args.manifest.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(rows)

    expected_dim = 84 if args.two_hands else 42
    final_dim = expected_dim if args.pool == "mean" else expected_dim * 3
    print(f"Pool mode: {args.pool}")
    print(f"Final feature dimension: {final_dim}")
    out_header = (
        ["label_gloss", "class_index", "split", "video_id", "num_frames_used"]
        + [f"f{i}" for i in range(final_dim)]
    )
    detect_report_path = args.out.parent / "detect_report_seg3_2h.csv"
    detect_report_header = [
        "label",
        "video_path",
        "total_frames",
        "detected_any",
        "detected_both",
        "seg1_any",
        "seg2_any",
        "seg3_any",
        "seg1_both",
        "seg2_both",
        "seg3_both",
        "longest_run_any",
        "longest_run_both",
        "keep_or_skip",
        "skip_reason",
        "threshold_total",
        "threshold_seg",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    total_rows = len(rows)
    kept_samples = 0
    skipped_decode = 0
    skipped_low = 0
    kept_per_label: Counter[str] = Counter()
    warn_state = {"seg3_small_warned": False}
    detect_report_rows: list[dict[str, object]] = []

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2 if args.two_hands else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, args.out.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(out_header)

        for row in rows:
            status, frame_feats, used_frames, debug_row = process_video_row(
                row=row,
                manifest_path=args.manifest,
                hands=hands,
                expected_dim=expected_dim,
                max_frames=args.max_frames,
                sample_fps=args.sample_fps,
                min_detected_ratio=args.min_detected_ratio,
                pool_mode=args.pool,
                resize=args.resize,
                two_hands=args.two_hands,
            )
            detect_report_rows.append(debug_row)
            if args.debug_detect:
                print(
                    "[DEBUG] "
                    f"{debug_row['label']} {debug_row['video_path']} "
                    f"total_frames={debug_row['total_frames']} "
                    f"detected_any={debug_row['detected_any']} "
                    f"detected_both={debug_row['detected_both']} "
                    f"seg_counts_any=[{debug_row['seg1_any']},{debug_row['seg2_any']},{debug_row['seg3_any']}] "
                    f"seg_counts_both=[{debug_row['seg1_both']},{debug_row['seg2_both']},{debug_row['seg3_both']}] "
                    f"longest_run_any={debug_row['longest_run_any']} "
                    f"longest_run_both={debug_row['longest_run_both']} "
                    f"threshold_total={debug_row['threshold_total']} "
                    f"threshold_seg={debug_row['threshold_seg']} "
                    f"skip_reason={debug_row['skip_reason']}"
                )

            if status == "decode_fail":
                skipped_decode += 1
                continue
            if status == "low_detected":
                skipped_low += 1
                continue
            if frame_feats is None:
                raise RuntimeError("Internal error: status ok but frame features are None.")

            pooled = pool_feature_sequence(
                frame_feats=frame_feats,
                base_dim=expected_dim,
                pool_mode=args.pool,
                warn_state=warn_state,
            )
            if pooled.shape[0] != final_dim:
                raise RuntimeError(
                    f"Inconsistent row feature dimension: got {pooled.shape[0]}, expected {final_dim}."
                )

            label = str(row.get("label_gloss", ""))
            writer.writerow(
                [
                    label,
                    row.get("class_index", ""),
                    row.get("split", ""),
                    row.get("video_id", ""),
                    used_frames,
                    *pooled.astype(np.float32).tolist(),
                ]
            )
            kept_samples += 1
            kept_per_label[label] += 1

    with detect_report_path.open("w", newline="", encoding="utf-8") as f_report:
        writer = csv.DictWriter(f_report, fieldnames=detect_report_header)
        writer.writeheader()
        for report_row in detect_report_rows:
            writer.writerow(report_row)

    print(f"Saved CSV: {args.out}")
    print(f"Saved detect report: {detect_report_path}")
    print(f"Total manifest rows: {total_rows}")
    print(f"Kept samples written: {kept_samples}")
    print(f"Skipped due to decode failure: {skipped_decode}")
    print(f"Skipped due to low detected frames: {skipped_low}")
    print("Per-label kept counts:")
    for label in sorted(kept_per_label):
        print(f"  {label}: {kept_per_label[label]}")


if __name__ == "__main__":
    main()

# Example run:
# python scripts_words/make_features_csv.py --manifest data/words_manifest.csv --out data/words_features_seg3.csv --pool seg3
