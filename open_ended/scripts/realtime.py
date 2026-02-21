#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime ASL inference with MediaPipe Hands + RandomForest."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/rf_asl.joblib"),
        help="Trained model path.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Majority-vote history length.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for display.",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.05,
        help="Step used by key control to adjust threshold.",
    )
    parser.add_argument(
        "--two_hands",
        action="store_true",
        help="Enable two-hand feature mode (84 dims: left 42 + right 42).",
    )
    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="MediaPipe min_detection_confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="MediaPipe min_tracking_confidence.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror camera frame horizontally for selfie view.",
    )
    return parser.parse_args()


def normalize_landmarks(hand_landmarks) -> np.ndarray | None:
    coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = float(np.linalg.norm(coords[12]))
    if scale <= 1e-6:
        return None
    coords /= scale
    return coords.reshape(-1)


def extract_feature_one_hand(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None
    return normalize_landmarks(results.multi_hand_landmarks[0])


def extract_feature_two_hands(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None

    left = np.zeros(42, dtype=np.float32)
    right = np.zeros(42, dtype=np.float32)
    left_set = False
    right_set = False

    handedness_list = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        feature = normalize_landmarks(hand_landmarks)
        if feature is None:
            continue

        hand_label = None
        if idx < len(handedness_list) and handedness_list[idx].classification:
            hand_label = handedness_list[idx].classification[0].label

        if hand_label == "Left" and not left_set:
            left = feature
            left_set = True
        elif hand_label == "Right" and not right_set:
            right = feature
            right_set = True
        elif not left_set:
            left = feature
            left_set = True
        elif not right_set:
            right = feature
            right_set = True

    if not left_set and not right_set:
        return None
    return np.concatenate([left, right], axis=0)


def majority_vote(history: deque[tuple[str, float]]) -> tuple[str, float]:
    if not history:
        return "UNKNOWN", 0.0

    labels = [item[0] for item in history]
    counts = Counter(labels)
    top_label, _ = counts.most_common(1)[0]

    confs = [c for lbl, c in history if lbl == top_label]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return top_label, avg_conf


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if args.history <= 0:
        raise ValueError("--history must be > 0.")
    if args.threshold_step <= 0:
        raise ValueError("--threshold_step must be > 0.")
    if not (0.0 <= args.threshold <= 1.0):
        raise ValueError("--threshold must be in [0, 1].")

    model = joblib.load(args.model)
    expected_features = int(getattr(model, "n_features_in_", 42))
    current_features = 84 if args.two_hands else 42
    if expected_features != current_features:
        raise ValueError(
            f"Feature size mismatch: model expects {expected_features}, "
            f"but current mode provides {current_features}. "
            f"Use matching mode or retrain the model."
        )

    history: deque[tuple[str, float]] = deque(maxlen=args.history)
    threshold = args.threshold

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {args.camera}")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2 if args.two_hands else 1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    prev_time = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            feature = (
                extract_feature_two_hands(results)
                if args.two_hands
                else extract_feature_one_hand(results)
            )

            instant_label = "NO_HAND"
            instant_conf = 0.0
            if feature is not None:
                probs = model.predict_proba(feature.reshape(1, -1))[0]
                top_idx = int(np.argmax(probs))
                instant_label = str(model.classes_[top_idx])
                instant_conf = float(probs[top_idx])
                history.append((instant_label, instant_conf))

            voted_label, voted_conf = majority_vote(history)
            if voted_conf < threshold:
                display_label = "UNKNOWN"
            else:
                display_label = voted_label

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            cv2.putText(
                frame,
                f"Label: {display_label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if display_label != "UNKNOWN" else (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Conf: {voted_conf:.2f}  (instant {instant_label}:{instant_conf:.2f})",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}  Th: {threshold:.2f}  Hist: {len(history)}/{args.history}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                "Keys: q=quit, r=reset, t=th+ , T=th-",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.imshow("ASL Realtime", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                history.clear()
            if key == ord("t"):
                threshold = min(1.0, threshold + args.threshold_step)
            if key == ord("T"):
                threshold = max(0.0, threshold - args.threshold_step)
            if key == ord("="):
                threshold = min(1.0, threshold + args.threshold_step)
            if key == ord("-"):
                threshold = max(0.0, threshold - args.threshold_step)
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
