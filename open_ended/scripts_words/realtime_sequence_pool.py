#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

# Keep realtime compatible in environments where GPU context creation fails.
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime WLASL word inference with sequence pooling + RandomForest."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained .joblib model.",
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["mean", "seg3"],
        default="seg3",
        help="Temporal pooling mode (must match training features).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Sliding window size in frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Run inference every N captured frames.",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=20,
        help="Minimum buffered frames required before inference.",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=10,
        help="Prediction history length for majority vote.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror camera frame horizontally.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=640,
        help="Resize frame width while keeping aspect ratio.",
    )
    parser.add_argument(
        "--two_hands",
        action="store_true",
        help="Use left42+right42 per-frame features (84D frame-level).",
    )
    parser.add_argument(
        "--show_landmarks",
        action="store_true",
        help="Draw detected hand landmarks.",
    )
    return parser.parse_args()


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def normalize_hand_landmarks(hand_landmarks) -> np.ndarray | None:
    coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    coords -= coords[0]
    scale = float(np.linalg.norm(coords[12]))
    if scale <= 1e-6:
        return None
    coords /= scale
    return coords.reshape(-1)


def extract_one_hand_feature(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None
    return normalize_hand_landmarks(results.multi_hand_landmarks[0])


def extract_two_hands_feature(results) -> np.ndarray | None:
    if not results.multi_hand_landmarks:
        return None

    left = np.zeros(42, dtype=np.float32)
    right = np.zeros(42, dtype=np.float32)
    left_set = False
    right_set = False

    handedness = results.multi_handedness or []
    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        feat = normalize_hand_landmarks(hand_landmarks)
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


def feature_dim_for_mode(frame_dim: int, pool_mode: str) -> int:
    return frame_dim if pool_mode == "mean" else frame_dim * 3


def pool_sequence(seq: np.ndarray, pool_mode: str) -> np.ndarray:
    # seq shape: K x D_frame
    if pool_mode == "mean":
        return np.mean(seq, axis=0)

    k = seq.shape[0]
    if k < 3:
        m = np.mean(seq, axis=0)
        return np.concatenate([m, m, m], axis=0)

    b1 = k // 3
    b2 = (2 * k) // 3
    b1 = max(1, min(b1, k - 2))
    b2 = max(b1 + 1, min(b2, k - 1))

    s1 = seq[:b1]
    s2 = seq[b1:b2]
    s3 = seq[b2:]
    if len(s1) == 0 or len(s2) == 0 or len(s3) == 0:
        m = np.mean(seq, axis=0)
        return np.concatenate([m, m, m], axis=0)

    return np.concatenate(
        [np.mean(s1, axis=0), np.mean(s2, axis=0), np.mean(s3, axis=0)], axis=0
    )


def infer_expected_dim(model, frame_dim_hint: int) -> int:
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None:
        return int(n_features)

    candidates = sorted(set([frame_dim_hint, frame_dim_hint * 3, 42, 84, 126, 252]))
    for dim in candidates:
        try:
            dummy = np.zeros((1, dim), dtype=np.float32)
            model.predict_proba(dummy)
            return dim
        except Exception:
            continue
    fail("Could not infer model input feature dimension.")


def load_display_classes(model, model_path: Path) -> list[str]:
    classes = [str(c) for c in getattr(model, "classes_", [])]
    label_path = model_path.parent / "label_classes.txt"
    if not label_path.exists():
        return classes

    file_classes = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines()]
    file_classes = [c for c in file_classes if c]
    if not file_classes:
        return classes
    if classes and len(file_classes) != len(classes):
        print(
            "WARNING: label_classes.txt length differs from model.classes_. "
            "Using model.classes_ for inference output."
        )
        return classes
    if classes and any(a != b for a, b in zip(file_classes, classes)):
        print(
            "WARNING: label_classes.txt order differs from model.classes_. "
            "Using model.classes_ for inference output."
        )
        return classes
    return file_classes if file_classes else classes


def majority_vote(history: deque[str]) -> str:
    if not history:
        return "UNKNOWN"
    return Counter(history).most_common(1)[0][0]


def set_status(status_box: dict[str, object], text: str, seconds: float = 2.0) -> None:
    status_box["text"] = text
    status_box["until"] = time.time() + seconds
    print(text)


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        fail(f"Model file not found: {args.model}")
    if args.window <= 0:
        fail("--window must be > 0.")
    if args.stride <= 0:
        fail("--stride must be > 0.")
    if args.min_frames <= 0:
        fail("--min_frames must be > 0.")
    if args.history <= 0:
        fail("--history must be > 0.")
    if not (0.0 <= args.threshold <= 1.0):
        fail("--threshold must be in [0, 1].")
    if args.resize is not None and args.resize <= 0:
        fail("--resize must be > 0.")

    model = joblib.load(args.model)
    frame_dim = 84 if args.two_hands else 42
    expected_dim = infer_expected_dim(model, frame_dim_hint=frame_dim)
    display_classes = load_display_classes(model, args.model)
    if not display_classes:
        fail("Model does not provide classes_ and label_classes.txt is not usable.")

    active_pool = args.pool
    current_dim = feature_dim_for_mode(frame_dim, active_pool)
    if current_dim != expected_dim:
        fail(
            f"Model expects {expected_dim} dims but got {current_dim}. "
            "Recreate features/model with matching pool and two_hands settings."
        )

    print("Startup summary")
    print(f"  model: {args.model}")
    print(f"  expected_dims: {expected_dim}")
    print(f"  pool: {active_pool}")
    print(f"  window: {args.window}")
    print(f"  stride: {args.stride}")
    print(f"  min_frames: {args.min_frames}")
    print(f"  history: {args.history}")
    print(f"  threshold: {args.threshold:.2f}")
    print(f"  two_hands: {args.two_hands}")

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2 if args.two_hands else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    draw = mp.solutions.drawing_utils
    hands_conn = mp.solutions.hands.HAND_CONNECTIONS

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        hands.close()
        fail(f"Failed to open camera index {args.camera}")

    feat_buffer: deque[np.ndarray] = deque(maxlen=args.window)
    pred_history: deque[str] = deque(maxlen=args.history)

    threshold = args.threshold
    latest_label = "UNKNOWN"
    majority_label = "UNKNOWN"
    latest_conf = 0.0
    frame_counter = 0
    status_box: dict[str, object] = {"text": "", "until": 0.0}

    prev_t = time.time()
    fps_buf: deque[float] = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.resize is not None:
                h, w = frame.shape[:2]
                if w > 0 and w != args.resize:
                    new_w = args.resize
                    new_h = max(1, int(round(h * (new_w / float(w)))))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if args.show_landmarks and results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    draw.draw_landmarks(frame, hand_lms, hands_conn)

            frame_feat = None
            if results.multi_hand_landmarks:
                if args.two_hands:
                    frame_feat = extract_two_hands_feature(results)
                else:
                    frame_feat = extract_one_hand_feature(results)
            if frame_feat is not None:
                feat_buffer.append(frame_feat)

            frame_counter += 1
            if frame_counter % args.stride == 0 and len(feat_buffer) >= args.min_frames:
                seq = np.stack(feat_buffer, axis=0)
                pooled = pool_sequence(seq, active_pool)
                got_dim = int(pooled.shape[0])
                if got_dim != expected_dim:
                    raise RuntimeError(
                        f"Model expects {expected_dim} dims but got {got_dim}. "
                        "Recreate features/model with matching pool and two_hands settings."
                    )

                probs = model.predict_proba(pooled.reshape(1, -1))[0]
                top_idx = int(np.argmax(probs))
                top_label = display_classes[top_idx]
                top_conf = float(probs[top_idx])

                latest_conf = top_conf
                latest_label = top_label if top_conf >= threshold else "UNKNOWN"
                pred_history.append(latest_label)
                majority_label = majority_vote(pred_history)

            now = time.time()
            dt = max(1e-6, now - prev_t)
            prev_t = now
            fps_buf.append(1.0 / dt)
            fps = float(np.mean(fps_buf)) if fps_buf else 0.0

            waiting_for_hand = frame_feat is None
            hand_text = "WAITING FOR HAND" if waiting_for_hand else "HAND DETECTED"
            hand_color = (0, 0, 255) if waiting_for_hand else (0, 200, 0)

            cv2.putText(
                frame,
                f"Label: {majority_label} (latest: {latest_label})",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Conf: {latest_conf:.2f}  FPS: {fps:.1f}",
                (16, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Buffer: {len(feat_buffer)}/{args.window}  Min: {args.min_frames}",
                (16, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Th: {threshold:.2f}  Pool: {active_pool}  Expected dims: {expected_dim}",
                (16, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                hand_text,
                (16, 138),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                hand_color,
                2,
            )
            cv2.putText(
                frame,
                "q quit | r reset | t/+ up | T/- down | 1 mean | 3 seg3",
                (16, 164),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            if time.time() < float(status_box["until"]):
                cv2.putText(
                    frame,
                    str(status_box["text"]),
                    (16, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 180, 255),
                    2,
                )

            cv2.imshow("WLASL Realtime Sequence Pool", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                feat_buffer.clear()
                pred_history.clear()
                latest_label = "UNKNOWN"
                majority_label = "UNKNOWN"
                latest_conf = 0.0
                set_status(status_box, "Buffers reset.")
            if key in (ord("t"), ord("=")):
                threshold = min(0.99, threshold + 0.05)
                set_status(status_box, f"Threshold: {threshold:.2f}")
            if key in (ord("T"), ord("-")):
                threshold = max(0.0, threshold - 0.05)
                set_status(status_box, f"Threshold: {threshold:.2f}")
            if key == ord("1"):
                new_pool = "mean"
                new_dim = feature_dim_for_mode(frame_dim, new_pool)
                if new_dim != expected_dim:
                    set_status(
                        status_box,
                        f"Pool mean blocked: model expects {expected_dim}, mean gives {new_dim}.",
                    )
                else:
                    active_pool = new_pool
                    set_status(status_box, "Pool mode switched to mean.")
            if key == ord("3"):
                new_pool = "seg3"
                new_dim = feature_dim_for_mode(frame_dim, new_pool)
                if new_dim != expected_dim:
                    set_status(
                        status_box,
                        f"Pool seg3 blocked: model expects {expected_dim}, seg3 gives {new_dim}.",
                    )
                else:
                    active_pool = new_pool
                    set_status(status_box, "Pool mode switched to seg3.")
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# Example command:
# python scripts_words/realtime_sequence_pool.py --model models_words/rf_words_seg3.joblib --pool seg3 --window 60 --stride 5 --history 10 --threshold 0.6 --mirror
