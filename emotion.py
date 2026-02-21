from __future__ import annotations

import math

SURPRISE_OPEN_T = 0.22
HAPPY_SMILE_T = 0.03


def euclidean(p1, p2) -> float:
    dx = float(p1.x) - float(p2.x)
    dy = float(p1.y) - float(p2.y)
    return math.hypot(dx, dy)


def compute_emotion_from_facemesh(landmarks, image_w: int, image_h: int) -> str:
    # image_w/image_h are part of the interface for future pixel-space tuning.
    _ = image_w
    _ = image_h

    eye_dist = euclidean(landmarks[33], landmarks[263])
    if eye_dist <= 1e-6:
        return "neutral"

    mouth_open = euclidean(landmarks[13], landmarks[14]) / eye_dist

    center_y = (float(landmarks[13].y) + float(landmarks[14].y)) / 2.0
    left_corner_y = float(landmarks[61].y)
    right_corner_y = float(landmarks[291].y)
    smile_score = (
        ((center_y - left_corner_y) + (center_y - right_corner_y)) / (2.0 * eye_dist)
    )

    if mouth_open > SURPRISE_OPEN_T:
        return "surprised"
    if smile_score > HAPPY_SMILE_T:
        return "happy"
    return "neutral"

