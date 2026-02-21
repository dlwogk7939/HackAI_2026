import cv2
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

BODY_LANDMARKS = set(range(11, 33))
BODY_CONNECTIONS = [
    c
    for c in mp_holistic.POSE_CONNECTIONS
    if c[0] in BODY_LANDMARKS and c[1] in BODY_LANDMARKS
]


def draw_pose_body(image, pose_landmarks) -> None:
    h, w, _ = image.shape

    for start_idx, end_idx in BODY_CONNECTIONS:
        start = pose_landmarks.landmark[start_idx]
        end = pose_landmarks.landmark[end_idx]
        cv2.line(
            image,
            (int(start.x * w), int(start.y * h)),
            (int(end.x * w), int(end.y * h)),
            (0, 255, 0),
            2,
        )

    for idx in BODY_LANDMARKS:
        lm = pose_landmarks.landmark[idx]
        cv2.circle(
            image,
            (int(lm.x * w), int(lm.y * h)),
            3,
            (0, 0, 255),
            -1,
        )

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 좌우반전 보정
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = holistic.process(rgb)
        rgb.flags.writeable = True

        # 포즈(얼굴 제외)
        if results.pose_landmarks:
            draw_pose_body(frame, results.pose_landmarks)

        # 왼손
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        # 오른손
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
            )

        cv2.imshow("MediaPipe Holistic", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
