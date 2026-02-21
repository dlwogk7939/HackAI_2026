import glob
import cv2
import mediapipe as mp

root = "dataset/alphabet"
paths = glob.glob(f"{root}/**/*.*", recursive=True)
paths = [p for p in paths if p.lower().endswith((".jpg", ".jpeg", ".png"))][:100]

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

ok = 0
for p in paths:
    img = cv2.imread(p)
    if img is None:
        print("read fail:", p)
        continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        ok += 1

print(f"Detected hands in {ok}/{len(paths)} images")