import cv2
import os
import time
import mediapipe as mp
import math
import threading
import numpy as np

cap = cv2.VideoCapture(0)
prev_time = time.time()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

target_volume = 50
last_volume = -1
last_update_time = 0
UPDATE_INTERVAL = 0.05

def volume_worker():
    global target_volume
    last_sent = -1
    while True:
        if abs(target_volume - last_sent) >= 2:
            os.system(f"osascript -e 'set volume output volume {target_volume}'")
            last_sent = target_volume
        time.sleep(0.05)

threading.Thread(target=volume_worker, daemon=True).start()

hold = False
prev_left_present = False

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    left_present = False

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            hand_label = results.multi_handedness[idx].classification[0].label
            lm = hand_landmarks.landmark
            now = time.time()

            if hand_label == "Left":
                left_present = True
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                continue

            if hand_label == "Right":
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                if hold:
                    continue

                x1, y1 = int(lm[4].x * w), int(lm[4].y * h)
                x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

                cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
                cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                length = math.hypot(x2 - x1, y2 - y1)
                vol_percent = int(np.interp(length, [60, 500], [0, 100]))

                if (
                    abs(vol_percent - last_volume) >= 2 and
                    now - last_update_time > UPDATE_INTERVAL
                ):
                    target_volume = vol_percent
                    last_volume = vol_percent
                    last_update_time = now

    if left_present and not prev_left_present:
        hold = not hold

    prev_left_present = left_present

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, f"Volume: {target_volume}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f"HOLD: {'ON' if hold else 'OFF'}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if hold else (0, 255, 0), 2)

    cv2.imshow("Hand Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
