import cv2
import os
import time
import mediapipe as mp
import math
import threading
import numpy as np

cap = cv2.VideoCapture(0)
prev_time = time.time()


target_volume = 0
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def volume_worker():
    global target_volume
    last_sent = -1
    while True:
        if abs(target_volume - last_sent) >= 2:
            os.system(f"osascript -e 'set volume output volume {target_volume}'")
            last_sent = target_volume
        time.sleep(0.05)


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

last_volume = -1
last_update_time = 0
UPDATE_INTERVAL = 0.05

threading.Thread(target=volume_worker, daemon=True).start()

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            x1, y1 = int(lm[4].x * w), int(lm[4].y * h) # getitng x and y depending on what mediapipe says times times the screen dimesion
            x2, y2 = int(lm[8].x * w), int(lm[8].y * h)

            cv2.circle(frame, (x1, y1), 10, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 10, (0, 255, 0), -1)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

            length = math.hypot(x2 - x1, y2 - y1)

            vol_percent = int(np.interp(length, [60, 500], [0, 100]))

            current_time = time.time()

            if (
                    abs(vol_percent - last_volume) >= 2 and
                    current_time - last_update_time > UPDATE_INTERVAL
            ):
                target_volume = vol_percent
                last_volume = vol_percent
                last_update_time = current_time

            cv2.putText(
                frame,
                f"Distance: {int(length)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

# FPS COUNTER LOGIC
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Hand Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
