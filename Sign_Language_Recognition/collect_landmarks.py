import cv2
import mediapipe as mp
import csv
import os
import argparse

# -----------------------------
# ARGUMENT
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
args = parser.parse_args()
label = args.label

# -----------------------------
# SETUP
# -----------------------------
os.makedirs("landmark_data", exist_ok=True)
file_path = f"landmark_data/{label}.csv"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)  # change to 0 if needed
count = 0

print(f"[INFO] Collecting landmarks for label: {label}")
print("[INFO] Hold the sign steady for 5–8 seconds")
print("[INFO] Press Q to stop")

# -----------------------------
# WRITE CSV
# -----------------------------
with open(file_path, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        h, w, _ = frame.shape

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                row = []
                for lm in hand.landmark:
                    row.extend([lm.x, lm.y, lm.z])
                writer.writerow(row)
                count += 1

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # -----------------------------
        # UI OVERLAY
        # -----------------------------
        cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"Collecting label: {label}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Samples collected: {count}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.putText(frame, "Hold sign steady | Press Q to stop",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Landmark Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print(f"[DONE] Collected {count} samples for label {label}")