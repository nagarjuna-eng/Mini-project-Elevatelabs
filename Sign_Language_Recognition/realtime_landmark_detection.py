import cv2
import mediapipe as mp
import numpy as np
import os
import pyttsx3
from collections import deque, Counter
from tensorflow.keras.models import load_model
from spellchecker import SpellChecker

# =============================
# CONFIGURATION
# =============================
MODEL_PATH = "model/landmark_model.h5"
DATA_DIR = "landmark_data"
CAMERA_INDEX = 1
SMOOTHING_FRAMES = 15
CONFIDENCE_THRESHOLD = 0.75
WORD_LOG_FILE = "words_log.txt"

# =============================
# LOAD MODEL & LABELS
# =============================
model = load_model(MODEL_PATH)
labels = sorted([f.replace(".csv", "") for f in os.listdir(DATA_DIR)])

# =============================
# SPELL CHECKER
# =============================
spell = SpellChecker()

# =============================
# TEXT TO SPEECH
# =============================
engine = pyttsx3.init()

# =============================
# MEDIAPIPE
# =============================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(CAMERA_INDEX)

prediction_buffer = deque(maxlen=SMOOTHING_FRAMES)
current_letter = ""
current_word = ""
suggestions = []

# =============================
# UI FUNCTION
# =============================
def draw_ui(frame, h, w):
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)
    cv2.putText(frame, "Sign Language Recognition (Landmark + NLP)",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2)

    cv2.rectangle(frame, (0, h - 70), (w, h), (30, 30, 30), -1)
    cv2.putText(frame,
                "SPACE: Letter | TAB: Suggestion | ENTER: Save | BACKSPACE | S: Speak | Q",
                (20, h - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)

# =============================
# MAIN LOOP
# =============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    draw_ui(frame, h, w)

    detected = False
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row = np.array(row).reshape(1, -1)
            preds = model.predict(row, verbose=0)
            confidence = np.max(preds)
            label = labels[np.argmax(preds)]

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(label)
                current_letter = Counter(prediction_buffer).most_common(1)[0][0]
                detected = True

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    if not detected:
        prediction_buffer.clear()
        current_letter = ""

    # =============================
    # SUGGESTIONS
    # =============================
    if len(current_word) >= 2:
        suggestions = list(spell.candidates(current_word.upper()))[:3]
    else:
        suggestions = []

    # =============================
    # DISPLAY LETTER
    # =============================
    cv2.rectangle(frame, (20, 80), (300, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Detected: {current_letter}",
                (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 2)

    # =============================
    # DISPLAY WORD
    # =============================
    cv2.rectangle(frame, (320, 80), (w - 20, 150), (0, 0, 0), -1)
    cv2.putText(frame, f"Word: {current_word}",
                (330, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # =============================
    # DISPLAY SUGGESTIONS
    # =============================
    cv2.putText(frame, "Suggestions:",
                (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    for i, sug in enumerate(suggestions):
        cv2.putText(frame, f"{i+1}. {sug}",
                    (20, 220 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 200, 100), 2)

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    # =============================
    # CONTROLS
    # =============================
    if key == ord(' '):  # Confirm letter
        if current_letter:
            current_word += current_letter
            prediction_buffer.clear()

    elif key == 9:  # TAB → Accept suggestion
        if suggestions:
            current_word = suggestions[0]

    elif key == 13:  # ENTER → Save word
        if current_word:
            corrected = spell.correction(current_word)
            with open(WORD_LOG_FILE, "a") as f:
                f.write(corrected.upper() + "\n")
            current_word = ""

    elif key == 8:  # BACKSPACE
        current_word = current_word[:-1]

    elif key == ord('s'):  # Speak
        if current_word:
            engine.say(current_word)
            engine.runAndWait()

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()