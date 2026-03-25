# 🖐️ Real-Time Sign Language Recognition (Landmark-Based)

By Pranav SP

A real-time sign language recognition system that converts hand gestures (A–Z) into text using MediaPipe hand landmarks and a deep learning model, enhanced with spell correction and intelligent word suggestions.

This project is stable, accurate, and demo/viva-ready, designed as a major-project-level implementation.

# ✨ Features

✅ Real-time recognition of A–Z hand signs

✅ Landmark-based model (21 hand landmarks × x, y, z)

✅ High accuracy & stable predictions

✅ Background and lighting independent

✅ Manual letter confirmation (no random characters)

✅ Word formation and storage

✅ Spell correction (e.g., HLELO → HELLO)

✅ Smart word suggestions (e.g., H-E-L → HELLO)

✅ Text-to-speech output

✅ Clean, professional UI overlay

# 🧠 Why Landmark-Based?

Instead of using raw images, this project uses MediaPipe hand landmarks, which makes the system:

Robust to lighting changes

Independent of background noise

Faster to train

Much more stable in real time

This approach is commonly used in production-level gesture recognition systems.

# 🛠️ Tech Stack

Python 3.9+

OpenCV – Webcam & UI rendering

MediaPipe – Hand landmark detection

TensorFlow / Keras – Deep learning model

NumPy, Pandas – Data processing

pyttsx3 – Text-to-speech

pyspellchecker – Spell correction & suggestions

# 📁 Project Structure
sign_language_recognition/

│
├── landmark_data/

│   ├── A.csv

│   ├── B.csv

│   └── ... Z.csv

│
├── model/

│   └── landmark_model.h5

│
├── collect_landmarks.py

├── train_landmark_model.py

├── realtime_landmark_detection.py

├── requirements.txt

├── words_log.txt

└── README.md

# ⚙️ Installation
pip install opencv-python mediapipe tensorflow numpy pandas pyttsx3 pyspellchecker

# 🚀 How to Run the Project
1️⃣ Collect Landmark Data

Capture hand landmarks for each letter:

python collect_landmarks.py --label A
python collect_landmarks.py --label B
...
python collect_landmarks.py --label Z

Hold each sign for 5–8 seconds

Collect ~150–300 samples per letter

2️⃣ Train the Landmark Model
python train_landmark_model.py

Training time: seconds

Typical accuracy: 90–98%

Model saved as: model/landmark_model.h5

3️⃣ Run Real-Time Detection
python realtime_landmark_detection.py

# 🎮 Controls
Key	          Action

SPACE	     ----   Confirm detected letter

TAB	     ----     Accept suggested word

ENTER	   ----     Save current word

BACKSPACE	 ----   Delete last letter

S	        ----    Speak the word

Q	       ----     Quit application

# 📝 Output

words_log.txt
Stores all finalized words, one per line
