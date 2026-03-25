# 📰 Fake News Detection System

A Machine Learning and Deep Learning based system that classifies news articles as **Real or Fake** using Natural Language Processing (NLP) techniques.

---

## 📌 Project Overview

Fake news spreads rapidly through digital platforms and can mislead people.  
This project builds an automated system to detect fake news articles using:

- Machine Learning (TF-IDF + Logistic Regression)
- Deep Learning (LSTM Neural Network)

The system allows users to input news text and receive:
- 🟢 Real News
- 🔴 Fake News
- 📊 Confidence Score

---

## 🚀 Features

- Text preprocessing (cleaning & normalization)
- TF-IDF feature extraction
- Logistic Regression model
- Naive Bayes model (comparison)
- Deep Learning model using LSTM
- Accuracy, Confusion Matrix & ROC Curve
- Real-time news prediction
- Streamlit Web Interface

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow & Keras
- Matplotlib
- Streamlit
- Kaggle Fake & Real News Dataset

---

## 📂 Project Structure

FakeNewsProject/
│── main.py
│── deep_model.py
│── app.py
│── model.pkl
│── vectorizer.pkl
│── Fake.csv
│── True.csv
│── README.md

---

## ⚙️ Installation

Install required libraries:

py -m pip install pandas scikit-learn nltk tensorflow streamlit matplotlib

---

## ▶️ Run the Project

Run Machine Learning Model:
py main.py

Run Deep Learning Model:
py deep_model.py

Run Streamlit Web App:
py -m streamlit run app.py

---

## 📊 Model Performance

- Logistic Regression Accuracy: ~95–99%
- LSTM Deep Learning Accuracy: ~99%

---

## 👨‍💻 Author

Nagarjun  
AI & ML Internship Project  
2026