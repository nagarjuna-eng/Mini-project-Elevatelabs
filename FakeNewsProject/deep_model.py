import pandas as pd
import numpy as np
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score

print("Loading dataset...")

# Load dataset
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

fake = fake.head(5000)
true = true.head(5000)

data = pd.concat([fake, true])
data = data.sample(frac=1)

X = data["text"]
y = data["label"]

# Clean text
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

X = X.apply(clean_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Tokenization
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

print("Text tokenized and padded ✅")

# Build LSTM Model
model = Sequential()
model.add(Embedding(max_words, 128))
model.add(LSTM(64))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

print("Training LSTM model...")

model.fit(X_train_pad, y_train, epochs=3, batch_size=64)

# Evaluate
loss, accuracy = model.evaluate(X_test_pad, y_test)

print("Deep Learning Accuracy:", accuracy)

# ===============================
# Predict Custom News
# ===============================

def predict_news(news_text):
    news_text = clean_text(news_text)
    seq = tokenizer.texts_to_sequences([news_text])
    padded = pad_sequences(seq, maxlen=max_len)

    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        print(f"\n🟢 Real News (Confidence: {prediction*100:.2f}%)")
    else:
        print(f"\n🔴 Fake News (Confidence: {(1-prediction)*100:.2f}%)")

# Take input from user
user_news = input("\nEnter a news article to check:\n")
predict_news(user_news)