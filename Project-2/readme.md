# 🎬 Movie Recommendation System

## 📌 Overview

This project is a Machine Learning based Movie Recommendation System that suggests movies based on user preferences. It uses content-based filtering techniques and integrates TMDB API to display movie posters along with recommendations.

---

## 🚀 Features

* 🎯 Content-based movie recommendation using cosine similarity
* 🔍 Search functionality to find movies quickly
* 🎭 Genre-based filtering for better personalization
* ⭐ Displays average ratings of movies
* 🎬 Shows movie posters using TMDB API
* 💻 Interactive user interface built with Streamlit

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* TMDB API

---

## 📂 Project Structure

movie-recommender/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│
├── model/
│   ├── recommender.py
│   ├── recommend.py
│
├── app/
│   ├── app.py
│
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

git clone <your-repo-link>
cd movie-recommender

### 2️⃣ Create Virtual Environment

python -m venv venv
venv\Scripts\activate

### 3️⃣ Install Dependencies

pip install -r requirements.txt

### 4️⃣ Run Model File

python model/recommender.py

### 5️⃣ Run Application

streamlit run app/app.py

---

## 🔑 API Setup

1. Create an account on TMDB
2. Generate API Key
3. Add the key in `model/recommend.py`

API_KEY = "your_api_key_here"

---

## 🧠 How It Works

1. Movie genres are converted into numerical vectors using CountVectorizer
2. Cosine similarity is calculated between movies
3. The system finds the most similar movies
4. Top 5 movies are recommended
5. Posters are fetched using TMDB API
6. Ratings are calculated from user data

---

## 📊 Dataset

* MovieLens Latest Small Dataset
* Contains movie details and user ratings

---

## 🎯 Future Improvements

* Add collaborative filtering
* Add user login system
* Deploy application online
* Improve recommendation accuracy

---

## 👨‍💻 Author

Nagarjuna T R
