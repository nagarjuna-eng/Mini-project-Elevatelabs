import streamlit as st
import pickle
import sys
import os

# =========================================================
# FIX IMPORT PATH
# =========================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.recommend import recommend

# =========================================================
# LOAD MOVIES DATA
# =========================================================

movies = pickle.load(open('model/movies.pkl', 'rb'))

# =========================================================
# EXTRACT GENRES
# =========================================================

all_genres = set()

for g in movies['genres']:
    for genre in g.split('|'):
        all_genres.add(genre)

genre_list = sorted(list(all_genres))

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# =========================================================
# CUSTOM CSS (NETFLIX STYLE)
# =========================================================

st.markdown("""
<style>
body {
    background-color: #141414;
    color: white;
}

h1 {
    color: #E50914;
    text-align: center;
    font-size: 40px;
}

.stTextInput input {
    background-color: #333;
    color: white;
}

.stSelectbox label {
    color: white;
}

.stButton>button {
    background-color: #E50914;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}

img {
    border-radius: 12px;
    transition: transform 0.3s ease;
}

img:hover {
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown("<h1>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# =========================================================
# SEARCH BAR
# =========================================================

search = st.text_input("🔍 Search for a movie")

movie_list = movies['title'].values

if search:
    filtered_movies = [m for m in movie_list if search.lower() in m.lower()]
else:
    filtered_movies = movie_list

# =========================================================
# GENRE FILTER
# =========================================================

selected_genre = st.selectbox("🎭 Filter by Genre (optional)", ["All"] + genre_list)

if selected_genre == "All":
    selected_genre = None

# =========================================================
# SELECT MOVIE
# =========================================================

selected_movie = st.selectbox("🎥 Select a movie", filtered_movies)

# =========================================================
# BUTTON ACTION
# =========================================================

if st.button("🎯 Show Recommendations"):

    names, posters, ratings = recommend(selected_movie, selected_genre)

    # Warning if no results initially
    if len(names) == 0:
        st.warning("⚠️ No exact match found. Showing similar movies instead.")

    st.markdown("## 🍿 Recommended for You")

    cols = st.columns(5)

    for i in range(len(names)):
        with cols[i]:
            st.image(posters[i])
            st.markdown(f"**{names[i]}**")
            st.markdown(f"⭐ Rating: {ratings[i]}")