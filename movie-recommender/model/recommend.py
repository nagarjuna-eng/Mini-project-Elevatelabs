import pickle
import requests
import re
import urllib.parse
import pandas as pd

# =========================================================
# LOAD SAVED DATA (MOVIES + SIMILARITY MATRIX)
# =========================================================

movies = pickle.load(open('model/movies.pkl', 'rb'))
similarity = pickle.load(open('model/similarity.pkl', 'rb'))

# =========================================================
# LOAD RATINGS DATA
# =========================================================

ratings = pd.read_csv('data/ratings.csv')

# =========================================================
# COMPUTE AVERAGE RATINGS FOR EACH MOVIE
# =========================================================

avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']

# Merge ratings with movies dataset
movies = movies.merge(avg_ratings, on='movieId', how='left')

# Fill missing ratings with 0
movies['avg_rating'] = movies['avg_rating'].fillna(0)

# =========================================================
# TMDB API KEY
# =========================================================

API_KEY = "YOUR_API_KEY_HERE"

# =========================================================
# FUNCTION: FETCH MOVIE POSTER FROM TMDB
# =========================================================

def fetch_poster(movie_title):
    """
    Fetch poster image from TMDB using movie title.
    Includes cleaning, encoding, and fallback handling.
    """

    try:
        # Remove year from movie title
        clean_title = re.sub(r"\(\d{4}\)", "", movie_title).strip()

        # Encode query to handle spaces and special characters
        query = urllib.parse.quote(clean_title)

        # Create API URL
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={query}"

        # Send request
        response = requests.get(url)

        # Check response status
        if response.status_code != 200:
            print("API Error:", response.status_code)
            return "https://via.placeholder.com/500x750?text=No+Image"

        data = response.json()

        # Check if results exist
        if not data.get('results'):
            print("No poster found for:", clean_title)
            return "https://via.placeholder.com/500x750?text=No+Image"

        # Get poster path
        poster_path = data['results'][0].get('poster_path')

        # Return full image URL
        if poster_path:
            return "https://image.tmdb.org/t/p/w500" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"

    except Exception as e:
        print("Poster Fetch Error:", e)
        return "https://via.placeholder.com/500x750?text=No+Image"

# =========================================================
# FUNCTION: RECOMMEND MOVIES
# =========================================================

def recommend(movie_name, selected_genre=None):
    """
    Returns:
    - Top 5 recommended movie names
    - Their posters
    - Their ratings
    """

    recommended_movies = []
    posters = []
    ratings_list = []

    # =====================================================
    # FIND SELECTED MOVIE INDEX
    # =====================================================

    try:
        movie_index = movies[movies['title'] == movie_name].index[0]
    except IndexError:
        return [], [], []

    # =====================================================
    # GET SIMILARITY SCORES
    # =====================================================

    distances = similarity[movie_index]

    # Sort movies based on similarity score
    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:25]   # take more for filtering

    # =====================================================
    # FIRST PASS: APPLY GENRE FILTER
    # =====================================================

    for i in movie_list:
        movie = movies.iloc[i[0]]

        if selected_genre:
            if selected_genre not in movie['genres']:
                continue

        recommended_movies.append(movie['title'])
        posters.append(fetch_poster(movie['title']))
        ratings_list.append(round(movie['avg_rating'], 2))

        if len(recommended_movies) == 5:
            return recommended_movies, posters, ratings_list

    # =====================================================
    # SECOND PASS: FALLBACK (NO FILTER)
    # =====================================================

    for i in movie_list:
        movie = movies.iloc[i[0]]

        if movie['title'] not in recommended_movies:
            recommended_movies.append(movie['title'])
            posters.append(fetch_poster(movie['title']))
            ratings_list.append(round(movie['avg_rating'], 2))

        if len(recommended_movies) == 5:
            break

    return recommended_movies, posters, ratings_list