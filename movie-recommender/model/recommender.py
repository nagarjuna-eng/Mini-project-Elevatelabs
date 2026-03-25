import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import pickle

movies = pd.read_csv('data/movies.csv')

movies['genres'] = movies['genres'].str.replace('|', ' ')

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['genres']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(similarity, open('model/similarity.pkl', 'wb'))
pickle.dump(movies, open('model/movies.pkl', 'wb'))

print("Model saved successfully!")