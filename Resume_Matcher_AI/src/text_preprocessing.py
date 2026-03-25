import re
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenize
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Join back
    return " ".join(words)