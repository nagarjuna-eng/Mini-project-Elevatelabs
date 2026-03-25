from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(job_description, resumes_dict):
    """
    Compute similarity scores between job description and resumes
    and normalize them into percentage (0–100)
    """

    if len(resumes_dict) == 0:
        raise ValueError("No resumes to compare.")

    # Extract names and text
    resume_names = list(resumes_dict.keys())
    resume_texts = list(resumes_dict.values())

    # Combine JD + resumes
    documents = [job_description] + resume_texts

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Separate vectors
    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    # Compute cosine similarity
    similarities = cosine_similarity(jd_vector, resume_vectors)

    # Raw similarity scores
    similarity_scores = {
        resume_names[i]: similarities[0][i]
        for i in range(len(resume_names))
    }

    # ---------------- NORMALIZATION (0–100%) ----------------
    max_score = max(similarity_scores.values())

    normalized_scores = {
        k: (v / max_score) * 100 if max_score > 0 else 0
        for k, v in similarity_scores.items()
    }

    return normalized_scores