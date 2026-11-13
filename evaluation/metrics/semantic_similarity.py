
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
_model = None

def get_model():
    global _model

    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    return _model


def calculate_similarity(expected_answer, actual_answer):
    """Calculate semantic similarity between two answers."""

    if not str(expected_answer).strip():
        return 0.0

    if not str(actual_answer).strip():
        return 0.0

    model = get_model()

    embeddings = model.encode(
        [str(expected_answer), str(actual_answer)]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(float(similarity), 4)
