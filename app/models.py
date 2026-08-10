"""
Loads all trained models and supporting components once at startup.
Keeping this separate means the models load a single time, not on every request.
"""
import joblib
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Path to the models folder, relative to the project root
MODELS_DIR = Path(__file__).parent.parent / "models"


class ModelStore:
    """Holds all loaded models and data in one place."""

    def __init__(self):
        self.department_pipeline = joblib.load(MODELS_DIR / "department_pipeline_rf.pkl")
        self.priority_pipeline = joblib.load(MODELS_DIR / "priority_pipeline_final.pkl")
        self.retrieval_vectorizer = joblib.load(MODELS_DIR / "retrieval_vectorizer.pkl")

        with open(MODELS_DIR / "response_templates.pkl", "rb") as f:
            self.templates = pickle.load(f)

        with open(MODELS_DIR / "urgency_lexicon.pkl", "rb") as f:
            self.urgency_lexicon = pickle.load(f)

        self.dataset = pd.read_pickle(MODELS_DIR / "dataset_for_retrieval.pkl")
        self.historical_vectors = self.retrieval_vectorizer.transform(
            self.dataset["request_text"]
        )

        # Flatten the urgency lexicon into one list for easy checking
        self.all_urgency_words = [
            word for group in self.urgency_lexicon.values() for word in group
        ]


# A single shared instance the rest of the app imports and uses
store = ModelStore()