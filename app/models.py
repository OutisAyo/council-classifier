"""
Loads all trained models and supporting components once, on first use.

Loading is lazy so that importing this package (for tests, linting, ``--help``)
does not require the ~200 MB of pickled artifacts to be present. The first call
to :func:`get_store` builds the singleton; every later call returns it.
"""
import pickle
from pathlib import Path

import joblib
import pandas as pd

# Path to the models folder, relative to the project root
MODELS_DIR = Path(__file__).parent.parent / "models"

REQUIRED_FILES = (
    "department_pipeline_rf.pkl",
    "priority_pipeline_final.pkl",
    "retrieval_vectorizer.pkl",
    "response_templates.pkl",
    "urgency_lexicon.pkl",
    "dataset_for_retrieval.pkl",
)


class ModelsNotFoundError(RuntimeError):
    """Raised when the trained artifacts are missing from ``models/``."""


class ModelStore:
    """Holds all loaded models and data in one place."""

    def __init__(self) -> None:
        missing = [f for f in REQUIRED_FILES if not (MODELS_DIR / f).exists()]
        if missing:
            raise ModelsNotFoundError(
                f"Missing model artifacts in {MODELS_DIR}: {', '.join(missing)}. "
                "Run `python scripts/download_models.py` to fetch them."
            )

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


_store: "ModelStore | None" = None


def get_store() -> "ModelStore":
    """Return the shared :class:`ModelStore`, building it on first call."""
    global _store
    if _store is None:
        _store = ModelStore()
    return _store
