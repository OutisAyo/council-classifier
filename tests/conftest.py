import pytest

from app.models import MODELS_DIR, REQUIRED_FILES

MODELS_PRESENT = all((MODELS_DIR / f).exists() for f in REQUIRED_FILES)

requires_models = pytest.mark.skipif(
    not MODELS_PRESENT,
    reason="model artifacts not present (run `python scripts/download_models.py`)",
)
