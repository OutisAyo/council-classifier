import pytest

from app.models import MODELS_DIR, REQUIRED_FILES

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def _is_real_artifact(path):
    """True if the file exists and is not an unresolved Git LFS pointer."""
    if not path.exists():
        return False
    with open(path, "rb") as f:
        return not f.read(len(_LFS_POINTER_PREFIX)).startswith(_LFS_POINTER_PREFIX)


MODELS_PRESENT = all(_is_real_artifact(MODELS_DIR / f) for f in REQUIRED_FILES)

requires_models = pytest.mark.skipif(
    not MODELS_PRESENT,
    reason="model artifacts not present (run `git lfs pull`)",
)
