import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.inference.predictor import resolve_model_dir  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def model_dir():
    """Skip the real-model tests when no checkpoint exists yet, rather than
    passing on mocks and pretending the model was verified."""
    try:
        return resolve_model_dir()
    except FileNotFoundError:
        pytest.skip("no trained DistilBERT checkpoint available")


@pytest.fixture(scope="session")
def predictor(model_dir, cfg):
    from src.inference.predictor import DistilBertPredictor

    return DistilBertPredictor(model_dir, cfg=cfg, device="cpu")
