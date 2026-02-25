import sys
import os
import pytest

# Add the parent directory to the system path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import load_artifacts, predict

# Test cases for the model prediction
@pytest.fixture(scope="session")
def model_setup():
    tokenizer, model, label_mapping = load_artifacts()
    return tokenizer, model, label_mapping

# Test that prediction returns a valid class and probabilities
def test_prediction_returns_valid_class(model_setup):
    tokenizer, model, label_mapping = model_setup

    text = "How intermittent fasting impacts long-term wellness"

    pred_id, probs = predict(text, tokenizer, model)

    assert 0 <= pred_id < len(label_mapping)
    assert len(probs) == len(label_mapping)

# Test that empty input raises a ValueError
def test_empty_input_raises_error(model_setup):
    tokenizer, model, _ = model_setup

    with pytest.raises(ValueError):
        predict("", tokenizer, model)