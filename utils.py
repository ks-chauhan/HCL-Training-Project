import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

MODEL_PATH = "the-kshitij-chauhan/news-topic-classifier"

# Load model artifacts (tokenizer, model, label mapping)
def load_artifacts():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        label_path = hf_hub_download(
            repo_id=MODEL_PATH,
            filename="label_mapping.json"
        )

        with open(label_path, "r") as f:
            label_mapping = json.load(f)
        
        return tokenizer, model, label_mapping
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {e}")

# Predict function
def predict(text, tokenizer, model):
    
    # Validate input
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]
    
    # Test inference
    with torch.inference_mode():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    predicted_label_id = torch.argmax(probs, dim=1).item()

    return predicted_label_id, probs.squeeze().tolist()
