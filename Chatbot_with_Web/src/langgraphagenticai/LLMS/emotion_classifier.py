from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd

# Load model once at top level
MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

# Emotion labels from the model
id2label = model.config.id2label

def predict_emotions(text: str) -> dict:
    """
    Predict GoEmotions (joy, sadness, etc.) for given input text.
    Returns top emotions as dict of scores.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.sigmoid(outputs.logits)[0].cpu().numpy()

    result = {id2label[i]: float(score) for i, score in enumerate(scores)}

    # Optional: filter most relevant emotions or scale/normalize
    top_emotions = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:5])
    return top_emotions
