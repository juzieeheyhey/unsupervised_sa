import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# CLASSIFIER = pipeline(
#     "sentiment-analysis",
#     model=model,
#     tokenizer=tokenizer,
#     device=0 if torch.cuda.is_available() else -1
# )


def zero_shot_predictions(texts, classifier, batch_size=16):
    outputs = classifier(texts, batch_size=batch_size)

    predictions = {
        "negative": [],
        "neutral": [],
        "positive": [],
    }

    for i, (text, out) in enumerate(zip(texts, outputs)):
        prediction = {
            "id": i,
            "text": text,
            "label": out["label"],
            "confidence": out["score"]
        }

        if prediction["label"] in predictions:
            predictions[prediction["label"]].append(prediction)

    return predictions


def top_n_predictions(predictions, percent=0.10):
    sorted_preds = sorted(predictions, key=lambda x: x["confidence"], reverse=True)
    k = max(1, int(len(sorted_preds) * percent))
    return sorted_preds[:k]


def select_and_split(predictions, percent=0.10):
    """
    Selects the top 10% predictions for each label and splits the predictions into selected and remaining.
    """
    selected = []
    all_predictions = []

    for label_predictions in predictions.values(): # each label has a list of predictions, go through each list and add to all_predictions
        all_predictions.extend(label_predictions)
        selected.extend(top_n_predictions(label_predictions, percent)) # select the top 10% predictions for each label

    selected_ids = {pred["id"] for pred in selected}
    remaining = [
    {
        "id": pred["id"],
        "text": pred["text"]
    }
    for pred in all_predictions
    if pred["id"] not in selected_ids]

    return selected, remaining

# def test():
#     texts = ["I love this product!", "I hate this product!", "I am neutral about this product.", 'Aaaaaaaaaaaaaa']
#     predictions = zero_shot_predictions(texts)
#     print(predictions)
#     selected, remaining = select_and_split(predictions)
#     print(selected)
#     print(remaining)

# if __name__ == "__main__":
#     test()