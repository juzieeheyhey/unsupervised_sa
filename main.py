import pandas as pd
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from zero_shot import zero_shot_predictions, select_and_split
from finetune import fine_tune_model, ft_predictions
from data_preprocess import load_data

MODEL_NAME = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

classifier = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_unlabeled_data(texts):
    return [{"id": i, "text": text} for i, text in enumerate(texts)]


def print_pseudo_labels(data, title="Pseudo-labels", max_examples=10):
    print(f"\n=== {title} ===")
    for item in data[:max_examples]:
        print(f'[{item["label"]}] ({item["confidence"]:.3f}) -> {item["text"]}')
    if len(data) > max_examples:
        print(f"... showing {max_examples} of {len(data)} examples")


def self_training_loop(
    texts,
    classifier,
    percent=0.10,
    max_iterations=15,
    min_new_samples=1,
    model_name=MODEL_NAME,
    zero_shot_batch_size=16,
    ft_batch_size=16,
    verbose=True
):
    unlabeled_data = initialize_unlabeled_data(texts)
    output_dir = "./finetuned_model"

    ### INITIALIZATION BLOCK: ZERO-SHOT PREDICTIONS 
    zero_shot_preds = zero_shot_predictions(
        unlabeled_data,
        classifier,
        batch_size=zero_shot_batch_size
    )

    train_pool, remaining = select_and_split(zero_shot_preds, percent)

    if verbose:
        print("Zero-shot predictions complete.")
        print_pseudo_labels(train_pool, title="Initial pseudo-labels")

    history = [{
        "iteration": 0,
        "new_selected": len(train_pool),
        "train_pool_size": len(train_pool),
        "remaining_size": len(remaining),
        "new_negative": sum(1 for x in train_pool if x["label"] == "negative"),
        "new_neutral": sum(1 for x in train_pool if x["label"] == "neutral"),
        "new_positive": sum(1 for x in train_pool if x["label"] == "positive"),
    }]

    current_model_name = model_name
    model = None
    tokenizer = None

    ### SELF-TRAINING LOOP 
    for i in range(1, max_iterations + 1):
        if verbose:
            print(f"\n--- Iteration {i} ---")

        if len(remaining) == 0:
            if verbose:
                print("No remaining data. Stop.")
            break

        if verbose:
            print("Fine-tuning model...")

        model, tokenizer = fine_tune_model(
            selected_data=train_pool,
            model_name=current_model_name,
            output_dir=output_dir
        )

        current_model_name = output_dir

        if verbose:
            print("Predicting on remaining data...")

        predictions = ft_predictions(
            data=remaining,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=ft_batch_size
        )

        new_selected, new_remaining = select_and_split(predictions, percent)

        if verbose:
            print_pseudo_labels(new_selected, title=f"Iteration {i} new pseudo-labels")

        if len(new_selected) < min_new_samples:
            if verbose:
                print(f"Fewer than {min_new_samples} new samples selected. Stop.")
            break

        train_pool.extend(new_selected)
        remaining = new_remaining

        history.append({
            "iteration": i,
            "new_selected": len(new_selected),
            "train_pool_size": len(train_pool),
            "remaining_size": len(remaining),
            "new_negative": sum(1 for x in new_selected if x["label"] == "negative"),
            "new_neutral": sum(1 for x in new_selected if x["label"] == "neutral"),
            "new_positive": sum(1 for x in new_selected if x["label"] == "positive"),
        })

    return train_pool, remaining, history


if __name__ == "__main__":
    FILE_PATH = "train_Vietnamese_eng.csv"
    texts = load_data(FILE_PATH)["text"].tolist()[:100]
   
    train_pool, remaining, history = self_training_loop(
        texts=texts,
        classifier=classifier,
        percent=0.10,
        max_iterations=15,
        min_new_samples=1,
        model_name=MODEL_NAME,
        zero_shot_batch_size=8,
        ft_batch_size=8,
        verbose=True
    )

    history_df = pd.DataFrame(history)
    print("\n=== History ===")
    print(history_df)

    print("\nFinal train pool size:", len(train_pool))
    print("Final remaining size:", len(remaining))