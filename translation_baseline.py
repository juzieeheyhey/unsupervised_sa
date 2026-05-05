import argparse
import ast
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)

LABELS = ["negative", "neutral", "positive"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
SENTIMENT_LABEL_MAP = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df, text_column="text", label_column="label"):
    df = df[[text_column, label_column]].copy()
    df = df.dropna(subset=[text_column, label_column])
    df[text_column] = df[text_column].astype(str).str.strip()
    df[label_column] = df[label_column].astype(str).str.lower().str.strip()
    df = df[df[text_column] != ""]
    df = df[df[label_column].isin(label2id.keys())]
    return df.reset_index(drop=True)


# ── Translation ───────────────────────────────────────────────────────────────

def load_translation_model():
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME, use_safetensors=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, src_lang="rus_Cyrl", batch_size=16):
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = "eng_Latn"
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
                max_length=512
            )
        results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return results


def get_translations(df, text_column, tokenizer, model, device, src_lang="rus_Cyrl", batch_size=16):
    texts = df[text_column].tolist()
    translations = translate_batch(texts, tokenizer, model, device, src_lang=src_lang, batch_size=batch_size)
    df = df.copy()
    df["translated_text"] = translations
    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────

def run_sentiment(texts, batch_size=16):
    device = 0 if torch.cuda.is_available() else -1
    
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME, use_safetensors=True)
    
    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    outputs = clf(texts, batch_size=batch_size, truncation=True, max_length=512)
    pred_labels = [SENTIMENT_LABEL_MAP.get(out["label"], out["label"]) for out in outputs]
    confidences = [float(out["score"]) for out in outputs]
    return pred_labels, confidences


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_predictions(gold_labels, pred_labels):
    gold_ids = [label2id[label] for label in gold_labels]
    pred_ids = [label2id[label] for label in pred_labels]
    return {
        "accuracy": accuracy_score(gold_ids, pred_ids),
        "macro_f1": f1_score(gold_ids, pred_ids, average="macro"),
        "weighted_f1": f1_score(gold_ids, pred_ids, average="weighted"),
        "report": classification_report(
            gold_ids,
            pred_ids,
            target_names=LABELS,
            digits=4
        ),
        "confusion_matrix": confusion_matrix(gold_ids, pred_ids)
    }


def print_results(results):
    print("\n=== TRANSLATION BASELINE RESULTS ===")
    print(f"Accuracy:     {results['accuracy']:.4f}")
    print(f"Macro-F1:     {results['macro_f1']:.4f}")
    print(f"Weighted-F1:  {results['weighted_f1']:.4f}")
    print("\n=== Classification Report ===")
    print(results["report"])
    print("=== Confusion Matrix ===")
    print(results["confusion_matrix"])


def save_outputs(df, pred_labels, confidences, output_file, label_column="label"):
    out = df.copy()
    out["translation_pred"] = pred_labels
    out["translation_confidence"] = confidences
    out["correct"] = out[label_column] == out["translation_pred"]
    out.to_csv(output_file, index=False)
    print(f"\nSaved predictions to: {output_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Translation baseline evaluation")
    parser.add_argument("--test_file", type=str, required=True, help="Annotated test CSV")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--label_column", type=str, default="label", help="Gold label column name")
    parser.add_argument("--src_lang", type=str, default="rus_Cyrl", help="NLLB source language code")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for translation and sentiment")
    parser.add_argument("--output_file", type=str, default="translation_baseline_results.csv", help="Output CSV")
    args = parser.parse_args()

    print("Loading data...")
    df = load_data(args.test_file)
    df = preprocess_data(df, text_column=args.text_column, label_column=args.label_column)
    print(f"Loaded {len(df)} samples.")

    print("\nLoading translation model...")
    tokenizer, model, device = load_translation_model()

    print("\nTranslating...")
    df = get_translations(
        df, args.text_column, tokenizer, model, device,
        src_lang=args.src_lang, batch_size=args.batch_size
    )

    print("\nRunning sentiment classifier on translated text...")
    pred_labels, confidences = run_sentiment(
        df["translated_text"].tolist(),
        batch_size=args.batch_size
    )

    gold_labels = df[args.label_column].tolist()
    results = evaluate_predictions(gold_labels, pred_labels)
    print_results(results)

    save_outputs(df, pred_labels, confidences, args.output_file, label_column=args.label_column)


if __name__ == "__main__":
    main()