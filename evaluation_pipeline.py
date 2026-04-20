import argparse
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification
)

LABELS = ["negative", "neutral", "positive"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}


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


def normalize_zero_shot_label(label):
    label = str(label).lower().strip()
    if label in label2id:
        return label
    raise ValueError(f"Unexpected zero-shot label: {label}")


def run_zero_shot(texts, model_name, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    clf = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    outputs = clf(texts, batch_size=batch_size)

    pred_labels = [normalize_zero_shot_label(out["label"]) for out in outputs]
    confidences = [float(out["score"]) for out in outputs]

    return pred_labels, confidences


def run_finetuned(texts, model_path, batch_size=16, max_length=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    model.eval()
    model.to(device)

    pred_labels = []
    confidences = []

    for start_idx in range(0, len(texts), batch_size):
        batch_texts = texts[start_idx:start_idx + batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

        pred_labels.extend([id2label[p.item()] for p in preds])
        confidences.extend([float(c.item()) for c in confs])

    return pred_labels, confidences


def print_comparison(zero_shot_results, finetuned_results):
    print("\n=== Comparison ===")
    print(f"{'Metric':<15}{'Zero-shot':<15}{'Fine-tuned':<15}")
    print(f"{'-'*45}")
    print(f"{'Accuracy':<15}{zero_shot_results['accuracy']:<15.4f}{finetuned_results['accuracy']:<15.4f}")
    print(f"{'Macro-F1':<15}{zero_shot_results['macro_f1']:<15.4f}{finetuned_results['macro_f1']:<15.4f}")
    print(f"{'Weighted-F1':<15}{zero_shot_results['weighted_f1']:<15.4f}{finetuned_results['weighted_f1']:<15.4f}")

    print("\n=== Zero-shot Classification Report ===")
    print(zero_shot_results["report"])

    print("=== Fine-tuned Classification Report ===")
    print(finetuned_results["report"])

    print("=== Zero-shot Confusion Matrix ===")
    print(zero_shot_results["confusion_matrix"])

    print("\n=== Fine-tuned Confusion Matrix ===")
    print(finetuned_results["confusion_matrix"])


def save_outputs(df, zero_shot_labels, zero_shot_conf, ft_labels, ft_conf, output_file):
    output_df = df.copy()
    output_df["zero_shot_pred"] = zero_shot_labels
    output_df["zero_shot_confidence"] = zero_shot_conf
    output_df["finetuned_pred"] = ft_labels
    output_df["finetuned_confidence"] = ft_conf
    output_df["zero_shot_correct"] = output_df["label"] == output_df["zero_shot_pred"]
    output_df["finetuned_correct"] = output_df["label"] == output_df["finetuned_pred"]
    output_df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Annotated test CSV")
    parser.add_argument("--finetuned_model_path", type=str, default="./finetuned_model", help="Path to saved fine-tuned model")
    parser.add_argument(
        "--zero_shot_model_name",
        type=str,
        default="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        help="Base zero-shot model name"
    )
    parser.add_argument("--text_column", type=str, default="text", help="Text column name")
    parser.add_argument("--label_column", type=str, default="label", help="Gold label column name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--output_file",
        type=str,
        default="test_set_comparison.csv",
        help="Output CSV with predictions"
    )

    args = parser.parse_args()

    df = load_data(args.test_file)
    df = preprocess_data(df, text_column=args.text_column, label_column=args.label_column)

    texts = df[args.text_column].tolist()
    gold_labels = df[args.label_column].tolist()

    print("Running zero-shot model...")
    zero_shot_labels, zero_shot_conf = run_zero_shot(
        texts=texts,
        model_name=args.zero_shot_model_name,
        batch_size=args.batch_size
    )

    print("Running fine-tuned model...")
    ft_labels, ft_conf = run_finetuned(
        texts=texts,
        model_path=args.finetuned_model_path,
        batch_size=args.batch_size
    )

    zero_shot_results = evaluate_predictions(gold_labels, zero_shot_labels)
    finetuned_results = evaluate_predictions(gold_labels, ft_labels)

    print_comparison(zero_shot_results, finetuned_results)

    save_outputs(
        df=df,
        zero_shot_labels=zero_shot_labels,
        zero_shot_conf=zero_shot_conf,
        ft_labels=ft_labels,
        ft_conf=ft_conf,
        output_file=args.output_file
    )
    print(f"\nSaved comparison file to: {args.output_file}")


if __name__ == "__main__":
    main()