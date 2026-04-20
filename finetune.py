import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = [str(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def prepare_training_data(selected_data):
    texts = []
    labels = []

    for item in selected_data:
        text = item.get("text")
        label = item.get("label")

        if text is None or label is None:
            continue

        text = str(text).strip()
        label = str(label).lower().strip()

        if text == "":
            continue

        if label not in label2id:
            continue

        texts.append(text)
        labels.append(label2id[label])

    return texts, labels


def fine_tune_model(selected_data, model_name, output_dir="./finetuned_model"):
    texts, labels = prepare_training_data(selected_data)

    if len(texts) == 0:
        raise ValueError("No valid training samples found in selected_data.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    )

    train_dataset = SentimentDataset(texts, labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="no",
        logging_strategy="steps",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return model, tokenizer


def ft_predictions(data, model, tokenizer, device, batch_size=16):
    model.eval()
    model.to(device)

    predictions = {
        "negative": [],
        "neutral": [],
        "positive": [],
    }

    for start_idx in range(0, len(data), batch_size):
        batch_items = data[start_idx:start_idx + batch_size]
        batch_texts = [str(item["text"]) for item in batch_items]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

        for item, conf, pred in zip(batch_items, confs, preds):
            label = id2label[pred.item()]
            prediction = {
                "id": item["id"],
                "text": str(item["text"]),
                "label": label,
                "confidence": conf.item()
            }
            predictions[label].append(prediction)

    return predictions