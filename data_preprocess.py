import os
import ast
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def data_split(df, test_size=200, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train_df, test_df


def flatten_text(cell):
    try:
        lst = ast.literal_eval(cell)
        if isinstance(lst, list):
            return ' '.join(map(str, lst))
        return str(lst)
    except:
        return str(cell)


def preprocess_data(df, text_column, topic_column='topic', include_topic=False, include_label=False):
    output_df = pd.DataFrame({
        'id': range(len(df)),
        'text': df[text_column].apply(flatten_text)
    })
    if include_topic:
        output_df['topic'] = df[topic_column]
    if include_label:
        output_df['label'] = ''
    return output_df


def save_data(df, file_path):
    df.to_csv(file_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Preprocess and split raw CSV into train/test sets")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory e.g. data/russian")
    parser.add_argument("--text_column", type=str, default="data_generation_result", help="Column to preprocess")
    parser.add_argument("--test_size", type=int, default=200, help="Number of test samples")
    parser.add_argument("--hand_labeled_size", type=int, default=100, help="Number of hand-labeled training samples to set aside")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    lang = os.path.splitext(os.path.basename(args.input))[0]

    df = load_data(args.input)

    # Split into test and remaining
    remaining_df, test_df = data_split(df, test_size=args.test_size, random_state=args.random_state)

    # Split remaining into hand-labeled and unlabeled train
    unlabeled_df, hand_labeled_df = data_split(remaining_df, test_size=args.hand_labeled_size, random_state=args.random_state)

    # Preprocess each split
    train_output = preprocess_data(unlabeled_df, args.text_column, include_topic=False, include_label=False)
    test_output = preprocess_data(test_df, args.text_column, include_topic=True, include_label=True)
    hand_labeled_output = preprocess_data(hand_labeled_df, args.text_column, include_topic=True, include_label=True)

    # Save
    train_path = os.path.join(args.output_dir, f"train_{lang}.csv")
    test_path = os.path.join(args.output_dir, f"test_{lang}_annotation.csv")
    hand_labeled_path = os.path.join(args.output_dir, f"hand_labeled_{lang}.csv")

    save_data(train_output, train_path)
    save_data(test_output, test_path)
    save_data(hand_labeled_output, hand_labeled_path)

    print("Saved files:")
    print(f"  - {train_path} ({len(train_output)} examples)")
    print(f"  - {test_path} ({len(test_output)} examples)")
    print(f"  - {hand_labeled_path} ({len(hand_labeled_output)} examples)")


if __name__ == "__main__":
    main()