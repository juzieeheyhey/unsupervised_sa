import pandas as pd
import ast
import argparse
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def data_split(df, test_size=100, random_state=42):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--text_column", type=str, default="data_generation_result", help="Column to preprocess")
    parser.add_argument("--test_size", type=int, default=100, help="Number of test samples")

    args = parser.parse_args()

    df = load_data(args.input)
    train_df, test_df = data_split(df, test_size=args.test_size)

    train_output = preprocess_data(train_df, args.text_column, include_topic=False, include_label=False)
    test_annotation_output = preprocess_data(test_df, args.text_column, include_topic=True, include_label=True)
    
    train_path = f"train_{args.input}"
    test_annotation_path = f"test_{args.input}_annotation.csv"

    save_data(train_output, train_path)
    save_data(test_annotation_output, test_annotation_path)

    print("Saved files:")
    print(f"- {train_path}")
    print(f"- {test_annotation_path}")
    print(f"Train size: {len(train_output)}")
    print(f"Test size: {len(test_annotation_output)}")


if __name__ == "__main__":
    main()