from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from steering_classificator.model import SUPPORTED_LABELS, build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a differential equation type classifier.")
    parser.add_argument("--data", required=True, type=Path, help="Path to a CSV dataset.")
    parser.add_argument("--text-column", default="equation", help="Column with equation text.")
    parser.add_argument("--label-column", default="label", help="Column with target class.")
    parser.add_argument(
        "--model-out",
        default=Path("models/equation_classifier.joblib"),
        type=Path,
        help="Where to save the trained model.",
    )
    parser.add_argument("--test-size", default=0.2, type=float, help="Validation split size.")
    parser.add_argument("--random-state", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--allow-extra-labels",
        action="store_true",
        help="Train even if labels outside the three supported classes are present.",
    )
    return parser.parse_args()


def load_dataset(path: Path, text_column: str, label_column: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    missing_columns = {text_column, label_column} - set(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(data.columns))
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing column(s): {missing}. Available columns: {columns}")

    data = data[[text_column, label_column]].dropna()
    data[text_column] = data[text_column].astype(str)
    data[label_column] = data[label_column].astype(str).str.strip()
    data = data[data[text_column].str.strip().ne("")]

    if data.empty:
        raise ValueError("Dataset is empty after removing missing and blank rows.")

    return data


def validate_labels(labels: pd.Series, allow_extra_labels: bool) -> None:
    found = set(labels.unique())
    supported = set(SUPPORTED_LABELS)
    unknown = found - supported
    missing = supported - found

    if unknown and not allow_extra_labels:
        raise ValueError(
            "Dataset contains unsupported labels: "
            f"{sorted(unknown)}. Expected only: {sorted(supported)}."
        )

    if missing:
        raise ValueError(f"Dataset is missing supported label(s): {sorted(missing)}.")


def can_stratify(labels: pd.Series, test_size: float) -> bool:
    class_count = labels.nunique()
    validation_size = math.ceil(len(labels) * test_size)
    train_size = len(labels) - validation_size
    return labels.value_counts().min() >= 2 and validation_size >= class_count and train_size >= class_count


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    data = load_dataset(args.data, args.text_column, args.label_column)
    validate_labels(data[args.label_column], args.allow_extra_labels)

    stratify = data[args.label_column] if can_stratify(data[args.label_column], args.test_size) else None
    if stratify is None:
        print("Warning: validation split is too small for stratification; using a random split.")

    train_x, valid_x, train_y, valid_y = train_test_split(
        data[args.text_column],
        data[args.label_column],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = build_model()
    model.fit(train_x, train_y)

    predictions = model.predict(valid_x)
    labels = sorted(data[args.label_column].unique())

    print("Labels:", ", ".join(labels))
    print()
    print("Classification report:")
    print(classification_report(valid_y, predictions, labels=labels, zero_division=0))
    print("Confusion matrix:")
    print(pd.DataFrame(confusion_matrix(valid_y, predictions, labels=labels), index=labels, columns=labels))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": labels,
            "text_column": args.text_column,
            "label_column": args.label_column,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
        args.model_out,
    )
    print()
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
