from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict differential equation types.")
    parser.add_argument("--model", required=True, type=Path, help="Path to a saved .joblib model.")
    parser.add_argument("--equation", help="Equation text for a single prediction.")
    parser.add_argument("--input", type=Path, help="CSV file with equations.")
    parser.add_argument("--text-column", default="equation", help="Column with equation text.")
    parser.add_argument("--output", type=Path, help="Where to save CSV predictions.")
    return parser.parse_args()


def load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def predict_many(model, equations: pd.Series) -> pd.DataFrame:
    result = pd.DataFrame({"equation": equations.astype(str)})
    result["predicted_label"] = model.predict(equations)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(equations)
        for index, label in enumerate(model.classes_):
            result[f"probability_{label}"] = probabilities[:, index]

    return result


def main() -> None:
    args = parse_args()
    if bool(args.equation) == bool(args.input):
        raise ValueError("Pass exactly one of --equation or --input.")

    bundle = load_bundle(args.model)
    model = bundle["model"]

    if args.equation:
        predictions = predict_many(model, pd.Series([args.equation]))
        print(predictions.to_string(index=False))
        return

    data = pd.read_csv(args.input)
    if args.text_column not in data.columns:
        columns = ", ".join(sorted(data.columns))
        raise ValueError(f"Missing text column: {args.text_column}. Available columns: {columns}")

    predictions = predict_many(model, data[args.text_column])
    output = pd.concat([data.reset_index(drop=True), predictions.drop(columns=["equation"])], axis=1)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
        print(f"Saved predictions to {args.output}")
    else:
        print(output.to_string(index=False))


if __name__ == "__main__":
    main()
