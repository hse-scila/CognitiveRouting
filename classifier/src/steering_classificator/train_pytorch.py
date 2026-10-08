"""Train the paper's CNN with a separate, protected external test set."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from steering_classificator.preprocessing import equation_key, preprocess_equation
from steering_classificator.pytorch_model import MathTextCNN, encode, load_checkpoint

CLASSIFIER_DIR = Path(__file__).resolve().parents[2]
LABELS = ["polynomial", "separable", "unhomogenous"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_datasets(data: list[dict], test: list[dict], reserved: list[dict]) -> None:
    if not data or not test:
        raise ValueError("Training or external test dataset is empty")
    if any(row.get("split") not in {"train", "valid"} for row in data):
        raise ValueError("Training workbook may contain only train and valid rows")
    for name, rows in (("training", data), ("external test", test)):
        keys = [equation_key(row.get("equation", "")) for row in rows]
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise ValueError(f"Blank or duplicate equation in {name}")
        if set(row["label"] for row in rows) != set(LABELS):
            raise ValueError(f"Unexpected or missing class in {name}")
    protected = {equation_key(row["equation"]) for row in test + reserved}
    if any(equation_key(row["equation"]) in protected for row in data):
        raise ValueError("A current or original test equation occurs in training data")
    if any("test" in Path(row.get("source_file", "")).stem.lower() for row in data):
        raise ValueError("A row has a test source filename")
    for split in ("train", "valid"):
        if set(row["label"] for row in data if row["split"] == split) != set(LABELS):
            raise ValueError(f"Missing class in {split}")


def metrics(true: list[int], predicted: list[int]) -> dict:
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for a, b in zip(true, predicted, strict=True):
        matrix[a][b] += 1
    per_class = {}
    for index, label in enumerate(LABELS):
        correct = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = correct / predicted_count if predicted_count else 0.0
        recall = correct / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    count = len(true)
    return {"accuracy": sum(matrix[i][i] for i in range(len(LABELS))) / count,
            "macro_f1": sum(item["f1"] for item in per_class.values()) / len(LABELS),
            "weighted_f1": sum(item["f1"] * item["support"] for item in per_class.values()) / count,
            "correct": sum(matrix[i][i] for i in range(len(LABELS))), "total": count,
            "per_class": per_class, "confusion_matrix": matrix,
            "confusion_matrix_labels": LABELS, "confusion_matrix_axes": "rows=true, columns=predicted"}


def make_loader(rows, vocab, max_len, batch_size, shuffle=False):
    x = torch.tensor([encode(row["equation"], vocab, max_len) for row in rows], dtype=torch.long)
    y = torch.tensor([LABELS.index(row["label"]) for row in rows], dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, true, predicted, probabilities = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
                optimizer.step()
        total_loss += loss.item() * len(y)
        true.extend(y.detach().cpu().tolist())
        predicted.extend(logits.argmax(dim=1).detach().cpu().tolist())
        probabilities.extend(logits.softmax(dim=1).detach().cpu().tolist())
    return {"loss": total_loss / len(loader.dataset), **metrics(true, predicted)}, predicted, probabilities


def evaluate_checkpoint(path, rows, device):
    model, bundle = load_checkpoint(path, device)
    if bundle["label_names"] != LABELS:
        raise ValueError("Checkpoint label order differs from the deployed classifier")
    loader = make_loader(rows, bundle["vocab"], bundle["max_len"], 64)
    return run_epoch(model, loader, nn.CrossEntropyLoss(), device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CLASSIFIER_DIR / "data/combined_equations_with_split.csv")
    parser.add_argument("--test", type=Path, default=CLASSIFIER_DIR / "data/test_equations.csv")
    parser.add_argument("--reserved-test", type=Path, default=CLASSIFIER_DIR / "reports/excluded_test_equations.csv")
    parser.add_argument("--model-out", type=Path, default=CLASSIFIER_DIR / "pytorch_equation_classifier.pt")
    parser.add_argument("--reports", type=Path, default=CLASSIFIER_DIR / "reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    return parser.parse_args()


def main():
    args = parse_args()
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, test, reserved = read_csv(args.data), read_csv(args.test), read_csv(args.reserved_test)
    validate_datasets(data, test, reserved)
    audit_path = CLASSIFIER_DIR / "reports/data_integrity.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    external_xlsx = CLASSIFIER_DIR.parent / "data/datasets/test_all.xlsx"
    if sha256(external_xlsx) != audit["test_all_sha256"] or sha256(args.test) != audit["external_test_csv_sha256"]:
        raise ValueError("External test has changed; refresh and audit the test CSV before retraining")
    train = [row for row in data if row["split"] == "train"]
    valid = [row for row in data if row["split"] == "valid"]
    args.reports.mkdir(parents=True, exist_ok=True)
    previous = None
    previous_sha = None
    if args.model_out.exists():
        previous_sha = sha256(args.model_out)
        previous, _, _ = evaluate_checkpoint(args.model_out, test, device)
    # Loading/evaluating the previous model must not change the training seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    counter = Counter(token for row in train for token in preprocess_equation(row["equation"]).split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    lengths = [len(preprocess_equation(row["equation"]).split()) for row in train]
    max_len = max(16, min(int(np.percentile(lengths, 95)), max(lengths)))
    config = {"embed_dim": 96, "num_filters": 128, "dropout": 0.25, "min_token_freq": 1,
              "normalization": "normalize_equation + tokenize_math", "kernel_sizes": [3, 5, 7],
              "batch_size": 64, "learning_rate": 1e-3, "weight_decay": 1e-4,
              "max_len_percentile": 95, "seed": args.seed, "epochs_limit": args.epochs,
              "patience": args.patience, "selection_metric": "validation macro F1"}
    model = MathTextCNN(len(vocab), len(LABELS)).to(device)
    counts = Counter(row["label"] for row in train)
    weights = torch.tensor([len(train) / counts[label] for label in LABELS], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = make_loader(train, vocab, max_len, 64, True)
    valid_loader = make_loader(valid, vocab, max_len, 64)
    best_f1, best_epoch, best_state, bad_epochs = -1.0, 0, None, 0
    history = []
    print(f"Train={len(train)}, validation={len(valid)}, external test={len(test)}, vocabulary={len(vocab)}, max_len={max_len}", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        valid_metrics, _, _ = run_epoch(model, valid_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_f1": train_metrics["macro_f1"],
                        "train_acc": train_metrics["accuracy"], "valid_loss": valid_metrics["loss"],
                        "valid_f1": valid_metrics["macro_f1"], "valid_acc": valid_metrics["accuracy"]})
        print(f"Epoch {epoch:02d}: train F1={train_metrics['macro_f1']:.4f}; validation F1={valid_metrics['macro_f1']:.4f}", flush=True)
        if valid_metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = valid_metrics["macro_f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("No trained checkpoint was selected")
    bundle = {"model_state_dict": best_state, "label_names": LABELS,
              "label_to_id": {label: i for i, label in enumerate(LABELS)},
              "id_to_label": dict(enumerate(LABELS)), "vocab": vocab, "max_len": max_len,
              "config": config, "history": history, "best_epoch": best_epoch,
              "trained_at": datetime.now(timezone.utc).isoformat(),
              "train_data_sha256": sha256(args.data), "external_test_sha256": sha256(external_xlsx)}
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    staged_model = args.model_out.with_suffix(".pt.new")
    torch.save(bundle, staged_model)
    # Evaluate the reloaded checkpoint that will be used by the routing notebook.
    result, predicted, probabilities = evaluate_checkpoint(staged_model, test, device)
    os.replace(staged_model, args.model_out)
    report = {"trained_at": bundle["trained_at"], "device": str(device), "torch_version": str(torch.__version__),
              "configuration": config, "training_rows": len(train), "validation_rows": len(valid),
              "test_rows": len(test), "best_epoch": best_epoch, "epochs_run": len(history),
              "best_validation_macro_f1": best_f1, "vocab_size": len(vocab), "max_len": max_len,
              "training_labels": dict(Counter(row["label"] for row in train)),
              "validation_labels": dict(Counter(row["label"] for row in valid)),
              "test_metrics": result, "previous_checkpoint_test_metrics": previous,
              "previous_checkpoint_sha256": previous_sha,
              "previous_checkpoint_matches_original_leaked_model": previous_sha == audit["original_checkpoint_sha256"],
              "previous_checkpoint_note": (
                  "The original checkpoint was trained using test-source records. Its score is not an unbiased baseline."
                  if previous_sha == audit["original_checkpoint_sha256"] else
                  "The previous checkpoint was evaluated on the same revised external test; see its recorded hash for provenance."
              ),
              "model_sha256": sha256(args.model_out), "training_data_sha256": sha256(args.data),
              "test_xlsx_sha256": sha256(external_xlsx), "test_csv_sha256": sha256(args.test),
              "overlap_train_validation_test": 0, "duration_seconds": time.monotonic() - started}
    (args.reports / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.reports / "training_history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.reports / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = list(test[0]) + ["predicted_label", "correct"] + [f"probability_{label}" for label in LABELS]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row, prediction, probability in zip(test, predicted, probabilities, strict=True):
            writer.writerow({**row, "predicted_label": LABELS[prediction], "correct": row["label"] == LABELS[prediction],
                             **{f"probability_{label}": probability[i] for i, label in enumerate(LABELS)}})
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved compatible checkpoint: {args.model_out}", flush=True)


if __name__ == "__main__":
    main()
