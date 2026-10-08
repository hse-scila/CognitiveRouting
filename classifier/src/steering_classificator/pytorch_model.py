"""Compatible with the checkpoint consumed by CognitiveRouting.ipynb."""
from __future__ import annotations

import torch
from torch import nn

from steering_classificator.preprocessing import preprocess_equation


class MathTextCNN(nn.Module):
    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int = 96,
                 num_filters: int = 128, kernel_sizes: tuple[int, ...] = (3, 5, 7),
                 dropout: float = 0.25, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        pooled = [torch.amax(self.activation(conv(embedded)), dim=-1) for conv in self.convs]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))


def encode(equation: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(token, vocab["<UNK>"]) for token in preprocess_equation(equation).split()]
    ids = ids[:max_len]
    return ids + [vocab["<PAD>"]] * (max_len - len(ids))


def load_checkpoint(path, device="cpu"):
    bundle = torch.load(path, map_location=device, weights_only=False)
    config = bundle["config"]
    model = MathTextCNN(len(bundle["vocab"]), len(bundle["label_names"]),
                        embed_dim=config.get("embed_dim", 96),
                        num_filters=config.get("num_filters", 128),
                        dropout=config.get("dropout", 0.25)).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()
    return model, bundle
