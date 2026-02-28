from __future__ import annotations

import torch


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float().mean()
