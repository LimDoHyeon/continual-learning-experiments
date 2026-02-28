from __future__ import annotations

import torch.nn as nn


def build_cross_entropy_loss(label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
