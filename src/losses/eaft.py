from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _entropy_weights_from_logits(
    logits: torch.Tensor,
    *,
    alpha: float,
    entropy_norm: float,
    topk: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_classes = int(logits.shape[-1])
    k = max(1, min(int(topk), num_classes))
    if k < num_classes:
        topk_logits, _ = torch.topk(logits, k=k, dim=-1)
    else:
        topk_logits = logits

    logsumexp_topk = torch.logsumexp(topk_logits, dim=-1, keepdim=True)
    log_probs_topk = topk_logits - logsumexp_topk
    probs_topk = torch.exp(log_probs_topk)
    entropy = -(probs_topk * log_probs_topk).sum(dim=-1)

    norm = max(float(entropy_norm), 1e-6)
    normalized_entropy = (entropy / norm).clamp_min(0.0)
    weights = torch.pow(normalized_entropy, float(alpha))
    return weights, entropy


def compute_eaft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 1.0,
    entropy_norm: float = 3.0,
    topk: int = 20,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute entropy-adaptive weighted cross-entropy for sample-level classification."""
    logits_float = logits.float()
    per_sample_loss = F.cross_entropy(
        logits_float,
        labels,
        ignore_index=ignore_index,
        reduction='none',
        label_smoothing=label_smoothing,
    )
    valid_mask = labels != ignore_index

    if not torch.any(valid_mask):
        loss = per_sample_loss.mean()
        ones = torch.ones((), device=logits.device, dtype=loss.dtype)
        zeros = torch.zeros((), device=logits.device, dtype=loss.dtype)
        return loss, {
            'weight_mean': ones,
            'weight_min': ones,
            'weight_max': ones,
            'entropy_mean': zeros,
        }

    with torch.no_grad():
        logits_valid = logits_float[valid_mask].detach()
        weights_valid, entropy = _entropy_weights_from_logits(
            logits_valid,
            alpha=alpha,
            entropy_norm=entropy_norm,
            topk=topk,
        )

        weights = torch.ones_like(per_sample_loss)
        weights[valid_mask] = weights_valid

    weighted_loss = per_sample_loss * weights
    denom = valid_mask.sum().clamp(min=1)
    loss = weighted_loss.sum() / denom
    stats = {
        'weight_mean': weights_valid.mean(),
        'weight_min': weights_valid.min(),
        'weight_max': weights_valid.max(),
        'entropy_mean': entropy.mean(),
    }
    return loss, stats


def compute_eaft_token_loss(
    token_logits: torch.Tensor,
    labels: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    alpha: float = 1.0,
    entropy_norm: float = 3.0,
    topk: int = 20,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute EAFT on token/frame logits using clip-level labels broadcast to valid tokens."""
    logits_float = token_logits.float()
    batch_size, seq_len, num_classes = logits_float.shape

    expanded_labels = labels.unsqueeze(1).expand(batch_size, seq_len)
    flat_logits = logits_float.reshape(-1, num_classes)
    flat_labels = expanded_labels.reshape(-1)
    flat_mask = token_mask.reshape(-1).bool()

    if not torch.any(flat_mask):
        zeros = torch.zeros((), device=token_logits.device, dtype=logits_float.dtype)
        ones = torch.ones((), device=token_logits.device, dtype=logits_float.dtype)
        return zeros, {
            'weight_mean': ones,
            'weight_min': ones,
            'weight_max': ones,
            'entropy_mean': zeros,
        }

    valid_logits = flat_logits[flat_mask]
    valid_labels = flat_labels[flat_mask]
    valid_losses = F.cross_entropy(
        valid_logits,
        valid_labels,
        reduction='none',
        label_smoothing=label_smoothing,
    )

    with torch.no_grad():
        weights_valid, entropy = _entropy_weights_from_logits(
            valid_logits.detach(),
            alpha=alpha,
            entropy_norm=entropy_norm,
            topk=topk,
        )

    weighted_loss = valid_losses * weights_valid
    loss = weighted_loss.mean()
    stats = {
        'weight_mean': weights_valid.mean(),
        'weight_min': weights_valid.min(),
        'weight_max': weights_valid.max(),
        'entropy_mean': entropy.mean(),
    }
    return loss, stats
