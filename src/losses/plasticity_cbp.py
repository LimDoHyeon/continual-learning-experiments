from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.optim import Optimizer


def build_activation_name(cfg: Dict[str, Any]) -> str:
    pcfg = cfg.get("plasticity", {})
    return str(pcfg.get("activation", "relu")).lower()


def make_activation_module(name: str) -> nn.Module:
    lname = name.lower()
    if lname == "relu":
        return nn.ReLU()
    if lname == "elu":
        return nn.ELU()
    if lname == "selu":
        return nn.SELU()
    if lname == "tanh":
        return nn.Tanh()
    if lname == "swish":
        return nn.SiLU()
    if lname == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported plasticity activation: {name}")


@dataclass(frozen=True)
class CBPConfig:
    enabled: bool = False
    replacement_rate: float = 0.001
    decay_rate: float = 0.9
    maturity_threshold: int = 100
    util_type: str = "contribution"
    init: str = "kaiming"
    accumulate: bool = False


def build_cbp_config(cfg: Dict[str, Any]) -> CBPConfig:
    pcfg = cfg.get("plasticity", {})
    return CBPConfig(
        enabled=bool(pcfg.get("enabled", False)),
        replacement_rate=float(pcfg.get("replacement_rate", 0.001)),
        decay_rate=float(pcfg.get("decay_rate", 0.9)),
        maturity_threshold=int(pcfg.get("maturity_threshold", 100)),
        util_type=str(pcfg.get("util_type", "contribution")),
        init=str(pcfg.get("init", "kaiming")),
        accumulate=bool(pcfg.get("accumulate", False)),
    )


def _ensure_lop_path() -> None:
    lop_root = Path(__file__).resolve().parent / "loss-of-plasticity"
    if not lop_root.exists():
        raise FileNotFoundError(f"loss-of-plasticity directory not found: {lop_root}")
    lop_root_str = str(lop_root)
    if lop_root_str not in sys.path:
        sys.path.append(lop_root_str)


def create_gnt(
    layers,
    activation_name: str,
    optimizer: Optimizer,
    config: CBPConfig,
    device: torch.device,
):
    _ensure_lop_path()
    from lop.algos.gnt import GnT

    return GnT(
        net=layers,
        hidden_activation=activation_name,
        opt=optimizer,
        replacement_rate=config.replacement_rate,
        decay_rate=config.decay_rate,
        maturity_threshold=config.maturity_threshold,
        util_type=config.util_type,
        device=str(device),
        loss_func=torch.nn.functional.cross_entropy,
        init=config.init,
        accumulate=config.accumulate,
    )


def run_gnt_step(gnt, features: List[torch.Tensor]) -> None:
    detached = [feature.detach() for feature in features]
    gnt.gen_and_test(features=detached)
