from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy

from ...datamodule.dataset import CLASSES
from .train_asc_titanmag import ASCDataModule, ASCTitanMAGSystem, DOMAIN_CHOICES

DOMAIN_SEQUENCE = ("europe6", "lisbon", "lyon", "prague", "korea")


def _extract_test_acc(metrics: Dict[str, Any]) -> float:
    if "test_acc" in metrics:
        return float(metrics["test_acc"])
    for key, value in metrics.items():
        if key.endswith("test_acc"):
            return float(value)
    raise KeyError(f"test_acc metric not found in metrics keys: {list(metrics.keys())}")


def main(
    config_path: str,
    checkpoint_path: str,
    workspace_path: str,
    dataset_name: str,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    assert dataset_name in DOMAIN_CHOICES

    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    label_names = cfg.get("dataset", {}).get("label_names", list(CLASSES.keys()))
    label2idx = {name: idx for idx, name in enumerate(label_names)}

    tcfg: Dict[str, Any] = cfg["train"]
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    devices_cfg = tcfg.get("devices", "auto")
    is_multi_device = False
    if isinstance(devices_cfg, int):
        is_multi_device = devices_cfg > 1
    elif isinstance(devices_cfg, (list, tuple)):
        is_multi_device = len(devices_cfg) > 1
    elif devices_cfg == "auto":
        accelerator_cfg = str(tcfg.get("accelerator", "auto"))
        if accelerator_cfg in {"auto", "gpu", "cuda"}:
            is_multi_device = torch.cuda.device_count() > 1

    strategy = tcfg.get("strategy", "auto")
    if strategy == "auto" and is_multi_device:
        strategy = DDPStrategy(find_unused_parameters=bool(tcfg.get("find_unused_parameters", False)))

    trainer = pl.Trainer(
        accelerator=tcfg.get("accelerator", "auto"),
        devices=devices_cfg,
        strategy=strategy,
        precision=tcfg.get("precision", "32-true"),
        deterministic=bool(tcfg.get("deterministic", False)),
        default_root_dir=str(workspace),
        logger=False,
    )

    test_domains: List[str] = list(DOMAIN_SEQUENCE) if dataset_name == "all" else [dataset_name]
    domain_acc: Dict[str, float] = {}

    for domain in test_domains:
        print(f"[TEST] domain={domain}, checkpoint={checkpoint_path}")
        datamodule = ASCDataModule(cfg=cfg, label2idx=label2idx, dataset_name=domain)
        system = ASCTitanMAGSystem.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            num_classes=len(label2idx),
            map_location="cpu",
        )
        test_results = trainer.test(model=system, datamodule=datamodule)
        if not test_results:
            raise RuntimeError(f"No test result returned for domain: {domain}")
        acc = _extract_test_acc(test_results[0])
        domain_acc[domain] = acc
        print(f"[RESULT] {domain}: test_acc={acc:.4f}")

    plots_dir = workspace / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(checkpoint_path).stem
    plot_path = plots_dir / f"{ckpt_stem}_eval.png"

    x_domains = list(domain_acc.keys())
    y_acc = [domain_acc[d] for d in x_domains]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x_domains, y_acc, color="steelblue")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Domain")
    plt.ylabel("Accuracy")
    plt.title(f"Domain-wise Test Accuracy ({ckpt_stem})")
    for bar, value in zip(bars, y_acc):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print(f"[PLOT] saved: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASC with Titans MAG backbone")
    parser.add_argument("-c", "--config", type=str, default="config/train_asc_titanmag.yaml")
    parser.add_argument("-k", "--checkpoint", type=str, required=True)
    parser.add_argument("-w", "--workspace", type=str, default="workspace/test_asc_titanmag")
    parser.add_argument("-d", "--dataset", type=str, default="all", choices=DOMAIN_CHOICES)
    args = parser.parse_args()

    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        workspace_path=args.workspace,
        dataset_name=args.dataset,
    )
