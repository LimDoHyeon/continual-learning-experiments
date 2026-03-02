from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.strategies import DDPStrategy

from ...datamodule.dataset import CLASSES
from .train_asc_titanmag import ASCDataModule, ASCTitanMAGSystem, DOMAIN_CHOICES


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

    datamodule = ASCDataModule(cfg=cfg, label2idx=label2idx, dataset_name=dataset_name)
    system = ASCTitanMAGSystem.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        num_classes=len(label2idx),
        map_location="cpu",
    )

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

    trainer.test(model=system, datamodule=datamodule)


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
