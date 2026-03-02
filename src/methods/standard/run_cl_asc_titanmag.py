from __future__ import annotations

import argparse
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy

from ...datamodule.dataset import CLASSES
from .train_asc_titanmag import ASCDataModule, ASCTitanMAGSystem, build_logger

DOMAIN_SEQUENCE = ("europe6", "lisbon", "lyon", "prague", "korea")


def _build_strategy(train_cfg: Dict[str, Any]):
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    devices_cfg = train_cfg.get("devices", "auto")
    is_multi_device = False
    if isinstance(devices_cfg, int):
        is_multi_device = devices_cfg > 1
    elif isinstance(devices_cfg, (list, tuple)):
        is_multi_device = len(devices_cfg) > 1
    elif devices_cfg == "auto":
        accelerator_cfg = str(train_cfg.get("accelerator", "auto"))
        if accelerator_cfg in {"auto", "gpu", "cuda"}:
            is_multi_device = torch.cuda.device_count() > 1

    strategy = train_cfg.get("strategy", "auto")
    if strategy == "auto" and is_multi_device:
        strategy = DDPStrategy(find_unused_parameters=bool(train_cfg.get("find_unused_parameters", False)))
    return strategy


def _build_system(cfg: Dict[str, Any], num_classes: int, init_ckpt: Optional[Path]) -> ASCTitanMAGSystem:
    if init_ckpt is None:
        return ASCTitanMAGSystem(cfg=cfg, num_classes=num_classes)
    return ASCTitanMAGSystem.load_from_checkpoint(
        checkpoint_path=str(init_ckpt),
        cfg=cfg,
        num_classes=num_classes,
        map_location="cpu",
    )


def run_continual(
    config_path: str,
    workspace_path: str,
    resume_from: Optional[str],
    disable_wandb: bool,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    assert isinstance(base_cfg, dict)

    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)
    ckpt_dir = workspace / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(base_cfg.get("seed", 42)), workers=True)

    label_names = base_cfg.get("dataset", {}).get("label_names", list(CLASSES.keys()))
    label2idx = {name: idx for idx, name in enumerate(label_names)}

    prev_ckpt = Path(resume_from).resolve() if resume_from is not None else None

    for domain in DOMAIN_SEQUENCE:
        cfg = deepcopy(base_cfg)
        cfg.setdefault("train", {})
        cfg["train"]["max_epochs"] = 30
        cfg.setdefault("logging", {})
        cfg["logging"]["save_top_k"] = 0
        cfg["logging"]["run_name"] = f"cl_asc_titanmag_{domain}"
        if disable_wandb:
            cfg["logging"]["use_wandb"] = False

        run_dir = workspace / f"stage_{domain}"
        run_dir.mkdir(parents=True, exist_ok=True)

        datamodule = ASCDataModule(cfg=cfg, label2idx=label2idx, dataset_name=domain)
        system = _build_system(cfg=cfg, num_classes=len(label2idx), init_ckpt=prev_ckpt)

        checkpoint_callback = ModelCheckpoint(
            dirpath=str(run_dir / "checkpoints"),
            save_last=True,
            save_top_k=0,
        )
        callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
        logger = build_logger(cfg=cfg, workspace=run_dir, wandb_id=None)

        trainer = pl.Trainer(
            accelerator=cfg["train"].get("accelerator", "auto"),
            devices=cfg["train"].get("devices", "auto"),
            strategy=_build_strategy(cfg["train"]),
            max_epochs=int(cfg["train"]["max_epochs"]),
            precision=cfg["train"].get("precision", "32-true"),
            gradient_clip_val=float(cfg["train"].get("gradient_clip_val", 0.0)),
            log_every_n_steps=int(cfg["train"].get("log_every_n_steps", 20)),
            deterministic=bool(cfg["train"].get("deterministic", False)),
            default_root_dir=str(run_dir),
            logger=logger,
            callbacks=callbacks,
        )

        trainer.fit(model=system, datamodule=datamodule)

        stage_last_ckpt = run_dir / "checkpoints" / "last.ckpt"
        if not stage_last_ckpt.exists():
            raise FileNotFoundError(f"last checkpoint was not created: {stage_last_ckpt}")

        domain_ckpt = ckpt_dir / f"{domain}.ckpt"
        shutil.copy2(stage_last_ckpt, domain_ckpt)
        prev_ckpt = domain_ckpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual ASC training with Titans MAG backbone")
    parser.add_argument("-c", "--config", type=str, default="config/train_asc_titanmag.yaml")
    parser.add_argument("-w", "--workspace", type=str, default="workspace/run_cl_asc_titanmag")
    parser.add_argument(
        "-r",
        "--resume-from",
        type=str,
        default=None,
        help="Optional warm-start checkpoint for the first domain(europe6).",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable wandb and use CSV logger instead.",
    )
    args = parser.parse_args()

    run_continual(
        config_path=args.config,
        workspace_path=args.workspace,
        resume_from=args.resume_from,
        disable_wandb=args.disable_wandb,
    )
