from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ..datamodule.dataloaderv2 import AllDataLoader, LoaderConfig, SingleDataLoader
from ..datamodule.datasetv2 import CLASSES
from ..losses.ce import build_cross_entropy_loss
from ..losses.eaft import compute_eaft_token_loss
from ..losses.plasticity_cbp import (
    build_activation_name,
    build_cbp_config,
    create_gnt,
    make_activation_module,
    run_gnt_step,
)
from ..metrics.acc import top1_accuracy
from ..models.BEATs.beats_backbone import make_beats_block

DOMAIN_CHOICES = (
    "lisbon",
    "lyon",
    "prague",
    "barcelona",
    "helsinki",
    "london",
    "milan",
    "paris",
    "stockholm",
    "vienna",
    "all",
)


class ASCDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any], label2idx: Dict[str, int], dataset_name: str):
        super().__init__()
        self.cfg = cfg
        self.label2idx = label2idx
        self.dataset_name = dataset_name
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    @staticmethod
    def _collate_batch(batch: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        waves: List[torch.Tensor] = []
        labels: List[int] = []

        for sample in batch:
            item = sample[0] if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], str) else sample
            waveform, label = item

            if waveform.ndim == 2:
                waveform = waveform[0]
            elif waveform.ndim > 2:
                waveform = waveform.reshape(-1)

            waves.append(waveform)
            labels.append(int(label))

        lengths = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
        padded = pad_sequence(waves, batch_first=True)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        return padded, lengths, label_tensor

    def _loader_cfg(self, split: str, shuffle: bool) -> LoaderConfig:
        dcfg = self.cfg["dataset"]
        return LoaderConfig(
            split=split,
            batch_size=int(dcfg["batch_size"]),
            num_workers=int(dcfg["num_workers"]),
            shuffle=shuffle,
            drop_last=(split == "train"),
            pin_memory=bool(dcfg.get("pin_memory", True)),
            persistent_workers=bool(dcfg.get("persistent_workers", True)),
            prefetch_factor=int(dcfg.get("prefetch_factor", 2)),
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dcfg = self.cfg["dataset"]
        common_kwargs = {
            "label2idx": self.label2idx,
            "seed": int(self.cfg.get("seed", 42)),
            "target_sample_rate": int(dcfg["target_sample_rate"]),
            "return_path": False,
            "collate_fn": self._collate_batch,
            "tau2019_root": dcfg["tau2019_root"],
            "lisbon_meta": dcfg.get("lisbon_meta", "lisbon_meta.csv"),
            "lyon_meta": dcfg.get("lyon_meta", "lyon_meta.csv"),
            "prague_meta": dcfg.get("prague_meta", "prague_meta.csv"),
            "barcelona_meta": dcfg.get("barcelona_meta", "barcelona_meta.csv"),
            "helsinki_meta": dcfg.get("helsinki_meta", "helsinki_meta.csv"),
            "london_meta": dcfg.get("london_meta", "london_meta.csv"),
            "milan_meta": dcfg.get("milan_meta", "milan_meta.csv"),
            "paris_meta": dcfg.get("paris_meta", "paris_meta.csv"),
            "stockholm_meta": dcfg.get("stockholm_meta", "stockholm_meta.csv"),
            "vienna_meta": dcfg.get("vienna_meta", "vienna_meta.csv"),
        }
        all_loader_kwargs = {**common_kwargs, "return_domain": False}

        use_all_loader = self.dataset_name == "all"

        if self.train_loader is None:
            if use_all_loader:
                self.train_loader = AllDataLoader(cfg=self._loader_cfg(split="train", shuffle=True), **all_loader_kwargs).dataloader
            else:
                self.train_loader = SingleDataLoader(
                    dataset_name=self.dataset_name,
                    cfg=self._loader_cfg(split="train", shuffle=True),
                    **common_kwargs,
                ).dataloader
        if self.val_loader is None:
            if use_all_loader:
                self.val_loader = AllDataLoader(cfg=self._loader_cfg(split="val", shuffle=False), **all_loader_kwargs).dataloader
            else:
                self.val_loader = SingleDataLoader(
                    dataset_name=self.dataset_name,
                    cfg=self._loader_cfg(split="val", shuffle=False),
                    **common_kwargs,
                ).dataloader
        if self.test_loader is None:
            if use_all_loader:
                self.test_loader = AllDataLoader(cfg=self._loader_cfg(split="test", shuffle=False), **all_loader_kwargs).dataloader
            else:
                self.test_loader = SingleDataLoader(
                    dataset_name=self.dataset_name,
                    cfg=self._loader_cfg(split="test", shuffle=False),
                    **common_kwargs,
                ).dataloader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


class BEATsASCModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        mcfg = cfg["model"]
        pcfg = cfg.get("plasticity", {})

        checkpoint_path = str(mcfg.get("checkpoint_path", "checkpoints/BEATs_iter3_plus_AS2M.pt"))
        freeze_backbone = bool(mcfg.get("freeze_backbone", False))
        self.max_layer = mcfg.get("max_layer", None)

        self.backbone = make_beats_block(
            checkpoint_path=checkpoint_path,
            freeze=freeze_backbone,
        )

        backbone_dim = int(getattr(self.backbone, "embed_dim", 768))
        self.embed_dim = int(mcfg.get("dim", backbone_dim))

        if self.embed_dim != backbone_dim:
            self.input_proj = nn.Linear(backbone_dim, self.embed_dim)
        else:
            self.input_proj = nn.Identity()

        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(float(mcfg.get("dropout", 0.0)))
        self.cbp_enabled = bool(pcfg.get("enabled", False))
        self.cbp_act_type = build_activation_name(cfg)

        if self.cbp_enabled:
            cbp_hidden_dim = int(pcfg.get("head_hidden_dim", self.embed_dim))
            self.cbp_hidden = nn.Linear(self.embed_dim, cbp_hidden_dim)
            self.cbp_activation = make_activation_module(self.cbp_act_type)
            self.classifier = nn.Linear(cbp_hidden_dim, num_classes)
            self.cbp_layers = nn.ModuleList([self.cbp_hidden, self.cbp_activation, self.classifier])
        else:
            self.cbp_hidden = None
            self.cbp_activation = None
            self.classifier = nn.Linear(self.embed_dim, num_classes)
            self.cbp_layers = None

    def forward(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor,
        return_cbp_features: bool = False,
        return_token_info: bool = False,
    ):
        features, padding_mask = self.backbone(
            waveforms=waveforms,
            lengths=lengths,
            max_layer=self.max_layer,
            return_padding_mask=True,
        )
        features = self.input_proj(features)
        features = self.norm(features)

        if padding_mask is None:
            valid_mask = torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        else:
            valid_mask = ~padding_mask

        token_features = features
        denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(token_features.dtype)
        cbp_features: List[torch.Tensor] = []
        if self.cbp_enabled:
            token_features = self.cbp_hidden(token_features)
            token_features = self.cbp_activation(token_features)

        pooled = (token_features * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        if self.cbp_enabled:
            cbp_features.append(pooled)
        logits = self.classifier(self.dropout(pooled))

        if return_cbp_features and return_token_info:
            token_logits = self.classifier(self.dropout(token_features))
            return logits, cbp_features, token_logits, valid_mask
        if return_cbp_features:
            return logits, cbp_features
        if return_token_info:
            token_logits = self.classifier(self.dropout(token_features))
            return logits, token_logits, valid_mask
        return logits


class ASCBEATsSystem(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = BEATsASCModel(cfg=cfg, num_classes=num_classes)
        loss_cfg = cfg.get("loss", {})
        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        eaft_cfg = loss_cfg.get("eaft", {})
        if not isinstance(eaft_cfg, dict):
            eaft_cfg = {}

        self.label_smoothing = label_smoothing
        self.eaft_enabled = bool(eaft_cfg.get("enabled", False))
        self.eaft_alpha = float(eaft_cfg.get("alpha", 1.0))
        self.eaft_topk = int(eaft_cfg.get("topk", 20))
        self.eaft_entropy_norm = float(eaft_cfg.get("entropy_norm", 3.0))

        self.criterion = build_cross_entropy_loss(label_smoothing=label_smoothing)
        self.cbp_cfg = build_cbp_config(cfg)
        self.use_plasticity = self.cbp_cfg.enabled
        self._gnt = None
        self._latest_cbp_features: List[torch.Tensor] = []

        if self.use_plasticity and self.model.cbp_layers is None:
            raise ValueError("plasticity.enabled requires a valid CBP head.")

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(waveforms, lengths)

    def on_fit_start(self) -> None:
        if not self.use_plasticity:
            return

        optimizer = self.optimizers(use_pl_optimizer=False)
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        self._gnt = create_gnt(
            layers=self.model.cbp_layers,
            activation_name=self.model.cbp_act_type,
            optimizer=optimizer,
            config=self.cbp_cfg,
            device=self.device,
        )

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        waveforms, lengths, labels = batch
        logits = self(waveforms, lengths)
        loss = self.criterion(logits, labels)
        acc = top1_accuracy(logits, labels)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "test"), sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=(stage != "test"), sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        waveforms, lengths, labels = batch
        token_logits = None
        token_mask = None
        if self.use_plasticity and self.eaft_enabled:
            logits, cbp_features, token_logits, token_mask = self.model(
                waveforms, lengths, return_cbp_features=True, return_token_info=True)
            # Keep only detached activations so the autograd graph from this
            # iteration is not kept alive across hooks/DDP bookkeeping.
            self._latest_cbp_features = [feature.detach() for feature in cbp_features]
        elif self.use_plasticity:
            logits, cbp_features = self.model(waveforms, lengths, return_cbp_features=True)
            self._latest_cbp_features = [feature.detach() for feature in cbp_features]
        elif self.eaft_enabled:
            logits, token_logits, token_mask = self.model(waveforms, lengths, return_token_info=True)
            self._latest_cbp_features = []
        else:
            logits = self(waveforms, lengths)
            self._latest_cbp_features = []

        if self.eaft_enabled:
            loss, eaft_stats = compute_eaft_token_loss(
                token_logits=token_logits,
                labels=labels,
                token_mask=token_mask,
                alpha=self.eaft_alpha,
                entropy_norm=self.eaft_entropy_norm,
                topk=self.eaft_topk,
                label_smoothing=self.label_smoothing,
            )
            self.log("train_eaft_weight_mean", eaft_stats["weight_mean"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train_eaft_entropy_mean", eaft_stats["entropy_mean"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = self.criterion(logits, labels)
        acc = top1_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        if self.use_plasticity and self._gnt is not None and self._latest_cbp_features:
            run_gnt_step(self._gnt, self._latest_cbp_features)
        self._latest_cbp_features = []

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        ocfg = self.cfg["optimizer"]
        scfg = self.cfg.get("scheduler", {})
        tcfg = self.cfg["train"]

        optimizer = AdamW(
            self.parameters(),
            lr=float(ocfg["lr"]),
            weight_decay=float(ocfg.get("weight_decay", 0.0)),
            betas=(float(ocfg.get("beta1", 0.9)), float(ocfg.get("beta2", 0.999))),
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(tcfg["max_epochs"]),
            eta_min=float(scfg.get("eta_min", 1e-6)),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def build_logger(cfg: Dict[str, Any], workspace: Path, wandb_id: Optional[str]):
    lcfg = cfg.get("logging", {})
    project = lcfg.get("project", "asc-beats")
    run_name = lcfg.get("run_name", "train_asc_beats")
    use_wandb = bool(lcfg.get("use_wandb", True))

    if use_wandb:
        return WandbLogger(
            project=project,
            name=run_name,
            id=wandb_id,
            save_dir=str(workspace),
            log_model=False,
        )

    return CSVLogger(save_dir=str(workspace), name="csv_logs")


def main(
    config_path: str,
    resume: Optional[str],
    workspace_path: str,
    wandb_id: Optional[str],
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
    system = ASCBEATsSystem(cfg=cfg, num_classes=len(label2idx))

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(workspace / "checkpoints"),
        filename="epoch{epoch:02d}-val_acc{val_acc:.4f}",
        monitor=cfg.get("logging", {}).get("monitor", "val_acc"),
        mode=cfg.get("logging", {}).get("mode", "max"),
        save_top_k=int(cfg.get("logging", {}).get("save_top_k", 1)),
        save_last=True,
    )

    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    logger = build_logger(cfg=cfg, workspace=workspace, wandb_id=wandb_id)

    tcfg = cfg["train"]
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
        max_epochs=int(tcfg["max_epochs"]),
        precision=tcfg.get("precision", "32-true"),
        gradient_clip_val=float(tcfg.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(tcfg.get("log_every_n_steps", 20)),
        deterministic=bool(tcfg.get("deterministic", False)),
        default_root_dir=str(workspace),
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(system, datamodule=datamodule, ckpt_path=resume)
    test_ckpt = str(tcfg.get("test_ckpt", "last"))
    trainer.test(system, datamodule=datamodule, ckpt_path=test_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASC with BEATs backbone")
    parser.add_argument("-c", "--config", type=str, default="config/train_asc_beats_v2.yaml")
    parser.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument("-w", "--workspace", type=str, default="workspace/train_asc_beats")
    parser.add_argument("-id", "--wandb-id", type=str, default=None)
    parser.add_argument("-d", "--dataset", type=str, default="all", choices=DOMAIN_CHOICES)
    args = parser.parse_args()

    main(
        config_path=args.config,
        resume=args.resume,
        workspace_path=args.workspace,
        wandb_id=args.wandb_id,
        dataset_name=args.dataset,
    )
