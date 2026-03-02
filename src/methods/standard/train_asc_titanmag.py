from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils.rnn import pad_sequence
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ...datamodule.dataloader import AllDataLoader, LoaderConfig, SingleDataLoader
from ...datamodule.dataset import CLASSES
from ...losses.ce import build_cross_entropy_loss
from ...metrics.acc import top1_accuracy
from ...models.titans.src.titan_backbone import make_titan_block

DOMAIN_CHOICES = ("europe6", "lisbon", "lyon", "prague", "korea", "all")


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
        # Aligns data length utilizing pad_sequence
        waves: List[torch.Tensor] = []
        labels: List[int] = []

        for sample in batch:
            item = sample[0] if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], str) else sample
            waveform, label = item

            if waveform.ndim == 2:
                waveform = waveform.mean(dim=0)
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
            "europe6_root": dcfg["europe6_root"],
            "europe6_meta": dcfg.get("europe6_meta", "meta.csv"),
            "tau2019_root": dcfg["tau2019_root"],
            "lisbon_meta": dcfg.get("lisbon_meta", "lisbon_meta.csv"),
            "lyon_meta": dcfg.get("lyon_meta", "lyon_meta.csv"),
            "prague_meta": dcfg.get("prague_meta", "prague_meta.csv"),
            "korea_root": dcfg["korea_root"],
            "korea_csv": dcfg.get("korea_csv", "cochlscene_meta.csv"),
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


class TitanASCModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        mcfg = cfg["model"]

        self.hop_length = int(mcfg["hop_length"])
        self.n_mels = int(mcfg["n_mels"])
        self.embed_dim = int(mcfg["dim"])

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(mcfg["sample_rate"]),
            n_fft=int(mcfg["n_fft"]),
            hop_length=self.hop_length,
            win_length=int(mcfg.get("win_length", mcfg["n_fft"])),
            n_mels=self.n_mels,
            center=True,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        self.input_proj = nn.Linear(self.n_mels, self.embed_dim)
        self.backbone = make_titan_block(
            dim=self.embed_dim,
            heads=int(mcfg["heads"]),
            head_dim=int(mcfg["head_dim"]),
            window_size=int(mcfg["window_size"]),
            num_persistent=int(mcfg["num_persistent"]),
            store_chunk_size=int(mcfg["store_chunk_size"]),
            max_ltm_lr=float(mcfg["max_ltm_lr"]),
            ttt_batch_size=int(mcfg["ttt_batch_size"]),
            max_grad_norm=float(mcfg["max_grad_norm"]),
            test_time_update=bool(mcfg.get("test_time_update", True)),
            use_accelerated_scan=bool(mcfg.get("use_accelerated_scan", False)),
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def _frame_mask(self, lengths: torch.Tensor, n_frames: int, device: torch.device) -> torch.Tensor:
        frame_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1
        frame_lengths = frame_lengths.clamp(min=1, max=n_frames)
        indices = torch.arange(n_frames, device=device).unsqueeze(0)
        return indices < frame_lengths.unsqueeze(1)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mel = self.to_db(self.melspec(waveforms))
        tokens = self.input_proj(mel.transpose(1, 2))

        mask = self._frame_mask(lengths=lengths, n_frames=tokens.shape[1], device=tokens.device)
        features = self.backbone(tokens, attn_mask=mask)
        features = self.norm(features)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1).to(features.dtype)
        pooled = (features * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.classifier(pooled)


class ASCTitanMAGSystem(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])

        self.model = TitanASCModel(cfg=cfg, num_classes=num_classes)
        label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
        self.criterion = build_cross_entropy_loss(label_smoothing=label_smoothing)

    def forward(self, waveforms: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.model(waveforms, lengths)

    def _shared_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        waveforms, lengths, labels = batch
        logits = self(waveforms, lengths)
        loss = self.criterion(logits, labels)
        acc = top1_accuracy(logits, labels)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=(stage != "test"), sync_dist=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=(stage != "test"), sync_dist=True)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

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
    project = lcfg.get("project", "asc-titanmag")
    run_name = lcfg.get("run_name", "train_asc_titanmag")
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
    system = ASCTitanMAGSystem(cfg=cfg, num_classes=len(label2idx))

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
    # Ensure DDP subprocess uses its own CUDA device before process-group init.
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
    # Test using a checkpoint chosen by config. Prefer "last" for reproducibility.
    # Allowed values: "best", "last", or an explicit checkpoint path.
    test_ckpt = str(tcfg.get("test_ckpt", "last"))
    trainer.test(system, datamodule=datamodule, ckpt_path=test_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASC with Titans MAG backbone")
    parser.add_argument("-c", "--config", type=str, default="config/train_asc_titanmag.yaml")
    parser.add_argument("-r", "--resume", type=str, default=None)
    parser.add_argument("-w", "--workspace", type=str, default="workspace/train_asc_titanmag")
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
