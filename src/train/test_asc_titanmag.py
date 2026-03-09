from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import yaml
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.utils.rnn import pad_sequence

from ..datamodule.dataset import CLASSES
from .train_asc_titanmag import ASCDataModule, ASCTitanMAGSystem, DOMAIN_CHOICES

DOMAIN_SEQUENCE = ("europe6", "lisbon", "lyon", "prague", "korea")


def _extract_test_acc(metrics: Dict[str, Any]) -> float:
    if "test_acc" in metrics:
        return float(metrics["test_acc"])
    for key, value in metrics.items():
        if key.endswith("test_acc"):
            return float(value)
    raise KeyError(f"test_acc metric not found in metrics keys: {list(metrics.keys())}")


def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(checkpoint)}")
    return checkpoint


def _align_plasticity_cfg_with_checkpoint(cfg: Dict[str, Any], checkpoint_path: str) -> None:
    checkpoint = _load_checkpoint(checkpoint_path=checkpoint_path)
    state_dict = checkpoint.get("state_dict", {})
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint state_dict is missing or invalid.")

    has_cbp_head = "model.cbp_hidden.weight" in state_dict
    pcfg = cfg.setdefault("plasticity", {})
    prev_enabled = bool(pcfg.get("enabled", False))
    pcfg["enabled"] = has_cbp_head

    if has_cbp_head:
        cbp_hidden_weight = state_dict["model.cbp_hidden.weight"]
        if not isinstance(cbp_hidden_weight, torch.Tensor):
            raise TypeError("model.cbp_hidden.weight in checkpoint is not a tensor.")
        inferred_hidden_dim = int(cbp_hidden_weight.shape[0])
        pcfg["head_hidden_dim"] = inferred_hidden_dim
        pcfg.setdefault("activation", "relu")
        print(
            f"[CFG] plasticity enabled from checkpoint (head_hidden_dim={inferred_hidden_dim})."
        )
    elif prev_enabled:
        print("[CFG] plasticity disabled to match checkpoint (no CBP head found).")


def _collate_batch_with_paths(
    batch: List[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    waves: List[torch.Tensor] = []
    labels: List[int] = []
    paths: List[str] = []

    for sample in batch:
        if isinstance(sample, tuple) and len(sample) == 2 and isinstance(sample[1], str):
            item, path = sample
        else:
            item, path = sample, ""
        waveform, label = item

        if waveform.ndim == 2:
            waveform = waveform[0]
        elif waveform.ndim > 2:
            waveform = waveform.reshape(-1)

        waves.append(waveform)
        labels.append(int(label))
        paths.append(path)

    lengths = torch.tensor([w.shape[-1] for w in waves], dtype=torch.long)
    padded = pad_sequence(waves, batch_first=True)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, label_tensor, paths


def _build_test_loader_for_paths(
    cfg: Dict[str, Any],
    label2idx: Dict[str, int],
    domain: str,
):
    dcfg = cfg["dataset"]
    # Use a single-process loader for CSV export stability.
    loader_cfg = LoaderConfig(
        split="test",
        batch_size=int(dcfg["batch_size"]),
        num_workers=0,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
    )
    loader = SingleDataLoader(
        dataset_name=domain,
        cfg=loader_cfg,
        label2idx=label2idx,
        seed=int(cfg.get("seed", 42)),
        target_sample_rate=int(dcfg["target_sample_rate"]),
        return_path=True,
        collate_fn=_collate_batch_with_paths,
        europe6_root=dcfg["europe6_root"],
        europe6_meta=dcfg.get("europe6_meta", "meta.csv"),
        tau2019_root=dcfg["tau2019_root"],
        lisbon_meta=dcfg.get("lisbon_meta", "lisbon_meta.csv"),
        lyon_meta=dcfg.get("lyon_meta", "lyon_meta.csv"),
        prague_meta=dcfg.get("prague_meta", "prague_meta.csv"),
        korea_root=dcfg["korea_root"],
        korea_csv=dcfg.get("korea_csv", "cochlscene_meta.csv"),
    )
    return loader.dataloader


class ASCTitanMAGEvalSystem(ASCTitanMAGSystem):
    def __init__(self, cfg: Dict[str, Any], num_classes: int):
        super().__init__(cfg=cfg, num_classes=num_classes)
        self._collect_rows = False
        self._rows: List[Dict[str, str]] = []
        self._idx2label: Dict[int, str] = {}
        self._domain = ""

    def enable_sample_collection(self, idx2label: Dict[int, str], domain: str) -> None:
        self._collect_rows = True
        self._idx2label = idx2label
        self._domain = domain
        self._rows = []

    def pop_rows(self) -> List[Dict[str, str]]:
        rows = self._rows
        self._rows = []
        return rows

    def test_step(self, batch, batch_idx: int) -> None:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            waveforms, lengths, labels, paths = batch
        else:
            waveforms, lengths, labels = batch
            paths = None

        logits = self(waveforms, lengths)
        loss = self.criterion(logits, labels)
        acc = top1_accuracy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        if self._collect_rows and paths is not None:
            pred_indices = torch.argmax(logits, dim=1).detach().cpu().tolist()
            gt_indices = labels.detach().cpu().tolist()
            for pred_idx, gt_idx, sample_path in zip(pred_indices, gt_indices, paths):
                self._rows.append(
                    {
                        "domain": self._domain,
                        "pred_class": self._idx2label.get(int(pred_idx), str(int(pred_idx))),
                        "gt_class": self._idx2label.get(int(gt_idx), str(int(gt_idx))),
                        "sample_path": sample_path,
                    }
                )


def _gather_rows_across_ranks(local_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not (dist.is_available() and dist.is_initialized()):
        return local_rows
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_rows, gathered, dst=0)
    if rank != 0:
        return []
    merged: List[Dict[str, str]] = []
    for rows in gathered:
        if rows:
            merged.extend(rows)
    return merged


def _save_sample_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["domain", "pred_class", "gt_class", "sample_path"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(
    config_path: str,
    checkpoint_path: str,
    workspace_path: str,
    dataset_name: str,
    save_sample_csv: bool,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    assert dataset_name in DOMAIN_CHOICES
    _align_plasticity_cfg_with_checkpoint(cfg=cfg, checkpoint_path=checkpoint_path)

    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)

    label_names = cfg.get("dataset", {}).get("label_names", list(CLASSES.keys()))
    label2idx = {name: idx for idx, name in enumerate(label_names)}
    idx2label = {idx: name for name, idx in label2idx.items()}

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
    is_global_zero = bool(getattr(trainer, "is_global_zero", True))
    world_size = int(getattr(trainer, "world_size", 1))

    test_domains: List[str] = list(DOMAIN_SEQUENCE) if dataset_name == "all" else [dataset_name]
    domain_acc: Dict[str, float] = {}
    sample_rows: List[Dict[str, str]] = []

    for domain in test_domains:
        if is_global_zero:
            print(f"[TEST] domain={domain}, checkpoint={checkpoint_path}")
        system = ASCTitanMAGEvalSystem.load_from_checkpoint(
            checkpoint_path,
            cfg=cfg,
            num_classes=len(label2idx),
            map_location="cpu",
        )
        if save_sample_csv:
            system.enable_sample_collection(idx2label=idx2label, domain=domain)
            test_loader = _build_test_loader_for_paths(cfg=cfg, label2idx=label2idx, domain=domain)
            test_results = trainer.test(model=system, dataloaders=test_loader)
        else:
            datamodule = ASCDataModule(cfg=cfg, label2idx=label2idx, dataset_name=domain)
            test_results = trainer.test(model=system, datamodule=datamodule)
        if not test_results and (world_size <= 1 or is_global_zero):
            raise RuntimeError(f"No test result returned for domain: {domain}")
        if test_results:
            acc = _extract_test_acc(test_results[0])
            domain_acc[domain] = acc
            if is_global_zero:
                print(f"[RESULT] {domain}: test_acc={acc:.4f}")
        if save_sample_csv:
            local_rows = system.pop_rows()
            gathered_rows = _gather_rows_across_ranks(local_rows)
            if is_global_zero:
                sample_rows.extend(gathered_rows)

    plots_dir = workspace / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = Path(checkpoint_path).stem
    plot_path = plots_dir / f"{ckpt_stem}_eval.png"

    x_domains = list(domain_acc.keys())
    y_acc = [domain_acc[d] for d in x_domains]
    if is_global_zero:
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

    if save_sample_csv and is_global_zero:
        csv_path = plots_dir / f"{ckpt_stem}_sample_predictions.csv"
        _save_sample_csv(csv_path=csv_path, rows=sample_rows)
        print(f"[CSV] saved: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ASC with Titans MAG backbone")
    parser.add_argument("-c", "--config", type=str, default="config/train_asc_titanmag.yaml")
    parser.add_argument("-k", "--checkpoint", type=str, required=True)
    parser.add_argument("-w", "--workspace", type=str, default="workspace/test_asc_titanmag")
    parser.add_argument("-d", "--dataset", type=str, default="all", choices=DOMAIN_CHOICES)
    parser.add_argument(
        "--save-sample-csv",
        action="store_true", default=True,
        help="Save per-test-sample predictions (pred/gt/path) CSV under workspace/plots.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        workspace_path=args.workspace,
        dataset_name=args.dataset,
        save_sample_csv=args.save_sample_csv,
    )
