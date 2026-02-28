from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .dataset import (
    AllDataset,
    Europe6Dataset,
    LisbonDataset,
    LyonDataset,
    PragueDataset,
    KoreaDataset,
    KoreaDatasetConfig,
)


@dataclass(frozen=True)
class LoaderConfig:
    split: str = "train"  # train | val | test
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2


class SingleDataLoader:
    """Wrap a single domain Dataset into a torch DataLoader.

    This chooses one of the 5 domain datasets (Europe6/Lisbon/Lyon/Prague/Korea)
    and returns a DataLoader.

    - label2idx must be injected from outside.
    - split is applied as in each Dataset implementation.
    """

    _NAME2CLS = {
        "europe6": Europe6Dataset,
        "lisbon": LisbonDataset,
        "lyon": LyonDataset,
        "prague": PragueDataset,
        "korea": KoreaDataset,
    }

    def __init__(
        self,
        dataset_name: str,
        label2idx: Dict[str, int],
        europe6_root: str,
        europe6_meta: str,
        tau2019_root: str,
        lisbon_meta: str,
        lyon_meta: str,
        prague_meta: str,
        korea_root: str,
        korea_csv: str,
        cfg: Optional[LoaderConfig] = None,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name.lower().strip()
        assert self.dataset_name in self._NAME2CLS

        self.label2idx = label2idx
        self.cfg = cfg or LoaderConfig()
        self.seed = seed
        self.target_sample_rate = target_sample_rate
        self.return_path = return_path
        self.transform = transform
        self.collate_fn = collate_fn

        # save overrides
        self._paths = {
            "europe6_root": europe6_root,
            "europe6_meta": europe6_meta,
            "tau2019_root": tau2019_root,
            "lisbon_meta": lisbon_meta,
            "lyon_meta": lyon_meta,
            "prague_meta": prague_meta,
            "korea_root": korea_root,
            "korea_csv": korea_csv,
        }

        self.dataset = self._build_dataset()
        self.dataloader = self._build_dataloader()

    def _build_dataset(self):
        split = self.cfg.split

        if self.dataset_name == "europe6":
            return Europe6Dataset(
                split=split,
                label2idx=self.label2idx,
                seed=self.seed,
                data_root=self._paths["europe6_root"],
                meta_csv=self._paths["europe6_meta"],
                target_sample_rate=self.target_sample_rate,
                return_path=self.return_path,
                transform=self.transform,
            )

        if self.dataset_name == "lisbon":
            return LisbonDataset(
                split=split,
                label2idx=self.label2idx,
                seed=self.seed,
                data_root=self._paths["tau2019_root"],
                meta_csv=self._paths["lisbon_meta"],
                target_sample_rate=self.target_sample_rate,
                return_path=self.return_path,
                transform=self.transform,
            )

        if self.dataset_name == "lyon":
            return LyonDataset(
                split=split,
                label2idx=self.label2idx,
                seed=self.seed,
                data_root=self._paths["tau2019_root"],
                meta_csv=self._paths["lyon_meta"],
                target_sample_rate=self.target_sample_rate,
                return_path=self.return_path,
                transform=self.transform,
            )

        if self.dataset_name == "prague":
            return PragueDataset(
                split=split,
                label2idx=self.label2idx,
                seed=self.seed,
                data_root=self._paths["tau2019_root"],
                meta_csv=self._paths["prague_meta"],
                target_sample_rate=self.target_sample_rate,
                return_path=self.return_path,
                transform=self.transform,
            )

        # else(korea)
        # KoreaDataset expects split prefix names Train/Val/Test
        split_l = split.lower()
        korea_split = {"train": "Train", "val": "Val", "test": "Test"}.get(split_l)
        assert korea_split is not None

        korea_cfg = KoreaDatasetConfig(
            data_root=self._paths["korea_root"],
            csv_name=self._paths["korea_csv"],
            split=korea_split,
            return_path=self.return_path,
            target_sample_rate=(self.target_sample_rate if self.target_sample_rate is not None else 16000),
        )

        return KoreaDataset(
            cfg=korea_cfg,
            transform=self.transform,
            label2idx=self.label2idx,
        )

    def _build_dataloader(self) -> DataLoader:
        # persistent_workers requires num_workers > 0
        persistent = self.cfg.persistent_workers and self.cfg.num_workers > 0

        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(self.cfg.shuffle if self.cfg.split.lower() == "train" else False),
            num_workers=self.cfg.num_workers,
            drop_last=self.cfg.drop_last,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=persistent,
            prefetch_factor=(self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None),
            collate_fn=self.collate_fn,
        )


class AllDataLoader:
    def __init__(
        self,
        label2idx: Dict[str, int],
        europe6_root: str,
        europe6_meta: str,
        tau2019_root: str,
        lisbon_meta: str,
        lyon_meta: str,
        prague_meta: str,
        korea_root: str,
        korea_csv: str,
        cfg: Optional[LoaderConfig] = None,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        collate_fn: Optional[Callable] = None,
        return_domain: bool = False,
    ):
        self.label2idx = label2idx
        self.cfg = cfg or LoaderConfig()
        self.seed = seed
        self.target_sample_rate = target_sample_rate
        self.return_path = return_path
        self.transform = transform
        self.collate_fn = collate_fn
        self.return_domain = return_domain

        self.dataset = AllDataset(
            split=self.cfg.split,
            label2idx=self.label2idx,
            seed=self.seed,
            target_sample_rate=self.target_sample_rate,
            return_path=self.return_path,
            transform=self.transform,
            return_domain=self.return_domain,
            europe6_root=europe6_root,
            europe6_meta=europe6_meta,
            tau2019_root=tau2019_root,
            lisbon_meta=lisbon_meta,
            lyon_meta=lyon_meta,
            prague_meta=prague_meta,
            korea_root=korea_root,
            korea_csv=korea_csv,
        )

        persistent = self.cfg.persistent_workers and self.cfg.num_workers > 0

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(self.cfg.shuffle if self.cfg.split.lower() == "train" else False),
            num_workers=self.cfg.num_workers,
            drop_last=self.cfg.drop_last,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=persistent,
            prefetch_factor=(self.cfg.prefetch_factor if self.cfg.num_workers > 0 else None),
            collate_fn=self.collate_fn,
        )
