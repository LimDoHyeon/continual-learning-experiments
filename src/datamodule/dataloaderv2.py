from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .datasetv2 import (
    AllDataset,
    BarcelonaDataset,
    HelsinkiDataset,
    LisbonDataset,
    LondonDataset,
    LyonDataset,
    MilanDataset,
    ParisDataset,
    PragueDataset,
    StockholmDataset,
    ViennaDataset,
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
    """Wrap one TAU2019 domain dataset into a torch DataLoader."""

    _NAME2CLS = {
        "lisbon": LisbonDataset,
        "lyon": LyonDataset,
        "prague": PragueDataset,
        "barcelona": BarcelonaDataset,
        "helsinki": HelsinkiDataset,
        "london": LondonDataset,
        "milan": MilanDataset,
        "paris": ParisDataset,
        "stockholm": StockholmDataset,
        "vienna": ViennaDataset,
    }

    _NAME2METAKEY = {
        "lisbon": "lisbon_meta",
        "lyon": "lyon_meta",
        "prague": "prague_meta",
        "barcelona": "barcelona_meta",
        "helsinki": "helsinki_meta",
        "london": "london_meta",
        "milan": "milan_meta",
        "paris": "paris_meta",
        "stockholm": "stockholm_meta",
        "vienna": "vienna_meta",
    }

    def __init__(
        self,
        dataset_name: str,
        label2idx: Dict[str, int],
        tau2019_root: str,
        lisbon_meta: str,
        lyon_meta: str,
        prague_meta: str,
        barcelona_meta: str,
        helsinki_meta: str,
        london_meta: str,
        milan_meta: str,
        paris_meta: str,
        stockholm_meta: str,
        vienna_meta: str,
        cfg: Optional[LoaderConfig] = None,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name.lower().strip()
        if self.dataset_name not in self._NAME2CLS:
            valid = ", ".join(sorted(self._NAME2CLS.keys()))
            raise ValueError(f"Invalid dataset_name={dataset_name}. Valid: {valid}")

        self.label2idx = label2idx
        self.cfg = cfg or LoaderConfig()
        self.seed = seed
        self.target_sample_rate = target_sample_rate
        self.return_path = return_path
        self.transform = transform
        self.collate_fn = collate_fn

        self._paths = {
            "tau2019_root": tau2019_root,
            "lisbon_meta": lisbon_meta,
            "lyon_meta": lyon_meta,
            "prague_meta": prague_meta,
            "barcelona_meta": barcelona_meta,
            "helsinki_meta": helsinki_meta,
            "london_meta": london_meta,
            "milan_meta": milan_meta,
            "paris_meta": paris_meta,
            "stockholm_meta": stockholm_meta,
            "vienna_meta": vienna_meta,
        }

        self.dataset = self._build_dataset()
        self.dataloader = self._build_dataloader()

    def _build_dataset(self):
        ds_cls = self._NAME2CLS[self.dataset_name]
        meta_key = self._NAME2METAKEY[self.dataset_name]
        return ds_cls(
            split=self.cfg.split,
            label2idx=self.label2idx,
            seed=self.seed,
            data_root=self._paths["tau2019_root"],
            meta_csv=self._paths[meta_key],
            target_sample_rate=self.target_sample_rate,
            return_path=self.return_path,
            transform=self.transform,
        )

    def _build_dataloader(self) -> DataLoader:
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
        tau2019_root: str,
        lisbon_meta: str,
        lyon_meta: str,
        prague_meta: str,
        barcelona_meta: str,
        helsinki_meta: str,
        london_meta: str,
        milan_meta: str,
        paris_meta: str,
        stockholm_meta: str,
        vienna_meta: str,
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
            tau2019_root=tau2019_root,
            lisbon_meta=lisbon_meta,
            lyon_meta=lyon_meta,
            prague_meta=prague_meta,
            barcelona_meta=barcelona_meta,
            helsinki_meta=helsinki_meta,
            london_meta=london_meta,
            milan_meta=milan_meta,
            paris_meta=paris_meta,
            stockholm_meta=stockholm_meta,
            vienna_meta=vienna_meta,
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
