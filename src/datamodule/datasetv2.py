from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

# TAU scene labels
CLASSES = {
    "airport": 0,
    "bus": 1,
    "metro": 2,
    "metro_station": 3,
    "park": 4,
    "public_square": 5,
    "shopping_mall": 6,
    "street_pedestrian": 7,
    "street_traffic": 8,
    "tram": 9,
}


@dataclass(frozen=True)
class TAUDatasetConfig:
    data_root: str
    meta_csv: str
    split: str = "train"  # train | val | test
    seed: int = 42
    target_sample_rate: Optional[int] = None
    return_path: bool = False


def _infer_sep(csv_path: Path) -> str:
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()
    return "\t" if "\t" in first_line else ","


def _read_meta(csv_path: Path) -> pd.DataFrame:
    sep = _infer_sep(csv_path)
    df = pd.read_csv(csv_path, sep=sep)
    if len(df.columns) == 1 and "\t" in df.columns[0]:
        df = pd.read_csv(csv_path, sep="\t")
    return df


def _stratified_split_indices(
    labels: List[str],
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    labels_arr = np.asarray(labels)

    train_idx_all: List[np.ndarray] = []
    val_idx_all: List[np.ndarray] = []
    test_idx_all: List[np.ndarray] = []

    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_idx_all.append(cls_idx[:n_train])
        val_idx_all.append(cls_idx[n_train:n_train + n_val])
        test_idx_all.append(cls_idx[n_train + n_val :])

    train_idx = np.concatenate(train_idx_all) if train_idx_all else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx_all) if val_idx_all else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx_all) if test_idx_all else np.array([], dtype=int)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return train_idx, val_idx, test_idx


class TAUDatasetBase(Dataset):
    """TAU metadata-driven dataset with stratified 80/10/10 split."""

    def __init__(
        self,
        cfg: TAUDatasetConfig,
        label2idx: Dict[str, int],
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        self.cfg = cfg
        self.transform = transform
        self.label2idx = label2idx

        split = cfg.split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split={cfg.split}. Use train/val/test.")

        self.data_root = Path(cfg.data_root)
        self.meta_path = self.data_root / cfg.meta_csv
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Meta CSV not found: {self.meta_path}")

        df = _read_meta(self.meta_path)
        required = {"filename", "scene_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Meta CSV missing columns: {sorted(missing)}. Got: {list(df.columns)}")

        self._all_filenames = df["filename"].astype(str).tolist()
        self._all_labels_str = df["scene_label"].astype(str).tolist()
        unknown = set(self._all_labels_str) - set(self.label2idx.keys())
        if unknown:
            raise ValueError(f"Unknown labels in {self.meta_path}: {sorted(unknown)}")

        train_idx, val_idx, test_idx = _stratified_split_indices(
            labels=self._all_labels_str,
            seed=cfg.seed,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        if split == "train":
            self.indices = train_idx
        elif split == "val":
            self.indices = val_idx
        else:
            self.indices = test_idx

    def __len__(self) -> int:
        return int(len(self.indices))

    def _maybe_resample(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        if self.cfg.target_sample_rate is None or sr == self.cfg.target_sample_rate:
            return waveform, sr
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.target_sample_rate)
        return resampler(waveform), self.cfg.target_sample_rate

    def __getitem__(self, i: int):
        idx = int(self.indices[i])

        rel_path = self._all_filenames[idx]
        label_str = self._all_labels_str[idx]
        label = self.label2idx[label_str]

        wav_path = self.data_root / rel_path
        waveform, sr = torchaudio.load(wav_path)
        waveform, sr = self._maybe_resample(waveform, sr)

        if self.transform is not None:
            out = self.transform(waveform, label)
        else:
            out = (waveform, label)

        if self.cfg.return_path:
            return out, str(wav_path)
        return out


def _build_tau_cfg(
    split: str,
    data_root: str,
    meta_csv: str,
    seed: int,
    target_sample_rate: Optional[int],
    return_path: bool,
) -> TAUDatasetConfig:
    return TAUDatasetConfig(
        data_root=data_root,
        meta_csv=meta_csv,
        split=split,
        seed=seed,
        target_sample_rate=target_sample_rate,
        return_path=return_path,
    )


class LisbonDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class LyonDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class PragueDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class BarcelonaDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class HelsinkiDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class LondonDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class MilanDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class ParisDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class StockholmDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class ViennaDataset(TAUDatasetBase):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        data_root: str,
        meta_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
    ):
        super().__init__(
            cfg=_build_tau_cfg(split, data_root, meta_csv, seed, target_sample_rate, return_path),
            label2idx=label2idx,
            transform=transform,
        )


class AllDataset(Dataset):
    def __init__(
        self,
        split: str,
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
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        return_domain: bool = False,
    ):
        split_l = split.lower()
        if split_l not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split={split}. Use train/val/test.")

        self.return_domain = return_domain

        dataset_defs = [
            (LisbonDataset, lisbon_meta, "Lisbon"),
            (LyonDataset, lyon_meta, "Lyon"),
            (PragueDataset, prague_meta, "Prague"),
            (BarcelonaDataset, barcelona_meta, "Barcelona"),
            (HelsinkiDataset, helsinki_meta, "Helsinki"),
            (LondonDataset, london_meta, "London"),
            (MilanDataset, milan_meta, "Milan"),
            (ParisDataset, paris_meta, "Paris"),
            (StockholmDataset, stockholm_meta, "Stockholm"),
            (ViennaDataset, vienna_meta, "Vienna"),
        ]

        self.datasets: List[Dataset] = []
        self.domain_names: List[str] = []
        for ds_cls, meta_csv, domain_name in dataset_defs:
            self.datasets.append(
                ds_cls(
                    split=split_l,
                    label2idx=label2idx,
                    seed=seed,
                    data_root=tau2019_root,
                    meta_csv=meta_csv,
                    target_sample_rate=target_sample_rate,
                    return_path=return_path,
                    transform=transform,
                )
            )
            self.domain_names.append(domain_name)

        self._cum_lengths: List[int] = []
        total = 0
        for ds in self.datasets:
            total += int(len(ds))
            self._cum_lengths.append(total)

    def __len__(self) -> int:
        return self._cum_lengths[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx} (len={len(self)})")

        lo, hi = 0, len(self._cum_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cum_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1
        ds_id = lo
        prev_cum = 0 if ds_id == 0 else self._cum_lengths[ds_id - 1]
        local_idx = idx - prev_cum
        return ds_id, local_idx

    def __getitem__(self, i: int):
        ds_id, local_idx = self._locate(int(i))
        sample = self.datasets[ds_id][local_idx]

        if not self.return_domain:
            return sample

        return sample, ds_id, self.domain_names[ds_id]
