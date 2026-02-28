from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union, Any, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio  # torchaudio.load 사용

# TUT2018, TAU2019
# NOTE: cochl's different class-names are converted correctly based on below.
CLASSES = {"airport": 0, "bus": 1, "metro": 2, "metro_station": 3, "park": 4, "public_square": 5,
           "shopping_mall": 6, "street_pedestrian": 7, "street_traffic": 8, "tram": 9}

# configs
@dataclass(frozen=True)
class TAUDatasetConfig:
    data_root: str
    meta_csv: str  # meta.csv or lisbon_meta.csv ...
    split: str = "train"  # "train" | "val" | "test"
    seed: int = 42
    audio_dir: str = "audio"
    target_sample_rate: Optional[int] = None
    return_path: bool = False

@dataclass(frozen=True)
class KoreaDatasetConfig:
    data_root: str  # CochlScene/CochlScene 경로
    csv_name: str
    split: str = "Train"  # "Train" | "Val" | "Test"
    return_path: bool = False
    target_sample_rate: Optional[int] = 16000  # 예: 16000, None이면 원본 유지


## Utility functions
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
    """
    scene_label 기준 stratified 80/10/10 split.
    - 각 클래스별로 shuffle 후 비율대로 나눈 뒤, 전체 인덱스를 합침.
    - seed 고정으로 재현 가능.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)

    train_idx_all = []
    val_idx_all = []
    test_idx_all = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_idx = cls_idx[:n_train]
        val_idx = cls_idx[n_train:n_train + n_val]
        test_idx = cls_idx[n_train + n_val:]

        train_idx_all.append(train_idx)
        val_idx_all.append(val_idx)
        test_idx_all.append(test_idx)

    train_idx_all = np.concatenate(train_idx_all) if train_idx_all else np.array([], dtype=int)
    val_idx_all = np.concatenate(val_idx_all) if val_idx_all else np.array([], dtype=int)
    test_idx_all = np.concatenate(test_idx_all) if test_idx_all else np.array([], dtype=int)

    rng.shuffle(train_idx_all)
    rng.shuffle(val_idx_all)
    rng.shuffle(test_idx_all)

    return train_idx_all, val_idx_all, test_idx_all



## Domain-wise dataset
class TAUDatasetBase(Dataset):
    """
    TAU meta CSV 기반 Dataset (80/10/10 stratified split 내장) 
    """

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
        self.audio_root = self.data_root / cfg.audio_dir

        assert self.meta_path.exists()
        assert self.audio_root.exists()

        df = _read_meta(self.meta_path)

        required = {"filename", "scene_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Meta CSV missing columns: {sorted(missing)}. Got: {list(df.columns)}")

        self._all_filenames = df["filename"].astype(str).tolist()
        self._all_labels_str = df["scene_label"].astype(str).tolist()
        assert not (set(self._all_labels_str) - set(self.label2idx.keys()))

        # split indices 생성 (stratified)
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

        rel = self._all_filenames[idx]
        label_str = self._all_labels_str[idx]
        label = self.label2idx[label_str]

        wav_path = self.audio_root / rel
        waveform, sr = torchaudio.load(wav_path)
        waveform, sr = self._maybe_resample(waveform, sr)

        if self.transform is not None:
            out = self.transform(waveform, label)
        else:
            out = (waveform, label)

        if self.cfg.return_path:
            return out, str(wav_path)
        return out


# -------------------------
# Concrete Datasets
# -------------------------
class Europe6Dataset(TAUDatasetBase):
    """
    2020 mobile development (Europe 6 cities) meta.csv 기반
    """
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
        cfg = TAUDatasetConfig(
            data_root=data_root,
            meta_csv=meta_csv,
            split=split,
            seed=seed,
            audio_dir="audio",
            target_sample_rate=target_sample_rate,
            return_path=return_path,
        )
        super().__init__(cfg=cfg, label2idx=label2idx, transform=transform)


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
        cfg = TAUDatasetConfig(
            data_root=data_root,
            meta_csv=meta_csv,
            split=split,
            seed=seed,
            audio_dir="audio",
            target_sample_rate=target_sample_rate,
            return_path=return_path,
        )
        super().__init__(cfg=cfg, label2idx=label2idx, transform=transform)


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
        cfg = TAUDatasetConfig(
            data_root=data_root,
            meta_csv=meta_csv,
            split=split,
            seed=seed,
            audio_dir="audio",
            target_sample_rate=target_sample_rate,
            return_path=return_path,
        )
        super().__init__(cfg=cfg, label2idx=label2idx, transform=transform)


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
        cfg = TAUDatasetConfig(
            data_root=data_root,
            meta_csv=meta_csv,
            split=split,
            seed=seed,
            audio_dir="audio",
            target_sample_rate=target_sample_rate,
            return_path=return_path,
        )
        super().__init__(cfg=cfg, label2idx=label2idx, transform=transform)

class KoreaDataset(Dataset):
    """
    CochlScene meta CSV(cochlscene_meta.csv)를 기반으로 오디오/라벨을 로드하는 Dataset.

    CSV columns: filename, scene_label
      - filename: data_root 기준 상대경로 (예: "Train/Bus/xxx.wav")
      - scene_label: {"bus", "metro_station", "pedestrian", "park"}

    __getitem__ returns:
      - (waveform, label)  where waveform is torch.Tensor [C, T], label is int
      - optional path 반환 가능
    """

    def __init__(
        self,
        cfg: KoreaDatasetConfig,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        label2idx: Optional[Dict[str, int]] = None,  # Converts label(str) into integer
    ):
        self.cfg = cfg
        self.data_root = Path(cfg.data_root)
        self.csv_path = self.data_root / cfg.csv_name
        self.transform = transform

        df = pd.read_csv(self.csv_path)

        # split 필터: filename이 "Train/..." 형태라고 가정
        split_prefix = cfg.split + "/"
        df = df[df["filename"].astype(str).str.startswith(split_prefix)].reset_index(drop=True)

        self.filenames = df["filename"].astype(str).tolist()
        self.scene_labels = df["scene_label"].astype(str).tolist()
        self.label2idx = label2idx
        assert not (set(self.scene_labels) - set(self.label2idx.keys()))

    def __len__(self) -> int:
        return len(self.filenames)

    def _maybe_resample(self, waveform: torch.Tensor, sr: int) -> Tuple[torch.Tensor, int]:
        if self.cfg.target_sample_rate is None or sr == self.cfg.target_sample_rate:
            return waveform, sr
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.cfg.target_sample_rate)
        return resampler(waveform), self.cfg.target_sample_rate

    def __getitem__(self, i: int):
        rel_path = self.filenames[i]
        wav_path = self.data_root / rel_path
        
        waveform, sr = torchaudio.load(wav_path)
        waveform, sr = self._maybe_resample(waveform, sr)

        label_str = self.scene_labels[i]
        label = self.label2idx[label_str]

        if self.transform is not None:
            out = self.transform(waveform, label)
        else:
            out = (waveform, label)

        if self.cfg.return_path:
            return out, str(wav_path)

        return out


## Whole Domains in one Dataset - not for incremental learning
class AllDataset(Dataset):
    def __init__(
        self,
        split: str,
        label2idx: Dict[str, int],
        europe6_root: str,
        europe6_meta: str,
        tau2019_root: str,
        lisbon_meta: str,
        lyon_meta: str,
        prague_meta: str,
        korea_root: str,
        korea_csv: str,
        seed: int = 42,
        target_sample_rate: Optional[int] = None,
        return_path: bool = False,
        transform: Optional[Callable[[torch.Tensor, int], Any]] = None,
        return_domain: bool = False,
    ):
        split_l = split.lower()
        assert split_l in {"train", "val", "test"}

        self.return_domain = return_domain

        # --- Build TAU datasets (train/val/test are internal 80/10/10 splits) ---
        self.datasets: List[Dataset] = [
            Europe6Dataset(
                split=split_l,
                label2idx=label2idx,
                seed=seed,
                data_root=europe6_root,
                meta_csv=europe6_meta,
                target_sample_rate=target_sample_rate,
                return_path=return_path,
                transform=transform,
            ),
            LisbonDataset(
                split=split_l,
                label2idx=label2idx,
                seed=seed,
                data_root=tau2019_root,
                meta_csv=lisbon_meta,
                target_sample_rate=target_sample_rate,
                return_path=return_path,
                transform=transform,
            ),
            LyonDataset(
                split=split_l,
                label2idx=label2idx,
                seed=seed,
                data_root=tau2019_root,
                meta_csv=lyon_meta,
                target_sample_rate=target_sample_rate,
                return_path=return_path,
                transform=transform,
            ),
            PragueDataset(
                split=split_l,
                label2idx=label2idx,
                seed=seed,
                data_root=tau2019_root,
                meta_csv=prague_meta,
                target_sample_rate=target_sample_rate,
                return_path=return_path,
                transform=transform,
            ),
        ]

        # --- Build Korea dataset ---
        # KoreaDataset uses split names "Train"|"Val"|"Test" in filenames
        korea_split = {"train": "Train", "val": "Val", "test": "Test"}[split_l]
        korea_cfg = KoreaDatasetConfig(
            data_root=korea_root,
            csv_name=korea_csv,
            split=korea_split,
            return_path=return_path,
            target_sample_rate=target_sample_rate if target_sample_rate is not None else 16000,
        )
        self.datasets.append(
            KoreaDataset(
                cfg=korea_cfg,
                transform=transform,
                label2idx=label2idx,
            )
        )

        self.domain_names = ["Europe6", "Lisbon", "Lyon", "Prague", "Korea"]
        self._cum_lengths: List[int] = []
        total = 0
        for ds in self.datasets:
            total += int(len(ds))
            self._cum_lengths.append(total)

    def __len__(self) -> int:
        return self._cum_lengths[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Map global idx -> (dataset_id, local_idx)."""
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

        domain_name = self.domain_names[ds_id]
        return sample, ds_id, domain_name
