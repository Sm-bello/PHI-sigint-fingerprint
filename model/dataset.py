"""
PHI-RF-CLASSIFIER: PyTorch I/Q Tensor Dataset & Stratified Split Builder
========================================================================
Converts synthetic or captured 2-channel I/Q frames (shape: 2 x 1024) into PyTorch tensors.
Strictly adheres to ML Best Practices:
  - Validates schema, shape (2, 1024), and checks for NaN / Inf / missing target values.
  - Performs stratified Train / Validation / Test splitting prior to any downstream fitting.
"""
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from simulator.spectrum_generator import SpectrumGenerator, CLASSES, CLASS_TO_IDX


class RFIQDataset(Dataset):
    """PyTorch Dataset wrapping 2-channel (I/Q) baseband tensors of shape (N, 2, 1024)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3 or X.shape[1] != 2:
            raise ValueError(f"Expected I/Q array shape (N, 2, chunk_size), got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"Feature/label length mismatch: {len(X)} vs {len(y)}")

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def generate_raw_corpus(
    num_samples: int = 10000,
    chunk_size: int = 1024,
    seed: int = 42,
    snr_range_db: Optional[Tuple[float, float]] = (-2.0, 18.0),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Synthesizes `num_samples` balanced 2-channel I/Q frames across all 5 tactical classes
    and performs schema/integrity validation.
    """
    gen = SpectrumGenerator(chunk_size=chunk_size, seed=seed)
    rng = np.random.default_rng(seed)

    X = np.empty((num_samples, 2, chunk_size), dtype=np.float32)
    y = np.empty(num_samples, dtype=np.int64)

    num_classes = len(CLASSES)
    for i in range(num_samples):
        cls_idx = i % num_classes
        cls_name = CLASSES[cls_idx]
        snr_db = float(rng.uniform(snr_range_db[0], snr_range_db[1])) if snr_range_db else None
        iq_chunk, label = gen.get_next_chunk(class_name=cls_name, snr_db=snr_db, crowded=True)
        X[i] = iq_chunk
        y[i] = CLASS_TO_IDX[label]

    # Shuffle corpus deterministically
    perm = rng.permutation(num_samples)
    X = X[perm]
    y = y[perm]

    # Data Integrity & Missing/NaN Verification (ML Best Practices)
    nan_rows = int(np.isnan(X).any(axis=(1, 2)).sum())
    inf_rows = int(np.isinf(X).any(axis=(1, 2)).sum())
    missing_targets = int((y < 0).sum() + (y >= num_classes).sum())

    if nan_rows > 0 or inf_rows > 0:
        valid_mask = ~(np.isnan(X).any(axis=(1, 2)) | np.isinf(X).any(axis=(1, 2)))
        X = X[valid_mask]
        y = y[valid_mask]

    integrity_report = {
        "total_generated": num_samples,
        "valid_samples": int(len(X)),
        "nan_rows_detected": nan_rows,
        "inf_rows_detected": inf_rows,
        "missing_target_rows": missing_targets,
        "class_distribution": {CLASSES[k]: int((y == k).sum()) for k in range(num_classes)},
    }
    return X, y, integrity_report


def build_splits(
    num_samples: int = 10000,
    chunk_size: int = 1024,
    seed: int = 42,
) -> Tuple[RFIQDataset, RFIQDataset, RFIQDataset, Dict[str, Any]]:
    """
    Generates `num_samples` I/Q chunks and performs a strict stratified split into
    Train (70%), Validation (15%), and Test (15%) sets before any model or scaler fitting.
    """
    X, y, integrity_report = generate_raw_corpus(
        num_samples=num_samples, chunk_size=chunk_size, seed=seed
    )

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Second split: 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed, stratify=y_temp
    )

    integrity_report["splits"] = {
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
    }

    return (
        RFIQDataset(X_train, y_train),
        RFIQDataset(X_val, y_val),
        RFIQDataset(X_test, y_test),
        integrity_report,
    )
