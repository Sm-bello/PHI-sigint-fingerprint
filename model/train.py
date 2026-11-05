"""
PHI-RF-CLASSIFIER: 1D-CNN Training, Baseline Comparison & Evaluation Pipeline
=============================================================================
Generates 10,000 simulated I/Q chunks across 5 battlefield RF classes, validates
data integrity, splits into Train/Val/Test sets, compares against naive & classical
ML baselines, trains `RFClassifierCNN` for 10 epochs using CrossEntropyLoss + Adam,
saves weights to `model/rf_model.pth`, and outputs full empirical evaluation metrics.
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulator.spectrum_generator import CLASSES
from model.dataset import build_splits
from model.network import RFClassifierCNN


def _extract_handcrafted_features(X: np.ndarray) -> np.ndarray:
    """
    Extracts classical statistical/spectral features per (2, 1024) I/Q chunk
    Strictly used for classical Logistic Regression baseline comparison.
    """
    iq = X[:, 0, :] + 1j * X[:, 1, :]
    mag = np.abs(iq)
    pwr = mag**2
    mean_pwr = np.mean(pwr, axis=1, keepdims=True) + 1e-12
    std_mag = np.std(mag, axis=1, keepdims=True)
    max_mag = np.max(mag, axis=1, keepdims=True)
    papr = max_mag**2 / mean_pwr

    # Instantaneous phase diff variance
    phase = np.unwrap(np.angle(iq), axis=1)
    dphase = np.diff(phase, axis=1)
    std_dphase = np.std(dphase, axis=1, keepdims=True)

    # FFT spectral flatness proxy
    fft_mag = np.abs(np.fft.fft(iq, axis=1)) + 1e-12
    p_norm = fft_mag / np.sum(fft_mag, axis=1, keepdims=True)
    spec_entropy = -np.sum(p_norm * np.log2(p_norm), axis=1, keepdims=True)

    return np.hstack([mean_pwr, std_mag, max_mag, papr, std_dphase, spec_entropy])


def evaluate_baselines(train_ds, test_ds) -> Dict[str, Any]:
    """
    Evaluates a Naive Majority/Stratified baseline and a Classical Logistic Regression
    baseline with strict featurization ordering (Scaler fit ONLY on X_train).
    """
    X_train_np = train_ds.X.numpy()
    y_train_np = train_ds.y.numpy()
    X_test_np = test_ds.X.numpy()
    y_test_np = test_ds.y.numpy()

    # 1. Naive Dummy Baseline
    dummy = DummyClassifier(strategy="prior", random_state=42)
    dummy.fit(X_train_np.reshape(len(X_train_np), -1), y_train_np)
    y_pred_dummy = dummy.predict(X_test_np.reshape(len(X_test_np), -1))
    dummy_acc = float(accuracy_score(y_test_np, y_pred_dummy))
    dummy_f1 = float(f1_score(y_test_np, y_pred_dummy, average="macro", zero_division=0))

    # 2. Classical Signal Feature + Logistic Regression Baseline (fit scaler strictly on train!)
    feat_train = _extract_handcrafted_features(X_train_np)
    feat_test = _extract_handcrafted_features(X_test_np)

    scaler = StandardScaler()
    feat_train_scaled = scaler.fit_transform(feat_train)  # Fit ONLY on train
    feat_test_scaled = scaler.transform(feat_test)        # Transform test independently

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(feat_train_scaled, y_train_np)
    y_pred_lr = clf.predict(feat_test_scaled)
    lr_acc = float(accuracy_score(y_test_np, y_pred_lr))
    lr_f1 = float(f1_score(y_test_np, y_pred_lr, average="macro"))

    return {
        "naive_baseline": {"accuracy": round(dummy_acc, 4), "macro_f1": round(dummy_f1, 4)},
        "logistic_regression_baseline": {"accuracy": round(lr_acc, 4), "macro_f1": round(lr_f1, 4)},
    }


def train_model(
    num_samples: int = 10000,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1.5e-3,
    weights_path: Path = ROOT_DIR / "model" / "rf_model.pth",
    metrics_path: Path = ROOT_DIR / "model" / "training_metrics.json",
) -> Dict[str, Any]:
    """Full end-to-end training & evaluation pipeline for RFClassifierCNN."""
    print("=" * 80)
    print("  PHI-RF-CLASSIFIER: HMGCC 1D-CNN DEEP LEARNING TRAINING PIPELINE")
    print("=" * 80)

    t0 = time.perf_counter()
    print(f"[*] Synthesizing {num_samples:,} multi-emitter battlefield I/Q chunks (shape: 2x1024)...")
    train_ds, val_ds, test_ds, integrity_report = build_splits(num_samples=num_samples, seed=42)
    print(
        f"[+] Data Integrity Verified: {integrity_report['valid_samples']:,} valid chunks "
        f"(NaNs: {integrity_report['nan_rows_detected']}, Missing Targets: {integrity_report['missing_target_rows']})"
    )
    print(
        f"[+] Stratified Splits -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}"
    )

    # Evaluate baselines first
    baselines = evaluate_baselines(train_ds, test_ds)
    print(
        f"[*] Baselines -> Naive Accuracy: {baselines['naive_baseline']['accuracy']:.2%} | "
        f"Classical Spectral LR Accuracy: {baselines['logistic_regression_baseline']['accuracy']:.2%}"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFClassifierCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_history = []
    best_val_acc = 0.0
    best_state = None

    print(f"[*] Training RFClassifierCNN on [{device}] for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == by).sum().item()
            total_train += bx.size(0)

        scheduler.step()
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train

        # Validation step
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                v_loss = criterion(v_logits, vy)
                val_loss += v_loss.item() * vx.size(0)
                v_preds = v_logits.argmax(dim=1)
                correct_val += (v_preds == vy).sum().item()
                total_val += vx.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 4),
        }
        epoch_history.append(epoch_record)
        print(
            f"    Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}"
        )

    # Restore best validation weights and save to disk
    if best_state is not None:
        model.load_state_dict(best_state)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"[+] Saved trained 1D-CNN weights to: {weights_path}")

    # Held-Out Test Set Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for tx, ty in test_loader:
            tx = tx.to(device)
            t_logits = model(tx)
            all_preds.extend(t_logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(ty.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    test_acc = float(accuracy_score(y_true, y_pred))
    test_f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred).tolist()
    cls_report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)

    # Single-chunk real-time inference latency benchmark (100 warm runs)
    sample_input = torch.randn(1, 2, 1024, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_input)
        t_lat_start = time.perf_counter()
        for _ in range(100):
            _ = model(sample_input)
        avg_latency_ms = ((time.perf_counter() - t_lat_start) / 100.0) * 1000.0

    total_train_time_s = time.perf_counter() - t0

    print("-" * 80)
    print(f"[✓] HELD-OUT TEST ACCURACY : {test_acc:.2%} (Macro F1: {test_f1:.4f})")
    print(f"[✓] SINGLE-CHUNK LATENCY   : {avg_latency_ms:.3f} ms / frame ({1000.0 / avg_latency_ms:,.0f} Hz throughput)")
    print(f"[✓] TOTAL PIPELINE TIME    : {total_train_time_s:.2f} s")
    print("-" * 80)
    print("Per-Class Test Metrics:")
    for cls_name in CLASSES:
        m = cls_report[cls_name]
        print(
            f"  - {cls_name:<18} | Precision: {m['precision']:.2%} | "
            f"Recall: {m['recall']:.2%} | F1: {m['f1-score']:.2%}"
        )
    print("=" * 80)

    summary = {
        "model_architecture": "RFClassifierCNN (3-Layer 1D-CNN + BatchNorm + AdaptiveAvgPool1d)",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "dataset_integrity": integrity_report,
        "baselines": baselines,
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "single_chunk_latency_ms": round(avg_latency_ms, 4),
        "training_time_sec": round(total_train_time_s, 2),
        "confusion_matrix": cm,
        "class_names": CLASSES,
        "per_class_metrics": {
            c: {
                "precision": round(float(cls_report[c]["precision"]), 4),
                "recall": round(float(cls_report[c]["recall"]), 4),
                "f1_score": round(float(cls_report[c]["f1-score"]), 4),
            }
            for c in CLASSES
        },
        "epoch_history": epoch_history,
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    train_model()
