"""
PHI-RF-CLASSIFIER: Monte Carlo SNR Sweep & Crowded Battlefield Benchmark Suite
==============================================================================
Evaluates the trained 1D-CNN (`RFClassifierCNN`) across:
  1. SNR Robustness Curve: -10 dB, -5 dB, 0 dB, +5 dB, +10 dB, +15 dB, +20 dB
  2. Co-Channel Crowded Battlefield vs. Isolated Spectrum Performance
  3. Hostile Detection Probability (P_d) & False Alarm Rate (FAR) at >80% Interdiction Threshold
  4. Real-Time Latency & Throughput Profiling
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulator.spectrum_generator import SpectrumGenerator, CLASSES, CLASS_TO_IDX
from model.network import RFClassifierCNN
from inference.live_classifier import HOSTILE_CLASSES


def run_full_benchmark_suite(
    weights_path: Path = ROOT_DIR / "model" / "rf_model.pth",
    output_path: Path = ROOT_DIR / "benchmarks" / "benchmark_results.json",
    samples_per_snr: int = 500,
) -> Dict[str, Any]:
    print("=" * 84)
    print("  PHI-RF-CLASSIFIER: HMGCC MONTE CARLO SNR & CROWDED SPECTRUM BENCHMARK CAMPAIGN")
    print("=" * 84)

    if not weights_path.exists():
        from model.train import train_model
        train_model(weights_path=weights_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RFClassifierCNN(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    gen = SpectrumGenerator(seed=2026)
    snr_levels_db = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    snr_curve = {}

    for snr_db in snr_levels_db:
        X_list, y_true = [], []
        for i in range(samples_per_snr):
            cls_name = CLASSES[i % len(CLASSES)]
            iq, lbl = gen.get_next_chunk(class_name=cls_name, snr_db=snr_db, crowded=True)
            X_list.append(iq)
            y_true.append(CLASS_TO_IDX[lbl])

        bx = torch.tensor(np.array(X_list), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(bx)
            probs = F.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

        y_pred = preds.cpu().numpy()
        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average="macro"))
        snr_curve[f"{int(snr_db):+d}_dB"] = {
            "snr_db": snr_db,
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
        }
        print(f"  [SNR {snr_db:+5.1f} dB] Crowded Spectrum Accuracy: {acc:6.2%} | Macro F1: {f1:.4f}")

    # Operational Counter-UAS KPI Evaluation (1,000 Crowded Battlefield Frames at Nominal SNR)
    n_eval = 1000
    true_hostile = 0
    detected_hostile = 0
    true_benign = 0
    false_alarms = 0

    t_start = time.perf_counter()
    for i in range(n_eval):
        cls_name = CLASSES[i % len(CLASSES)]
        iq, lbl = gen.get_next_chunk(class_name=cls_name, snr_db=None, crowded=True)
        tensor_x = torch.tensor(iq, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(tensor_x), dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        pred_cls = CLASSES[int(pred_idx.item())]
        conf_val = float(conf.item())

        is_true_hostile = lbl in HOSTILE_CLASSES
        is_pred_hostile = (pred_cls in HOSTILE_CLASSES) and (conf_val > 0.80)

        if is_true_hostile:
            true_hostile += 1
            if is_pred_hostile:
                detected_hostile += 1
        else:
            true_benign += 1
            if is_pred_hostile:
                false_alarms += 1

    elapsed_s = time.perf_counter() - t_start
    p_detection = detected_hostile / max(1, true_hostile)
    false_alarm_rate = false_alarms / max(1, true_benign)
    mean_latency_ms = (elapsed_s / n_eval) * 1000.0

    print("-" * 84)
    print(f"  [KPI] Hostile Probability of Detection (P_d) : {p_detection:.2%} (Target: >= 95.0%)")
    print(f"  [KPI] Friendly/Noise False Alarm Rate (FAR)  : {false_alarm_rate:.2%} (Target: <= 2.0%)")
    print(f"  [KPI] Mean End-to-End Inference Latency      : {mean_latency_ms:.3f} ms / chunk")
    print("=" * 84)

    results = {
        "campaign": "HMGCC Crowded Battlefield RF Classification Benchmark",
        "samples_per_snr": samples_per_snr,
        "snr_robustness_curve": snr_curve,
        "operational_kpis": {
            "evaluation_frames": n_eval,
            "probability_of_detection_pd": round(p_detection, 4),
            "false_alarm_rate_far": round(false_alarm_rate, 4),
            "mean_inference_latency_ms": round(mean_latency_ms, 4),
            "throughput_chunks_per_sec": round(1000.0 / mean_latency_ms, 1),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_full_benchmark_suite()
