"""
PHI-RF-CLASSIFIER: Real-Time 1D-CNN RF Classification Engine
============================================================
Loads trained `RFClassifierCNN` weights (`model/rf_model.pth`), ingests continuous
complex I/Q chunks (2 x 1024) from `SpectrumGenerator` (or physical SDR hooks),
computes Softmax confidence probabilities, and triggers colorized tactical alerts
plus optional PHI-GUARD interdiction telemetry dispatch.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulator.spectrum_generator import SpectrumGenerator, CLASSES
from model.network import RFClassifierCNN

HOSTILE_CLASSES = {"Hostile_MAVLink", "Hostile_OcuSync", "Jammer"}

# ANSI Color codes for terminal alerts
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


class LiveClassifier:
    """Real-time deep learning RF emitter classification engine."""

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        auto_train_if_missing: bool = True,
        enable_phi_guard_bridge: bool = True,
    ):
        self.weights_path = Path(weights_path) if weights_path else (ROOT_DIR / "model" / "rf_model.pth")
        self.generator = SpectrumGenerator()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RFClassifierCNN(num_classes=len(CLASSES)).to(self.device)
        self.threat_level = 0
        self.frames_processed = 0
        self.hostile_detections = 0
        self.correct_predictions = 0

        if not self.weights_path.exists() and auto_train_if_missing:
            print(f"[*] Weights not found at {self.weights_path}. Running initial 1D-CNN training...")
            from model.train import train_model
            train_model(weights_path=self.weights_path)

        if self.weights_path.exists():
            state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.weights_loaded = True
        else:
            self.weights_loaded = False

        self.model.eval()

        self.bridge = None
        if enable_phi_guard_bridge:
            from bridge.phi_guard_bridge import PhiGuardBridge
            self.bridge = PhiGuardBridge()

    def classify_chunk(
        self,
        class_override: Optional[str] = None,
        snr_db: Optional[float] = None,
        crowded: bool = True,
    ) -> Dict[str, Any]:
        """
        Pulls a single (2, 1024) I/Q chunk, runs 1D-CNN inference, computes Softmax
        confidence distribution, extracts PSD/RSSI metrics, and updates PHI-GUARD bridge.
        """
        t_start = time.perf_counter()
        iq_data, true_label, spectral_meta = self.generator.get_next_chunk_with_metadata(
            class_name=class_override, snr_db=snr_db, crowded=crowded
        )

        tensor_data = torch.tensor(iq_data, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor_data)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            prob_vector = probabilities.squeeze(0).cpu().tolist()

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        predicted_class = CLASSES[int(predicted_idx.item())]
        conf_score = float(confidence.item())

        is_hostile = predicted_class in HOSTILE_CLASSES
        self.frames_processed += 1
        if is_hostile:
            self.hostile_detections += 1
            self.threat_level = min(100, self.threat_level + 25)
        else:
            self.threat_level = max(0, self.threat_level - 10)

        if predicted_class == true_label:
            self.correct_predictions += 1

        interdiction_triggered = (
            predicted_class in ("Hostile_MAVLink", "Hostile_OcuSync", "Jammer") and conf_score > 0.80
        )

        bridge_event = None
        if self.bridge is not None:
            bridge_event = self.bridge.process_rf_classification(
                predicted_class=predicted_class,
                confidence=conf_score,
                spectral_meta=spectral_meta,
            )

        return {
            "frame_id": self.frames_processed,
            "predicted_class": predicted_class,
            "true_label": true_label,
            "confidence": conf_score,
            "probabilities": {CLASSES[i]: round(prob_vector[i], 4) for i in range(len(CLASSES))},
            "is_hostile": is_hostile,
            "interdiction_triggered": interdiction_triggered,
            "latency_ms": round(latency_ms, 3),
            "running_accuracy": round(self.correct_predictions / max(1, self.frames_processed), 4),
            "spectral": spectral_meta,
            "bridge_event": bridge_event,
        }

    def run(self, max_steps: Optional[int] = None, interval_sec: float = 1.0):
        """Runs continuous terminal RF classification loop with colorized tactical alerts."""
        print(
            f"{ANSI_CYAN}{ANSI_BOLD}[*] AI RF Classifier Online (Weights Loaded: {self.weights_loaded}). "
            f"Monitoring Battlefield Spectrum...{ANSI_RESET}"
        )
        step = 0
        try:
            while max_steps is None or step < max_steps:
                step += 1
                res = self.classify_chunk()
                predicted_class = res["predicted_class"]
                conf_score = res["confidence"]
                true_label = res["true_label"]
                rssi = res["spectral"]["rssi_dbm"]
                offset_khz = res["spectral"]["freq_offset_khz"]

                if res["is_hostile"]:
                    status = f"{ANSI_RED}{ANSI_BOLD}🔴 CRITICAL{ANSI_RESET}"
                    cls_str = f"{ANSI_RED}{ANSI_BOLD}{predicted_class:<16}{ANSI_RESET}"
                else:
                    status = f"{ANSI_GREEN}{ANSI_BOLD}🟢 SAFE{ANSI_RESET}"
                    cls_str = f"{ANSI_GREEN}{predicted_class:<16}{ANSI_RESET}"

                print(
                    f"[RF] Detected: {cls_str} | Confidence: {conf_score:6.2%} | "
                    f"RSSI: {rssi:6.1f} dBm ({offset_khz:+6.1f} kHz) | "
                    f"Status: {status} | (True: {true_label}) [{res['latency_ms']:.2f}ms]"
                )

                if predicted_class in ("Hostile_MAVLink", "Hostile_OcuSync") and conf_score > 0.80:
                    print(
                        f"{ANSI_RED}{ANSI_BOLD}[!!!] HOSTILE RF EMITTER DETECTED - "
                        f"INITIATING PHI-GUARD INTERDICTION !!!{ANSI_RESET}"
                    )
                elif predicted_class == "Jammer" and conf_score > 0.80:
                    print(
                        f"{ANSI_YELLOW}{ANSI_BOLD}[!!!] HIGH-POWER EW JAMMER DETECTED - "
                        f"ACTIVATING PHI-GUARD ECCM FREQUENCY AGILITY !!!{ANSI_RESET}"
                    )

                if max_steps is None or step < max_steps:
                    time.sleep(interval_sec)
        except KeyboardInterrupt:
            print(f"\n{ANSI_CYAN}[*] Shutting down RF Classifier.{ANSI_RESET}")


if __name__ == "__main__":
    classifier = LiveClassifier()
    classifier.run()
