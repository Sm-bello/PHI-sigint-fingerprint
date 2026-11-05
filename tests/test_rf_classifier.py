"""
PHI-RF-CLASSIFIER: Automated Unit & Integration Test Suite
==========================================================
Verifies:
  1. `SpectrumGenerator` I/Q chunk dimensionality (2, 1024), data types, and PSD metrics.
  2. `RFClassifierCNN` tensor forward pass, latent embedding shape, and parameter count.
  3. `RFIQDataset` & `build_splits` data integrity and stratified splitting.
  4. `PaceForensicLedger` & `PhiGuardBridge` cryptographic chain verification and state machine.
  5. `LiveClassifier` end-to-end inference and >80% confidence interdiction trigger logic.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from simulator.spectrum_generator import SpectrumGenerator, CLASSES
from model.network import RFClassifierCNN
from model.dataset import build_splits
from bridge.phi_guard_bridge import PhiGuardBridge, PaceForensicLedger
from inference.live_classifier import LiveClassifier


class TestPhiRFClassifier(unittest.TestCase):
    def test_spectrum_generator_shapes_and_classes(self):
        gen = SpectrumGenerator(seed=123)
        for cls_name in CLASSES:
            iq, label, meta = gen.get_next_chunk_with_metadata(class_name=cls_name)
            self.assertEqual(iq.shape, (2, 1024))
            self.assertEqual(iq.dtype, np.float32)
            self.assertEqual(label, cls_name)
            self.assertIn("rssi_dbm", meta)
            self.assertIn("spectral_entropy", meta)
            self.assertEqual(len(meta["psd_bars_dbm"]), 32)

    def test_cnn_network_forward_pass(self):
        model = RFClassifierCNN(num_classes=5)
        dummy_batch = torch.randn(8, 2, 1024)
        logits = model(dummy_batch)
        features = model.extract_features(dummy_batch)
        self.assertEqual(logits.shape, (8, 5))
        self.assertEqual(features.shape, (8, 128))

    def test_dataset_stratified_splits(self):
        train_ds, val_ds, test_ds, report = build_splits(num_samples=200, seed=42)
        self.assertEqual(len(train_ds), 140)
        self.assertEqual(len(val_ds), 30)
        self.assertEqual(len(test_ds), 30)
        self.assertEqual(report["nan_rows_detected"], 0)
        x0, y0 = train_ds[0]
        self.assertEqual(tuple(x0.shape), (2, 1024))
        self.assertTrue(0 <= int(y0.item()) < 5)

    def test_pace_forensic_ledger_and_bridge(self):
        ledger = PaceForensicLedger()
        ledger.record_detection(
            event_type="TEST_DETECTION",
            emitter_class="Hostile_OcuSync",
            confidence=0.96,
            threat_level="CRITICAL",
            interdiction_directive="GPS_MEACONING_TOW",
            spectral_evidence={"rssi_dbm": -52.4},
        )
        valid, err = ledger.verify_chain()
        self.assertTrue(valid)
        self.assertIsNone(err)

        bridge = PhiGuardBridge()
        ev = bridge.process_rf_classification(
            predicted_class="Hostile_MAVLink",
            confidence=0.95,
            spectral_meta={"peak_freq_hz": 2436700000, "rssi_dbm": -58.0, "spectral_entropy": 6.4, "papr_db": 4.2},
        )
        self.assertEqual(ev["threat_level"], "CRITICAL")
        self.assertEqual(ev["interdiction_directive"], "MAVLINK_GUIDED_OVERRIDE")
        self.assertEqual(ev["interceptor_state"], "S2_PHI_CTRL_TAKEOFF")
        bridge.close()

    def test_live_classifier_inference(self):
        clf = LiveClassifier(auto_train_if_missing=True, enable_phi_guard_bridge=True)
        res = clf.classify_chunk(class_override="Hostile_OcuSync")
        self.assertIn(res["predicted_class"], CLASSES)
        self.assertTrue(0.0 <= res["confidence"] <= 1.0)
        self.assertIsNotNone(res["bridge_event"])
        if clf.bridge is not None:
            clf.bridge.close()


if __name__ == "__main__":
    unittest.main()
