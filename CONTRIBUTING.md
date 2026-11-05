# Contributing to `PHI-sigint-fingerprint` (`phi-rf-classifier`)

We welcome contributions, technical partnerships, and Over-The-Air (OTA) Software-Defined Radio (SDR) validation datasets from defence researchers, aerospace engineers, and academic collaborators.

This repository is developed under the auspices of **DIMASA (Digital Intelligence Maintenance for Aircraft Systems in Africa)** and the **Physics Hybrid Integrity Lab**.

---

## 1. How to Contribute

1. **Open an Issue**: Before submitting major architectural changes, open a GitHub Issue describing the proposed enhancement, new emitter modulation class, or SDR hardware driver integration.
2. **Fork & Branch**: Create a feature branch from `main` (e.g., `feat/usrp-b210-driver` or `feat/elrs-lora-modulation`).
3. **Follow Google-Style Docstrings**: Every new module, class, and public function must include Google-style Python docstrings specifying `Args`, `Returns`, `Raises`, and `Attributes` for audit readiness.
4. **Verify Tests & Benchmarks**: Ensure all unit tests and SNR benchmark campaigns pass locally before opening a Pull Request:
   ```powershell
   python -m unittest discover -s tests -p "test_*.py" -v
   python benchmarks/run_benchmarks.py
   ```
5. **Submit a Pull Request**: Include empirical metrics (held-out accuracy, Macro F1, and single-frame latency) in your PR description.

---

## 2. Priority Collaboration Areas (Phase 2 Roadmap)

- **Physical SDR Hardware Drivers**: Native `SoapySDR`, `pyrtlsdr` (RTL-SDR Blog V4), and `uhd` (Ettus USRP B210 / BladeRF 2.0 micro) ring-buffer stream adapters.
- **Over-The-Air (OTA) RF Captures**: Anonymized `.sigmf` or `.npy` $I/Q$ captures of commercial COTS UAV links (DJI OcuSync 3/4, ExpressLRS 868/915 MHz & 2.4 GHz, Crossfire, SiK MAVLink).
- **Edge AI Quantization**: ONNX Runtime and NVIDIA TensorRT FP16/INT8 export pipelines for NVIDIA Jetson Orin Nano deployment.
- **Cryptographic Assurance**: Post-quantum signature extensions for the UK PACE `PaceForensicLedger`.

---

## 3. Responsible Disclosure & Operational Security

Do not commit classified frequencies, restricted government cryptographic keys, or proprietary third-party firmware binaries to this repository. All training and benchmark waveforms in the public repository must remain mathematically synthesized or collected from authorized, unlicensed ISM/laboratory testbeds.
