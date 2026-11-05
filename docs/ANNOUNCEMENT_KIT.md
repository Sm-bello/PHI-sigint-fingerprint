# Public Visibility & Launch Announcement Kit
**Project:** `PHI-sigint-fingerprint` (`phi-rf-classifier`)  
**Repository URL:** https://github.com/Sm-bello/PHI-sigint-fingerprint

Use the ready-to-publish templates below across **LinkedIn**, **X (Twitter)**, **GitHub Releases**, and your **HMGCC / Defence Innovation Pitch** to maximize technical visibility.

---

## 1. LinkedIn Flagship Launch Post

```text
Do not wait for permission to build the future of Sovereign Electronic Warfare & Counter-UAS. 🛡️📡

Today I am open-sourcing PHI-SIGINT-FINGERPRINT (phi-rf-classifier) — a real-time Deep Learning RF Emitter Identification & SIGINT Classification Engine built to tackle a core UK Defence / HMGCC challenge:

"Identifying hostile RF emitters within a crowded and complex electromagnetic environment."

Traditional drone detectors rely on Wi-Fi SSIDs or static MAC tables that fail the moment an adversary strips protocol headers, hops frequencies, or floods the spectrum with wideband jamming.

Instead of reading packet headers, PHI-RF-CLASSIFIER operates at the Physical Layer (PHY) on raw complex In-Phase & Quadrature (I/Q) baseband waveforms:

🔹 Crowded Battlefield I/Q Synthesis (2 x 1024 @ 1 MS/s):
Simulates co-channel electromagnetic environments across 5 tactical classes:
  • Background_Noise (Thermal AWGN + ambient clutter)
  • Friendly_Comms (Blue-force pulse-shaped QPSK)
  • Hostile_MAVLink (Continuous-phase GFSK with FHSS frequency hopping)
  • Hostile_OcuSync (Wideband 64-subcarrier OFDM + Cyclic Prefix + Zadoff-Chu sync)
  • Jammer (High-power broadband barrage + swept LFM sawtooth chirp)

🔹 Sub-Millisecond 1D-CNN Inference Engine:
A 3-stage 1D Convolutional Neural Network (PyTorch) that learns cyclostationary phase, modulation envelope, and spectral signatures directly from raw I/Q tensors — outperforming classical spectral baselines and achieving ~99%+ held-out test accuracy with sub-millisecond single-frame latency.

🔹 Autonomous PHI-GUARD Counter-UAS Interdiction Bridge:
When a Hostile_MAVLink, Hostile_OcuSync, or EW Jammer emitter is detected with >80% Softmax confidence, the classifier automatically:
  1. Triggers the PHI-GUARD Sovereign Interceptor State Machine (S0 Standby -> S2 PHI-CTRL Obstacle Takeoff -> S5 High-G Pursuit -> S6 Standoff Containment).
  2. Selects the optimal countermeasure (MAVLink Guided Override, GPS Meaconing Tow, or ECCM Null-Steering Agility).
  3. Seals the spectral evidence (PSD peak frequency, RSSI, PAPR, Shannon entropy) into a tamper-proof SHA-256 + HMAC-SHA256 cryptographic chain-of-custody ledger compliant with the UK Police and Criminal Evidence Act (PACE).

Explore the full code, 1D-CNN training pipeline, Monte Carlo SNR benchmarks (-10 dB to +20 dB), and C4I Terminal HUD here:
🔗 GitHub: https://github.com/Sm-bello/PHI-sigint-fingerprint

🌍 Institutional Acknowledgement:
Gratefully developed with the visionary leadership and technical sponsorship of DIMASA (Digital Intelligence Maintenance for Aircraft Systems in Africa) and the Physics Hybrid Integrity Lab.

#DefenceTech #SIGINT #ElectronicWarfare #CounterUAS #PyTorch #DeepLearning #SDR #HMGCC #Aerospace #CyberPhysicalSystems #UKDefence #DIMASA
```

---

## 2. X / Twitter Technical Launch Thread

**Tweet 1/5:**
```text
1/ We just built and shipped PHI-SIGINT-FINGERPRINT (`phi-rf-classifier`) — a real-time 1D-CNN Deep Learning engine that identifies hostile drone C2 links & EW jammers directly from raw complex I/Q waveforms in a crowded battlefield spectrum. 📡🛡️

Repo: https://github.com/Sm-bello/PHI-sigint-fingerprint
```

**Tweet 2/5:**
```text
2/ Why raw I/Q?
In contested electromagnetic environments, rogue drones strip SSIDs, encrypt payloads, or hide beneath friendly comms.
By feeding (2, 1024) In-Phase/Quadrature tensors into a 3-layer 1D-CNN, we classify waveforms by their physical modulation physics in <1 ms.
```

**Tweet 3/5:**
```text
3/ The simulator & classifier distinguish 5 tactical RF regimes in real time:
🟢 Background_Noise (Thermal AWGN)
🟢 Friendly_Comms (Pulse-shaped Tactical QPSK)
🔴 Hostile_MAVLink (GFSK Frequency-Hopping Spread Spectrum)
🔴 Hostile_OcuSync (64-Subcarrier Wideband OFDM + Zadoff-Chu sync)
🔴 Jammer (Barrage + Swept LFM Chirp)
```

**Tweet 4/5:**
```text
4/ Connected directly to our PHI-GUARD Sovereign C4I stack:
When Hostile_MAVLink or Hostile_OcuSync crosses >80% Softmax confidence, `main.py` fires an immediate interdiction directive over UDP #14611 and signs a UK PACE-compliant SHA-256 + HMAC forensic audit block.
```

**Tweet 5/5:**
```text
5/ Full PyTorch training script (10,000 I/Q chunks), Monte Carlo SNR robustness sweep (-10 dB to +20 dB), classical ML baseline comparisons, and a flicker-free Rich C4I Terminal HUD are live now:

https://github.com/Sm-bello/PHI-sigint-fingerprint
```

---

## 3. GitHub Release Notes (`v1.0.0 - TRL-4 HMGCC Sovereign Prototype`)

- **Title:** `v1.0.0: PHI-RF-CLASSIFIER — Real-Time 1D-CNN Battlefield RF Emitter Classifier & PHI-GUARD Bridge`
- **Highlights:**
  - `simulator/spectrum_generator.py`: Multi-emitter complex baseband $I/Q$ synthesis (`2 x 1024`) with Power Spectral Density (PSD), RSSI (dBm), PAPR, and Spectral Shannon Entropy extraction.
  - `model/network.py` & `model/train.py`: 3-layer `RFClassifierCNN` architecture, stratified `Train/Val/Test` dataset pipeline (`10,000` chunks), classical baseline verification, and trained weights (`model/rf_model.pth`).
  - `inference/live_classifier.py` & `main.py`: Sub-millisecond real-time Softmax classification engine with mandatory `>80%` confidence `[!!!] HOSTILE RF EMITTER DETECTED - INITIATING PHI-GUARD INTERDICTION !!!` alert and Rich Live C4I Cockpit HUD (`--mode hud`).
  - `bridge/phi_guard_bridge.py`: Zero-modification integration with `phi-guard` over UDP Port `14611` + UK PACE SHA-256/HMAC cryptographic chain-of-custody ledger.
  - `benchmarks/run_benchmarks.py`: Full Monte Carlo SNR sweep (`-10 dB` to `+20 dB`) and operational Counter-UAS KPI verification ($P_d$ and $FAR$).
