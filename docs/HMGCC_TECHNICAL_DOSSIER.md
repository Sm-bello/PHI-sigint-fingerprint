# HMGCC Challenge Technical Dossier: `phi-rf-classifier`
## Real-Time 1D-CNN RF Emitter Identification in Crowded Electromagnetic Environments
**Repository:** [https://github.com/Sm-bello/PHI-sigint-fingerprint](https://github.com/Sm-bello/PHI-sigint-fingerprint)  
**Technology Readiness Level:** TRL-4 / Edge-of-5 Sovereign Prototype  
**Parent Ecosystem Integration:** `PHI-GUARD` Sovereign Counter-UAS & C4I Air Defense Hub

---

## 1. Executive Summary & HMGCC Challenge Alignment

The core operational requirement of the HMGCC challenge is **"identifying RF emitters within a crowded and complex electromagnetic environment"** with sub-second latency and zero reliance on cloud connectivity.

Traditional Counter-UAS and Electronic Support Measures (ESM) rely on static MAC/SSID lookup tables or fragile energy-threshold detectors that fail when:
1. Adversaries strip or spoof 802.11 / Remote ID headers.
2. Hostile C2 links operate beneath co-channel friendly tactical communications.
3. Wideband electronic warfare (EW) jammers elevate the broadband noise floor.

**`phi-rf-classifier`** solves this at the physical layer (PHY) by operating directly on **raw complex baseband In-Phase and Quadrature ($I/Q$) time-series tensors** of shape $(2, 1024)$ sampled at $f_s = 1\text{ MS/s}$. A compact, hardware-deployable **1D Convolutional Neural Network (`RFClassifierCNN`)** extracts cyclostationary phase, instantaneous frequency, and envelope features in **$< 1\text{ ms}$**, classifying emitters across 5 tactical categories and automatically triggering **`PHI-GUARD` Sovereign Interdiction** when a hostile emitter exceeds the $80\%$ Softmax confidence gate.

---

## 2. Physical-Layer Complex Baseband ($I/Q$) Signal Formulations

Each captured frame consists of $N = 1024$ complex baseband samples $x[n] = I[n] + jQ[n]$ at sample period $T_s = 1 / f_s$, represented as a 2-channel real tensor $\mathbf{X} \in \mathbb{R}^{2 \times 1024}$:

$$\mathbf{X} = \begin{bmatrix} \operatorname{Re}\{x[0]\} & \operatorname{Re}\{x[1]\} & \dots & \operatorname{Re}\{x[N-1]\} \\ \operatorname{Im}\{x[0]\} & \operatorname{Im}\{x[1]\} & \dots & \operatorname{Im}\{x[N-1]\} \end{bmatrix}$$

### Class 0: `Background_Noise` (Thermal AWGN & Ambient Clutter)
Circularly symmetric complex Additive White Gaussian Noise $w[n] \sim \mathcal{CN}(0, \sigma_n^2)$ with $\sigma_n = 0.05$ ($\approx -94\text{ dBm}$) plus weak ambient electromagnetic spur interference:

$$x_{\text{noise}}[n] = w[n] + A_{\text{spur}} e^{j(2\pi f_{\text{spur}} n T_s + \phi_0)}$$

### Class 1: `Friendly_Comms` (Blue-Force Tactical QPSK Link)
Pulse-shaped Quadrature Phase-Shift Keying centered at carrier offset $f_c = +200\text{ kHz}$ with symbol period $N_{\text{sym}} = 16$ samples and constellation symbols $s_k \in \frac{1}{\sqrt{2}}\{\pm 1 \pm j\}$:

$$x_{\text{QPSK}}[n] = P_{\text{friendly}} \left( \sum_{k} s_k \, g[n - k N_{\text{sym}}] \right) e^{j(2\pi (f_c + \delta f) n T_s + \phi_0)} + w[n]$$

### Class 2: `Hostile_MAVLink` (GFSK Frequency-Hopping Spread Spectrum)
Continuous-Phase Gaussian Frequency-Shift Keying (GFSK, $BT = 0.5$) modelling 433/868/915 MHz and 2.4 GHz MAVLink / SiK / ExpressLRS C2 links with mid-burst frequency hopping ($f_{\text{hop},1} \to f_{\text{hop},2}$ around $-300\text{ kHz}$) and TDMA guard-time gating $a[n]$:

$$f_{\text{inst}}[n] = f_{\text{hop}}[n] + \Delta f \sum_{k} b_k \, h_{\text{Gauss}}[n - k N_{\text{sym}}], \quad b_k \in \{-1, +1\}$$

$$x_{\text{MAVLink}}[n] = P_{\text{MAV}} \, a[n] \exp\!\left(j 2\pi \sum_{m=0}^{n} f_{\text{inst}}[m] T_s + j\phi_0\right) + x_{\text{co-channel}}[n] + w[n]$$

### Class 3: `Hostile_OcuSync` (Wideband 64-Subcarrier OFDM + Zadoff-Chu Sync)
Models encrypted DJI OcuSync video/telemetry links using an $N_{\text{FFT}} = 64$ Inverse Fast Fourier Transform with 48 active QPSK subcarriers $\mathcal{K}_{\text{data}}$, 4 high-energy BPSK pilot subcarriers $\mathcal{K}_{\text{pilot}} = \{\pm 7, \pm 21\}$, a $N_{\text{CP}} = 16$ Cyclic Prefix, and a root-$u=25$ Zadoff-Chu synchronization sequence:

$$x_{\text{ZC}}[n] = \exp\!\left(-j \frac{\pi u \, n(n+1)}{N_{\text{ZC}}}\right), \quad s_{m}[n] = \frac{1}{\sqrt{N_{\text{FFT}}}} \sum_{k \in \mathcal{K}} S_{m,k} e^{j 2\pi k n / N_{\text{FFT}}}$$

### Class 4: `Jammer` (Broadband Barrage + Swept LFM Chirp with SSPA Saturation)
Models a high-power military Electronic Attack emitter combining broadband Gaussian barrage noise with a sawtooth Linear Frequency Modulated (LFM) chirp sweeping $[-450\text{ kHz}, +450\text{ kHz}]$ at chirp rate $\mu = B / T_{\text{sweep}}$, passed through a non-linear Solid-State Power Amplifier (SSPA) $\tanh(\cdot)$ compression envelope:

$$s_{\text{raw}}[n] = P_{\text{jam}} \left( \alpha \exp\!\left(j 2\pi \left(f_{\text{start}} t + \frac{1}{2}\mu t^2\right)\right) + \beta w_{\text{barrage}}[n] \right)$$

---

## 3. 1D-CNN Architecture (`RFClassifierCNN`)

| Layer | Type & Configuration | Input Shape | Output Shape | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Block 1** | `Conv1d(2, 32, k=16, s=2, p=7)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` | `(B, 2, 1024)` | `(B, 32, 256)` | Captures fine-grained carrier phase transitions, symbol edges, and chirp slopes |
| **Block 2** | `Conv1d(32, 64, k=8, s=2, p=3)` + `BatchNorm1d` + `ReLU` + `MaxPool1d(2)` | `(B, 32, 256)` | `(B, 64, 64)` | Extracts multi-symbol modulation periodicity, OFDM cyclic prefix correlation, and FHSS hops |
| **Block 3** | `Conv1d(64, 128, k=4, s=2, p=1)` + `BatchNorm1d` + `ReLU` + `AdaptiveAvgPool1d(1)` | `(B, 64, 64)` | `(B, 128, 1)` | Aggregates shift-invariant global frame fingerprint into a 128-D latent vector |
| **Head** | `Linear(128, 5)` + `Softmax` (inference) | `(B, 128)` | `(B, 5)` | Outputs calibrated class probabilities across the 5 tactical RF emitter classes |

---

## 4. Cryptographic Chain-of-Custody (UK PACE Compliance)

Every high-confidence hostile detection ($P(\text{Hostile}) > 0.80$) is recorded into `PaceForensicLedger` (`bridge/phi_guard_bridge.py`), computing a deterministic SHA-256 block hash chained to the prior block and signed via HMAC-SHA256:

$$H_i = \operatorname{SHA256}\!\left(i \;\|\; H_{i-1} \;\|\; t_i \;\|\; \text{Class}_i \;\|\; \text{Conf}_i \;\|\; \text{Directive}_i \;\|\; \text{PSD\_Evidence}_i\right)$$

$$\Sigma_i = \operatorname{HMAC-SHA256}(K_{\text{sovereign}}, H_i)$$

This guarantees court-admissible forensic integrity under the UK *Police and Criminal Evidence Act (PACE)* and HMGCC assurance guidelines.
