# Joint Operational Workflow: `phi-rf-classifier` $\times$ `phi-guard`

This workflow defines the non-invasive, zero-modification integration between **`phi-rf-classifier`** (the HMGCC 1D-CNN RF Emitter Classification & SIGINT Fingerprinting Engine in `c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-rf-classifier`) and **`phi-guard`** (the Sovereign Counter-UAS & C4I Air Defense Hub in `c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-guard`).

> [!IMPORTANT]
> **Strict Read-Only Boundary for `phi-guard`**: `phi-rf-classifier` communicates with `phi-guard` exclusively over standard network sockets (`UDP 127.0.0.1:14611`) and cryptographically compatible PACE SHA-256/HMAC structures. Zero files in `c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-guard` are modified.

---

## 1. End-to-End Cyber-Physical Architecture

```mermaid
flowchart LR
    subgraph SIGINT["PHI-RF-CLASSIFIER (HMGCC AI SIGINT Module)"]
        SG["SpectrumGenerator<br/>(2x1024 Complex I/Q @ 1 MS/s)"]
        CNN["RFClassifierCNN<br/>(3-Layer 1D-CNN + Softmax)"]
        PSD["Spectral Analyzer<br/>(PSD, RSSI, Entropy, PAPR)"]
        BR["PhiGuardBridge &<br/>PACE SHA-256 Ledger"]
        SG --> CNN
        SG --> PSD
        CNN --> BR
        PSD --> BR
    end

    subgraph PHIGUARD["PHI-GUARD C4I STACK (Read-Only Native Integration)"]
        UDP["UDP Port 14611<br/>(RF Sniffer Probe Ingress)"]
        S0S6["Sovereign Interceptor<br/>State Machine (S0 -> S6)"]
        QGC["QGroundControl 10Hz<br/>MAVLink Stream (14550/14570)"]
        DB["PostgreSQL + PACE<br/>Forensic Audit Chain"]
        UDP --> S0S6
        S0S6 --> QGC
        S0S6 --> DB
    end

    BR -- "Real-Time JSON RF Frame<br/>(127.0.0.1:14611)" --> UDP
```

---

## 2. Threat Class to PHI-GUARD Interdiction Mapping

When `RFClassifierCNN` classifies a `(2, 1024)` baseband I/Q frame with **Softmax Confidence $> 80\%$**, `PhiGuardBridge` (`bridge/phi_guard_bridge.py`) automatically maps the waveform class to the corresponding **PHI-GUARD Tiered Interdiction Directive** and advances the **Sovereign Interceptor State Machine**:

| 1D-CNN Emitter Class | Waveform Signature | Threat Level | PHI-GUARD Interdiction Directive | Sovereign Interceptor State Progression |
| :--- | :--- | :---: | :--- | :--- |
| **`Background_Noise`** | Thermal AWGN ($\sigma = 0.05$, $-94\text{ dBm}$) | `NOMINAL` | `S0_IDLE_MONITORING` | Remains in deep rooftop standby ($S_0$), saving **98.6%** operational battery energy |
| **`Friendly_Comms`** | Pulse-shaped QPSK ($+200\text{ kHz}$) | `LOW (WHITELISTED)` | `FRIENDLY_QPSK_PASSTHROUGH` | Whitelisted Blue-Force pass-through ($S_0$) |
| **`Hostile_MAVLink`** | GFSK FHSS Telemetry ($-300\text{ kHz}$) | `CRITICAL` | `MAVLINK_GUIDED_OVERRIDE` | $S_0 \to S_2\text{ (PHI-CTRL Takeoff)} \to S_5\text{ (Pursuit)} \to S_6\text{ (Protocol Hijack Escort)}$ |
| **`Hostile_OcuSync`** | 64-Subcarrier OFDM + ZC Sync ($+400\text{ kHz}$) | `CRITICAL` | `GPS_MEACONING_TOW` | $S_0 \to S_2\text{ (PHI-CTRL Takeoff)} \to S_5\text{ (Pursuit)} \to S_6\text{ (Standoff GPS Meaconing Tow)}$ |
| **`Jammer`** | Broadband Barrage + LFM Sweep Chirp | `CRITICAL` | `ECCM_NULL_STEERING_AGILITY` | Activates adaptive spatial null-steering & frequency hopping agility + Ghost Protocol EW deception |

---

## 3. Dual-Terminal Mission Launch Procedure

### Terminal 1: Launch `phi-rf-classifier` (AI SIGINT Sensor & Interdiction Trigger)

From `c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-rf-classifier`:

```powershell
# 1. Train or verify the 1D-CNN model weights (10,000 I/Q chunks, 10 epochs)
.\.venv\Scripts\python.exe main.py --mode train

# 2A. Launch Real-Time Stream Mode (Prints massive RED interdiction alerts on >80% Hostile detections)
.\.venv\Scripts\python.exe main.py --mode stream

# 2B. Or Launch Flicker-Free Rich C4I Cockpit HUD Mode
.\.venv\Scripts\python.exe main.py --mode hud
```

### Terminal 2 (Optional Joint Run): Launch `phi-guard` Sovereign C4I Stack

In a second terminal, launch the existing `phi-guard` stack without modifying any of its files:

```powershell
cd c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-guard\workspace-1
python run_native_stack.py
```

Or run the `phi-guard` Workspace-2 interactive radar & benchmark suite:

```powershell
$env:PYTHONPATH="c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-guard\workspace-2"
cd c:\Users\User\Desktop\COMPLETED_PROJECTS\phi-guard\workspace-2
python cli.py radar contraband
```

While both terminals are running:
1. `phi-rf-classifier` continuously synthesizes crowded battlefield I/Q chunks, classifies them in **$< 1\text{ ms}$** using `RFClassifierCNN`, signs a **PACE SHA-256 + HMAC-SHA256** forensic block, and dispatches the JSON RF probe packet to `127.0.0.1:14611`.
2. `phi-guard` receives the RF probe telemetry on port `14611`, correlates the hostile emitter (`Hostile_OcuSync` / `Hostile_MAVLink`), and scrambles the Sovereign Interceptor Drone (`Vehicle 1`) to intercept the Rogue Intruder (`Vehicle 2`).
