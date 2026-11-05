"""
PHI-RF-CLASSIFIER: Crowded Battlefield RF Spectrum Simulator
============================================================
Synthesizes 2-channel complex I/Q baseband waveforms (shape: 2 x 1024) for 5 tactical RF classes:
  0. Background_Noise : Thermal AWGN + weak ambient electromagnetic clutter floor
  1. Friendly_Comms   : Low-power Tactical QPSK data link (+200 kHz offset, pulse-shaped)
  2. Hostile_MAVLink  : GFSK Frequency-Hopping Spread Spectrum (FHSS) C2 link (-300 kHz nominal)
  3. Hostile_OcuSync  : Wideband 64-subcarrier OFDM + Cyclic Prefix + Zadoff-Chu preamble (+400 kHz)
  4. Jammer           : High-power Broadband Barrage + Swept-Chirp Electronic Attack (LFM sweep)

Also computes Power Spectral Density (PSD), Spectral Shannon Entropy, Peak-to-Average Power Ratio (PAPR),
and RSSI (dBm) compatible with the PHI-GUARD SDRReceiver telemetry standard.
"""
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

CLASSES: List[str] = [
    "Background_Noise",
    "Friendly_Comms",
    "Hostile_MAVLink",
    "Hostile_OcuSync",
    "Jammer",
]
CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASSES)}


class SpectrumGenerator:
    """Simulates a crowded battlefield RF spectrum emitting complex I/Q data."""

    def __init__(
        self,
        sample_rate: float = 1e6,
        chunk_size: int = 1024,
        center_freq_hz: float = 2_437_000_000.0,
        seed: Optional[int] = None,
    ):
        self.sample_rate = float(sample_rate)
        self.chunk_size = int(chunk_size)
        self.center_freq_hz = float(center_freq_hz)
        self.rng = np.random.default_rng(seed)

        self.signals: List[Dict[str, Any]] = [
            {"name": "Background_Noise", "power": 0.1, "freq": 0.0, "protocol": "awgn_floor", "hostile": False},
            {"name": "Friendly_Comms", "power": 0.3, "freq": 2e5, "protocol": "tactical_qpsk", "hostile": False},
            {"name": "Hostile_MAVLink", "power": 0.6, "freq": -3e5, "protocol": "mavlink_gfsk_fhss", "hostile": True},
            {"name": "Hostile_OcuSync", "power": 0.8, "freq": 4e5, "protocol": "dji_ocusync_ofdm", "hostile": True},
            {"name": "Jammer", "power": 0.9, "freq": 0.0, "protocol": "ew_barrage_chirp", "hostile": True},
        ]

    def _complex_awgn(self, std: float) -> np.ndarray:
        """Generates circularly symmetric complex Gaussian noise CN(0, std^2)."""
        scale = std / np.sqrt(2.0)
        return self.rng.normal(0.0, scale, self.chunk_size) + 1j * self.rng.normal(0.0, scale, self.chunk_size)

    def _generate_qpsk(self, t: np.ndarray, power: float, freq: float) -> np.ndarray:
        """Synthesizes a pulse-shaped QPSK friendly tactical communication waveform."""
        sps = 16  # Samples per symbol
        num_symbols = int(np.ceil(self.chunk_size / sps)) + 2
        # QPSK constellation points: {+-1 +- 1j} / sqrt(2)
        bits_i = self.rng.choice([-1.0, 1.0], size=num_symbols)
        bits_q = self.rng.choice([-1.0, 1.0], size=num_symbols)
        symbols = (bits_i + 1j * bits_q) / np.sqrt(2.0)

        # Upsample and apply raised-cosine-like smoothing window
        baseband = np.repeat(symbols, sps)[: self.chunk_size]
        kernel = np.hanning(9)
        kernel /= np.sum(kernel)
        baseband_smooth = np.convolve(baseband.real, kernel, mode="same") + 1j * np.convolve(
            baseband.imag, kernel, mode="same"
        )

        # Random carrier phase & slight CFO jitter
        phase_offset = self.rng.uniform(0.0, 2.0 * np.pi)
        cfo = freq + self.rng.normal(0.0, 3e3)
        carrier = np.exp(1j * (2.0 * np.pi * cfo * t + phase_offset))
        return power * baseband_smooth * carrier

    def _generate_mavlink_gfsk(self, t: np.ndarray, power: float, freq: float) -> np.ndarray:
        """Synthesizes a frequency-hopping GFSK MAVLink / SiK / ELRS telemetry burst."""
        sps = 8  # Fast symbol rate for telemetry frame
        num_symbols = int(np.ceil(self.chunk_size / sps))
        symbols = self.rng.choice([-1.0, 1.0], size=num_symbols)
        raw_pam = np.repeat(symbols, sps)[: self.chunk_size]

        # Gaussian pulse-shaping filter (BT = 0.5 approximation)
        gauss_t = np.linspace(-2.0, 2.0, 9)
        gauss_kernel = np.exp(-2.0 * (gauss_t**2))
        gauss_kernel /= np.sum(gauss_kernel)
        filtered_pam = np.convolve(raw_pam, gauss_kernel, mode="same")

        # Continuous phase modulation (CPM) with mid-chunk FHSS frequency hop
        freq_dev = 45e3  # 45 kHz GFSK deviation
        hop_offset_1 = freq + self.rng.normal(0.0, 1.5e4)
        hop_offset_2 = freq + self.rng.choice([-8e4, 8e4]) + self.rng.normal(0.0, 1.5e4)
        hop_idx = self.rng.integers(self.chunk_size // 3, 2 * self.chunk_size // 3)

        inst_freq = np.empty(self.chunk_size, dtype=np.float64)
        inst_freq[:hop_idx] = hop_offset_1 + freq_dev * filtered_pam[:hop_idx]
        inst_freq[hop_idx:] = hop_offset_2 + freq_dev * filtered_pam[hop_idx:]

        phase = 2.0 * np.pi * np.cumsum(inst_freq) / self.sample_rate + self.rng.uniform(0.0, 2.0 * np.pi)
        # Burst envelope (MAVLink TDMA packet framing)
        envelope = np.ones(self.chunk_size, dtype=np.float64)
        guard_len = 64
        gate_start = self.rng.integers(0, self.chunk_size - guard_len)
        envelope[gate_start : gate_start + guard_len] = 0.08

        return power * envelope * np.exp(1j * phase)

    def _generate_ocusync_ofdm(self, t: np.ndarray, power: float, freq: float) -> np.ndarray:
        """
        Synthesizes a DJI OcuSync wideband OFDM frame:
        64-point IFFT with 48 active QPSK subcarriers + 4 BPSK pilots + 16-sample Cyclic Prefix,
        prefixed with a Zadoff-Chu synchronization sequence.
        """
        n_fft = 64
        cp_len = 16
        sym_len = n_fft + cp_len  # 80 samples per OFDM symbol
        num_ofdm_syms = self.chunk_size // sym_len + 1

        # 1. Zadoff-Chu root-25 synchronization sequence in the first symbol slot
        n_zc = np.arange(sym_len)
        zc_seq = np.exp(-1j * np.pi * 25.0 * n_zc * (n_zc + 1.0) / sym_len)

        ofdm_stream = [1.35 * zc_seq]
        active_bins = [k for k in range(-26, 27) if k not in (0, -21, -7, 7, 21)]
        pilot_bins = [-21, -7, 7, 21]

        for sym_idx in range(1, num_ofdm_syms):
            freq_grid = np.zeros(n_fft, dtype=np.complex128)
            # Load QPSK data onto active subcarriers
            qpsk_data = (
                self.rng.choice([-1.0, 1.0], size=len(active_bins))
                + 1j * self.rng.choice([-1.0, 1.0], size=len(active_bins))
            ) / np.sqrt(2.0)
            for b, val in zip(active_bins, qpsk_data):
                freq_grid[b % n_fft] = val
            # Load high-energy BPSK pilots
            pilot_sign = 1.0 if (sym_idx % 2 == 0) else -1.0
            for pb in pilot_bins:
                freq_grid[pb % n_fft] = 2.0 * pilot_sign

            time_sym = np.fft.ifft(freq_grid) * np.sqrt(n_fft)
            # Prepend Cyclic Prefix (CP)
            sym_with_cp = np.concatenate([time_sym[-cp_len:], time_sym])
            ofdm_stream.append(sym_with_cp)

        baseband = np.concatenate(ofdm_stream)[: self.chunk_size]
        # Normalize RMS to unit power before scaling by `power`
        rms = np.sqrt(np.mean(np.abs(baseband) ** 2)) + 1e-8
        baseband = baseband / rms

        cfo = freq + self.rng.normal(0.0, 5e3)
        phase_offset = self.rng.uniform(0.0, 2.0 * np.pi)
        return power * baseband * np.exp(1j * (2.0 * np.pi * cfo * t + phase_offset))

    def _generate_jammer(self, t: np.ndarray, power: float, freq: float) -> np.ndarray:
        """
        Synthesizes a high-power military EW Jammer combining broadband barrage noise
        and a swept Linear Frequency Modulated (LFM) sawtooth chirp with amplifier compression.
        """
        # Sawtooth LFM chirp sweeping [-450 kHz, +450 kHz] every 256 samples
        sweep_period = 256
        num_sweeps = int(np.ceil(self.chunk_size / sweep_period))
        t_local = np.arange(sweep_period) / self.sample_rate
        T_sweep = sweep_period / self.sample_rate
        f_start = -4.5e5 + self.rng.uniform(-2e4, 2e4)
        f_stop = 4.5e5 + self.rng.uniform(-2e4, 2e4)
        chirp_rate = (f_stop - f_start) / T_sweep

        single_chirp = np.exp(1j * 2.0 * np.pi * (f_start * t_local + 0.5 * chirp_rate * (t_local**2)))
        chirp_wave = np.tile(single_chirp, num_sweeps)[: self.chunk_size]

        # Broadband barrage component
        barrage = self._complex_awgn(std=0.85)
        raw_jammer = power * (0.75 * chirp_wave + 0.65 * barrage)

        # Non-linear solid-state power amplifier (SSPA) soft saturation signature
        mag = np.abs(raw_jammer) + 1e-8
        saturated_mag = np.tanh(1.6 * mag) * 1.15
        return raw_jammer * (saturated_mag / mag)

    def _generate_signal(
        self,
        signal_type: str,
        power: float,
        freq: float,
        snr_db: Optional[float] = None,
        crowded: bool = True,
    ) -> np.ndarray:
        """
        Generates a 1024-sample complex I/Q array for the specified emitter class,
        including battlefield AWGN and optional co-channel background interference.
        """
        t = np.arange(self.chunk_size, dtype=np.float64) / self.sample_rate

        # Base thermal noise floor
        noise_std = 0.05
        noise = self._complex_awgn(std=noise_std)

        if signal_type == "Background_Noise":
            # Pure thermal noise + faint distant atmospheric clutter spur
            spur_freq = self.rng.uniform(-4e5, 4e5)
            faint_spur = 0.015 * np.exp(1j * (2.0 * np.pi * spur_freq * t + self.rng.uniform(0, 2 * np.pi)))
            return noise + faint_spur

        # Scale signal power with realistic battlefield path-loss variation (0.40x to 1.60x)
        # so emitters overlap in received amplitude and require true modulation/waveform recognition
        eff_power = float(power) * self.rng.uniform(0.40, 1.60)

        if signal_type == "Friendly_Comms":
            clean_sig = self._generate_qpsk(t, eff_power, freq)
        elif signal_type == "Hostile_MAVLink":
            clean_sig = self._generate_mavlink_gfsk(t, eff_power, freq)
        elif signal_type == "Hostile_OcuSync":
            clean_sig = self._generate_ocusync_ofdm(t, eff_power, freq)
        elif signal_type == "Jammer":
            clean_sig = self._generate_jammer(t, eff_power, freq)
        else:
            return noise

        # In a crowded battlefield spectrum, low-level friendly background traffic coexists
        if crowded and signal_type in ("Hostile_MAVLink", "Hostile_OcuSync", "Jammer"):
            if self.rng.random() < 0.45:
                co_channel_qpsk = self._generate_qpsk(t, power=0.08, freq=2e5)
                clean_sig = clean_sig + co_channel_qpsk

        # Apply explicit SNR override if requested (used in Monte Carlo SNR sweeps)
        if snr_db is not None:
            sig_pwr = float(np.mean(np.abs(clean_sig) ** 2)) + 1e-12
            target_noise_pwr = sig_pwr / (10.0 ** (float(snr_db) / 10.0))
            noise = self._complex_awgn(std=np.sqrt(target_noise_pwr))

        return clean_sig + noise

    @staticmethod
    def compute_spectral_metrics(
        iq_2ch: np.ndarray,
        sample_rate: float = 1e6,
        center_freq_hz: float = 2_437_000_000.0,
        n_bins: int = 32,
    ) -> Dict[str, Any]:
        """
        Computes Power Spectral Density (PSD), peak carrier frequency, RSSI (dBm),
        Spectral Shannon Entropy, and PAPR (dB) aligned with PHI-GUARD SDRReceiver.
        """
        iq = iq_2ch[0].astype(np.float64) + 1j * iq_2ch[1].astype(np.float64)
        n = len(iq)
        window = np.hanning(n)
        fft_vals = np.fft.fftshift(np.fft.fft(iq * window, n=n))
        psd_linear = (np.abs(fft_vals) ** 2) / (n * sample_rate)
        psd_dbm = 10.0 * np.log10(np.maximum(1e-15, psd_linear)) + 30.0

        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate)) + center_freq_hz
        peak_idx = int(np.argmax(psd_dbm))
        peak_freq_hz = float(freqs[peak_idx])

        # Broadband average power -> RSSI estimate (calibrated to -95 dBm floor .. -42 dBm jammer)
        mean_pwr = float(np.mean(np.abs(iq) ** 2))
        peak_pwr = float(np.max(np.abs(iq) ** 2))
        rssi_dbm = round(float(-40.0 + 18.0 * np.log10(max(mean_pwr, 1e-6))), 1)
        papr_db = round(float(10.0 * np.log10(max(peak_pwr / max(mean_pwr, 1e-9), 1.0))), 2)

        # Normalized Spectral Shannon Entropy (0..8 bits scale)
        p_norm = psd_linear / (np.sum(psd_linear) + 1e-15)
        p_norm = np.maximum(p_norm, 1e-15)
        shannon_entropy = float(-np.sum(p_norm * np.log2(p_norm)) * (8.0 / np.log2(n)))

        # Downsample PSD to `n_bins` for compact HUD spectrum analyzer visualization
        bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
        psd_bars = [
            float(np.mean(psd_dbm[bin_edges[i] : bin_edges[i + 1]])) for i in range(n_bins)
        ]

        return {
            "peak_freq_hz": int(peak_freq_hz),
            "freq_offset_khz": round((peak_freq_hz - center_freq_hz) / 1e3, 1),
            "rssi_dbm": rssi_dbm,
            "papr_db": papr_db,
            "spectral_entropy": round(shannon_entropy, 2),
            "mean_power": round(mean_pwr, 4),
            "psd_bars_dbm": psd_bars,
        }

    def get_next_chunk(
        self,
        class_name: Optional[str] = None,
        snr_db: Optional[float] = None,
        crowded: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        Returns a 2-channel real NumPy array of shape (2, 1024) [I, Q]
        and the ground-truth emitter class name.
        """
        if class_name is not None:
            matching = [s for s in self.signals if s["name"] == class_name]
            active_signal = matching[0] if matching else random.choice(self.signals)
        else:
            active_signal = random.choice(self.signals)

        iq_data = self._generate_signal(
            active_signal["name"],
            active_signal["power"],
            active_signal["freq"],
            snr_db=snr_db,
            crowded=crowded,
        )

        real = np.real(iq_data).astype(np.float32)
        imag = np.imag(iq_data).astype(np.float32)
        return np.stack([real, imag], axis=0), active_signal["name"]

    def get_next_chunk_with_metadata(
        self,
        class_name: Optional[str] = None,
        snr_db: Optional[float] = None,
        crowded: bool = True,
    ) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        """Returns (iq_2ch, true_label, spectral_metadata)."""
        iq_2ch, label = self.get_next_chunk(class_name=class_name, snr_db=snr_db, crowded=crowded)
        meta = self.compute_spectral_metrics(
            iq_2ch, sample_rate=self.sample_rate, center_freq_hz=self.center_freq_hz
        )
        sig_info = next(s for s in self.signals if s["name"] == label)
        meta["protocol"] = sig_info["protocol"]
        meta["is_hostile"] = sig_info["hostile"]
        return iq_2ch, label, meta
