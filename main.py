#!/usr/bin/env python3
"""
========================================================================================
  PHI-RF-CLASSIFIER: HMGCC SOVEREIGN AI SIGINT & RF EMITTER CLASSIFICATION ENGINE
  Master Terminal Orchestrator & PHI-GUARD C4I Interdiction Trigger
========================================================================================
Modes:
  - Stream Mode (Default): `python main.py` or `python main.py --mode stream`
      Real-time terminal telemetry stream with massive RED interdiction alerts:
      `[!!!] HOSTILE RF EMITTER DETECTED - INITIATING PHI-GUARD INTERDICTION !!!`
  - Web C4I Dashboard    : `python main.py --mode web`
      Launches real-time browser-based military tactical cockpit with live I/Q oscillograms,
      polar constellation, 64-bin PSD waterfall, Softmax bars, and circular intercept radar.
  - Rich C4I HUD Mode    : `python main.py --mode hud`
      Flicker-free Rich Live Military SIGINT Cockpit with 32-bin PSD spectrum bar chart,
      1D-CNN Softmax probability bars, PHI-GUARD S0-S6 Interceptor state, and PACE ledger.
  - Train Mode           : `python main.py --mode train`
      Synthesizes 10,000 I/Q chunks, trains the 1D-CNN for 10 epochs, and saves `model/rf_model.pth`.
  - Benchmark Mode       : `python main.py --mode benchmark`
      Executes the Monte Carlo SNR sweep (-10 dB to +20 dB) and crowded spectrum stress test.
========================================================================================
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inference.live_classifier import (
    LiveClassifier,
    CLASSES,
    ANSI_RED,
    ANSI_GREEN,
    ANSI_YELLOW,
    ANSI_CYAN,
    ANSI_BOLD,
    ANSI_RESET,
)


def _render_psd_sparkline(psd_bars_dbm: List[float], width: int = 32) -> str:
    """Renders a compact Unicode spectral density sparkline across the sampled RF band.

    Args:
        psd_bars_dbm (List[float]): Downsampled Power Spectral Density values in dBm.
        width (int, optional): Number of character bins to render. Defaults to 32.

    Returns:
        str: Unicode block sparkline string representing relative spectral power across the band.
    """
    blocks = " ▂▃▄▅▆▇█"
    lo, hi = -95.0, -35.0
    chars = []
    for val in psd_bars_dbm[:width]:
        norm = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        idx = int(round(norm * (len(blocks) - 1)))
        chars.append(blocks[idx])
    return "".join(chars)


def run_stream_orchestrator(max_steps: Optional[int] = None, interval_sec: float = 0.85) -> None:
    """Runs the master real-time RF classifier stream and triggers PHI-GUARD interdiction.

    Continuously pulls `(2, 1024)` complex baseband I/Q frames, evaluates them with the
    trained 1D-CNN (`RFClassifierCNN`), prints live PSD sparklines and classification telemetry,
    and emits a high-visibility RED interdiction banner whenever a `Hostile_MAVLink` or
    `Hostile_OcuSync` emitter is identified with `>80%` Softmax confidence.

    Args:
        max_steps (Optional[int], optional): Maximum number of frames to process before
            returning. If `None`, runs indefinitely until `KeyboardInterrupt`. Defaults to None.
        interval_sec (float, optional): Delay in seconds between consecutive frames.
            Defaults to 0.85.
    """
    classifier = LiveClassifier(auto_train_if_missing=True, enable_phi_guard_bridge=True)

    print("=" * 96)
    print(
        f"{ANSI_CYAN}{ANSI_BOLD}  [#] PHI-RF-CLASSIFIER | HMGCC AI SIGINT & COUNTER-UAS SPECTRUM MONITOR [#]{ANSI_RESET}"
    )
    print(
        "  1D-CNN Architecture: 3x Conv1d + BatchNorm + AdaptiveAvgPool1d | Input: (batch, 2, 1024) I/Q"
    )
    print(
        "  PHI-GUARD Bridge   : ACTIVE (UDP Port 14611 + PACE SHA-256/HMAC Forensic Chain-of-Custody)"
    )
    print("=" * 96)

    step = 0
    try:
        while max_steps is None or step < max_steps:
            step += 1
            res = classifier.classify_chunk()
            pred = res["predicted_class"]
            conf = res["confidence"]
            true_lbl = res["true_label"]
            spec = res["spectral"]
            bridge_ev = res["bridge_event"]
            spark = _render_psd_sparkline(spec["psd_bars_dbm"], width=24)

            if res["is_hostile"]:
                status_badge = f"{ANSI_RED}{ANSI_BOLD}🔴 CRITICAL{ANSI_RESET}"
                pred_colored = f"{ANSI_RED}{ANSI_BOLD}{pred:<16}{ANSI_RESET}"
            else:
                status_badge = f"{ANSI_GREEN}{ANSI_BOLD}🟢 SAFE    {ANSI_RESET}"
                pred_colored = f"{ANSI_GREEN}{pred:<16}{ANSI_RESET}"

            print(
                f"[RF #{step:03d}] Detected: {pred_colored} | Confidence: {conf:6.2%} | "
                f"Status: {status_badge} | PSD:[{spark}] {spec['rssi_dbm']:5.1f}dBm | "
                f"(True: {true_lbl:<16}) [{res['latency_ms']:.2f}ms]"
            )

            # Task 6 Mandatory High-Confidence Interdiction Alert
            if pred in ("Hostile_MAVLink", "Hostile_OcuSync") and conf > 0.80:
                print(
                    f"\n{ANSI_RED}{ANSI_BOLD}"
                    f"╔══════════════════════════════════════════════════════════════════════════════════════════════╗\n"
                    f"║  [!!!] HOSTILE RF EMITTER DETECTED - INITIATING PHI-GUARD INTERDICTION !!!                   ║\n"
                    f"║  ► Threat Class : {pred:<20} | 1D-CNN Confidence : {conf:6.2%}                          ║\n"
                    f"║  ► Directive    : {bridge_ev['interdiction_directive']:<24} | Interceptor State : {bridge_ev['interceptor_state']:<18}  ║\n"
                    f"║  ► PACE Block   : #{bridge_ev['pace_block_height']:04d}                    | SHA-256 Hash      : {bridge_ev['pace_latest_hash'][:24]}... ║\n"
                    f"╚══════════════════════════════════════════════════════════════════════════════════════════════╝"
                    f"{ANSI_RESET}\n"
                )
            elif pred == "Jammer" and conf > 0.80:
                print(
                    f"\n{ANSI_YELLOW}{ANSI_BOLD}"
                    f"  [!!!] HOSTILE RF EMITTER DETECTED - INITIATING PHI-GUARD INTERDICTION !!! "
                    f"(EW JAMMER -> {bridge_ev['interdiction_directive']} | PACE #{bridge_ev['pace_block_height']:04d})"
                    f"{ANSI_RESET}\n"
                )

            if max_steps is None or step < max_steps:
                time.sleep(interval_sec)
    except KeyboardInterrupt:
        print(f"\n{ANSI_CYAN}[+] PHI-RF-CLASSIFIER safely powered down. PACE Forensic Chain verified.{ANSI_RESET}")


def _build_hud_renderable(last_res: Dict[str, Any], classifier: LiveClassifier):
    """Constructs a flicker-free Rich C4I SIGINT HUD matching the PHI-GUARD cockpit aesthetic.

    Args:
        last_res (Dict[str, Any]): Dictionary containing the latest frame classification,
            Softmax probabilities, spectral metrics, and `PhiGuardBridge` event telemetry.
        classifier (LiveClassifier): Active `LiveClassifier` instance tracking cumulative statistics.

    Returns:
        rich.console.Group: Renderable Rich layout group containing header, telemetry table, and alert ticker.
    """
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box

    pred = last_res["predicted_class"]
    conf = last_res["confidence"]
    spec = last_res["spectral"]
    bridge_ev = last_res["bridge_event"]

    header = Panel(
        Text(
            "[#] SOVEREIGN AI SIGINT HUB: PHI-RF-CLASSIFIER x PHI-GUARD C4I [#]\n"
            "Real-Time 1D-CNN Baseband I/Q Classifier | Crowded Battlefield Spectrum Analyzer\n"
            "Automated Counter-UAS Interdiction Trigger | UK PACE Cryptographic Chain-of-Custody",
            justify="center",
            style="bold yellow",
        ),
        box=box.ROUNDED,
        border_style="blue",
    )

    # 1. Live Spectrum & Classification Table
    table = Table(box=box.ROUNDED, expand=True, show_header=True, header_style="bold cyan")
    table.add_column("SIGINT SUBSYSTEM", style="bold cyan", width=26)
    table.add_column("LIVE ELECTROMAGNETIC & AI TELEMETRY", style="bold white")
    table.add_column("STATUS / VERDICT", justify="center", width=24)

    spark = _render_psd_sparkline(spec["psd_bars_dbm"], width=32)
    table.add_row(
        "[ RF SPECTRUM (2.437 GHz) ]",
        f"PSD: [{spark}] | Peak: {spec['freq_offset_khz']:+6.1f} kHz | RSSI: {spec['rssi_dbm']:.1f} dBm | Entropy: {spec['spectral_entropy']:.2f} bits",
        "[bold green]SDR LOCKED (1 MS/s)[/]",
    )

    cls_style = "bold white on red" if last_res["is_hostile"] else "bold green"
    table.add_row(
        "[ 1D-CNN CLASSIFICATION ]",
        f"Emitter: {pred} (True: {last_res['true_label']}) | Softmax Confidence: {conf:.2%} | Latency: {last_res['latency_ms']:.2f} ms",
        f"[{cls_style}]{'CRITICAL THREAT' if last_res['is_hostile'] else 'SAFE / NOMINAL'}[/]",
    )

    # Probability distribution bar breakdown
    prob_parts = []
    for c in CLASSES:
        p = last_res["probabilities"][c]
        short_name = c.replace("Background_", "").replace("Friendly_", "").replace("Hostile_", "")
        prob_parts.append(f"{short_name}: {p:.0%}")
    table.add_row(
        "[ SOFTMAX DISTRIBUTION ]",
        " | ".join(prob_parts),
        f"[bold cyan]ACC: {last_res['running_accuracy']:.1%}[/]",
    )

    ic_style = "bold red on white" if "SCRAMBLE" in bridge_ev["interceptor_state"] or "TOW" in bridge_ev["interceptor_state"] else "bold blue"
    table.add_row(
        "[ PHI-GUARD INTERCEPTOR ]",
        f"State: {bridge_ev['interceptor_state']} | Directive: {bridge_ev['interdiction_directive']} | Target: {bridge_ev['target_id']}",
        f"[{ic_style}]{bridge_ev['interceptor_state']}[/]",
    )

    table.add_row(
        "[ PACE FORENSIC LEDGER ]",
        f"Block Height: #{bridge_ev['pace_block_height']} | SHA-256: {bridge_ev['pace_latest_hash'][:22]}... | UDP #14611 Sent: {bridge_ev['udp_packets_dispatched']}",
        "[bold green]HMAC VERIFIED[/]",
    )

    # 2. Tactical Alert Banner
    if pred in ("Hostile_MAVLink", "Hostile_OcuSync", "Jammer") and conf > 0.80:
        alert_text = (
            "[!!!] HOSTILE RF EMITTER DETECTED - INITIATING PHI-GUARD INTERDICTION !!!\n"
            f"Emitter: {pred} ({conf:.2%}) -> Executing {bridge_ev['interdiction_directive']} "
            f"[{bridge_ev['interceptor_state']}]"
        )
        alert_style = "bold white on red"
        border_col = "red"
    else:
        alert_text = (
            f"SPECTRUM MONITOR NOMINAL: Classified {pred} ({conf:.2%}). "
            "Sovereign Interceptor in standby (98.6% energy conserved)."
        )
        alert_style = "bold green on black"
        border_col = "green"

    ticker = Panel(
        Text(alert_text, justify="center", style=alert_style),
        box=box.ROUNDED,
        border_style=border_col,
    )
    footer = Text(
        f"Frames Processed: {classifier.frames_processed} | Hostile Interdictions: {bridge_ev['interdictions_initiated']} | Press Ctrl+C to exit",
        justify="center",
        style="dim",
    )
    return Group(header, table, ticker, footer)


def run_hud_orchestrator(max_steps: Optional[int] = None, interval_sec: float = 0.65) -> None:
    """Runs the flicker-free Rich Live C4I SIGINT HUD in the terminal.

    Args:
        max_steps (Optional[int], optional): Number of frames to render before exiting.
            Defaults to None (continuous loop).
        interval_sec (float, optional): Refresh interval in seconds between simulated RF frames.
            Defaults to 0.65.
    """
    from rich.console import Console
    from rich.live import Live

    console = Console(safe_box=True, force_terminal=True)
    classifier = LiveClassifier(auto_train_if_missing=True, enable_phi_guard_bridge=True)
    first_res = classifier.classify_chunk()

    step = 1
    try:
        with Live(_build_hud_renderable(first_res, classifier), refresh_per_second=4, console=console) as live:
            while max_steps is None or step < max_steps:
                time.sleep(interval_sec)
                step += 1
                res = classifier.classify_chunk()
                live.update(_build_hud_renderable(res, classifier))
    except KeyboardInterrupt:
        console.print("\n[bold green][+] PHI-RF-CLASSIFIER C4I HUD safely powered down.[/bold green]")


def main() -> None:
    """Parses command-line arguments and dispatches the selected execution mode."""
    parser = argparse.ArgumentParser(
        description="PHI-RF-CLASSIFIER: HMGCC 1D-CNN Battlefield RF Classifier & PHI-GUARD Bridge"
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "hud", "web", "train", "benchmark"],
        default="stream",
        help="Execution mode: 'stream' (default CLI), 'hud' (Rich C4I Cockpit), 'web' (Browser C4I Dashboard), 'train', or 'benchmark'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional number of classification frames to run before exiting (default: continuous)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.85,
        help="Update interval in seconds between I/Q frames (default: 0.85s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port for '--mode web' dashboard (default: 8080)",
    )
    args = parser.parse_args()

    if args.mode == "web":
        from dashboard.app import start_server
        start_server(port=args.port)
    elif args.mode == "train":
        from model.train import train_model
        train_model()
    elif args.mode == "benchmark":
        from benchmarks.run_benchmarks import run_full_benchmark_suite
        run_full_benchmark_suite()
    elif args.mode == "hud":
        run_hud_orchestrator(max_steps=args.steps, interval_sec=args.interval)
    else:
        run_stream_orchestrator(max_steps=args.steps, interval_sec=args.interval)


if __name__ == "__main__":
    main()
