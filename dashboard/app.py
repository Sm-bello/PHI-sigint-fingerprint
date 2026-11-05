"""
PHI-RF-CLASSIFIER: Real-Time Web C4I Dashboard Server
=====================================================
Hosts a zero-dependency, local HTTP server streaming live PyTorch 1D-CNN classifications,
Power Spectral Density (PSD), baseband I/Q waveforms, and PHI-GUARD interdiction states
to a web-based military tactical cockpit (`dashboard/index.html`).
"""
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, Any, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from inference.live_classifier import LiveClassifier, CLASSES


class DashboardState:
    """Thread-safe state container shared between the background inference loop and HTTP handler."""

    def __init__(self):
        self.lock = threading.Lock()
        self.emitter_override: Optional[str] = None
        self.snr_db: Optional[float] = 12.0
        self.latest_telemetry: Dict[str, Any] = {}
        self.is_running = True


GLOBAL_STATE = DashboardState()
HTML_PATH = ROOT_DIR / "dashboard" / "index.html"


class DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    """Custom HTTP request handler serving the dashboard UI and live JSON telemetry API."""

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP access logs for clean console
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            if not HTML_PATH.exists():
                self.send_error(404, "Dashboard HTML not found")
                return
            content = HTML_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        elif self.path == "/api/telemetry":
            with GLOBAL_STATE.lock:
                telemetry = dict(GLOBAL_STATE.latest_telemetry)

            payload = json.dumps(telemetry).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        elif self.path == "/api/status":
            status = {"status": "ONLINE", "model": "RFClassifierCNN", "classes": CLASSES}
            payload = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self.path == "/api/control":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                with GLOBAL_STATE.lock:
                    if "emitter_override" in data:
                        GLOBAL_STATE.emitter_override = data["emitter_override"]
                    if "snr_db" in data and data["snr_db"] is not None:
                        GLOBAL_STATE.snr_db = float(data["snr_db"])

                resp = {"status": "SUCCESS", "override": GLOBAL_STATE.emitter_override, "snr": GLOBAL_STATE.snr_db}
                payload = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as e:
                self.send_error(400, f"Invalid JSON payload: {e}")
        else:
            self.send_error(404, "Endpoint not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def _background_inference_worker(classifier: LiveClassifier):
    """Background worker continuously invoking 1D-CNN inference and updating telemetry state."""
    while GLOBAL_STATE.is_running:
        with GLOBAL_STATE.lock:
            override = GLOBAL_STATE.emitter_override
            snr = GLOBAL_STATE.snr_db

        res = classifier.classify_chunk(class_override=override, snr_db=snr, crowded=True)

        with GLOBAL_STATE.lock:
            GLOBAL_STATE.latest_telemetry = res

        time.sleep(0.08)  # ~12.5 Hz refresh rate for smooth web streaming


def start_server(host: str = "127.0.0.1", port: int = 8080, open_browser: bool = True):
    """Starts the real-time C4I Web Dashboard HTTP server and launches the browser."""
    print("=" * 84)
    print("  PHI-RF-CLASSIFIER: SOVEREIGN 1D-CNN AI SIGINT WEB C4I DASHBOARD")
    print("=" * 84)
    print(f"[*] Initializing PyTorch 1D-CNN LiveClassifier...")
    classifier = LiveClassifier(auto_train_if_missing=True, enable_phi_guard_bridge=True)
    print(f"[+] 1D-CNN Engine Ready (Device: {classifier.device})")

    # Start background inference thread
    infer_thread = threading.Thread(target=_background_inference_worker, args=(classifier,), daemon=True)
    infer_thread.start()

    # Create multi-threaded HTTP server allowing address reuse
    socketserver.TCPServer.allow_reuse_address = True
    server_address = (host, port)

    try:
        httpd = socketserver.ThreadingTCPServer(server_address, DashboardRequestHandler)
    except OSError:
        # Fallback to alternate port if 8080 is occupied
        port = 8088
        server_address = (host, port)
        httpd = socketserver.ThreadingTCPServer(server_address, DashboardRequestHandler)

    url = f"http://{host}:{port}/"
    print(f"[+] Real-Time C4I Dashboard Online: {url}")
    print(f"[*] Serving live baseband I/Q oscillograms, PSD spectrum, 1D-CNN Softmax & radar...")
    print(f"[*] Press Ctrl+C in this terminal to safely stop the server.")
    print("=" * 84)

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down C4I Web Dashboard server...")
        GLOBAL_STATE.is_running = False
        httpd.shutdown()
        httpd.server_close()
        print("[+] Dashboard server successfully terminated.")


if __name__ == "__main__":
    start_server()
