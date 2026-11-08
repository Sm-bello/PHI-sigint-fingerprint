"""
PHI-RF-CLASSIFIER: Automated Project Image & Publication Assets Generator
========================================================================
Uses Microsoft Edge / Chrome headless to render pixel-perfect, high-resolution
PNG assets for LinkedIn publishing, newsletter headers, logos, and GitHub documentation:
  1. `newsletter_logo_300x300.png`      : 300x300 px emblem matching LinkedIn modal requirement
  2. `linkedin_article_cover_1200x640.png`: 1200x640 px flagship cover banner for LinkedIn edition
  3. `c4i_tactical_dashboard_full.png`  : 1600x1000 px full tactical C4I cockpit screenshot
"""
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "docs" / "images"
TEMPLATES_DIR = ROOT_DIR / "docs" / "templates"
DASHBOARD_HTML = ROOT_DIR / "dashboard" / "index.html"
BRAIN_DIR = Path(r"C:\Users\User\.gemini\antigravity\brain\ca4b404b-0cce-4b31-889f-8f4d9eef34ec")

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

EDGE_PATHS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]


def find_browser_executable() -> Path:
    for p in EDGE_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome found on system.")


def capture_html(browser: Path, html_file: Path, output_png: Path, width: int, height: int):
    print(f"[*] Rendering {html_file.name} -> {output_png.name} ({width}x{height})...")
    cmd = [
        str(browser),
        "--headless",
        "--disable-gpu",
        f"--window-size={width},{height}",
        "--hide-scrollbars",
        f"--screenshot={str(output_png)}",
        str(html_file.resolve()),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

    # Post-process with PIL to ensure exact dimension crop if needed
    if output_png.exists():
        with Image.open(output_png) as img:
            if img.size != (width, height):
                cropped = img.crop((0, 0, width, height))
                cropped.save(output_png)
        print(f"  [+] Saved: {output_png} ({output_png.stat().st_size:,} bytes)")


def main():
    browser = find_browser_executable()
    print(f"[+] Using browser engine: {browser}")

    # 1. 300x300 Newsletter Logo
    logo_html = TEMPLATES_DIR / "logo_300x300.html"
    logo_out = IMAGES_DIR / "newsletter_logo_300x300.png"
    capture_html(browser, logo_html, logo_out, 300, 300)

    # 2. 1200x640 LinkedIn Article Cover
    cover_html = TEMPLATES_DIR / "cover_1200x640.html"
    cover_out = IMAGES_DIR / "linkedin_article_cover_1200x640.png"
    capture_html(browser, cover_html, cover_out, 1200, 640)

    # 3. 1600x1000 Full C4I Tactical Dashboard
    dash_out = IMAGES_DIR / "c4i_tactical_dashboard_full.png"
    capture_html(browser, DASHBOARD_HTML, dash_out, 1600, 1000)

    # Copy generated assets to Artifact Directory for embedded review
    if BRAIN_DIR.exists():
        import shutil
        for f in [logo_out, cover_out, dash_out]:
            if f.exists():
                dest = BRAIN_DIR / f.name
                shutil.copy2(f, dest)
                print(f"  [+] Copied to Brain: {dest.name}")

    print("\n[✓] All publication assets successfully generated!")


if __name__ == "__main__":
    main()
