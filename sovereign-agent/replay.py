"""
Bundle a run's traces into a self-contained replay page and open it.

    python replay.py                                # latest run in traces/
    python replay.py traces/run-20260922-151146     # a specific run

Writes <run folder>/replay.html (replay.html with the traces built in).
"""

import json
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).parent
TEMPLATE = HERE / "replay.html"
PLACEHOLDER = "/*TRACE_DATA*/null"


def latest_run() -> Path:
    runs = sorted(p for p in (HERE / "traces").glob("run-*") if p.is_dir())
    if not runs:
        sys.exit("No run folders in traces/ yet — run orchestrator.py first.")
    return runs[-1]


def open_in_browser(page: Path) -> None:
    """Open the page; on WSL hand it to the Windows browser."""
    if "microsoft" in platform.release().lower():
        win_path = subprocess.run(["wslpath", "-w", str(page)], capture_output=True, text=True).stdout.strip()
        subprocess.run(["explorer.exe", win_path])  # exits non-zero even on success
    else:
        webbrowser.open(page.as_uri())


def main() -> None:
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    files = sorted(run.glob("*.jsonl"))
    if not files:
        sys.exit(f"No .jsonl trace files in {run}")

    data = {"name": run.name,
            "files": [{"name": f.name, "text": f.read_text(encoding="utf-8")} for f in files]}
    blob = json.dumps(data).replace("</", "<\\/")  # keep "</script>" in traces from ending the tag
    page = run / "replay.html"
    page.write_text(TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, blob, 1), encoding="utf-8")

    print(f"Replay page: {page.resolve()}")
    open_in_browser(page.resolve())


if __name__ == "__main__":
    main()
