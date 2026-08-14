#!/usr/bin/env python
"""Download the official Kilosort4 demo recording and the default probe files.

**Not a pipeline stage** -- deliberately outside the numbered ``NN_`` sequence.
It is a one-off fetch that gives ``configs/demo.yaml`` something to point at:

    python scripts/fetch_demo_data.py                        # once
    python scripts/run_pipeline.py --config configs/demo.yaml

The recording is a short excerpt of an IBL Neuropixels 1.0 session (385 channels:
384 AP + the SY word), the same file the Kilosort documentation uses. It arrives
as a bare ``.bin`` with no ``.meta``, which is why ``configs/demo.yaml`` states
``n_chan_bin`` and ``sample_rate`` explicitly.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

# Same OSF archive the Kilosort4 documentation notebook uses (cropped dataset).
DEMO_URL = "https://osf.io/download/67effd64f74150d8738b7f34/"
DEMO_NAME = "ZFM-02370_mini.imec0.ap.short.bin"


def downloads_dir() -> Path:
    """Kilosort's downloads folder, matching what configs/demo.yaml resolves."""
    try:
        from kilosort.utils import DOWNLOADS_DIR

        return Path(DOWNLOADS_DIR)
    except Exception:
        return Path.home() / ".kilosort"


def _make_reporter():
    """Progress callback that prints at most once per whole percent.

    Rewriting the line every block produces megabytes of carriage returns when
    stdout is a log file rather than a terminal.
    """
    state = {"last": -1}
    interactive = sys.stdout.isatty()

    def report(count: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(count * block_size, total)
        pct = int(100.0 * done / total)
        if pct == state["last"]:
            return
        state["last"] = pct
        end = "\r" if interactive else "\n"
        sys.stdout.write(f"  {done / 1e6:8.1f} / {total / 1e6:.1f} MB ({pct:3d}%){end}")
        sys.stdout.flush()

    return report


def download_demo(target: Path, url: str = DEMO_URL, force: bool = False) -> Path:
    """Download and unzip the demo binary to ``target``."""
    if target.exists() and not force:
        size_mb = target.stat().st_size / 1e6
        print(f"already present: {target} ({size_mb:.1f} MB)")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_suffix(".zip")
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, filename=archive, reporthook=_make_reporter())
    print()

    print(f"unzipping -> {target.parent}")
    with zipfile.ZipFile(archive, "r") as handle:
        handle.extractall(target.parent)
    archive.unlink()

    if not target.exists():
        raise FileNotFoundError(
            f"archive did not contain {target.name}; found "
            f"{[p.name for p in target.parent.iterdir()]}"
        )
    print(f"ready: {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def download_default_probes() -> bool:
    """Fetch Kilosort's bundled probe files (NeuroPix1_default.mat and friends)."""
    try:
        from kilosort.utils import download_probes
    except Exception as error:
        print(f"skipped probe download: kilosort is not importable here ({error})")
        return False
    download_probes()
    print(f"probe files in {downloads_dir() / 'probes'}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--url", default=DEMO_URL, help="override the download URL")
    parser.add_argument(
        "--dest",
        default=None,
        help="destination folder (default: <kilosort downloads>/.test_data)",
    )
    args = parser.parse_args()

    dest = Path(args.dest) if args.dest else downloads_dir() / ".test_data"
    download_demo(dest / DEMO_NAME, args.url, args.force)
    download_default_probes()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
