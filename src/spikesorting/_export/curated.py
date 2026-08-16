"""Read a Kilosort/Phy results folder, before or after manual curation (step 7).

Phy writes curation back into the same folder: ``cluster_group.tsv`` holds the
good/mua/noise labels a human assigned. Kilosort's own automatic labels are in
``cluster_KSLabel.tsv``. When both exist the human wins -- that is the entire
point of the curation step.

NumPy and pandas only. No Kilosort import, so this runs anywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["PhyResults", "parse_params_py", "load_phy_results", "select_units"]

_PARAM_RE = re.compile(r"^\s*(\w+)\s*=\s*(.+?)\s*$")


def parse_params_py(path: str | Path) -> dict[str, object]:
    """Parse Phy's ``params.py`` without executing it.

    It is a Python file, but it is also a file a human may have hand-edited, so it
    is read as data rather than run.
    """
    path = Path(path)
    if not path.exists():
        return {}

    params: dict[str, object] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PARAM_RE.match(line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2)
        raw = raw.split("#", 1)[0].strip()
        if raw.startswith(("'", '"', "r'", 'r"')):
            params[key] = raw.strip("r").strip("'\"")
            continue
        if raw in ("True", "False"):
            params[key] = raw == "True"
            continue
        try:
            params[key] = int(raw)
        except ValueError:
            try:
                params[key] = float(raw)
            except ValueError:
                params[key] = raw
    return params


@dataclass
class PhyResults:
    """Arrays from one sorting results folder."""

    results_dir: Path
    #: Spike times as sample indices, exactly as Kilosort wrote them.
    spike_samples: np.ndarray
    spike_clusters: np.ndarray
    fs: float
    amplitudes: np.ndarray | None = None
    templates: np.ndarray | None = None
    channel_map: np.ndarray | None = None
    #: Template index per spike. After a Phy merge a cluster spans several
    #: templates, so cluster id and template id are not interchangeable.
    spike_templates: np.ndarray | None = None
    #: cluster id -> "good" / "mua" / "noise" / "unsorted"
    labels: dict[int, str] = field(default_factory=dict)
    #: True when a human curated this folder in Phy.
    curated: bool = False

    @property
    def unit_ids(self) -> np.ndarray:
        return np.unique(self.spike_clusters)

    @property
    def spike_times_s(self) -> np.ndarray:
        return np.asarray(self.spike_samples, dtype=np.float64) / self.fs

    @property
    def duration_s(self) -> float:
        return float(self.spike_times_s.max()) if self.spike_samples.size else 0.0

    def times_for(self, unit_id: int) -> np.ndarray:
        """Spike times of one unit, in seconds."""
        return self.spike_times_s[self.spike_clusters == unit_id]

    def amplitudes_for(self, unit_id: int) -> np.ndarray | None:
        if self.amplitudes is None:
            return None
        return self.amplitudes[self.spike_clusters == unit_id]

    def best_channels(self) -> np.ndarray | None:
        """Peak channel per template, in recording-channel numbering."""
        if self.templates is None or self.channel_map is None:
            return None
        peak = (self.templates**2).sum(axis=1).argmax(axis=-1)
        return self.channel_map[peak]


def _read_label_tsv(path: Path, value_column: str | None = None) -> dict[int, str]:
    if not path.exists():
        return {}
    table = pd.read_csv(path, sep="\t")
    if table.empty:
        return {}
    id_col = "cluster_id" if "cluster_id" in table.columns else table.columns[0]
    if value_column is None:
        candidates = [c for c in table.columns if c != id_col]
        if not candidates:
            return {}
        value_column = candidates[0]
    return {int(row[id_col]): str(row[value_column]) for _, row in table.iterrows()}


def load_phy_results(results_dir: str | Path, fs: float | None = None) -> PhyResults:
    """Load a Kilosort4 results folder.

    ``fs`` overrides the sampling rate; otherwise it comes from ``params.py``.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results folder not found: {results_dir}")

    params = parse_params_py(results_dir / "params.py")
    if fs is None:
        fs = float(params.get("sample_rate", 0.0) or 0.0)
    if not fs:
        raise ValueError(
            f"no sample rate for {results_dir}: params.py has none and none was passed"
        )

    def maybe(name: str) -> np.ndarray | None:
        path = results_dir / name
        return np.load(path) if path.exists() else None

    spike_samples = np.load(results_dir / "spike_times.npy").reshape(-1)
    spike_clusters = np.load(results_dir / "spike_clusters.npy").reshape(-1)

    curated_labels = _read_label_tsv(results_dir / "cluster_group.tsv", "group")
    auto_labels = _read_label_tsv(results_dir / "cluster_KSLabel.tsv", "KSLabel")
    labels = {**auto_labels, **curated_labels}

    amplitudes = maybe("amplitudes.npy")
    spike_templates = maybe("spike_templates.npy")
    return PhyResults(
        results_dir=results_dir,
        spike_samples=spike_samples,
        spike_clusters=spike_clusters,
        fs=float(fs),
        amplitudes=amplitudes.reshape(-1) if amplitudes is not None else None,
        templates=maybe("templates.npy"),
        channel_map=maybe("channel_map.npy"),
        spike_templates=spike_templates.reshape(-1) if spike_templates is not None else None,
        labels=labels,
        curated=bool(curated_labels),
    )


def select_units(results: PhyResults, groups: tuple[str, ...] = ("good",)) -> np.ndarray:
    """Unit ids whose label is in ``groups``.

    With no labels at all, returns every unit rather than nothing -- an uncurated
    folder should still export.
    """
    if not results.labels:
        return results.unit_ids
    wanted = {g.lower() for g in groups}
    return np.array(
        [uid for uid in results.unit_ids if results.labels.get(int(uid), "").lower() in wanted],
        dtype=results.unit_ids.dtype,
    )
