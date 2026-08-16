"""Running Kilosort4 (pipeline steps 3 and 4).

One route for both systems. They arrive as SpikeInterface recordings with their
probes already attached -- see :func:`spikesorting.api.load_data` -- so nothing
here needs to know which system produced the samples.

Sorting runs inside the machine's SSD cache and the results are copied out
afterwards. Kilosort writes a whitened copy of the whole recording next to its
results; on a network share that is both slow and rude to everyone else on it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ._config import SessionConfig

__all__ = ["SortResult", "sort_recording", "summarize_results"]


@dataclass
class SortResult:
    """Where a sorting landed and how it went."""

    results_dir: Path
    n_units: int | None = None
    n_spikes: int | None = None
    fs: float | None = None
    work_dir: Path | None = None
    probe: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _lift(root: Path, subdir: str) -> None:
    """Move ``root/subdir``'s contents up into ``root`` and drop the husk.

    The in-place counterpart of publishing a subdirectory, for the no-cache
    fallback: both branches must leave the same tree behind, or a machine
    without a ``cache_dir`` would quietly produce a layout nothing can read.
    """
    source = root / subdir
    if not source.is_dir():
        raise FileNotFoundError(f"expected sorter output at {source}")
    for item in source.iterdir():
        shutil.move(str(item), str(root / item.name))
    shutil.rmtree(root / Path(subdir).parts[0], ignore_errors=True)


def _run_in_cache(
    final_dir: Path,
    cache_dir: Path | None,
    tag: str,
    body: Callable[[Path], Any],
    keep_dat: bool = False,
    publish_from: str | None = None,
) -> tuple[Any, Path]:
    """Run ``body(work_dir)`` on fast local storage, then publish the results.

    Returns ``(body_result, work_dir)``. The cache is deleted afterwards, which is
    the "released after kilosort4 is done" behaviour the path setup calls for.
    Falls back to sorting in place when the machine has no cache configured.

    ``publish_from`` names a subdirectory of the work dir to publish *as*
    ``final_dir``, for writers that nest their output. SpikeInterface is the one
    that does: it puts Kilosort's own files under ``<folder>/sorter_output/``,
    two levels below where every consumer looks for ``spike_times.npy``.
    """
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        result = body(final_dir)
        if publish_from:
            _lift(final_dir, publish_from)
        return result, final_dir

    work_dir = Path(cache_dir) / tag
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = body(work_dir)
        ignore = None if keep_dat else shutil.ignore_patterns("*.dat")
        source = work_dir / publish_from if publish_from else work_dir
        if not source.is_dir():
            raise FileNotFoundError(f"expected sorter output at {source}")
        shutil.copytree(source, final_dir, dirs_exist_ok=True, ignore=ignore)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result, work_dir


def sort_recording(
    config: SessionConfig,
    system: str,
    recording: Any,
    results_dir: Path | None = None,
) -> SortResult:
    """Sort an already-loaded, already-preprocessed recording. One path, both systems.

    This is what :func:`spikesorting.api.sort` calls. The recording arrives with
    its probe attached, so nothing here needs to know which system produced it --
    which is the whole point of loading both through SpikeInterface.

    SpikeInterface writes Kilosort's output under ``<folder>/sorter_output/``, so
    only that subdirectory is published: the result is a plain Kilosort/Phy
    directory holding ``spike_times.npy``, which is what every consumer expects.
    Sorting into a subdirectory of the work dir rather than the work dir itself is
    deliberate -- ``remove_existing_folder`` would otherwise wipe it, and in the
    no-cache fallback that directory *is* the final one.
    """
    import spikeinterface.sorters as ss

    final_dir = Path(results_dir) if results_dir else config.paths.sorted_for(system)

    params = {"device": config.machine.device}
    params.update(config.kilosort_settings)

    def body(work_dir: Path) -> Any:
        return ss.run_sorter(
            "kilosort4",
            recording,
            folder=str(work_dir / "kilosort4"),
            remove_existing_folder=True,
            **params,
        )

    sorting, work_dir = _run_in_cache(
        final_dir,
        config.cache_dir,
        f"{config.session}_kilosort4_{system}",
        body,
        publish_from="kilosort4/sorter_output",
    )

    probe = recording.get_probe() if hasattr(recording, "get_probe") else None
    result = SortResult(
        results_dir=final_dir,
        n_units=int(len(sorting.unit_ids)) if sorting is not None else None,
        n_spikes=None,
        fs=float(recording.get_sampling_frequency()),
        work_dir=work_dir,
        probe={"n_contacts": int(probe.get_contact_count())} if probe is not None else {},
    )
    _write_run_info(final_dir, config, result, stream=system)
    return result


def _write_run_info(
    final_dir: Path, config: SessionConfig, result: SortResult, stream: str
) -> None:
    """Record what produced this sorting, beside the sorting itself."""
    payload = {
        "session": config.session,
        "machine": config.machine.name,
        "device": config.machine.device,
        "source": stream,
        "n_units": result.n_units,
        "n_spikes": result.n_spikes,
        "fs": result.fs,
        "probe": result.probe,
        "kilosort_settings": config.kilosort_settings,
        "notes": result.notes,
    }
    with open(Path(final_dir) / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def summarize_results(results_dir: str | Path) -> dict[str, Any]:
    """Read a Kilosort results folder into a small summary dict. Pure.

    Reads only the ``.npy``/``.tsv`` outputs, so it works without Kilosort
    installed -- which is what lets export and QC run off the sorting machine.
    """
    results_dir = Path(results_dir)
    spike_times = np.load(results_dir / "spike_times.npy")
    clusters = np.load(results_dir / "spike_clusters.npy")
    templates = np.load(results_dir / "templates.npy")
    channel_map = np.load(results_dir / "channel_map.npy")

    best_channel = channel_map[(templates**2).sum(axis=1).argmax(axis=-1)]
    unit_ids, counts = np.unique(clusters, return_counts=True)

    return {
        "n_spikes": int(spike_times.size),
        "n_units": int(unit_ids.size),
        "unit_ids": unit_ids,
        "spike_counts": counts,
        "best_channel": best_channel,
        "duration_samples": int(spike_times.max()) if spike_times.size else 0,
    }
