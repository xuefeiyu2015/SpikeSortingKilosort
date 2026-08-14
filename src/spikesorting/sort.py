"""Kilosort4 entry points (pipeline steps 3 and 4).

Two routes, because the two systems arrive in different shapes:

* **Neuropixels** is already a flat int16 binary, so it goes straight into
  ``kilosort.run_kilosort``.
* **Blackrock** is read through SpikeInterface (which handles the ``.nsX``
  container and any preprocessing), then handed to Kilosort via
  ``spikeinterface.sorters.run_sorter``.

Both run inside the machine's SSD cache and copy their results out afterwards.
Kilosort writes a whitened copy of the whole recording next to its results; on a
network share that is both slow and rude to everyone else on the share.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import SessionConfig

__all__ = ["SortResult", "sort_neuropixels", "sort_blackrock", "summarize_results"]


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


def _torch_device(name: str) -> Any:
    import torch  # lazy

    return torch.device(name)


def _run_in_cache(
    final_dir: Path,
    cache_dir: Path | None,
    tag: str,
    body: Callable[[Path], Any],
    keep_dat: bool = False,
) -> tuple[Any, Path]:
    """Run ``body(work_dir)`` on fast local storage, then publish the results.

    Returns ``(body_result, work_dir)``. The cache is deleted afterwards, which is
    the "released after kilosort4 is done" behaviour the path setup calls for.
    Falls back to sorting in place when the machine has no cache configured.
    """
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    if cache_dir is None:
        return body(final_dir), final_dir

    work_dir = Path(cache_dir) / tag
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = body(work_dir)
        ignore = None if keep_dat else shutil.ignore_patterns("*.dat")
        shutil.copytree(work_dir, final_dir, dirs_exist_ok=True, ignore=ignore)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return result, work_dir


def sort_neuropixels(
    config: SessionConfig,
    probe: dict | None = None,
    probe_index: int = 0,
    results_dir: Path | None = None,
) -> SortResult:
    """Sort the Neuropixels AP binary with Kilosort4.

    ``probe`` overrides the channel map; otherwise the session's ``probe_name``
    is used, or the map is built from the run's ``.meta``.
    """
    from kilosort import run_kilosort  # lazy: pulls in torch

    from .io import spikeglx
    from .probes import neuropixels as np_probes

    npx = config.neuropixels

    if npx.bin_file is not None:
        info = spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate)
    else:
        files = spikeglx.find_run_files(
            npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe_index
        )
        if files["ap"] is None:
            raise FileNotFoundError(
                f"no AP binary found for run {npx.run_name} g{npx.gate} imec{probe_index}"
            )
        info = spikeglx.stream_info(files["ap"])

    probe_name = npx.probe_name
    if probe is None and probe_name is None:
        probe = np_probes.probe_from_meta(info.meta)

    settings: dict[str, Any] = {
        "filename": str(info.path),
        "n_chan_bin": info.n_chan,
        "fs": info.fs,
    }
    settings.update(config.kilosort_settings)

    final_dir = Path(results_dir) if results_dir else config.paths.sorted_np
    save_preprocessed = bool(settings.pop("save_preprocessed_copy", False))

    def body(work_dir: Path) -> Any:
        kwargs: dict[str, Any] = {
            "settings": settings,
            "results_dir": str(work_dir),
            "device": _torch_device(config.machine.device),
            "save_preprocessed_copy": save_preprocessed,
        }
        # run_kilosort accepts a probe dict OR a probe file name, never both.
        if probe is not None:
            kwargs["probe"] = probe
        else:
            kwargs["probe_name"] = probe_name
        return run_kilosort(**kwargs)

    outputs, work_dir = _run_in_cache(
        final_dir,
        config.cache_dir,
        f"{config.session}_kilosort4_np",
        body,
        keep_dat=save_preprocessed,
    )

    st, clu = outputs[1], outputs[2]
    result = SortResult(
        results_dir=final_dir,
        n_units=int(np.unique(clu).size) if clu is not None else None,
        n_spikes=int(np.asarray(st).size) if st is not None else None,
        fs=info.fs,
        work_dir=work_dir,
        probe={"source": "dict" if probe is not None else f"probe_name={probe_name}"},
    )
    _write_run_info(final_dir, config, result, stream=str(info.path))
    return result


def sort_blackrock(
    config: SessionConfig,
    probe: dict | None = None,
    results_dir: Path | None = None,
) -> SortResult:
    """Sort the Utah array file with Kilosort4 through SpikeInterface.

    A probe with real geometry must be attached before sorting -- see the warning
    in :mod:`spikesorting.probes.utah` about placeholder maps.
    """
    import spikeinterface.sorters as ss  # lazy

    from .io import blackrock
    from .preprocess import describe_preprocessing, preprocess_recording
    from .probes.common import probe_summary, to_probeinterface

    spec = config.blackrock
    if spec.spike_file is None:
        raise ValueError("blackrock.spike_file is required to sort Blackrock data")

    recording = blackrock.read_recording(
        spec.spike_file, stream_id=spec.stream_id, exclude_channels=spec.exclude_channels
    )

    notes: list[str] = []
    if probe is not None:
        if probe.get("_placeholder"):
            notes.append(
                "PLACEHOLDER Utah geometry in use -- channel positions are a guess. "
                "Supply the array's .cmp map before interpreting these results spatially."
            )
        recording = recording.set_probe(to_probeinterface(probe))

    pre = describe_preprocessing(config.preprocess)
    if pre.pop("apply", False):
        recording, pre_info = preprocess_recording(recording, **pre)
        notes.append(f"preprocessing applied: {pre_info}")

    final_dir = Path(results_dir) if results_dir else config.paths.sorted_br
    sorter_params = dict(config.kilosort_settings)

    def body(work_dir: Path) -> Any:
        return ss.run_sorter(
            "kilosort4",
            recording,
            folder=str(work_dir / "kilosort4"),
            remove_existing_folder=True,
            **sorter_params,
        )

    sorting, work_dir = _run_in_cache(
        final_dir, config.cache_dir, f"{config.session}_kilosort4_br", body
    )

    result = SortResult(
        results_dir=final_dir,
        n_units=int(len(sorting.unit_ids)) if sorting is not None else None,
        n_spikes=None,
        fs=float(recording.get_sampling_frequency()),
        work_dir=work_dir,
        probe=probe_summary(probe) if probe else {},
        notes=notes,
    )
    _write_run_info(final_dir, config, result, stream=str(spec.spike_file))
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
