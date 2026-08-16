"""The pipeline: one verb per thing you do, in the order you do it.

This is the only file you need to read to understand the workflow. Everything
below it -- ``_io``, ``_probes``, ``_sync``, ``_export``, ``_plots`` -- is
machinery these verbs call.

Every per-system verb takes ``system``: ``"neuropixels"`` or ``"blackrock"``. The
two acquisition systems therefore read identically, which is the point of loading
both through SpikeInterface::

    config = load_session_config("configs/athos.yaml", "windows_rig")

    probe  = setup_probe(config, "blackrock")
    rec    = load_spike_continuous(config, "blackrock", probe)
    rec    = preprocess(rec, config, "blackrock")
    sort_with_kilosort(rec, config, "blackrock")

    extract_sync(config, "blackrock")
    time_remapping(config, "neuropixels")
    validate_remapping(config)
    export_results(config, "neuropixels")

**Verbs return values, not status wrappers** -- a recording, a report, a path --
so a notebook cell shows the real object and the next verb takes it directly.

**The session config decides whether a verb does anything.** Each one returns
``None`` when the YAML says not to: ``sort_with_kilosort`` returns immediately if
``kilosort_on_<system>`` is false. ``None`` propagates, so a sequence of calls
needs no branching. :func:`skip_reason` says *why*, for callers that report.

Heavy imports (``kilosort``, ``torch``, ``spikeinterface``) stay inside the
functions that need them, so importing this module costs nothing and it loads on
a machine with none of them installed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._config import (  # noqa: F401  (re-exported: the config verb and its types)
    MachineProfile,
    OutputPaths,
    SessionConfig,
    load_machine,
    load_session_config,
)
from ._sync import align, burst, catgt, extract

log = logging.getLogger("spikesorting")

__all__ = [
    "SYSTEMS",
    "REFERENCE_SYSTEM",
    "skip_reason",
    "load_session_config",
    "setup_probe",
    "load_spike_continuous",
    "preprocess",
    "sort_with_kilosort",
    "extract_sync",
    "extract_lfp",
    "time_remapping",
    "validate_remapping",
    "export_results",
    "TimeMap",
    "SessionConfig",
    "MachineProfile",
    "OutputPaths",
    "load_machine",
]

SYSTEMS = ("neuropixels", "blackrock")

#: The timebase everything else is mapped onto. Never the other way round.
REFERENCE_SYSTEM = "blackrock"


def _check(system: str) -> str:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}")
    return system


def skip_reason(config: SessionConfig, verb: Any, system: str | None = None) -> str | None:
    """Why ``verb`` will do nothing for ``system``, or None if it will run.

    The verbs guard themselves and return ``None``; this exists so a caller can
    *report* the reason without re-deriving it, which keeps status formatting out
    of the pipeline and in the runner scripts.
    """
    name = getattr(verb, "__name__", str(verb))

    if system is not None and name in {
        "setup_probe",
        "load_spike_continuous",
        "preprocess",
        "sort_with_kilosort",
        "extract_sync",
        "extract_lfp",
        "export_results",
    }:
        if not getattr(config, f"has_{system}_data"):
            return f"has_{system}_data is false"

    if name == "sort_with_kilosort" and system is not None:
        if not getattr(config, f"kilosort_on_{system}"):
            return f"kilosort_on_{system} is false"

    if name == "extract_lfp" and system == "blackrock":
        return "Blackrock LFPs are saved separately by Central"

    if name in {"time_remapping", "validate_remapping"} and config.skip_sync:
        return "skip_sync is set (no cross-system alignment for this session)"

    if name == "time_remapping" and system == REFERENCE_SYSTEM:
        return f"{REFERENCE_SYSTEM} is the reference timebase; nothing to map it onto"

    return None


# ----------------------------------------------------------------------------
# Step 1: the channel map
# ----------------------------------------------------------------------------


def setup_probe(config: SessionConfig, system: str) -> dict | None:
    """The channel map for one system, as a Kilosort probe dict.

    An explicit step rather than something hidden inside loading, because this is
    where a wrong choice does the most damage: units land on the wrong
    electrodes and nothing downstream can tell.

    Resolution order, the same for both systems:

    1. ``<system>.probe_file`` -- a JSON built once by ``tools/make_probe.py``
       and kept in ``configs/probes/``. Which array is in which monkey is a
       property of the session, so that is where it is named.
    2. the recording's own geometry: ``~snsGeomMap`` in the SpikeGLX ``.meta``,
       or the ``.cmp`` named by ``blackrock.cmp_file``.
    3. for Utah only, a placeholder 10x10 grid flagged ``_placeholder``. Sorting
       still runs and says so loudly, because a silently wrong map is worse.
    """
    _check(system)
    if not getattr(config, f"has_{system}_data"):
        return None

    spec = getattr(config, system)

    if spec.probe_file is not None:
        from ._probes.io import load_probe_json

        return load_probe_json(spec.probe_file)

    if system == "neuropixels":
        from ._probes import neuropixels as np_probes

        if spec.probe_name is not None:
            return _probe_from_kilosort_library(spec.probe_name)
        return np_probes.probe_from_meta(_neuropixels_stream(config).meta)

    from ._probes.utah import probe_from_cmp, utah_grid_probe

    if spec.cmp_file is not None:
        return probe_from_cmp(spec.cmp_file, independent=True)
    log.warning(
        "no blackrock.probe_file or cmp_file: using a PLACEHOLDER grid in channel "
        "order, so units will be attributed to the wrong electrodes"
    )
    return utah_grid_probe(96, independent=True)


def _probe_from_kilosort_library(name: str) -> dict:
    """Resolve a ``probe_name`` against Kilosort's own probe directory.

    ``run_kilosort(probe_name=...)`` used to do this itself, downloading the file
    when missing. Loading now goes through SpikeInterface, so the map has to be
    resolved *before* Kilosort is involved -- on a machine that may not have it.
    Say so plainly rather than failing on a guessed path.
    """
    from ._probes.io import probe_from_mat

    try:
        from kilosort.utils import PROBE_DIR  # type: ignore[import-not-found]

        directory = Path(PROBE_DIR)
    except Exception:
        directory = Path.home() / ".kilosort" / "probes"

    path = directory / name
    if not path.exists():
        raise FileNotFoundError(
            f"probe_name '{name}' not found at {path}. Kilosort ships these and "
            "downloads them on first use, so this needs the kilosort4 env. On a "
            "machine without it, set neuropixels.probe_file to a JSON built by "
            "tools/make_probe.py instead."
        )
    return probe_from_mat(path)


def _neuropixels_stream(config: SessionConfig, probe: int = 0) -> Any:
    """The AP stream this session points at, however it was specified."""
    from ._io import spikeglx

    npx = config.neuropixels
    if npx.bin_file is not None:
        return spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate)
    files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    if files["ap"] is None:
        raise FileNotFoundError(
            f"no AP binary for run {npx.run_name} g{npx.gate} imec{probe}"
        )
    return spikeglx.stream_info(files["ap"])


# ----------------------------------------------------------------------------
# Step 2: the recording
# ----------------------------------------------------------------------------


def load_spike_continuous(
    config: SessionConfig,
    system: str,
    probe: dict | None = None,
    probe_index: int = 0,
) -> Any | None:
    """The spike-band continuous data for one system, as a SpikeInterface recording.

    Not the LFP and not the sync channels -- those have verbs of their own. The
    probe is attached here, so everything downstream stops caring which system
    produced the samples. Pass ``probe`` to override the configured map for one
    run, for trying a map before committing it to the session file.
    """
    _check(system)
    if not getattr(config, f"has_{system}_data"):
        return None

    import spikeinterface.full as si

    from ._probes.common import to_probeinterface

    if system == "blackrock":
        from ._io import blackrock

        spec = config.blackrock
        if spec.spike_file is None:
            raise ValueError("blackrock.spike_file is required to load Blackrock data")
        recording = blackrock.read_recording(
            spec.spike_file, stream_id=spec.stream_id, exclude_channels=spec.exclude_channels
        )
    else:
        info = _neuropixels_stream(config, probe_index)
        recording = si.read_binary(
            file_paths=[str(info.path)],
            sampling_frequency=float(info.fs),
            num_channels=int(info.n_chan),
            dtype="int16",
        )

    probe = probe if probe is not None else setup_probe(config, system)
    return recording.set_probe(to_probeinterface(probe))


# ----------------------------------------------------------------------------
# Step 3: preprocessing
# ----------------------------------------------------------------------------


def preprocess(recording: Any, config: SessionConfig, system: str) -> Any | None:
    """Apply this system's ``preprocess:`` block.

    Returns the recording unchanged when the block says ``apply: false``, and
    ``None`` when given ``None``, so it is always safe to call in a sequence.

    Order matters: bad-channel detection *drops* channels, then bandpass, then
    the common reference. ``common_reference`` is ``"median"`` (CMR),
    ``"average"``/``"mean"`` (CAR), or ``null`` for none.
    """
    _check(system)
    if recording is None:
        return None

    from ._preprocess import describe_preprocessing, preprocess_recording

    settings = describe_preprocessing(config.preprocess_for(system))
    if not settings.pop("apply", False):
        return recording

    result, info = preprocess_recording(recording, **settings)
    log.info("preprocessing applied to %s: %s", system, info)
    return result


# ----------------------------------------------------------------------------
# Step 4: sorting
# ----------------------------------------------------------------------------


def sort_with_kilosort(
    recording: Any,
    config: SessionConfig,
    system: str,
    results_dir: Path | None = None,
) -> Any | None:
    """Run Kilosort4 on a loaded, preprocessed recording. Returns a ``SortResult``.

    Returns ``None`` without sorting when ``kilosort_on_<system>`` is false --
    the "extract the pulses and the LFP but do not sort" case -- or when given no
    recording.
    """
    _check(system)
    if recording is None or not getattr(config, f"sorts_{system}"):
        return None

    from ._sort import sort_recording

    return sort_recording(config, system, recording, results_dir=results_dir)


# ----------------------------------------------------------------------------
# Step 5: sync pulses and LFP
# ----------------------------------------------------------------------------


def extract_sync(config: SessionConfig, system: str, probe: int = 0) -> Any | None:
    """Extract one system's sync edges to its own ``sync/``. Returns a report.

    Runs CatGT where the machine has it, always runs the NumPy detector, and
    compares them -- that agreement is what lets the fallback be trusted on a
    machine without CatGT.
    """
    _check(system)
    if not getattr(config, f"has_{system}_data"):
        return None

    if system == "neuropixels":
        return extract.extract_neuropixels_edges(config, probe)
    return extract.extract_blackrock_edges(config)


def extract_lfp(
    config: SessionConfig, system: str, decimate: int = 1, probe: int = 0
) -> Path | None:
    """Export one system's LFP. Returns the path written, or ``None``.

    Blackrock always returns ``None``: Central already saves those separately.

    Bulk I/O, no compute -- the LF band is ~7 GB/hour at 385 channels and
    ``decimate=1`` writes an output the size of the input, so run this where the
    recording lives. Decimation is a plain stride with no anti-alias filter; the
    LF band is hardware-limited to ~500 Hz at 2500 Hz sampling, so 2 is safe and
    more aliases.
    """
    _check(system)
    if system == "blackrock" or not config.has_neuropixels_data:
        return None

    from ._io import spikeglx

    npx = config.neuropixels
    if npx.run_dir is None or not npx.run_name:
        return None
    try:
        files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    except FileNotFoundError:
        return None
    if files["lf"] is None:
        return None

    path = spikeglx.export_lfp(files["lf"], config.paths.lfp, decimate=decimate)
    log.info("wrote %s%s", path, f" (decimated {decimate}x)" if decimate > 1 else "")
    return path


def _load_edges(config: SessionConfig, name: str) -> np.ndarray | None:
    """One edge file. Each system's edges sit beside its own recording."""
    system = extract.EDGE_SYSTEM[name]
    if config.paths.dir_for(system) is None:
        return None
    path = config.paths.sync_for(system) / f"{name}.txt"
    return catgt.read_edge_file(path) if path.exists() else None


# ----------------------------------------------------------------------------
# Step 6: onto the Blackrock timebase
# ----------------------------------------------------------------------------


@dataclass
class TimeMap:
    """The fitted map from one system's clock onto Blackrock time."""

    mapping: align.LinearMap
    coarse_offset_s: float
    #: "tprime", "linear_fit", or "map_only" when there was no sorting to map.
    method: str
    aligned_path: Path | None = None
    burst_match: Any | None = None

    def summary(self) -> str:
        m = self.mapping
        return (
            f"coarse offset {self.coarse_offset_s:.6f} s; "
            f"slope {m.slope:.9f} ({m.drift_ppm:+.1f} ppm), "
            f"intercept {m.intercept:.6f} s, from {m.n_points} edges "
            f"[{self.method}]"
        )


def time_remapping(config: SessionConfig, system: str = "neuropixels") -> TimeMap | None:
    """Map ``system``'s spike times onto Blackrock time. Returns a :class:`TimeMap`.

    Blackrock is the reference timebase, so it is not an argument -- this maps
    onto it, never the reverse.

    Coarse offset from the 14 s coded bursts first: their onsets alone are
    periodic and ambiguous, so the full pulse trains are matched on the coded
    intra-burst pattern to pick the right cycle. Then both 1 Hz trains are
    trimmed to their overlap and a linear map is fitted -- by TPrime where it is
    installed, otherwise by least squares on the same matched edges. Both paths
    write the same output file.

    Returns ``None`` when ``skip_sync`` is set, when asked for Blackrock itself
    (there is nothing to map the reference onto), or when the edge files are
    missing.
    """
    _check(system)
    if config.skip_sync or system == REFERENCE_SYSTEM:
        return None

    from ._sync import tprime as tprime_mod

    br_1hz = _load_edges(config, extract.BR_1HZ)
    npx_1hz = _load_edges(config, extract.NPX_1HZ)
    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst = _load_edges(config, extract.NPX_BURST)

    missing = [
        name
        for name, values in ((extract.BR_1HZ, br_1hz), (extract.NPX_1HZ, npx_1hz))
        if values is None or values.size == 0
    ]
    if missing:
        log.warning(
            "missing or empty edge files: %s; run extract_sync first", ", ".join(missing)
        )
        return None

    # 1. Coarse offset from the 14 s coded bursts.
    match = None
    if br_burst is not None and npx_burst is not None and br_burst.size and npx_burst.size:
        match = burst.match_bursts(br_burst, npx_burst, min_gap_s=config.burst_interval_s / 2)
        offset = match.offset_s
        log.info(
            "coarse offset %.6f s from %d matched bursts (of %d Blackrock / %d SpikeGLX), "
            "max residual %.3f ms",
            offset, match.n_matched, match.n_reference, match.n_other,
            match.max_residual_s * 1e3,
        )
    else:
        log.warning(
            "no burst edges on one or both systems; falling back to a 1 Hz-only offset "
            "estimate, which cannot resolve whole-cycle ambiguity"
        )
        offset = burst.estimate_offset(br_1hz, npx_1hz, tolerance_s=0.1)
        log.info("coarse offset %.6f s from the 1 Hz trains", offset)

    # 2. Trim both 1 Hz trains to their overlapping window. TPrime aligns edge
    #    *sequences*, so an edge present in only one recording shifts the
    #    correspondence by whole cycles.
    br_trim, npx_trim = align.trim_pair_to_overlap(
        br_1hz, npx_1hz, offset, margin_s=config.sync_period_s / 4
    )
    log.info(
        "1 Hz trains trimmed to overlap: Blackrock %d -> %d, SpikeGLX %d -> %d",
        br_1hz.size, br_trim.size, npx_1hz.size, npx_trim.size,
    )
    aligned_dir = config.paths.aligned
    aligned_dir.mkdir(parents=True, exist_ok=True)
    br_trim_path = catgt.write_edge_file(aligned_dir / "blackrock_1hz_trimmed.txt", br_trim)
    npx_trim_path = catgt.write_edge_file(aligned_dir / "npx_1hz_trimmed.txt", npx_trim)

    # 3. Pair the trimmed edges and fit the map (also the TPrime fallback).
    ref_idx, other_idx = burst.match_times(
        br_trim, npx_trim, offset, tolerance_s=config.sync_period_s / 4
    )
    if ref_idx.size < 2:
        raise ValueError(f"only {ref_idx.size} matched 1 Hz edges; cannot fit a time map")

    mapping = align.fit_linear_map(npx_trim[other_idx], br_trim[ref_idx])
    log.info(
        "linear map from %d matched edges: slope %.9f (%+.1f ppm), intercept %.6f s, "
        "fit residual max %.1f us",
        mapping.n_points, mapping.slope, mapping.drift_ppm, mapping.intercept,
        np.abs(mapping.residuals_s).max() * 1e6,
    )

    # 4. Convert Kilosort sample indices to seconds, then map them.
    results_dir = config.paths.sorted_for(system)
    spike_times_path = results_dir / "spike_times.npy"
    if not spike_times_path.exists():
        log.info("no sorting at %s; wrote the time map only", results_dir)
        _save_map(aligned_dir, mapping, offset, config)
        return TimeMap(mapping, offset, "map_only", burst_match=match)

    from ._export.curated import parse_params_py

    params = parse_params_py(results_dir / "params.py")
    fs = float(params.get("sample_rate", 0.0) or 0.0)
    if not fs:
        raise ValueError(
            f"no sample_rate in {results_dir / 'params.py'}; cannot convert spike times"
        )

    # Kilosort writes sample indices; TPrime needs seconds. Skipping this is the
    # easiest way to produce a confidently wrong alignment.
    spike_seconds = tprime_mod.spike_times_to_seconds(np.load(spike_times_path), fs)
    seconds_path = aligned_dir / f"{system}_spike_seconds.npy"
    np.save(seconds_path, spike_seconds)

    out_path = aligned_dir / f"{system}_spike_seconds_blackrock.npy"
    if config.machine.has_tprime:
        args = tprime_mod.build_tprime_args(
            to_stream=br_trim_path,
            from_streams={1: npx_trim_path},
            events=[tprime_mod.TPrimeEvent(1, seconds_path, out_path)],
            sync_period_s=config.sync_period_s,
        )
        log.info("TPrime command: %s", " ".join(args))
        tprime_mod.run_tprime(config.machine.tprime_dir, args)
        log.info("TPrime mapped %d spike times -> %s", spike_seconds.size, out_path.name)
        method = "tprime"
    else:
        np.save(out_path, mapping.apply(spike_seconds))
        log.info(
            "TPrime not available on machine '%s'; applied the least-squares map to "
            "%d spike times instead",
            config.machine.name, spike_seconds.size,
        )
        method = "linear_fit"

    _save_map(aligned_dir, mapping, offset, config)
    return TimeMap(mapping, offset, method, aligned_path=out_path, burst_match=match)


def _save_map(
    aligned_dir: Path, mapping: align.LinearMap, offset: float, config: SessionConfig
) -> None:
    payload = {
        "session": config.session,
        "reference_timebase": "blackrock",
        "coarse_offset_s": offset,
        "slope": mapping.slope,
        "intercept": mapping.intercept,
        "drift_ppm": mapping.drift_ppm,
        "n_points": mapping.n_points,
        "fit_max_residual_s": float(np.abs(mapping.residuals_s).max()),
        "formula": "blackrock_time = slope * spikeglx_time + intercept",
    }
    with open(Path(aligned_dir) / "time_map.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_remapping(config: SessionConfig, figures: bool = True) -> Any | None:
    """Check the map against the 14 s burst onsets. Returns a ``ValidationReport``.

    A real check precisely because the bursts were **held out** of the 1 Hz fit.
    Uncorrected clock drift of tens of ppm is 72 ms/hour at 20 ppm and fails the
    1 ms tolerance, which is the point.

    Returns ``None`` when ``skip_sync`` is set, no map has been fitted, or
    neither system recorded bursts -- in which case the 1 Hz fit residuals in
    ``time_map.json`` are the only quality measure, and they are not held-out.
    """
    from ._plots import summary as plots

    if config.skip_sync:
        return None

    map_path = config.paths.aligned / "time_map.json"
    if not map_path.exists():
        log.warning("no time map at %s; run time_remapping first", map_path)
        return None

    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst = _load_edges(config, extract.NPX_BURST)
    if br_burst is None or npx_burst is None or not br_burst.size or not npx_burst.size:
        log.warning(
            "no 14 s burst edges on one or both systems, so the alignment cannot be "
            "independently validated. The 1 Hz fit residuals in time_map.json are the "
            "only available quality measure, and they are not held-out data."
        )
        return None

    with open(map_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mapping = align.LinearMap(
        slope=float(payload["slope"]),
        intercept=float(payload["intercept"]),
        n_points=int(payload["n_points"]),
        residuals_s=np.empty(0),
    )

    br_onsets = burst.group_burst_onsets(br_burst, config.burst_interval_s / 2)
    npx_onsets = burst.group_burst_onsets(npx_burst, config.burst_interval_s / 2)
    report = align.validate_alignment(
        mapping.apply(npx_onsets), br_onsets, tolerance_s=config.alignment_tolerance_s
    )
    log.info("%s", report.summary())

    with open(config.paths.aligned / "validation.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "n_checked": report.n_checked,
                "max_residual_s": report.max_residual_s,
                "median_residual_s": report.median_residual_s,
                "tolerance_s": report.tolerance_s,
                "passed": report.passed,
            },
            handle,
            indent=2,
        )

    if figures and report.n_checked:
        axes = plots.plot_alignment_residuals(
            report.times_s, report.residuals_s, report.tolerance_s
        )
        path = config.paths.figures / "alignment_residuals.png"
        plots.save_figure(axes.figure, path)
        log.info("wrote %s", path)

    return report


# ----------------------------------------------------------------------------
# Step 7: export
# ----------------------------------------------------------------------------


def export_results(
    config: SessionConfig,
    system: str = "neuropixels",
    groups: tuple[str, ...] = ("good", "mua"),
    figures: bool = True,
    max_unit_figures: int = 40,
) -> dict | None:
    """Export metrics, figures and the final bundle for a sorted folder.

    Uses aligned spike times when ``time_remapping`` has written them, and
    records which timebase it used. Reads Phy's ``cluster_group.tsv`` where it
    exists, so curated labels override Kilosort's own.

    Returns ``None`` when that system has no sorting output.
    """
    _check(system)
    from ._export import final, metrics
    from ._export.curated import load_phy_results, select_units
    from ._plots import summary as plots

    if not getattr(config, f"has_{system}_data"):
        return None

    results_dir = config.paths.sorted_for(system)
    if not (results_dir / "spike_times.npy").exists():
        log.info("no sorting results in %s", results_dir)
        return None

    phy = load_phy_results(results_dir)
    unit_ids = select_units(phy, groups)
    log.info(
        "%d units total, %d selected (%s; groups=%s)",
        phy.unit_ids.size, unit_ids.size,
        "curated" if phy.curated else "not curated in Phy", groups,
    )

    aligned_path = config.paths.aligned / f"{system}_spike_seconds_blackrock.npy"
    spike_times_s: dict[int, np.ndarray] | None = None
    timebase = "sorter"
    if aligned_path.exists():
        aligned = np.load(aligned_path)
        if aligned.size != phy.spike_samples.size:
            log.warning(
                "aligned times (%d) do not match spike count (%d); exporting in the "
                "sorter timebase instead",
                aligned.size, phy.spike_samples.size,
            )
        else:
            spike_times_s = {int(uid): aligned[phy.spike_clusters == uid] for uid in unit_ids}
            timebase = "blackrock"
    log.info("timebase: %s", timebase)

    out_dir = config.paths.aligned if timebase == "blackrock" else results_dir / "export"
    paths = final.export_units(
        out_dir,
        phy,
        unit_ids=unit_ids,
        spike_times_s=spike_times_s,
        timebase=timebase,
        provenance={"session": config.session, "system": system, "machine": config.machine.name},
    )
    log.info("exported %d units -> %s", unit_ids.size, out_dir)

    if figures and unit_ids.size:
        figure_dir = config.paths.figures / system
        duration = phy.duration_s
        for unit_id in unit_ids[:max_unit_figures]:
            unit_id = int(unit_id)
            times = spike_times_s[unit_id] if spike_times_s is not None else phy.times_for(unit_id)
            waveform, channel = final.mean_template_waveform(phy, unit_id)
            isi_counts, isi_edges = metrics.compute_isi_histogram(metrics.compute_isi(times))
            centers, rate = metrics.compute_firing_rate(
                times, duration, bin_s=max(1.0, duration / 100)
            )
            figure = plots.plot_unit_summary(
                {
                    "unit_id": unit_id,
                    "label": phy.labels.get(unit_id, "unsorted"),
                    "channel": channel,
                    "n_spikes": int(times.size),
                    "waveform": waveform,
                    "waveform_t_ms": (
                        np.arange(waveform.size) / phy.fs * 1000.0 if waveform.size else None
                    ),
                    "isi_counts": isi_counts,
                    "isi_edges_ms": isi_edges,
                    "rate_centers_s": centers,
                    "rate_hz": rate,
                    "amp_times_s": times,
                    "amplitudes": phy.amplitudes_for(unit_id),
                }
            )
            plots.save_figure(figure, figure_dir / f"unit_{unit_id:04d}.png")

        table = final.build_unit_table(phy, unit_ids, spike_times_s)
        overview = plots.plot_sorting_overview(
            table["firing_rate_hz"].to_numpy(),
            amplitudes=table["amp_median"].to_numpy() if "amp_median" in table else None,
            contamination_pct=(
                table["isi_fraction"].to_numpy() * 100 if "isi_fraction" in table else None
            ),
        )
        plots.save_figure(overview, figure_dir / "overview.png")
        drawn = min(unit_ids.size, max_unit_figures)
        log.info("wrote %d unit figures + overview -> %s", drawn, figure_dir)
        if unit_ids.size > max_unit_figures:
            log.warning(
                "only the first %d of %d units were plotted (--max-unit-figures to change)",
                max_unit_figures, unit_ids.size,
            )

    return {"paths": paths, "timebase": timebase, "n_units": int(unit_ids.size)}
