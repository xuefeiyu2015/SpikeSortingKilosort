"""Pipeline steps, one function per stage.

The ``scripts/`` entry points are thin argparse wrappers around these, so the same
stages can be driven from a notebook, a batch job, or another script without
shelling out.

Every step returns a ``StepResult`` and, crucially, *reports* what it skipped
rather than failing. A demo session with no Blackrock data must still complete;
so must a Blackrock-only session. That is what the ``has_*_data`` and
``kilosort_on_*`` config flags do.

The stages are independent of each other: sorting reads no edge files, and
extraction needs no sorter. Any subset can therefore run on any machine, which is
what lets sorting go to a cluster while the CatGT/TPrime half stays on the rig.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import SessionConfig
from .sync import align, burst, catgt, extract

__all__ = [
    "StepResult",
    "step_extract_sync",
    "step_sort_neuropixels",
    "step_sort_blackrock",
    "step_export",
    "step_align",
    "step_validate",
    "STEPS",
    "step_lfp",
]


@dataclass
class StepResult:
    """Outcome of one pipeline stage."""

    name: str
    status: str  # "ok" | "skipped" | "failed"
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def note(self, message: str) -> None:
        self.notes.append(message)

    def render(self) -> str:
        marker = {"ok": "[ok]", "skipped": "[--]", "failed": "[!!]"}.get(self.status, "[??]")
        lines = [f"{marker} {self.name}"]
        lines.extend(f"       {note}" for note in self.notes)
        return "\n".join(lines)


# ----------------------------------------------------------------------------
# Step 2 + 4: sync extraction
# ----------------------------------------------------------------------------


def step_extract_sync(config: SessionConfig, probe: int = 0) -> StepResult:
    """Extract sync edges from both systems and write the canonical edge files.

    Runs even when ``skip_sync`` is set: ``skip_sync`` skips cross-system
    *alignment*, not extraction. Extracting anyway is cheap, and it is what
    exercises the CatGT path and the NumPy fallback against each other.
    """
    result = StepResult("extract_sync", "ok")
    report = extract.extract_session_edges(config, probe)

    for name, edge_set in sorted(report.edge_sets.items()):
        result.note(
            f"{name}: {edge_set.n} edges via {edge_set.source} "
            f"spanning {edge_set.span_s:.1f} s <- {edge_set.stream}"
        )
    for name, comparison in sorted(report.comparisons.items()):
        verdict = "agree" if comparison["agree"] else "DISAGREE"
        result.note(
            f"{name}: CatGT vs NumPy {verdict} "
            f"({comparison['n_a']} vs {comparison['n_b']} edges, "
            f"max |diff| {comparison['max_abs_diff_s'] * 1e6:.1f} us)"
        )
    for note in report.notes:
        result.note(note)
    for command in report.catgt_commands:
        result.note(f"CatGT command: {command}")

    result.data["edge_sets"] = report.edge_sets
    result.data["comparisons"] = report.comparisons
    if not report.edge_sets:
        result.status = "skipped"
    return result


def step_lfp(config: SessionConfig, probe: int = 0, decimate: int = 1) -> StepResult:
    """Export the Neuropixels LF band (part of step 2, run on its own).

    A stage of its own rather than a flag on extraction, because its cost is
    unlike anything else here: no compute, no GPU, just bulk I/O. The LF band is
    ~7 GB per hour for a 385-channel probe and ``decimate=1`` writes an output
    the same size as the input, so this wants to run where the recording lives.

    Deliberately *not* in the default order -- a full run should not silently
    write another copy of the recording.
    """
    from .io import spikeglx

    result = StepResult("lfp", "ok")
    if not config.has_neuropixels_data:
        result.status = "skipped"
        result.note("has_neuropixels_data is false")
        return result

    npx = config.neuropixels
    if npx.run_dir is None or not npx.run_name:
        result.status = "skipped"
        result.note("no SpikeGLX run (run_dir + run_name); nothing to export")
        return result

    try:
        files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    except FileNotFoundError as error:
        result.status = "skipped"
        result.note(f"no .lf.bin stream to export: {error}")
        return result
    if files["lf"] is None:
        result.status = "skipped"
        result.note("no .lf.bin stream in this run")
        return result

    path = spikeglx.export_lfp(files["lf"], config.paths.lfp, decimate=decimate)
    result.note(f"wrote {path}" + (f" (decimated {decimate}x)" if decimate > 1 else ""))
    result.data["lfp"] = path
    return result


# ----------------------------------------------------------------------------
# Step 3 + 4: sorting
# ----------------------------------------------------------------------------


def _sorting_environment_note(error: ImportError) -> str:
    """A missing sorter is an environment problem; say which one and how to fix it."""
    return (
        f"cannot import the sorting stack ({error}). Sorting needs the 'kilosort4' "
        "conda environment with a CUDA build of torch:\n"
        "         conda activate kilosort4\n"
        "       Every other stage -- sync extraction, alignment, export -- runs "
        "without it."
    )


def step_sort_neuropixels(config: SessionConfig, probe_index: int = 0) -> StepResult:
    """Run Kilosort4 on the Neuropixels AP binary."""
    result = StepResult("sort_neuropixels", "ok")
    if not config.sorts_neuropixels:
        result.status = "skipped"
        result.note(
            "has_neuropixels_data is false"
            if not config.has_neuropixels_data
            else "kilosort_on_neuropixels is false"
        )
        return result

    try:
        from .api import load_data, preprocess, sort
    except ImportError as error:
        result.status = "failed"
        result.note(_sorting_environment_note(error))
        return result

    try:
        recording = load_data(config, "neuropixels", probe_index=probe_index)
        recording = preprocess(recording, config, "neuropixels")
        sorted_result = sort(recording, config, "neuropixels")
    except ImportError as error:
        result.status = "failed"
        result.note(_sorting_environment_note(error))
        return result
    result.note(
        f"{sorted_result.n_units} units, {sorted_result.n_spikes} spikes "
        f"-> {sorted_result.results_dir}"
    )
    result.notes.extend(sorted_result.notes)
    result.data["sort"] = sorted_result
    return result


def step_sort_blackrock(config: SessionConfig, probe: dict | None = None) -> StepResult:
    """Run Kilosort4 on the Utah array file through SpikeInterface."""
    result = StepResult("sort_blackrock", "ok")
    if not config.sorts_blackrock:
        result.status = "skipped"
        result.note(
            "has_blackrock_data is false"
            if not config.has_blackrock_data
            else "kilosort_on_blackrock is false"
        )
        return result
    if config.blackrock.spike_file is None:
        result.status = "skipped"
        result.note("no blackrock.spike_file configured")
        return result

    try:
        from .api import load_data, preprocess, sort

        recording = load_data(config, "blackrock", probe=probe)
        recording = preprocess(recording, config, "blackrock")
        sorted_result = sort(recording, config, "blackrock")
    except ImportError as error:
        result.status = "failed"
        result.note(_sorting_environment_note(error))
        return result
    result.note(f"{sorted_result.n_units} units -> {sorted_result.results_dir}")
    result.notes.extend(sorted_result.notes)
    result.data["sort"] = sorted_result
    return result


# ----------------------------------------------------------------------------
# Step 7 + 10: export
# ----------------------------------------------------------------------------


def step_export(
    config: SessionConfig,
    system: str = "neuropixels",
    groups: tuple[str, ...] = ("good", "mua"),
    figures: bool = True,
    max_unit_figures: int = 40,
) -> StepResult:
    """Export metrics, figures and the final bundle for a sorted folder.

    Uses aligned spike times when ``aligned/spike_seconds_blackrock.npy`` exists
    (written by :func:`step_align`), and says which timebase it used.
    """
    from .export import final, metrics
    from .export.curated import load_phy_results, select_units
    from .plots import summary as plots

    result = StepResult(f"export_{system}", "ok")
    results_dir = config.paths.sorted_np if system == "neuropixels" else config.paths.sorted_br
    if not (results_dir / "spike_times.npy").exists():
        result.status = "skipped"
        result.note(f"no sorting results in {results_dir}")
        return result

    phy = load_phy_results(results_dir)
    unit_ids = select_units(phy, groups)
    result.note(
        f"{phy.unit_ids.size} units total, {unit_ids.size} selected "
        f"({'curated' if phy.curated else 'not curated in Phy'}; groups={groups})"
    )

    aligned_path = config.paths.aligned / f"{system}_spike_seconds_blackrock.npy"
    spike_times_s: dict[int, np.ndarray] | None = None
    timebase = "sorter"
    if aligned_path.exists():
        aligned = np.load(aligned_path)
        if aligned.size != phy.spike_samples.size:
            result.note(
                f"aligned times ({aligned.size}) do not match spike count "
                f"({phy.spike_samples.size}); exporting in the sorter timebase instead"
            )
        else:
            spike_times_s = {
                int(uid): aligned[phy.spike_clusters == uid] for uid in unit_ids
            }
            timebase = "blackrock"
    result.note(f"timebase: {timebase}")

    out_dir = config.paths.aligned if timebase == "blackrock" else results_dir / "export"
    paths = final.export_units(
        out_dir,
        phy,
        unit_ids=unit_ids,
        spike_times_s=spike_times_s,
        timebase=timebase,
        provenance={"session": config.session, "system": system, "machine": config.machine.name},
    )
    result.note(f"exported {len(unit_ids)} units -> {out_dir}")
    result.data["paths"] = paths
    result.data["timebase"] = timebase

    if figures and unit_ids.size:
        figure_dir = config.paths.figures / system
        duration = phy.duration_s
        drawn = 0
        for unit_id in unit_ids[:max_unit_figures]:
            unit_id = int(unit_id)
            times = (
                spike_times_s[unit_id] if spike_times_s is not None else phy.times_for(unit_id)
            )
            waveform, channel = final.mean_template_waveform(phy, unit_id)
            isi = metrics.compute_isi(times)
            isi_counts, isi_edges = metrics.compute_isi_histogram(isi)
            centers, rate = metrics.compute_firing_rate(times, duration, bin_s=max(1.0, duration / 100))
            amps = phy.amplitudes_for(unit_id)

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
                    "amplitudes": amps,
                }
            )
            plots.save_figure(figure, figure_dir / f"unit_{unit_id:04d}.png")
            drawn += 1

        table = final.build_unit_table(phy, unit_ids, spike_times_s)
        overview = plots.plot_sorting_overview(
            table["firing_rate_hz"].to_numpy(),
            amplitudes=table["amp_median"].to_numpy() if "amp_median" in table else None,
            contamination_pct=(
                table["isi_fraction"].to_numpy() * 100 if "isi_fraction" in table else None
            ),
        )
        plots.save_figure(overview, figure_dir / "overview.png")
        result.note(f"wrote {drawn} unit figures + overview -> {figure_dir}")
        if unit_ids.size > max_unit_figures:
            result.note(
                f"NOTE: only the first {max_unit_figures} of {unit_ids.size} units were plotted "
                f"(--max-unit-figures to change)"
            )

    return result


# ----------------------------------------------------------------------------
# Step 8: alignment
# ----------------------------------------------------------------------------


def _load_edges(config: SessionConfig, name: str) -> np.ndarray | None:
    # Each system's edges sit beside its own recording, so the file to read
    # depends on which system produced it.
    system = extract.EDGE_SYSTEM[name]
    if config.paths.dir_for(system) is None:
        return None
    path = config.paths.sync_for(system) / f"{name}.txt"
    return catgt.read_edge_file(path) if path.exists() else None


def step_align(config: SessionConfig, system: str = "neuropixels") -> StepResult:
    """Map sorted spike times onto the Blackrock timebase (step 8).

    Coarse offset from the 14 s bursts, then fine alignment on the 1 Hz train --
    by TPrime when it is installed, otherwise by a least-squares fit of the same
    matched edges. Both paths write the same output file.
    """
    from .sync import tprime as tprime_mod

    result = StepResult("align", "ok")
    if config.skip_sync:
        result.status = "skipped"
        result.note("skip_sync is set (no cross-system alignment for this session)")
        return result

    br_1hz = _load_edges(config, extract.BR_1HZ)
    npx_1hz = _load_edges(config, extract.NPX_1HZ)
    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst = _load_edges(config, extract.NPX_BURST)

    missing = [
        name
        for name, values in [
            (extract.BR_1HZ, br_1hz),
            (extract.NPX_1HZ, npx_1hz),
        ]
        if values is None or values.size == 0
    ]
    if missing:
        result.status = "skipped"
        result.note(f"missing or empty edge files: {', '.join(missing)}; run extract_sync first")
        return result

    # 1. Coarse offset from the 14 s coded bursts.
    offset = 0.0
    if br_burst is not None and npx_burst is not None and br_burst.size and npx_burst.size:
        match = burst.match_bursts(br_burst, npx_burst, min_gap_s=config.burst_interval_s / 2)
        offset = match.offset_s
        result.note(
            f"coarse offset {offset:.6f} s from {match.n_matched} matched bursts "
            f"(of {match.n_reference} Blackrock / {match.n_other} SpikeGLX), "
            f"max residual {match.max_residual_s * 1e3:.3f} ms"
        )
        result.data["burst_match"] = match
    else:
        result.note(
            "no burst edges on one or both systems; falling back to a 1 Hz-only offset "
            "estimate, which cannot resolve whole-cycle ambiguity"
        )
        offset = burst.estimate_offset(br_1hz, npx_1hz, tolerance_s=0.1)
        result.note(f"coarse offset {offset:.6f} s from the 1 Hz trains")

    # 2. Trim both 1 Hz trains to their overlapping window.
    br_trim, npx_trim = align.trim_pair_to_overlap(
        br_1hz, npx_1hz, offset, margin_s=config.sync_period_s / 4
    )
    result.note(
        f"1 Hz trains trimmed to overlap: Blackrock {br_1hz.size} -> {br_trim.size}, "
        f"SpikeGLX {npx_1hz.size} -> {npx_trim.size}"
    )
    aligned_dir = config.paths.aligned
    aligned_dir.mkdir(parents=True, exist_ok=True)
    br_trim_path = catgt.write_edge_file(aligned_dir / "blackrock_1hz_trimmed.txt", br_trim)
    npx_trim_path = catgt.write_edge_file(aligned_dir / "npx_1hz_trimmed.txt", npx_trim)

    # 3. Pair the trimmed edges and fit the linear map (also the TPrime fallback).
    ref_idx, other_idx = burst.match_times(
        br_trim, npx_trim, offset, tolerance_s=config.sync_period_s / 4
    )
    if ref_idx.size < 2:
        result.status = "failed"
        result.note(f"only {ref_idx.size} matched 1 Hz edges; cannot fit a time map")
        return result

    mapping = align.fit_linear_map(npx_trim[other_idx], br_trim[ref_idx])
    result.note(
        f"linear map from {mapping.n_points} matched edges: "
        f"slope {mapping.slope:.9f} ({mapping.drift_ppm:+.1f} ppm), "
        f"intercept {mapping.intercept:.6f} s, "
        f"fit residual max {np.abs(mapping.residuals_s).max() * 1e6:.1f} us"
    )
    result.data["map"] = mapping

    # 4. Convert Kilosort sample indices to seconds, then map them.
    results_dir = config.paths.sorted_np if system == "neuropixels" else config.paths.sorted_br
    spike_times_path = results_dir / "spike_times.npy"
    if not spike_times_path.exists():
        result.note(f"no sorting at {results_dir}; wrote the time map only")
        _save_map(aligned_dir, mapping, offset, config)
        return result

    from .export.curated import parse_params_py

    params = parse_params_py(results_dir / "params.py")
    fs = float(params.get("sample_rate", 0.0) or 0.0)
    if not fs:
        result.status = "failed"
        result.note(f"no sample_rate in {results_dir / 'params.py'}; cannot convert spike times")
        return result

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
        result.note("TPrime command: " + " ".join(args))
        tprime_mod.run_tprime(config.machine.tprime_dir, args)
        result.note(f"TPrime mapped {spike_seconds.size} spike times -> {out_path.name}")
        result.data["method"] = "tprime"
    else:
        np.save(out_path, mapping.apply(spike_seconds))
        result.note(
            f"TPrime not available on machine '{config.machine.name}'; applied the "
            f"least-squares map to {spike_seconds.size} spike times instead"
        )
        result.data["method"] = "linear_fit"

    _save_map(aligned_dir, mapping, offset, config)
    result.data["aligned_path"] = out_path
    return result


def _save_map(aligned_dir: Path, mapping: align.LinearMap, offset: float, config: SessionConfig) -> None:
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


# ----------------------------------------------------------------------------
# Step 9: validation
# ----------------------------------------------------------------------------


def step_validate(config: SessionConfig, figures: bool = True) -> StepResult:
    """Check the alignment against the 14 s bursts (step 9).

    This is the independent check: the mapping was fit on the 1 Hz train, so the
    bursts are held-out data. Residuals must stay under
    ``alignment_tolerance_s``.
    """
    from .plots import summary as plots

    result = StepResult("validate_alignment", "ok")
    if config.skip_sync:
        result.status = "skipped"
        result.note("skip_sync is set")
        return result

    map_path = config.paths.aligned / "time_map.json"
    if not map_path.exists():
        result.status = "skipped"
        result.note(f"no time map at {map_path}; run align first")
        return result

    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst = _load_edges(config, extract.NPX_BURST)
    if br_burst is None or npx_burst is None or not br_burst.size or not npx_burst.size:
        result.status = "skipped"
        result.note(
            "no 14 s burst edges on one or both systems, so the alignment cannot be "
            "independently validated. The 1 Hz fit residuals in time_map.json are the "
            "only available quality measure, and they are not held-out data."
        )
        return result

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
    mapped = mapping.apply(npx_onsets)

    report = align.validate_alignment(
        mapped, br_onsets, tolerance_s=config.alignment_tolerance_s
    )
    result.note(report.summary())
    result.status = "ok" if report.passed else "failed"
    result.data["report"] = report

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
        plots.save_figure(axes.figure, config.paths.figures / "alignment_residuals.png")
        result.note(f"wrote {config.paths.figures / 'alignment_residuals.png'}")

    return result


#: Step name -> callable, for the two ``run_*_pipeline.py --steps`` scripts.
STEPS = {
    "extract_sync": step_extract_sync,
    "lfp": step_lfp,
    "sort_neuropixels": step_sort_neuropixels,
    "sort_blackrock": step_sort_blackrock,
    "align": step_align,
    "validate": step_validate,
    "export": step_export,
}
