"""The pipeline as a flat set of verbs, one per thing you actually do.

Every per-system function takes ``system`` -- ``"neuropixels"`` or
``"blackrock"`` -- so the two acquisition systems read the same way::

    config = load_session_config("configs/athos.yaml", "windows_rig")  # the YAML
    rec    = load_data(config, "blackrock")        # through SpikeInterface
    rec    = preprocess(rec, config, "blackrock")  # bandpass, reference
    sort(rec, config, "blackrock")                 # Kilosort4

Each step returns what the next one takes, so they stack, and each is a real
function you can stop at and inspect. ``load_preprocess_sort`` runs all three
when you do not want to see them.

Two pipelines wrap the verbs, split where manual curation in Phy sits:

``run_sorting_pipeline``
    load, preprocess, sort, extract the sync edges, optionally the LFP. Needs a
    GPU for the sorting; everything else in it does not.

``run_exporting_pipeline``
    coarse align, fit the timing map, validate it, export. Needs no sorter, so it
    runs on a laptop or the rig after curation.

Heavy imports (``kilosort``, ``torch``, ``spikeinterface``) stay inside the
functions that need them, so importing this module costs nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import SessionConfig

__all__ = [
    "SYSTEMS",
    "load_probe",
    "load_data",
    "preprocess",
    "sort",
    "load_preprocess_sort",
    "extract_sync",
    "extract_lfp",
    "rough_align",
    "map_timing",
    "validate_mapping",
    "export_plot",
    "export_data",
    "run_sorting_pipeline",
    "run_exporting_pipeline",
]

SYSTEMS = ("neuropixels", "blackrock")


def _check(system: str) -> str:
    if system not in SYSTEMS:
        raise ValueError(f"system must be one of {SYSTEMS}, got {system!r}")
    return system


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------


def load_probe(config: SessionConfig, system: str) -> dict:
    """The channel map for one system, as a Kilosort probe dict.

    Resolution order, the same for both systems:

    1. ``<system>.probe_file`` -- a JSON built once by ``make_probe.py`` and kept
       in ``configs/probes/``. Which array is in which monkey is a property of
       the session, so this is where it belongs.
    2. the recording's own geometry -- ``~snsGeomMap`` in the SpikeGLX ``.meta``
       for Neuropixels, the ``.cmp`` named by ``blackrock.cmp_file`` for a Utah
       array.
    3. for Utah only, a placeholder 10x10 grid flagged ``_placeholder``. Sorting
       still runs and says so loudly, because a silently wrong map is worse.
    """
    _check(system)
    spec = getattr(config, system)

    if spec.probe_file is not None:
        from .probes.io import load_probe_json

        return load_probe_json(spec.probe_file)

    if system == "neuropixels":
        from .io import spikeglx
        from .probes import neuropixels as np_probes

        if spec.probe_name is not None:
            from .probes.io import probe_from_mat

            return probe_from_mat(Path.home() / ".kilosort" / "probes" / spec.probe_name)
        return np_probes.probe_from_meta(_neuropixels_stream(config).meta)

    from .probes.utah import probe_from_cmp, utah_grid_probe

    if spec.cmp_file is not None:
        return probe_from_cmp(spec.cmp_file, independent=True)
    return utah_grid_probe(96, independent=True)


def _neuropixels_stream(config: SessionConfig, probe: int = 0) -> Any:
    """The AP stream this session points at, however it was specified."""
    from .io import spikeglx

    npx = config.neuropixels
    if npx.bin_file is not None:
        return spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate)
    files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    if files["ap"] is None:
        raise FileNotFoundError(
            f"no AP binary for run {npx.run_name} g{npx.gate} imec{probe}"
        )
    return spikeglx.stream_info(files["ap"])


def load_data(
    config: SessionConfig,
    system: str,
    probe_index: int = 0,
    probe: dict | None = None,
) -> Any:
    """One SpikeInterface recording for one system, with its probe attached.

    Both systems arrive as the same object, which is what lets everything
    downstream stop caring which one it is. ``probe`` overrides the configured
    map for one run, for trying a map before committing it to the session file.
    """
    _check(system)
    import spikeinterface.full as si

    from .probes.common import to_probeinterface

    if system == "blackrock":
        from .io import blackrock

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

    return recording.set_probe(to_probeinterface(probe or load_probe(config, system)))


# ----------------------------------------------------------------------------
# Per-system processing
# ----------------------------------------------------------------------------


def preprocess(recording: Any, config: SessionConfig, system: str) -> Any:
    """Apply this system's ``preprocess:`` block. Returns the recording unchanged
    when the block says ``apply: false``, so it is always safe to call."""
    _check(system)
    from .preprocess import describe_preprocessing, preprocess_recording

    settings = describe_preprocessing(config.preprocess_for(system))
    if not settings.pop("apply", False):
        return recording
    result, _ = preprocess_recording(recording, **settings)
    return result


def sort(
    recording: Any,
    config: SessionConfig,
    system: str,
    results_dir: Path | None = None,
) -> Any:
    """Run Kilosort4 on an already-loaded recording. Returns a ``SortResult``.

    The recording is a required argument, not something this loads for itself:
    the three steps are worth seeing separately, and hiding two of them inside
    the third is what made the workflow hard to follow.
    """
    _check(system)
    from .sort import sort_recording

    return sort_recording(config, system, recording, results_dir=results_dir)


def load_preprocess_sort(
    config: SessionConfig,
    system: str,
    probe: dict | None = None,
    results_dir: Path | None = None,
) -> Any:
    """The three steps in order, for callers that want the whole thing at once.

    Exactly equivalent to writing them out, and written out here so that reading
    this function tells you what the workflow is::

        recording = load_data(config, system)
        recording = preprocess(recording, config, system)
        sort(recording, config, system)
    """
    recording = load_data(config, system, probe=probe)
    recording = preprocess(recording, config, system)
    return sort(recording, config, system, results_dir=results_dir)


def extract_sync(config: SessionConfig, system: str, probe: int = 0) -> Any:
    """Extract one system's sync edges. Returns an ``ExtractionReport``."""
    _check(system)
    from .sync import extract as extract_mod

    if system == "neuropixels":
        return extract_mod.extract_neuropixels_edges(config, probe)
    return extract_mod.extract_blackrock_edges(config)


def extract_lfp(config: SessionConfig, system: str, decimate: int = 1, probe: int = 0):
    """Export one system's LFP. Returns the path written, or None.

    Blackrock returns None by design: Central already saves those LFPs
    separately, so there is nothing here to extract.
    """
    _check(system)
    if system == "blackrock":
        return None

    from .io import spikeglx

    npx = config.neuropixels
    if npx.run_dir is None or not npx.run_name:
        return None
    try:
        files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    except FileNotFoundError:
        return None
    if files["lf"] is None:
        return None
    return spikeglx.export_lfp(files["lf"], config.paths.lfp, decimate=decimate)


# ----------------------------------------------------------------------------
# Cross-system: getting onto the Blackrock timebase
# ----------------------------------------------------------------------------


def rough_align(config: SessionConfig) -> float:
    """Coarse offset between the two systems, in seconds, from the 14 s bursts.

    The bursts are what resolve *which* 1 Hz cycle: their onsets alone are
    periodic and ambiguous, so the full pulse trains are matched on their coded
    intra-burst pattern. Falls back to a 1 Hz-only estimate, which cannot resolve
    whole-cycle ambiguity, when a burst train is missing.
    """
    from .pipeline import _load_edges
    from .sync import burst, extract as extract_mod

    br_burst = _load_edges(config, extract_mod.BR_BURST)
    npx_burst = _load_edges(config, extract_mod.NPX_BURST)
    if br_burst is not None and npx_burst is not None and br_burst.size and npx_burst.size:
        return float(
            burst.match_bursts(
                br_burst, npx_burst, min_gap_s=config.burst_interval_s / 2
            ).offset_s
        )

    br_1hz = _load_edges(config, extract_mod.BR_1HZ)
    npx_1hz = _load_edges(config, extract_mod.NPX_1HZ)
    if br_1hz is None or npx_1hz is None:
        raise FileNotFoundError("no edge files; run extract_sync for both systems first")
    return float(burst.estimate_offset(br_1hz, npx_1hz, tolerance_s=0.1))


def map_timing(config: SessionConfig, system: str = "neuropixels"):
    """Fit the timing map onto Blackrock time and apply it to the spikes.

    Blackrock is the reference timebase; this maps onto it, never the reverse.
    Returns the stage's ``StepResult``.
    """
    _check(system)
    from .pipeline import step_align

    return step_align(config, system=system)


def validate_mapping(config: SessionConfig, figures: bool = True):
    """Check the map against the held-out burst onsets. Returns a ``StepResult``.

    A real check precisely because the bursts were kept out of the 1 Hz fit:
    uncorrected drift of tens of ppm is milliseconds per minute and fails.
    """
    from .pipeline import step_validate

    return step_validate(config, figures=figures)


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------


def export_data(config: SessionConfig, system: str = "neuropixels", groups=("good", "mua")):
    """Aligned spike times, waveforms and metrics for one system."""
    _check(system)
    from .pipeline import step_export

    return step_export(config, system=system, groups=groups, figures=False)


def export_plot(config: SessionConfig, system: str = "neuropixels", groups=("good", "mua")):
    """Summary figures for one system."""
    _check(system)
    from .pipeline import step_export

    return step_export(config, system=system, groups=groups, figures=True)


# ----------------------------------------------------------------------------
# The two pipelines, split where manual curation sits
# ----------------------------------------------------------------------------


def run_sorting_pipeline(
    config: SessionConfig,
    systems: tuple[str, ...] = SYSTEMS,
    lfp: bool = False,
    decimate: int = 1,
) -> dict[str, Any]:
    """Everything before curation: sync edges, optional LFP, and sorting.

    Skips a system that did not record, and sorts only where
    ``kilosort_on_<system>`` asks for it. Returns a dict keyed
    ``"<verb>_<system>"`` so a caller can see what ran.
    """
    out: dict[str, Any] = {}
    for system in systems:
        _check(system)
        if not getattr(config, f"has_{system}_data"):
            out[f"skipped_{system}"] = f"has_{system}_data is false"
            continue

        out[f"extract_sync_{system}"] = extract_sync(config, system)
        if lfp:
            out[f"extract_lfp_{system}"] = extract_lfp(config, system, decimate=decimate)
        if getattr(config, f"sorts_{system}"):
            recording = load_data(config, system)
            recording = preprocess(recording, config, system)
            out[f"sort_{system}"] = sort(recording, config, system)
        else:
            out[f"skipped_sort_{system}"] = f"kilosort_on_{system} is false"
    return out


def run_exporting_pipeline(
    config: SessionConfig,
    systems: tuple[str, ...] = SYSTEMS,
    groups=("good", "mua"),
    figures: bool = True,
) -> dict[str, Any]:
    """Everything after curation: align onto Blackrock time, validate, export.

    Needs no sorter and no GPU -- only the edge files and the sorted output.
    """
    out: dict[str, Any] = {}
    if not config.skip_sync:
        out["map_timing"] = {s: map_timing(config, s) for s in systems if _check(s)}
        out["validate_mapping"] = validate_mapping(config, figures=figures)

    for system in systems:
        if not getattr(config, f"has_{system}_data"):
            continue
        out[f"export_{system}"] = (export_plot if figures else export_data)(
            config, system, groups=groups
        )
    return out
