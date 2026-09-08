"""The pipeline: one verb per thing you do, in the order you do it.

This is the only file you need to read to understand the workflow. Everything
below it -- ``_io``, ``_probes``, ``_sync``, ``_export``, ``_plots`` -- is
machinery these verbs call.

Every per-system verb takes ``system``: ``"neuropixels"`` or ``"blackrock"``, so
the two acquisition systems read identically::

    config = load_session_config("configs/athos.yaml", "windows_rig")

    sort_with_kilosort(config, "blackrock")        # pipeline 1, and all of it

    extract_sync(config, "blackrock")              # pipeline 2
    extract_lfp(config, "neuropixels")
    time_remapping(config, "neuropixels")
    validate_remapping(config)
    export_results(config, "neuropixels")

Sorting is Kilosort4's own ``run_kilosort`` on a flat int16 binary. SpikeGLX
already writes one; a Blackrock ``.ns6`` is read by SpikeInterface and
transformed into one once, beside the recording. That is SpikeInterface's only
job here.

A session names one binary per system, so every verb works on one stream and
everything it derives lands in one folder: ``<system>_dir/kilosort4/``, with the
export bundle inside it. Two probes is two session files.

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

import ast
import json
import logging
import shutil
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
    stream_label,
)
from ._io import reachable
from ._sync import align, burst, catgt, extract

log = logging.getLogger("spikesorting")

__all__ = [
    "SYSTEMS",
    "REFERENCE_SYSTEM",
    "skip_reason",
    "load_session_config",
    "setup_probe",
    "sort_with_kilosort",
    "extract_sync",
    "extract_lfp",
    "time_remapping",
    "validate_remapping",
    "export_results",
    "export_waveforms",
    "stamp_lfp_timebase",
    "Binary",
    "SortSummary",
    "TimeMap",
    "SessionConfig",
    "MachineProfile",
    "OutputPaths",
    "load_machine",
    "stream_label",
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
        "sort_with_kilosort",
        "extract_sync",
        "extract_lfp",
        "export_results",
        "export_waveforms",
    }:
        if not config.has_data(system):
            return f"no {system} paths declared in the session"

    if name == "sort_with_kilosort" and system is not None:
        if not getattr(config, f"kilosort_on_{system}"):
            # For Blackrock this is also how a sync-only session says it has no
            # Utah array. The flag with no spike_file is refused at load, so
            # there is no third answer here -- see _config._reject_sorting_nothing.
            return f"kilosort_on_{system} is false"

    if name == "extract_lfp":
        if system == "blackrock":
            return "Blackrock LFPs are saved separately by Central"
        if not config.export_lfp:
            return "export_lfp is false"

    if name in {"time_remapping", "validate_remapping"} and not config.aligns_systems:
        return "only one system declared, so there is nothing to align against"

    if name == "time_remapping" and system == REFERENCE_SYSTEM:
        return f"{REFERENCE_SYSTEM} is the reference timebase; nothing to map it onto"

    return None


# ----------------------------------------------------------------------------
# Step 1: the channel map
# ----------------------------------------------------------------------------


def setup_probe(config: SessionConfig, system: str) -> dict | None:
    """The channel map for one system, as a Kilosort probe dict.

    An explicit step rather than something hidden inside loading, because this is
    where a wrong choice does the most damage: units land on the wrong electrodes
    and nothing downstream can tell.

    **The recording's own geometry wins.** A SpikeGLX ``.meta`` records which
    sites were actually active for *that* run -- an imro choice made per recording
    -- so it cannot be wrong in the way a file built from another run can. A
    ``.cmp`` is the same thing for a Utah array: it describes how this array is
    wired.

    ================================ =========================================
    Neuropixels                      Blackrock
    ================================ =========================================
    1. the ``.meta`` beside bin_file  1. ``blackrock.probe_file``
    2. ``neuropixels.probe_file``     2. raises: no honest default
    3. raises: no honest default
    ================================ =========================================

    **One key per system, whatever format the map is in.** ``probe_file`` is read
    by its extension -- ``.cmp``, ``.json`` or ``.mat``, see
    ``_probes.io.load_probe_file`` -- so naming a Utah array's own wiring map is
    the same act as naming a built one. There used to be a second Blackrock key,
    ``cmp_file``, tried first, which meant a session could name both with nothing
    in the output to say which one was used.

    **Neither system invents a layout.** Kilosort has no probe library to fall
    back on -- it ships no probe files, and its own API
    (``kilosort.io.load_probe``) also takes a path -- so there is nothing to guess
    a Neuropixels map from. A Utah array could be given a square grid in channel
    order, and used to be: but real arrays are rarely wired that way, so units
    land on the wrong electrodes, and the grid's channel count silently drops
    every electrode past it. A deliberate grid is ``make_probe.py utah`` with no
    ``--cmp``, written to ``configs/probes/`` and named in the session, so the
    choice is recorded rather than assumed.

    The ``.meta`` read is the one beside ``bin_file``, so it describes exactly the
    binary being sorted. A session naming a second probe names a second binary in
    its own file, and reads that probe's own meta.
    """
    _check(system)
    if not config.has_data(system):
        return None

    spec = getattr(config, system)

    if system == "neuropixels":
        probe = _probe_from_meta(config)
        if probe is not None:
            return probe
        if spec.probe_file is not None:
            from ._probes.io import load_probe_file

            log.info("no usable .meta; using neuropixels.probe_file %s", spec.probe_file)
            return load_probe_file(spec.probe_file)
        raise FileNotFoundError(
            "no channel map for neuropixels: this run has no .meta to build one "
            "from, and neuropixels.probe_file is not set. Build one once and name "
            "it in the session:\n"
            "    python tools/make_probe.py from-mat --mat <probe>.mat "
            "--out configs/probes/<name>.json --plot\n"
            "then set  neuropixels.probe_file: configs/probes/<name>.json"
        )

    if spec.probe_file is not None:
        from ._probes.io import load_probe_file

        return load_probe_file(spec.probe_file)
    raise FileNotFoundError(
        "no channel map for the Utah array: blackrock.probe_file is not set, and "
        "there is no honest default. A grid in channel order attributes units to "
        "the wrong electrodes, and its channel count silently drops every "
        "electrode past it. Name the array's own wiring map:\n"
        "    blackrock.probe_file: <array>.cmp\n"
        "or build one once and name that instead:\n"
        "    python tools/make_probe.py utah --cmp <array>.cmp "
        "--out configs/probes/utah_<array>.json --plot\n"
        "Either way it is the same key -- .cmp, .json and .mat are told apart by "
        "extension. A deliberate placeholder grid is that command without --cmp: "
        "built, written down and named, rather than assumed."
    )


def _probe_from_meta(config: SessionConfig) -> dict | None:
    """The map from the run's own ``.meta``, or None when there is not one.

    Two readers, in order: this repo's ``~snsGeomMap`` parser, which needs only
    NumPy, then ``probeinterface.read_spikeglx``, which also knows the older
    layouts (``~snsShankMap``, imro-derived geometry) that predate it. Both read
    *this run's* meta, which is the rule that matters -- which sites were active
    is an imro choice made per recording.

    Returns None rather than raising for the ordinary case of a bare binary with
    no ``.meta`` beside it, or a meta neither reader can use, so the caller can
    fall through to ``probe_file``.
    """
    from ._probes import neuropixels as np_probes

    try:
        info = _neuropixels_stream(config)
    except FileNotFoundError:
        return None
    if not info.meta:
        return None

    try:
        return np_probes.probe_from_meta(info.meta)
    except ValueError as error:
        reason = error

    from ._io import spikeglx

    try:
        probe = np_probes.probe_from_meta_file(spikeglx.meta_path_for(info.path))
    except Exception as fallback_error:  # missing probeinterface, or a meta it rejects
        log.info(
            "no map from %s: %s; probeinterface could not read it either (%s)",
            spikeglx.meta_path_for(info.path).name, reason, fallback_error,
        )
        return None
    log.info(
        "%s has no ~snsGeomMap; read its geometry with probeinterface instead",
        spikeglx.meta_path_for(info.path).name,
    )
    return probe


def _neuropixels_stream(config: SessionConfig) -> Any:
    """The AP stream this session names.

    ``n_chan_bin`` and ``sample_rate`` override the ``.meta`` beside the binary
    and stand in for it entirely when there is none -- which is the demo file's
    situation, and the only reason those two keys exist.
    """
    from ._io import spikeglx

    npx = config.neuropixels
    return spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate)


# ----------------------------------------------------------------------------
# Step 2: the binary Kilosort reads
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Binary:
    """A flat int16 recording file, and the two facts Kilosort needs about it."""

    path: Path
    n_chan_bin: int
    fs: float

    def summary(self) -> str:
        return f"{self.path} ({self.n_chan_bin} channels at {self.fs:g} Hz)"


def _binary_for(config: SessionConfig, system: str) -> Binary:
    """The file ``run_kilosort`` is pointed at, for either system.

    SpikeGLX already writes one: the AP binary is flat, sample-interleaved int16,
    which is exactly Kilosort's own format, so nothing is converted. A Blackrock
    ``.ns6`` is not, so it is transformed once -- see :func:`_blackrock_binary`.
    """
    if system == "blackrock":
        return _blackrock_binary(config)
    info = _neuropixels_stream(config)
    return Binary(Path(info.path), int(info.n_chan), float(info.fs))


def _blackrock_binary(config: SessionConfig) -> Binary:
    """Transform the Utah array's ``.ns6`` into a Kilosort binary, once.

    SpikeInterface reads the ``.ns6`` and Kilosort's own
    ``io.spikeinterface_to_binary`` writes it out flat -- this is the whole of
    SpikeInterface's job in the sorting path. **Every channel is written**,
    including any sync or analog inputs sharing the file: the probe's ``chanMap``
    is what selects electrodes, and dropping them here would shift every contact
    after the removed one.

    The result sits beside the recording rather than in the cache, and is reused
    when it is already there at the expected size. It has to persist: it is what
    the sorting's ``params.py`` points at, so Phy can only show raw traces while
    it exists, and re-transforming an unchanged ``.ns6`` on every re-sort is
    minutes of I/O for nothing.
    """
    spec = config.blackrock
    if spec.spike_file is None:
        raise ValueError("blackrock.spike_file is required to sort the Utah array")

    from spikeinterface.extractors import read_blackrock

    # gap_tolerance_ms is not optional in practice: neo raises on any timestamp
    # jump larger than two sampling periods, and PTP recordings carry occasional
    # corrupted packet timestamps, so most files will not open without it.
    kwargs = {"gap_tolerance_ms": spec.gap_tolerance_ms}
    if spec.stream_id:
        kwargs["stream_id"] = spec.stream_id
    recording = read_blackrock(spec.spike_file, **kwargs)

    n_segments = int(recording.get_num_segments())
    if n_segments > 1 and not spec.allow_segments:
        raise ValueError(
            f"{spec.spike_file.name} splits into {n_segments} segments at "
            f"gap_tolerance_ms={spec.gap_tolerance_ms}: a jump that large is a paused "
            "recording, not a timestamp glitch. Sorting across a pause is a decision, "
            "so state it: blackrock.allow_segments: true to concatenate them, or raise "
            "blackrock.gap_tolerance_ms if the jump really is noise."
        )

    n_chan = int(recording.get_num_channels())
    fs = float(recording.get_sampling_frequency())
    n_bytes = n_chan * sum(
        int(recording.get_num_frames(segment_index=s)) for s in range(n_segments)
    ) * 2  # int16

    # One level above the sorting it feeds, so re-sorting neither wipes it nor
    # reconverts. Taken from sorted_for rather than dir_for so a session with no
    # blackrock_dir raises OutputPaths' message naming what to set.
    directory = config.paths.sorted_for("blackrock").parent
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{spec.spike_file.stem}.bin"
    path = directory / name

    if path.exists() and path.stat().st_size == n_bytes:
        log.info("reusing %s (%.1f GB) -- delete it to force a re-transform", path, n_bytes / 1e9)
        return Binary(path, n_chan, fs)

    from kilosort.io import spikeinterface_to_binary

    log.info(
        "transforming %s -> %s (%.1f GB)", spec.spike_file.name, path, n_bytes / 1e9
    )
    written = spikeinterface_to_binary(recording, directory, data_name=name, dtype="int16")
    # It returns a tuple whose first element is the path; the channel count and
    # rate come from the recording it was written from, which cannot disagree
    # with the file and does not depend on the tuple's layout in this release.
    written = written[0] if isinstance(written, tuple) else written
    return Binary(Path(written), n_chan, fs)


def write_nsp_time_map(config: SessionConfig, out_dir: Path, source: Path) -> Any | None:
    """Measure ``source``'s sample -> NSP-clock map and write it beside ``out_dir``.

    The axis every other Blackrock product in the lab uses -- ``.nev`` markers,
    eye traces, online spikes -- and the one Kilosort does not give back. Written
    as JSON so an export can be traced to the file and the fit it came from.
    """
    from ._io import blackrock

    spec = config.blackrock
    reader = blackrock.open_reader(source, gap_tolerance_ms=spec.gap_tolerance_ms)
    time_map = blackrock.nsp_time_map(
        reader, blackrock.nsx_number(source), tolerance_s=config.alignment_tolerance_s
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "nsp_time_map.json"
    payload = dict(time_map.to_dict(), source=str(source))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    log.info("NSP clock: %s -> %s", time_map.summary(), path.name)
    return time_map


def read_nsp_time_map(directory: Path) -> Any | None:
    """The map written by :func:`write_nsp_time_map`, or None if there is none."""
    from ._io.blackrock import NspTimeMap

    path = Path(directory) / "nsp_time_map.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return NspTimeMap.from_dict(json.load(handle))


def _input_path(config: SessionConfig, system: str) -> Path | None:
    """The path the session names for this system's spike data."""
    if system == "blackrock":
        return config.blackrock.spike_file
    return config.neuropixels.bin_file


def _check_input_reachable(config: SessionConfig, system: str) -> None:
    """Fail fast, and loudly, when the recording's filesystem is not answering."""
    path = _input_path(config, system)
    if path is None:
        return                             # nothing named; the caller reports that

    probe = reachable.check_reachable(path)
    log.info("%s: %s", path, probe.summary())
    if probe.total_bytes and probe.estimated_seconds() > 60:
        log.warning(
            "%s reads at %.0f MB/s -- sorting streams the whole file, so expect "
            "about %.0f min of I/O. Sort beside the data, or stage it to the "
            "machine's cache_dir first.",
            path.name, probe.bytes_per_s / 1e6, probe.estimated_seconds() / 60,
        )


# ----------------------------------------------------------------------------
# Step 3: sorting
# ----------------------------------------------------------------------------


@dataclass
class SortSummary:
    """Where a sorting landed and how it went."""

    results_dir: Path
    n_units: int
    n_spikes: int
    fs: float
    binary: Path
    settings: dict[str, Any]

    def summary(self) -> str:
        return (
            f"{self.n_units} units, {self.n_spikes} spikes at {self.fs:g} Hz "
            f"-> {self.results_dir}"
        )


#: Arguments this verb decides for itself. A session naming one under ``kilosort:``
#: would either be ignored or collide with the value passed here, so it is refused
#: with the key that actually controls it.
_RESERVED = {
    "filename": "neuropixels.bin_file, or blackrock.spike_file",
    "data_dir": "neuropixels.bin_file names the binary outright",
    "file_object": "not used: both systems arrive as a binary",
    "results_dir": "derived from the session directory",
    "probe": "the run's .meta, or <system>.probe_file",
    "probe_name": "probe_file -- Kilosort ships no probe library",
    "n_chan_bin": "read from the .meta, or neuropixels.n_chan_bin",
    "fs": "read from the .meta, or neuropixels.sample_rate",
    "data_dtype": "both systems are int16",
    "device": "the machine profile's device:",
}


def _kilosort_arguments(
    config: SessionConfig, system: str, binary: Binary
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the session's ``kilosort:`` block into ``settings`` and keyword args.

    Kilosort takes its parameters in two places -- a ``settings`` dict and
    ``run_kilosort``'s own arguments (``do_CAR``, ``bad_channels``,
    ``invert_sign``) -- and which parameter lives where has moved between
    releases. So the split is read from the installed package rather than
    hardcoded, and a key in neither is refused here, naming both sets, instead of
    failing hours later or being silently dropped.
    """
    import inspect

    from kilosort import run_kilosort
    from kilosort.run_kilosort import DEFAULT_SETTINGS

    settings: dict[str, Any] = {"n_chan_bin": binary.n_chan_bin, "fs": binary.fs}
    kwargs: dict[str, Any] = {}
    accepted = set(inspect.signature(run_kilosort).parameters)

    for key, value in config.kilosort_for(system).items():
        if key in _RESERVED:
            raise ValueError(
                f"kilosort.{key} is set by the pipeline, not by the session file. "
                f"It comes from: {_RESERVED[key]}"
            )
        if key in DEFAULT_SETTINGS:
            settings[key] = value
        elif key in accepted:
            kwargs[key] = value
        else:
            raise ValueError(
                f"Kilosort has no parameter '{key}'.\n"
                f"  settings: {', '.join(sorted(DEFAULT_SETTINGS))}\n"
                f"  run_kilosort arguments: "
                f"{', '.join(sorted(accepted - {'settings', 'probe'}))}"
            )
    return settings, kwargs


def sort_with_kilosort(
    config: SessionConfig,
    system: str,
    dry_run: bool = False,
) -> Any | None:
    """Sort one stream with Kilosort4. Returns a :class:`SortSummary`.

    The whole of pipeline 1: resolve the channel map, name the binary, hand both
    to Kilosort's own ``run_kilosort``, and put the results where Phy and the
    export stages look for them. Nothing is loaded into memory here and no
    recording object is passed around -- Kilosort streams the file itself.

    Returns ``None`` without sorting when ``kilosort_on_<system>`` is false, the
    "extract the pulses and the LFP but do not sort" case.

    There is no preprocessing step before this. Kilosort does its own on every
    batch: it subtracts the median across channels when ``do_CAR`` is set (the
    default), highpasses at ``highpass_cutoff`` (300 Hz), then whitens and
    drift-corrects. Those are settings, named under ``kilosort:`` in the session
    and resolved by :meth:`SessionConfig.kilosort_for`.

    Sorting runs in the machine's ``cache_dir`` and the results are copied out,
    because Kilosort writes beside its results and a network share is both slow
    and rude to everyone else on it. ``dry_run`` reports what would be run --
    map, binary, settings -- and sorts nothing.
    """
    _check(system)
    if not getattr(config, f"sorts_{system}"):
        return None

    _check_input_reachable(config, system)

    probe = setup_probe(config, system)
    binary = _binary_for(config, system)
    final_dir = config.paths.sorted_for(system)

    if dry_run:
        return _dry_run(config, system, probe, binary, final_dir)

    settings, kwargs = _kilosort_arguments(config, system, binary)

    import torch
    from kilosort import run_kilosort

    work_dir = _work_dir(config, system) or final_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    log.info("sorting %s -> %s", binary.summary(), work_dir)

    try:
        _ops, spike_times, clusters, *_rest = run_kilosort(
            settings=settings,
            probe=probe,
            filename=binary.path,
            data_dtype="int16",
            results_dir=work_dir,
            device=torch.device(config.machine.device),
            **kwargs,
        )
        if work_dir != final_dir:
            _publish(work_dir, final_dir)
            _repoint_params(final_dir, binary)
    finally:
        if work_dir != final_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    result = SortSummary(
        results_dir=final_dir,
        n_units=int(np.unique(clusters).size),
        n_spikes=int(np.asarray(spike_times).size),
        fs=binary.fs,
        binary=binary.path,
        settings=settings,
    )
    _write_run_info(config, system, result, kwargs)

    # Blackrock only: record what turns these sample indices into NSP-clock
    # seconds, beside the sorting, while the file that produced them is known.
    if system == "blackrock":
        try:
            write_nsp_time_map(config, final_dir, config.blackrock.spike_file)
        except Exception as error:  # a bad clock must not discard a finished sort
            log.warning(
                "could not measure the NSP clock for %s (%s: %s); spike times will "
                "export in the sorter's own timebase",
                config.blackrock.spike_file, type(error).__name__, error,
            )
    return result


def _work_dir(config: SessionConfig, system: str) -> Path | None:
    """Fast local scratch for this stream's sort, or None to sort in place.

    Tagged with the probe when the session states a ``probes:`` list. The session
    label is the same for every probe of one config file, so without the tag two
    probes would share a work directory -- and therefore one ``temp.dat`` and one
    set of results, the second sort landing on top of the first before either was
    published.
    """
    if config.cache_dir is None:
        log.warning(
            "machine '%s' has no cache_dir, so Kilosort writes straight to the "
            "session directory -- slow on a share",
            config.machine.name,
        )
        return None
    name = f"{config.session}_{stream_label(system)}"
    if system == "neuropixels" and config.probe_tag is not None:
        name = f"{name}_{config.probe_tag}"
    return Path(config.cache_dir) / name


def _publish(work_dir: Path, final_dir: Path) -> None:
    """Copy a finished sorting out of the cache. ``.dat`` stays behind.

    The only ``.dat`` Kilosort writes is the whitened copy of the recording, made
    on request and the size of the input; the results themselves are the ``.npy``
    and ``.tsv`` files beside it.
    """
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        work_dir, final_dir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.dat")
    )


def _repoint_params(final_dir: Path, binary: Binary) -> None:
    """Point ``params.py`` at a file that still exists, consistently.

    Phy reads the raw traces through ``dat_path``, and Kilosort writes that line
    for the file *it* sorted in the cache -- either ``temp_wh.dat``, which
    :func:`_publish` skips because it is the size of the recording, or a path
    relative to a directory that has since been deleted. Either way it no longer
    resolves from where the results now live, so it is repointed at the binary
    the sorter actually read. A path that does resolve is left exactly as
    Kilosort wrote it.

    **Three lines move together, not one.** ``temp_wh.dat`` holds only the
    probe's channels while the raw binary holds every channel in the file, and
    ``n_channels_dat`` is the stride Phy walks the file with -- leave it behind
    and every trace is silently sheared rather than erroring. ``hp_filtered``
    goes with them: the raw file is not the filtered copy, so Phy applies its own
    150 Hz pass for display.

    Written with forward slashes, as Kilosort writes it
    (``kilosort/io.py``: ``dat_path.resolve().as_posix()``). Phy *executes*
    ``params.py``, so a Windows path in a single-quoted literal makes ``\\N`` a
    named-unicode escape and the GUI dies on ``exec`` before it opens.
    """
    params = final_dir / "params.py"
    if not params.exists():
        return

    lines = params.read_text(encoding="utf-8").splitlines(keepends=True)
    stated = _stated_dat_path(lines)
    if stated is not None and ((final_dir / stated).exists() or Path(stated).exists()):
        return

    replacements = {
        "dat_path": f"dat_path = '{binary.path.as_posix()}'\n",
        "n_channels_dat": f"n_channels_dat = {binary.n_chan_bin}\n",
        "hp_filtered": "hp_filtered = False\n",
    }
    out = [replacements.pop(_assigned_name(line), line) for line in lines]
    out.extend(replacements.values())   # keys this params.py did not carry
    params.write_text("".join(out), encoding="utf-8")
    log.info(
        "params.py: dat_path repointed at %s (%d channels)",
        binary.path,
        binary.n_chan_bin,
    )


def _assigned_name(line: str) -> str:
    """The name a ``params.py`` line assigns to, or ``""`` for anything else."""
    name, sep, _ = line.partition("=")
    return name.strip() if sep else ""


def _stated_dat_path(lines: list[str]) -> str | None:
    """The path ``params.py`` currently names, or None if it names none.

    Kilosort writes this field two ways -- a bare string for ``temp_wh.dat``, a
    *list* of the source files otherwise -- so the value is parsed rather than
    unquoted. A value that will not parse is a Windows path whose backslashes
    are unicode escapes: exactly the case being repaired, and one that must not
    raise here.
    """
    for line in lines:
        if _assigned_name(line) != "dat_path":
            continue
        value = line.partition("=")[2].strip()
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip("'\"")
        if isinstance(parsed, (list, tuple)):
            return str(parsed[0]) if parsed else None
        return None if parsed is None else str(parsed)
    return None


def _dry_run(
    config: SessionConfig,
    system: str,
    probe: dict,
    binary: Binary,
    final_dir: Path,
) -> dict[str, Any]:
    """What a real run would do, without doing it. Needs no GPU."""
    from ._probes.common import probe_summary

    report: dict[str, Any] = {
        "probe": probe_summary(probe),
        "binary": binary.summary(),
        "results_dir": final_dir,
    }
    try:
        settings, kwargs = _kilosort_arguments(config, system, binary)
        report["settings"] = settings
        report["run_kilosort arguments"] = kwargs
    except ImportError:
        # No Kilosort here, so which parameter goes where cannot be looked up.
        # Report the block unsplit rather than nothing: the paths above are the
        # half a laptop can actually check.
        report["kilosort (unsplit -- Kilosort is not installed)"] = config.kilosort_for(system)
    return report


def _waveform_fields(measured: dict | None) -> dict | None:
    """The waveform arrays in the shape the .mat writer wants, or None."""
    if measured is None:
        return None
    result = measured["result"]
    return {
        "mean": result.mean,
        "std": result.std,
        "unit_ids": result.unit_ids,
        "units": measured.get("mean_units", "ADC"),
        "window_ms": measured.get("window_ms", float("nan")),
        # Not transposed: Chunked takes the HDF5 shape, and MATLAB reverses it.
        # (width, n_spikes) on disk is what MATLAB reads as nSpikes x nSamp,
        # which is the orientation the online .nev container uses.
        "snippets": result.snippets if measured.get("snippets_kept") else None,
    }


def _write_summary(
    config: SessionConfig,
    system: str,
    out_dir: Path,
    timebase: str,
    unit_ids: np.ndarray,
    measured: dict | None,
) -> Path:
    """The manifest that makes the bundle self-describing.

    What ran, on what, and where everything it used came from -- so a folder
    copied to another machine still says which recording produced it and which
    time map its spike times are on.
    """
    results_dir = config.paths.sorted_for(system)
    run_info_path = results_dir / "run_info.json"
    run_info = {}
    if run_info_path.exists():
        with open(run_info_path, "r", encoding="utf-8") as handle:
            run_info = json.load(handle)

    aligned_dir = config.paths.aligned
    payload = {
        "session": config.session,
        "system": system,
        "stream": stream_label(system),
        "sorter": config.sorter,
        "timebase": timebase,
        "n_units_exported": int(np.asarray(unit_ids).size),
        "waveforms": (
            {
                "n_spikes": measured["n_spikes"],
                "n_available": measured.get("n_available"),
                "n_dropped": measured["n_dropped"],
                "snippets_kept": measured["snippets_kept"],
                "units": measured.get("mean_units"),
                "highpass_hz": measured.get("highpass_hz"),
                "timebase": measured.get("timebase"),
            }
            if measured
            else "not measured: the sorted binary was not reachable"
        ),
        "sorting": run_info,
        "inputs": {
            "sorted_dir": str(results_dir),
            "aligned_dir": str(aligned_dir) if aligned_dir.exists() else None,
            "time_map": str(aligned_dir / "time_map.json")
            if (aligned_dir / "time_map.json").exists()
            else None,
        },
    }
    path = out_dir / "sorting_summary_info.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def _write_run_info(
    config: SessionConfig,
    system: str,
    result: SortSummary,
    kwargs: dict[str, Any],
) -> None:
    """Record what produced this sorting, beside the sorting itself."""
    payload = {
        "session": config.session,
        "machine": config.machine.name,
        "device": config.machine.device,
        "system": system,
        # Which stream, not just which system: a run can hold several probes.
        "stream": stream_label(system),
        "binary": str(result.binary),
        "n_units": result.n_units,
        "n_spikes": result.n_spikes,
        "fs": result.fs,
        "settings": result.settings,
        "run_kilosort_arguments": kwargs,
    }
    with open(result.results_dir / "run_info.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


# ----------------------------------------------------------------------------
# Step 4: sync pulses and LFP
# ----------------------------------------------------------------------------


def extract_sync(config: SessionConfig, system: str) -> Any | None:
    """Extract one system's sync edges into its own folder. Returns a report.

    Runs CatGT first where the machine has it, always runs the NumPy detector,
    and compares them -- that agreement is what lets the fallback be trusted on a
    machine without CatGT. CatGT's arguments come from the AP binary's own
    SpikeGLX filename, so the session names only ``bin_file``.

    Edges land beside the sorting, in ``<system>_dir/kilosort4/``, which is
    computable before anything is sorted.
    """
    _check(system)
    if not config.has_data(system):
        return None

    if system == "neuropixels":
        return extract.extract_neuropixels_edges(config)
    return extract.extract_blackrock_edges(config)


def extract_lfp(config: SessionConfig, system: str) -> Path | None:
    """Export one system's LFP. Returns the path written, or ``None``.

    Runs when the session says ``export_lfp: true``, and with the stride that
    session names in ``lfp_decimate``. Blackrock always returns ``None``: Central
    already saves those separately.

    Bulk I/O, no compute -- the LF band is ~7 GB/hour at 385 channels and
    ``lfp_decimate: 1`` writes an output the size of the input, so run this where
    the recording lives. Decimation is a plain stride with no anti-alias filter;
    the LF band is hardware-limited to ~500 Hz at 2500 Hz sampling, so 2 is safe
    and more aliases.
    """
    _check(system)
    if system == "blackrock" or not config.has_data("neuropixels"):
        return None
    if not config.export_lfp:
        return None

    decimate = config.lfp_decimate

    from ._io import spikeglx

    lf_file = _lf_binary(config)
    if lf_file is None:
        return None

    # Into the stream's own bundle, beside the sorting it will be analysed with.
    # sorted_for is computable whether or not sorting ran, so a session that only
    # exports the LFP still has one predictable home for it.
    path = spikeglx.export_lfp(
        lf_file, config.paths.export_for("neuropixels"), decimate=decimate
    )
    log.info("wrote %s%s", path, f" (decimated {decimate}x)" if decimate > 1 else "")

    # If a map was already fitted -- a re-export after alignment -- put it on the
    # Blackrock clock now rather than making the caller run time_remapping again.
    # Otherwise time_remapping stamps it when it runs.
    mapping = _fitted_map(config)
    if mapping is not None:
        stamp_lfp_timebase(config, mapping)
    return path


def _lf_binary(config: SessionConfig) -> Path | None:
    """The LF binary to export: the one named, else the AP file's own sibling.

    ``lf_file`` exists for a recording whose LF band is not where SpikeGLX would
    have put it. For an ordinary run it is unset and the file is found beside the
    AP binary, so the session states one path rather than two that must agree.
    """
    from ._io import spikeglx

    npx = config.neuropixels
    if npx.lf_file is not None:
        return npx.lf_file
    layout = spikeglx.run_layout(npx.bin_file) if npx.bin_file is not None else None
    return layout.sibling("lf") if layout is not None else None


def _fitted_map(config: SessionConfig) -> align.LinearMap | None:
    """This probe's fitted time map, if ``time_remapping`` has already run."""
    map_path = config.paths.aligned / "time_map.json"
    if not map_path.exists():
        return None
    with open(map_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return align.LinearMap(
        slope=float(payload["slope"]),
        intercept=float(payload["intercept"]),
        n_points=int(payload.get("n_points", 0)),
        residuals_s=np.empty(0),
    )


def stamp_lfp_timebase(
    config: SessionConfig, mapping: align.LinearMap
) -> Path | None:
    """Put the fitted map into an LFP export that was written before it existed.

    The LF band is extracted with the rest of the extraction, hours before there
    is a time map -- so it goes out on the probe's own clock. The LF and AP
    streams of a probe share that clock, so the fit from the AP stream's sync
    edges applies to it unchanged, and this fills in the Blackrock axis
    afterwards.

    Only the small fields are rewritten. The samples are gigabytes and are never
    re-read: see :func:`spikesorting._io.matlab.update_mat`.

    Returns the file it stamped, or ``None`` when this probe has no LFP export.
    """
    from ._io.matlab import update_mat

    lfp_dir = config.paths.export_for("neuropixels")
    exports = sorted(lfp_dir.glob("*.lfp.mat")) if lfp_dir.exists() else []
    if not exports:
        return None

    import h5py

    stamped = None
    for path in exports:
        with h5py.File(path, "r") as handle:
            fs = float(np.asarray(handle["lfp/fs"][()]).ravel()[0])
        # t_blackrock(k) = slope * (k / fs) + intercept, another uniform axis:
        # rate fs / slope, origin at the intercept.
        update_mat(
            path,
            {
                "lfp": {
                    "timebase": "blackrock",
                    "fs_blackrock": fs / mapping.slope,
                    "t0_blackrock": mapping.intercept,
                    "blackrock_slope": mapping.slope,
                    "blackrock_intercept": mapping.intercept,
                }
            },
        )
        log.info("stamped the Blackrock timebase into %s", path.name)
        stamped = path
    return stamped


def _load_edges(config: SessionConfig, name: str) -> np.ndarray | None:
    """One edge file, from the folder beside the recording it came from.

    ``EDGE_SYSTEM`` says which system owns the name, so a reader never has to
    guess which directory to look in from the name prefix.
    """
    system = extract.EDGE_SYSTEM[name]
    if config.paths.dir_for(system) is None:
        return None
    path = config.paths.sync_for(system) / f"{name}.txt"
    return catgt.read_edge_file(path) if path.exists() else None


# ----------------------------------------------------------------------------
# Step 5: onto the Blackrock timebase
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


def time_remapping(
    config: SessionConfig, system: str = "neuropixels"
) -> TimeMap | None:
    """Map one stream's spike times onto Blackrock time. Returns a :class:`TimeMap`.

    Blackrock is the reference timebase, so it is not an argument -- this maps
    onto it, never the reverse.

    One map per probe. Each probe has its own oscillator and its own SY word, so
    one probe's fit says nothing about another's -- which is why a session
    covering several writes each into its own ``<blackrock_dir>/<sorter>/imec<n>/``
    rather than one shared folder. Nothing here knows about that: it is handed a
    config already projected onto one probe, and ``paths.aligned`` is where that
    probe's map goes.

    Coarse offset from the 14 s coded bursts first: their onsets alone are
    periodic and ambiguous, so the full pulse trains are matched on the coded
    intra-burst pattern to pick the right cycle. Then both 1 Hz trains are
    trimmed to their overlap and a linear map is fitted -- by TPrime where it is
    installed, otherwise by least squares on the same matched edges. Both paths
    write the same output file.

    Returns ``None`` when only one system is declared, when asked for Blackrock itself
    (there is nothing to map the reference onto), or when the edge files are
    missing.
    """
    _check(system)
    if not config.aligns_systems or system == REFERENCE_SYSTEM:
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
        stamp_lfp_timebase(config, mapping)
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
    stamp_lfp_timebase(config, mapping)
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


def validate_remapping(
    config: SessionConfig, figures: bool = True
) -> Any | None:
    """Check one probe's map against the 14 s burst onsets. Returns a ``ValidationReport``.

    A real check precisely because the bursts were **held out** of the 1 Hz fit.
    Uncorrected clock drift of tens of ppm is 72 ms/hour at 20 ppm and fails the
    1 ms tolerance, which is the point.

    Returns ``None`` when only one system is declared, no map has been fitted, or
    neither system recorded bursts -- in which case the 1 Hz fit residuals in
    ``time_map.json`` are the only quality measure, and they are not held-out.
    """
    from ._plots import summary as plots

    if not config.aligns_systems:
        return None

    aligned_dir = config.paths.aligned
    map_path = aligned_dir / "time_map.json"
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

    with open(aligned_dir / "validation.json", "w", encoding="utf-8") as handle:
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
        path = (
            config.paths.figures_for("neuropixels") / "alignment_residuals.png"
        )
        plots.save_figure(axes.figure, path)
        log.info("wrote %s", path)

    return report


# ----------------------------------------------------------------------------
# Step 6a: waveforms from the raw samples
# ----------------------------------------------------------------------------


def _resolve_spike_seconds(
    config: SessionConfig, system: str, phy
) -> tuple[np.ndarray | None, str, object | None]:
    """``(seconds_per_spike, timebase, nsp_time_map)`` for one sorted stream.

    Flat and parallel to ``spike_times.npy``, so a caller can split it per unit
    or index it down to a subset of spikes.

    **One place, because two files in the same folder must agree.**
    ``sorted_spikes.mat`` and ``waveforms.mat`` both carry spike times, and a
    reader has every right to assume they are on the same axis. Deriving that
    axis twice is how they would quietly stop being.

    Order: the cross-system alignment if it ran; otherwise, for Blackrock, the
    NSP clock measured from the file's own timestamps -- which is not the axis
    Kilosort's sample indices imply, and is the one the .nev markers and eye
    traces are already on. Neither present leaves times in the sorter's timebase.
    """
    results_dir = config.paths.sorted_for(system)
    aligned_path = (
        config.paths.aligned / f"{system}_spike_seconds_blackrock.npy"
    )
    if aligned_path.exists():
        aligned = np.load(aligned_path)
        if aligned.size == phy.spike_samples.size:
            return aligned, "blackrock", None
        log.warning(
            "aligned times (%d) do not match spike count (%d); exporting in the "
            "sorter timebase instead",
            aligned.size, phy.spike_samples.size,
        )
    elif system == "blackrock":
        time_map = read_nsp_time_map(results_dir)
        if time_map is not None:
            log.info("NSP clock: %s", time_map.summary())
            return time_map.apply(phy.spike_samples), "nsp", time_map
    return None, "sorter", None


def _binary_blocks(
    path: Path,
    n_chan: int,
    overlap: int,
    block_samples: int = 2_000_000,
    regions: list[tuple[int, int]] | None = None,
):
    """Walk a flat int16 recording in order, yielding ``(start, samples)``.

    Without ``regions``, consecutive blocks overlap by ``overlap`` samples so a
    spike whose window straddles a boundary is still wholly inside one of them.

    With ``regions`` -- the merged ranges from ``_export.waveforms.snippet_regions``
    -- only those are yielded, still in increasing offset order. That is the
    difference between reading the whole file and reading the parts that hold a
    spike being measured: filtering forces a block to be *materialised*, so
    walking the file in 2M-sample steps would turn today's sparse read into a
    full one. Sequential order is kept either way, because it is what a share
    rewards.

    Row ranges of a memmap are contiguous reads; it is *column* slices that turn
    into scattered page faults, which is the distinction :mod:`._io.spikeglx` was
    rewritten around.
    """
    data = np.memmap(path, dtype=np.int16, mode="r").reshape(-1, n_chan)
    if regions is not None:
        for start, stop in regions:
            yield start, data[start:stop]
        return
    for start in range(0, data.shape[0], block_samples):
        yield start, data[start : start + block_samples + overlap]


def _uv_per_digit(config: SessionConfig, system: str, n_chan: int) -> tuple[np.ndarray, str]:
    """Per-channel microvolts per ADC unit, and a note on where it came from.

    Blackrock records the gain per channel and neo reports it, so those snippets
    come out in real microvolts. SpikeGLX keeps the neural gain in ``~imroTbl``,
    which nothing here parses, so those come back NaN rather than 1.0 -- a NaN
    propagates loudly through any scaling, where a fabricated unit gain would
    quietly produce plausible, wrong amplitudes.
    """
    if system == "blackrock" and config.blackrock.spike_file is not None:
        from ._io import blackrock

        try:
            reader = blackrock.open_reader(
                config.blackrock.spike_file,
                gap_tolerance_ms=config.blackrock.gap_tolerance_ms,
            )
            channels = reader.header["signal_channels"]
            gains = np.asarray(channels["gain"], dtype=np.float64)
            units = {str(u) for u in channels["units"]}
        except Exception as error:
            # The snippets come from the transformed binary, which is present;
            # only the scale is lost. Falling over here would throw away a
            # finished pass over the recording for a header we cannot read.
            log.warning(
                "could not read the gain from %s (%s: %s); snippets stay in ADC units",
                config.blackrock.spike_file, type(error).__name__, error,
            )
        else:
            if gains.size >= n_chan:
                return (
                    gains[:n_chan],
                    f"gain per channel from the .ns6 header, units {sorted(units)}",
                )
    return (
        np.full(n_chan, np.nan),
        "gain unknown -- SpikeGLX keeps it in ~imroTbl, which is not parsed here, "
        "and a Blackrock header that will not open says nothing either; snippets "
        "stay raw int16 and the mean is in those units",
    )


def export_waveforms(
    config: SessionConfig, system: str = "blackrock"
) -> dict | None:
    """Measure every spike's waveform from the binary the sorter read.

    Kilosort saves templates, not snippets: ``templates.npy`` is the shape it
    *fitted*, in whitened units, so nothing else in the export has a real
    amplitude. These are measured from the raw samples, and their per-unit mean
    does.

    **The mean is always computed; only the snippets are optional.** Reading the
    recording is the expense and it is the same single pass either way, so
    ``waveforms.export_snippets`` decides whether the ~120 MB per million
    snippets is *kept*, not whether the recording is read.

    **How much of the recording is read is bounded by the spikes, not the file.**
    ``waveforms.max_spikes`` caps the spikes per unit -- chosen uniformly over the
    session, so the subset spans it rather than clustering where the unit was
    busiest -- and only the ranges holding those spikes are read. Everything else
    about the cut lives in the same per-system ``waveforms:`` block: the window,
    the high-pass, and the pad that keeps the filter's transient out of the kept
    samples.

    Returns ``None`` when there is no sorting to read, or when the binary the
    sorting was made from is not reachable -- the rest of the export is still
    worth having, so that is a warning rather than a failure.
    """
    _check(system)
    if not config.has_data(system):
        return None

    from ._export.curated import load_phy_results, select_units
    from ._export.final import best_channel
    from ._export.waveforms import accumulate, plan_snippets, snippet_regions
    from ._io.matlab import write_mat

    results_dir = config.paths.sorted_for(system)
    info_path = results_dir / "run_info.json"
    if not (results_dir / "spike_times.npy").exists() or not info_path.exists():
        log.info("no sorting with a run_info.json in %s", results_dir)
        return None

    with open(info_path, "r", encoding="utf-8") as handle:
        run_info = json.load(handle)
    binary = Path(run_info["binary"])
    n_chan = int(run_info["settings"]["n_chan_bin"])
    fs = float(run_info["settings"]["fs"])
    if not binary.exists():
        log.warning(
            "the binary this sorting read is gone (%s), so there is nothing to "
            "measure a waveform from; the rest of the export is unaffected",
            binary,
        )
        return None

    phy = load_phy_results(results_dir)
    unit_ids = select_units(phy, tuple(config.export_groups))
    channel_for_unit = {
        int(u): best_channel(phy, int(u))
        for u in unit_ids
        if best_channel(phy, int(u)) is not None
    }
    if not channel_for_unit:
        log.warning("no unit has a template to pick a channel from; nothing to cut")
        return None

    wf = config.waveforms_for(system)
    width = max(2, int(round(wf.window_ms * fs / 1000.0)))
    before = width // 2
    after = width - before
    n_samples = binary.stat().st_size // (2 * n_chan)
    # Read either side of each snippet and throw it away, so the samples that are
    # kept carry no filter transient. Nothing to pad when nothing is filtered.
    pad = int(round(wf.filter_pad_ms * fs / 1000.0)) if wf.highpass_hz else 0

    plan = plan_snippets(
        phy.spike_samples,
        phy.spike_clusters,
        channel_for_unit,
        n_samples,
        before,
        after,
        max_per_unit=wf.max_spikes,
        margin=pad,
    )
    regions = snippet_regions(plan, pad)
    scale, scale_note = _uv_per_digit(config, system, n_chan)
    # An unknown gain must not destroy the mean: the shape and the relative
    # amplitude are still worth having, and on SpikeGLX the gain is never known
    # here. uv_per_digit stays NaN so nobody mistakes ADC units for microvolts,
    # and mean_units says which one the mean is in.
    known = not np.isnan(scale).all()
    mean_units = "uV" if known else "ADC"
    band = (
        f"high-passed at {wf.highpass_hz:g} Hz" if wf.highpass_hz else "unfiltered"
    )
    log.info(
        "cutting %d of %d spikes (%d samples each) from %s in %d range(s); "
        "%s, %s, mean in %s",
        plan.n_spikes, plan.n_available, width, binary.name, len(regions),
        band, scale_note, mean_units,
    )
    result = accumulate(
        plan,
        _binary_blocks(binary, n_chan, width, regions=regions),
        uv_per_digit=scale if known else 1.0,
        highpass_hz=wf.highpass_hz,
        fs=fs,
        highpass_order=wf.highpass_order,
        pad=pad,
    )
    for note in result.notes:
        log.warning("%s", note)

    # Which spikes these were, on the axis the .nev markers and eye traces are
    # already on -- so a snippet can be attributed to the task it happened in.
    # Same resolver export_results uses, so the two .mat files in this folder
    # cannot end up on different clocks.
    seconds, timebase, _ = _resolve_spike_seconds(config, system, phy)
    spike_time_s = (
        seconds[plan.spike_index].astype(np.float64)
        if seconds is not None
        else plan.sample.astype(np.float64) / fs
    )

    out_dir = results_dir / "export"
    out_path = out_dir / f"{config.session}_waveforms.mat"
    write_mat(
        out_path,
        {
            "waveforms": {
                **({"snippet": result.snippets} if wf.export_snippets else {}),
                "unit_id": plan.unit_id.astype(np.float64),
                "channel": plan.channel.astype(np.float64),
                "spike_sample": plan.sample.astype(np.float64),
                "spike_index": plan.spike_index.astype(np.float64),
                "spike_time_s": spike_time_s,
                "timebase": timebase,
                "mean": result.mean,
                "std": result.std,
                "mean_unit_id": result.unit_ids.astype(np.float64),
                "n_spikes": result.n_per_unit.astype(np.float64),
                "uv_per_digit": scale,
                "mean_units": mean_units,
                "fs": fs,
                "window_ms": float(wf.window_ms),
                "samples_before": float(before),
                "n_dropped": float(plan.n_dropped),
                "n_available": float(plan.n_available),
                "max_spikes_per_unit": (
                    float(wf.max_spikes) if wf.max_spikes is not None else np.nan
                ),
                "highpass_hz": (
                    float(wf.highpass_hz) if wf.highpass_hz is not None else np.nan
                ),
                "highpass_order": float(wf.highpass_order),
                "filter_pad_ms": float(wf.filter_pad_ms) if pad else 0.0,
                "source": str(binary),
                "kept_snippets": float(bool(wf.export_snippets)),
                "note": (
                    "snippet is int16, nSamples x nSpikes in MATLAB, and is "
                    "present only when waveforms.export_snippets is true; mean "
                    f"and std are nSamples x nUnits in {mean_units}, measured "
                    "from the recording rather than from Kilosort's templates. "
                    f"Signal was {band}. "
                    + (
                        f"Measured from {plan.n_spikes} of {plan.n_available} "
                        "spikes, sampled uniformly over the recording rather "
                        "than over the spike list, so the subset spans the "
                        "session. "
                        if wf.max_spikes is not None
                        else "Measured from every spike. "
                    )
                    + f"spike_time_s is on the {timebase} timebase. {scale_note}"
                ),
            }
        },
    )
    log.info("wrote %s", out_path)
    return {
        "path": out_path,
        "n_spikes": plan.n_spikes,
        "n_available": plan.n_available,
        "n_units": int(result.unit_ids.size),
        "n_dropped": plan.n_dropped,
        "snippets_kept": bool(wf.export_snippets),
        "mean_units": mean_units,
        "window_ms": float(wf.window_ms),
        "highpass_hz": wf.highpass_hz,
        "timebase": timebase,
        "result": result,
    }


# ----------------------------------------------------------------------------
# Step 6: export
# ----------------------------------------------------------------------------


def export_results(
    config: SessionConfig,
    system: str = "neuropixels",
    max_unit_figures: int = 40,
) -> dict | None:
    """Export metrics, figures and the final bundle for a sorted folder.

    Which unit labels to keep and whether to draw figures are the session's
    ``export_groups`` and ``export_figures``.

    Uses aligned spike times when ``time_remapping`` has written them, and
    records which timebase it used. Reads Phy's ``cluster_group.tsv`` where it
    exists, so curated labels override Kilosort's own.

    Returns ``None`` when that system has no sorting output.
    """
    _check(system)
    from ._export import final, metrics
    from ._export.curated import load_phy_results, select_units
    from ._plots import summary as plots

    if not config.has_data(system):
        return None

    groups = tuple(config.export_groups)
    figures = config.export_figures

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

    seconds, timebase, time_map = _resolve_spike_seconds(config, system, phy)
    spike_times_s: dict[int, np.ndarray] | None = None
    if seconds is not None:
        spike_times_s = {int(uid): seconds[phy.spike_clusters == uid] for uid in unit_ids}
    log.info("timebase: %s", timebase)

    # One bundle per stream, so an analysis gets a folder rather than a tour of
    # three trees. aligned/ stays where it is: the time map is cross-system.
    out_dir = config.paths.export_for(system)
    paths = final.export_units(
        out_dir,
        phy,
        unit_ids=unit_ids,
        spike_times_s=spike_times_s,
        timebase=timebase,
        provenance={
            "session": config.session,
            "system": system,
            "stream": stream_label(system),
            "machine": config.machine.name,
            **({"nsp_time_map": time_map.to_dict()} if time_map is not None else {}),
        },
    )
    log.info("exported %d units -> %s", unit_ids.size, out_dir)

    # The same spikes in the format the lab's online-spike code already reads,
    # plus the measured waveform when the recording was reachable.
    measured = export_waveforms(config, system)
    table = final.build_unit_table(phy, unit_ids, spike_times_s)
    mat_path = final.export_sorted_spikes_mat(
        out_dir,
        phy,
        unit_ids=unit_ids,
        spike_times_s=spike_times_s,
        timebase=timebase,
        table=table,
        waveforms=_waveform_fields(measured),
        provenance={
            "session": config.session,
            "system": system,
            "stream": stream_label(system),
        },
        filename=f"{config.session}_sorted_spikes.mat",
    )
    log.info("wrote %s", mat_path.name)
    _write_summary(config, system, out_dir, timebase, unit_ids, measured)

    if figures and unit_ids.size:
        figure_dir = config.paths.figures_for(system)
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
