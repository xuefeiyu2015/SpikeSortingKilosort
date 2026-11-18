"""The pipeline: one verb per thing you do, in the order you do it.

This is the only file you need to read to understand the workflow. Everything
below it -- ``_io``, ``_probes``, ``_sync``, ``_export``, ``_plots`` -- is
machinery these verbs call.

Every per-system verb takes ``system``: ``"neuropixels"`` or ``"blackrock"``, so
the two acquisition systems read identically::

    config = load_session_config("configs/athos.yaml", "windows_rig")

    preprocess(config, "blackrock")                # pipeline 1
    sort_with_kilosort(config, "blackrock")

    extract_sync(config, "blackrock")              # pipeline 2
    extract_lfp(config, "neuropixels")
    time_remapping(config, "neuropixels")
    validate_remapping(config)
    export_results(config, "neuropixels")

Sorting is Kilosort4's own ``run_kilosort`` on a flat int16 binary that has
already been preprocessed -- band-passed and re-referenced by CatGT or
SpikeInterface, as the session's ``preprocess:`` block says -- so Kilosort only
whitens. The preprocessed file is written once, beside the recording, and
reused while it still matches the session.

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
from ._preprocess import provenance
from ._sync import align, burst, catgt, extract

log = logging.getLogger("spikesorting")

__all__ = [
    "SYSTEMS",
    "REFERENCE_SYSTEM",
    "skip_reason",
    "load_session_config",
    "setup_probe",
    "preprocess",
    "sort_with_kilosort",
    "extract_sync",
    "extract_lfp",
    "extract_analog",
    "time_remapping",
    "validate_remapping",
    "export_results",
    "export_waveforms",
    "record_timing",
    "timing_table",
    "stamp_lfp_timebase",
    "stamp_analog_timebase",
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
        "preprocess",
        "sort_with_kilosort",
        "extract_sync",
        "extract_lfp",
        "extract_analog",
        "export_results",
        "export_waveforms",
        "record_timing",
        "exam_sync",
    }:
        if not config.has_data(system):
            return f"no {system} paths declared in the session"

    if name == "exam_sync" and system == "blackrock":
        if config.blackrock.sync_file is None:
            return "no blackrock.sync_file in the session: no Blackrock pulses to examine"
        if not config.has_data("neuropixels"):
            return "no Neuropixels in this session: Blackrock's pulses align nothing"

    if name in {"preprocess", "sort_with_kilosort"} and system is not None:
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

    if name == "extract_analog":
        if system == "blackrock":
            return "Blackrock's analog inputs are already on its own clock"
        if not config.export_analog:
            return "export_analog is false"
        if _onebox_binary(config) is None:
            return "no OneBox (.obx.bin) in this SpikeGLX run"

    if name in {"time_remapping", "validate_remapping"} and not config.aligns_systems:
        # Say what it means for the data, not just that a stage was skipped: a
        # session with no Blackrock ends with times on the probe's own clock.
        if config.has_data("neuropixels"):
            return (
                "no Blackrock in this session -- synchronization skipped; "
                "times stay on the probe clock (timebase: sorter)"
            )
        if config.has_data("blackrock"):
            return (
                "no Neuropixels in this session -- nothing to synchronize; "
                "Blackrock is already the reference clock"
            )
        return "neither system declares a recording -- nothing to synchronize"

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
    from ._io import spikeglx
    from ._probes import neuropixels as np_probes

    bin_file = config.neuropixels.bin_file
    if bin_file is None or not spikeglx.meta_path_for(bin_file).exists():
        return None
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

    ``n_chan_bin``, ``sample_rate`` and ``has_sy_word`` stand in for the
    ``.meta`` when there is none -- which is the demo file's situation, and the
    only reason those keys exist.
    """
    from ._io import spikeglx

    npx = config.neuropixels
    return spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate,
                                npx.has_sy_word)


# ----------------------------------------------------------------------------
# Step 2: preprocessing, and the binary Kilosort reads
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Binary:
    """A flat int16 recording file, and the two facts Kilosort needs about it."""

    path: Path
    n_chan_bin: int
    fs: float
    #: Whether the file was band-passed before Kilosort read it. Phy filters the
    #: traces it shows unless ``params.py`` says this, and filtering twice
    #: would show a band nobody sorted.
    filtered: bool = False

    def summary(self) -> str:
        return f"{self.path} ({self.n_chan_bin} channels at {self.fs:g} Hz)"


def preprocess(config: SessionConfig, system: str) -> Binary | None:
    """Write the binary Kilosort reads: the recording, preprocessed. Returns it.

    What is done is the session's ``<system>.preprocess:`` block -- CatGT
    (Neuropixels only) or SpikeInterface, band-pass, common reference, CatGT's
    ``gfix``, phase shift, optionally motion correction -- and Kilosort then only
    whitens. The file is written to ``<system>_dir/preprocessed/`` with a record
    of what made it (``preprocess.json``), and is reused for as long as that
    record still matches the session and the source file; anything else makes it
    again. It is the size of the recording.

    **CatGT does not fall back.** When it cannot run here -- no ``catgt_dir`` on
    this machine, a binary not named the SpikeGLX way, CatGT failing -- this
    raises, naming ``method: spikeinterface``. Which preprocessing a sorting had
    is written in the session, never decided by what happened to be installed.

    With CatGT the same pass extracts the sync edges ``extract_sync`` needs, so
    the AP binary is read once for both; ``extract_sync`` picks them up from the
    record.

    Returns ``None`` when ``kilosort_on_<system>`` is false: nothing will be
    sorted, so there is nothing to preprocess for. :func:`sort_with_kilosort`
    calls this itself, so running it first only splits the work into two steps.
    """
    _check(system)
    if not getattr(config, f"sorts_{system}"):
        return None
    _check_input_reachable(config, system)
    binary = _binary_for(config, system)
    if system == "neuropixels":
        checked = _preprocessing_sync_check(config)
        for problem in (checked or {}).get("problems", []):
            log.warning("sync pulses: %s -- synchronization will not be able to use it; "
                        "run_sorting_pipeline.py --sync-exam checks it on its own", problem)
    return binary


@dataclass
class SyncExam:
    """What :func:`exam_sync` found: each sync train, counted against the recording."""

    system: str
    checks: list[dict[str, Any]]
    notes: list[str]
    report_path: Path | None = None

    @property
    def ok(self) -> bool:
        """Every train there in full. Not ``passed``: one system's missing pulse
        must not stop the runner before the other system is examined."""
        return all(c["status"] == "ok" for c in self.checks)

    def summary(self) -> str:
        lines = []
        for c in self.checks:
            found = "-" if c["found"] is None else f"{c['found']:,}"
            lines.append(f"{c['status'].upper():13} {c['kind']:10} {c['name']:16} "
                         f"{found:>7} of ~{c['expected']:,} {c['unit']}  <- {c['stream']}")
        lines.extend(self.notes)
        if self.report_path is not None:
            lines.append(f"report: {self.report_path}")
        return "\n       ".join(lines)


#: Where the sync exam writes: its own folder, read by nothing else.
SYNC_EXAM_DIR = "sync_exam"


def exam_sync(config: SessionConfig, system: str) -> SyncExam | None:
    """Extract one system's sync pulses on their own and say whether they are there.

    A check to run before (or instead of) a whole pipeline, e.g. right after a
    recording: ``run_sorting_pipeline.py --sync-exam``. Neuropixels runs CatGT's
    edge extraction alone -- the probe's 1 Hz, the OneBox's 1 Hz and the 14 s
    burst, with the session's extractors -- and Blackrock reads the ``.ns5``'s
    two sync inputs with the detector ``extract_sync`` uses (CatGT cannot read a
    ``.ns5``). Each train is counted against the recording's length.

    **Changes nothing the pipeline uses.** It writes to
    ``<neuropixels_dir>/<export_folder_name>/sync_exam/`` only: the edge files the map is
    fitted from, the map and any accepted verdict stay as they are. Returns
    ``None`` when the session declares no such system.
    """
    from ._sync import edge_check

    _check(system)
    if skip_reason(config, exam_sync, system):
        return None
    if system == "neuropixels":
        config.paths.require_dir("neuropixels")
        _check_input_reachable(config, "neuropixels")
        folder = config.paths.sorted_for("neuropixels") / SYNC_EXAM_DIR
        command = extract.run_catgt_extraction_into(config, folder / "catgt")
        trains = extract.catgt_sync_trains(config, folder / "catgt")
        stream = _neuropixels_stream(config)
        duration, notes, folders = stream.n_samples / stream.fs, [command], [folder]
    else:
        source = config.blackrock.sync_file
        log.info("%s: %s", source, reachable.check_reachable(source).summary())
        trains, duration, notes = extract.blackrock_sync_trains(config)
        folders = [c.paths.sorted_for("neuropixels") / SYNC_EXAM_DIR
                   for _, c in config.per_probe()]
    checks = edge_check.check_sync_edges(trains, duration, config.sync_period_s,
                                         config.burst_interval_s)
    for problem in edge_check.sync_problems(checks):
        log.warning("sync pulses: %s", problem)
    payload = {"system": system, "duration_s": duration, "checks": checks, "notes": notes,
               "ok": all(c["status"] == "ok" for c in checks)}
    path = None
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"sync_exam_{system}.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return SyncExam(system, checks, notes, path)


def _preprocessing_sync_check(config: SessionConfig) -> dict[str, Any] | None:
    """Were the 1 Hz and the 14 s burst in the edges the CatGT pass extracted?

    None unless this probe's binary came from CatGT (only CatGT extracts edges
    while it preprocesses). Reads the record and the small edge files it points
    at; the result goes into ``run_info.json["sync_edges"]`` and the sort's
    summary, so a missing pulse is known at sorting time, not at synchronization.
    """
    from ._sync import edge_check

    record = provenance.read(config.paths.preprocessed_for("neuropixels")) or {}
    if record.get("engine") != "catgt" or not record.get("edge_dir"):
        return None
    try:
        duration = float(record["n_samples"]) / float(record["fs"])
        trains = extract.catgt_sync_trains(config, record["edge_dir"])
    except (KeyError, TypeError, ValueError, OSError) as error:
        log.info("sync pulses not checked: %s", error)
        return None
    checks = edge_check.check_sync_edges(trains, duration, config.sync_period_s,
                                         config.burst_interval_s)
    return {"source": "CatGT preprocessing pass", "edge_dir": record["edge_dir"],
            "duration_s": duration, "checks": checks,
            "problems": edge_check.sync_problems(checks)}


def _job_kwargs(config: SessionConfig) -> dict[str, Any]:
    """How SpikeInterface chunks the work on this machine."""
    return {"n_jobs": config.machine.n_jobs, "chunk_duration": "1s", "progress_bar": True}


def _preprocess_expected(config: SessionConfig, system: str) -> dict[str, Any]:
    """What a preprocessed binary for this session has to have been made from.

    Everything that changes the output is here -- the ``preprocess:`` block, and
    the reader settings that decide what the recording *is* (Blackrock's gap
    tolerance, a bare binary's stated channel count) -- plus the source file's
    fingerprint, so a re-recorded or re-copied file is not mistaken for the old.
    """
    from dataclasses import asdict

    spec = config.preprocess_for(system)
    settings: dict[str, Any] = asdict(spec)
    if spec.method == "spikeinterface" and spec.car_mode:
        settings["reference_rows"] = _reference_rows(config, system)
    if spec.method == "catgt":
        # Only when there is something to exclude: a binary made before this
        # key existed, with no exclusions, is still the same binary.
        excluded = _catgt_excluded(config)
        if excluded:
            settings["catgt_excluded"] = excluded
    if system == "blackrock":
        brk = config.blackrock
        settings["reader"] = {
            "gap_tolerance_ms": brk.gap_tolerance_ms,
            "stream_id": brk.stream_id,
            "allow_segments": brk.allow_segments,
        }
    else:
        npx = config.neuropixels
        settings["reader"] = {"n_chan_bin": npx.n_chan_bin, "sample_rate": npx.sample_rate}
        if npx.has_sy_word is not None:
            settings["reader"]["has_sy_word"] = npx.has_sy_word
    return {
        "engine": spec.method,
        "settings": settings,
        "source": provenance.source_fingerprint(_input_path(config, system)),
    }


def _reference_rows(config: SessionConfig, system: str) -> list[int]:
    """The rows the common median is taken over: the probe's, minus ``bad_channels``.

    A ``.ns6`` can carry rows that are not electrodes of this array (analog
    inputs, a second array's bank), and a bad channel is one Kilosort is told to
    ignore -- neither belongs in the reference every other channel has
    subtracted from it. Rows are binary rows, as ``chanMap`` and
    ``bad_channels`` both are. Recorded in ``preprocess.json``, so a changed
    ``bad_channels`` remakes the binary.
    """
    probe = setup_probe(config, system)
    rows = {int(r) for r in np.asarray(probe["chanMap"]).ravel()}
    bad = {int(r) for r in (config.kilosort_for(system).get("bad_channels") or [])}
    kept = sorted(rows - bad)
    if not kept:
        raise ValueError(f"every {system} channel is in bad_channels; nothing to reference")
    return kept


def _catgt_excluded(config: SessionConfig) -> list[int]:
    """``bad_channels`` as CatGT's ``-chnexcl`` wants them: acquisition channel IDs.

    The same exclusion SpikeInterface's median gets from :func:`_reference_rows`,
    translated: ``bad_channels`` are binary rows, CatGT counts the channels as
    acquired, and the two differ once the run saved a subset. Empty when CatGT
    would use neither a reference nor ``gfix``. Recorded in ``preprocess.json``,
    so a change remakes the binary.
    """
    from ._io import spikeglx
    from ._preprocess import catgt as catgt_pre

    bad = config.kilosort_for("neuropixels").get("bad_channels") or []
    spec = config.preprocess_for("neuropixels")
    if not bad or (spec.car_mode is None and spec.gfix is None):
        return []
    meta_path = spikeglx.meta_path_for(config.neuropixels.bin_file)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"bad_channels has to reach CatGT as acquisition channel IDs, read from "
            f"{meta_path}, which is missing"
        )
    ids = spikeglx.ap_acquisition_channels(spikeglx.read_meta(meta_path), bad)
    return catgt_pre.excluded_channels(spec, ids)


def _stale_reasons(config: SessionConfig, system: str) -> tuple[list[str], dict | None]:
    """Why the preprocessed binary on disk cannot be reused, and its record."""
    directory = config.paths.preprocessed_for(system)
    recorded = provenance.read(directory)
    reasons = provenance.stale_reasons(recorded, _preprocess_expected(config, system))
    if not reasons:
        path = Path(recorded["binary"])
        expected = int(recorded["n_chan"]) * int(recorded["n_samples"]) * 2
        if not path.exists():
            reasons.append(f"{path} is gone")
        elif path.stat().st_size != expected:
            reasons.append(f"{path} is {path.stat().st_size} bytes, not {expected}")
    return reasons, recorded


def _binary_from_record(record: dict[str, Any]) -> Binary:
    return Binary(Path(record["binary"]), int(record["n_chan"]), float(record["fs"]), True)


def _binary_for(config: SessionConfig, system: str) -> Binary:
    """The preprocessed binary for this stream: reused if still valid, else made."""
    reasons, recorded = _stale_reasons(config, system)
    if not reasons:
        log.info(
            "reusing %s -- preprocessed by %s, still matching the session",
            recorded["binary"], recorded["engine"],
        )
        return _binary_from_record(recorded)

    import time
    from datetime import datetime

    directory = config.paths.preprocessed_for(system)
    expected = _preprocess_expected(config, system)
    if recorded is not None:
        _keep_binary_record(recorded)
    kept = _kept_binary(directory, expected)
    if kept is not None:
        # Made earlier with exactly these settings (another motion preset was
        # used since): its record comes back verbatim, so the sorting made
        # from it still recognises its binary.
        provenance.write(directory, kept)
        log.info("reusing %s -- kept from an earlier run with these settings",
                 kept["binary"])
        return _binary_from_record(kept)

    log.info("preprocessing %s (%s)", system, "; ".join(reasons))
    provenance.discard(directory)
    started = time.perf_counter()
    record = _run_preprocessing(config, system, directory, previous=recorded)
    record.update(expected)
    # How long it took and where, so a sort that reuses this binary -- on this
    # machine or after copying it to another -- can still report the cost.
    # Not one of provenance's deciding keys: it never makes a binary stale.
    record["timing"] = {
        "seconds": round(time.perf_counter() - started, 2),
        "machine": config.machine.name,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    _write_binary_record(record)
    provenance.write(directory, record)
    log.info("preprocessed -> %s", record["binary"])
    _report_kept_binaries(directory, Path(record["binary"]))
    return _binary_from_record(record)


# ---------------------------------------------------------------------------
# One binary per variant, each with its own record
#
# preprocessed/preprocess.json says which binary Kilosort reads *now*. Every
# binary also has its record beside it, <binary>.json -- written after the
# binary, removed before it is rewritten -- so a variant made earlier (DREDge,
# then MEDiCINe, then DREDge again) is found and reused rather than remade.
# A motion-corrected binary lives in motion_<preset>/, so two presets never
# write to one place.
# ---------------------------------------------------------------------------


def _binary_record_path(binary: str | Path) -> Path:
    binary = Path(binary)
    return binary.with_name(binary.name + ".json")


def motion_folder(directory: Path, preset: str) -> Path:
    """Where a motion preset's binary, drift record and figures go."""
    return directory / f"motion_{preset}"


def _write_binary_record(record: dict[str, Any]) -> None:
    path = _binary_record_path(record["binary"])
    path.write_text(json.dumps(provenance.normalise(record), indent=2), encoding="utf-8")


def _discard_binary_record(binary: Path) -> None:
    """Before ``binary`` is (re)written: a torn write must not be found later."""
    path = _binary_record_path(binary)
    if path.exists():
        path.unlink()


def _read_record(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _kept_binary(directory: Path, expected: dict[str, Any]) -> dict | None:
    """A binary already in ``preprocessed/`` made with exactly ``expected``."""
    if not directory.is_dir():
        return None
    for path in sorted(directory.rglob("*.bin.json")):
        record = _read_record(path)
        if (record is not None
                and _binary_record_path(record.get("binary", "")) == path
                and not provenance.stale_reasons(record, expected)
                and _binary_size_problem(record) is None):
            return record
    return None


def _keep_binary_record(record: dict[str, Any]) -> None:
    """Give the binary ``record`` describes its own record, if it has none.

    For a binary made before records sat beside binaries -- the DREDge binary
    a sort already read, say -- so switching preset and back reuses it in
    place instead of making it again. ``record`` is ``preprocess.json``,
    written only after that binary was finished.
    """
    if _binary_size_problem(record) is not None:
        return
    path = _binary_record_path(record["binary"])
    if not path.exists():
        _write_binary_record(record)


def _report_kept_binaries(directory: Path, current: Path) -> None:
    """Say which other variants are still on disk: each is recording-sized."""
    others = []
    for path in sorted(directory.rglob("*.bin.json")):
        binary = path.with_name(path.name[: -len(".json")])
        if binary != current and binary.exists():
            others.append(f"{binary.relative_to(directory)} "
                          f"({binary.stat().st_size / 1e9:.1f} GB)")
    if others:
        log.info("also kept in %s, for reuse: %s -- delete a folder to free its space",
                 directory, ", ".join(others))


def _run_preprocessing(
    config: SessionConfig, system: str, directory: Path, previous: dict | None = None
) -> dict[str, Any]:
    """Make the preprocessed binary. Returns what it made, for the record.

    ``record["stages"]`` is what each stage cost -- CatGT, each step of motion
    correction, the write -- for the timing table.
    """
    import time

    from ._preprocess import si

    spec = config.preprocess_for(system)
    jobs = _job_kwargs(config)
    directory.mkdir(parents=True, exist_ok=True)

    # Only SpikeInterface takes reference rows; CatGT gets -chnexcl instead.
    rows = (_reference_rows(config, system)
            if spec.car_mode and spec.method != "catgt" else None)
    if system == "neuropixels" and spec.method == "catgt":
        return _run_catgt_preprocessing(config, directory, jobs, previous)

    if spec.motion == "medicine":
        si.check_medicine()
    folder = directory
    if spec.motion:
        folder = motion_folder(directory, spec.motion)
        if folder.exists():
            shutil.rmtree(folder)              # this preset's, made with other settings
    t0 = time.perf_counter()
    if system == "blackrock":
        recording, motion_info = si.build_chain(_read_blackrock(config), spec, jobs, rows)
        target = folder / f"{config.blackrock.spike_file.stem}.bin"
    else:
        info = _neuropixels_stream(config)
        has_meta = _has_spikeglx_meta(config)
        locations = None
        if spec.motion and not has_meta:
            n_ap = info.n_chan - (info.sy_index is not None)
            locations = locations_by_row(setup_probe(config, system), n_ap)
        recording, motion_info = si.build_chain(
            si.read_neuropixels(
                info.path, info.n_chan, info.fs, info.sy_index, has_meta, locations
            ),
            spec,
            jobs,
            rows,
        )
        target = folder / f"{info.path.stem}.bin"
    t_built = time.perf_counter()
    _discard_binary_record(target)
    path = si.write(recording, target, jobs)
    t_written = time.perf_counter()

    stages = _motion_stages(spec.motion, motion_info, t_built - t0)
    stages.append(_stage(
        "spikeinterface: filter, reference"
        + (", interpolate motion" if spec.motion else "") + " + write",
        t_written - t_built,
    ))
    if motion_info is not None:
        stages.append(_save_motion(config, system, folder, recording, motion_info,
                                   spec.motion))
    return _record_of(recording, path, steps=_si_steps(spec), stages=stages)


def _stage(step: str, seconds: float | None, note: str | None = None) -> dict[str, Any]:
    """One row of ``record["stages"]``."""
    row: dict[str, Any] = {"step": step,
                           "seconds": None if seconds is None else round(float(seconds), 2)}
    if note:
        row["note"] = note
    return row


def _motion_stages(preset: str | None, motion_info: dict | None,
                   construct_s: float) -> list[dict[str, Any]]:
    """Motion estimation's rows, from SpikeInterface's own ``run_times``.

    Those are measured inside ``correct_motion`` anyway. Whatever they do not
    cover of building the chain -- noise levels, set-up -- is its own row, so
    the rows add up to the time actually spent.
    """
    if motion_info is None:
        return []
    run_times = {k: float(v) for k, v in (motion_info.get("run_times") or {}).items()}
    rows = [_stage(f"{preset}: {k.replace('_', ' ')}", v) for k, v in run_times.items()]
    rest = construct_s - sum(run_times.values())
    rows.append(_stage(f"{preset}: noise levels & set-up", max(rest, 0.0)))
    return rows


def _save_motion(
    config: SessionConfig,
    system: str,
    folder: Path,
    recording: Any,
    motion_info: dict,
    preset: str,
) -> dict[str, Any]:
    """Keep what motion correction measured, and draw it. Returns its stage row.

    ``preprocessed/motion_<preset>/``, beside the binary: ``drift.npz`` (the motion, and a subsample of the
    peaks before and after correction) and the two figures. A failure here is
    a warning: the corrected binary is already written, and an hour of
    preprocessing is not discarded over a figure.
    """
    import time

    from ._preprocess import motion as motion_record
    from ._preprocess import si

    t0 = time.perf_counter()
    try:
        drift = si.peak_drift(recording, motion_info, motion_record.DRIFT_PEAKS)
        if drift is None:
            log.warning("%s estimated no motion; no drift figures", preset)
            return _stage("drift record + figures", time.perf_counter() - t0, "no motion")
        drift["preset"] = preset
        motion_record.save_drift(folder, drift)
        _draw_drift(config, system, folder, drift)
    except Exception as error:  # the binary is written; the figures are extra
        log.warning("could not save the drift record/figures in %s (%s: %s)",
                    folder, type(error).__name__, error)
        return _stage("drift record + figures", time.perf_counter() - t0, "failed")
    log.info("drift record and figures -> %s", folder)
    return _stage("drift record + figures", time.perf_counter() - t0)


#: The figures :func:`_draw_drift` writes. Not Kilosort's names
#: (``drift_scatter.png``/``drift_amount.png``), which it writes when *it*
#: corrects drift: a re-sort merges into the folder, and the two must not be
#: mistaken for each other.
DRIFT_FIGURES = ("motion_drift_scatter.png", "motion_drift_amount.png")


def _draw_drift(config: SessionConfig, system: str, folder: Path, drift: dict) -> None:
    """Render ``drift`` into ``folder``: before/after rasters, and the motion."""
    from ._plots import motion as motion_plots
    from ._plots.summary import save_figure
    from ._preprocess import motion as motion_record

    t_edges, d_edges = motion_record.drift_raster_edges(
        drift["times_s"], [drift["depth_raw_um"], drift["depth_corrected_um"]]
    )
    before = motion_record.compute_drift_raster(
        drift["times_s"], drift["depth_raw_um"], drift["amplitude"], t_edges, d_edges)
    after = motion_record.compute_drift_raster(
        drift["times_s"], drift["depth_corrected_um"], drift["amplitude"], t_edges, d_edges)
    label = f"{config.session} | {stream_label(system)} | {drift['preset']}"
    shown = len(drift["times_s"])
    save_figure(
        motion_plots.plot_drift_rasters(
            before, after, t_edges, d_edges,
            title=f"{label} | peak amplitude across time and depth "
                  f"({shown:,} of {drift['n_peaks']:,} peaks)",
        ),
        folder / DRIFT_FIGURES[0],
    )
    save_figure(
        motion_plots.plot_motion_traces(
            drift["temporal_bins_s"], drift["spatial_bins_um"], drift["displacement_um"],
            title=f"{label} | estimated motion",
        ),
        folder / DRIFT_FIGURES[1],
    )


def locations_by_row(probe: dict, n_rows: int) -> np.ndarray:
    """Each binary row's contact position, ``(n_rows, 2)`` in um, from a Kilosort probe.

    For motion correction on a binary with no ``.meta``: the probe dict lists
    contacts in its own order and ``chanMap`` says which row each reads. Every
    row needs a position -- a row with none could not be localised -- so a map
    that leaves one out (or names a row twice, or one past the end) raises
    rather than inventing it.
    """
    rows = np.asarray(probe["chanMap"], dtype=np.int64)
    xy = np.column_stack([np.asarray(probe["xc"], float), np.asarray(probe["yc"], float)])
    if rows.size != np.unique(rows).size:
        raise ValueError("the probe's chanMap names a binary row twice")
    if rows.size and (rows.min() < 0 or rows.max() >= n_rows):
        raise ValueError(f"the probe's chanMap names rows outside the binary's {n_rows}")
    missing = np.setdiff1d(np.arange(n_rows), rows)
    if missing.size:
        raise ValueError(
            f"motion correction needs a position for every channel, and the probe "
            f"file has none for {missing.size} of the binary's {n_rows} rows "
            f"(first: {missing[:5].tolist()}); name a probe_file covering them all"
        )
    out = np.empty((n_rows, 2))
    out[rows] = xy
    return out


def _has_spikeglx_meta(config: SessionConfig) -> bool:
    """Whether the binary has a ``.meta`` to read the probe and ADC offsets from."""
    from ._io import spikeglx

    return spikeglx.meta_path_for(config.neuropixels.bin_file).exists()


def _si_steps(spec: Any) -> list[str]:
    """The SpikeInterface chain, as it will be recorded."""
    steps = ["phase_shift"] if spec.phase_shift else []
    steps.append(f"bandpass_filter {spec.highpass_hz:g}-{spec.lowpass_hz:g} Hz")
    if spec.car_mode:
        steps.append("common_reference global median (gblcar), over the probe's "
                     "channels minus bad_channels")
    if spec.motion:
        steps.append(f"correct_motion preset={spec.motion}")
    return steps


def _record_of(recording: Any, path: Path, steps: list[str], **extra: Any) -> dict[str, Any]:
    """What the record says about a binary SpikeInterface wrote."""
    n_samples = sum(
        int(recording.get_num_frames(segment_index=s))
        for s in range(recording.get_num_segments())
    )
    return {
        "binary": str(path),
        "n_chan": int(recording.get_num_channels()),
        "fs": float(recording.get_sampling_frequency()),
        "n_samples": n_samples,
        "steps": steps,
        **extra,
    }


def _run_catgt_preprocessing(
    config: SessionConfig,
    directory: Path,
    jobs: dict[str, Any],
    previous: dict | None = None,
) -> dict[str, Any]:
    """Filter the AP band with CatGT (extracting the sync edges in the same
    pass), then correct motion when the session asks.

    **Two stages, cached separately.** CatGT's output does not depend on
    ``motion``, so ``preprocessed/catgt/`` has its own record
    (:func:`_catgt_stage`), and switching motion correction on, off or to
    another preset re-runs only that -- not a full CatGT pass over the raw
    AP binary.
    """
    import time

    from ._io import spikeglx
    from ._preprocess import si

    spec = config.neuropixels.preprocess
    if spec.motion == "medicine":
        si.check_medicine()
    stage, note = _catgt_stage(config, directory / "catgt", previous)
    record = {k: stage[k] for k in _CATGT_STAGE_FIELDS if k in stage}
    made = stage.get("timing") or {}
    stages = [_stage("CatGT: filter, reference, gfix, sync edges", made.get("seconds"), note)]
    if spec.motion is None:
        return {**record, "stages": stages}

    # CatGT has no motion correction, so its output goes through SpikeInterface
    # once more. The CatGT file stays: it is what the edges came from, and the
    # record names both.
    out = spikeglx.stream_info(Path(stage["binary"]))
    folder = motion_folder(directory, spec.motion)
    if folder.exists():
        shutil.rmtree(folder)                  # this preset's, made with other settings
    t0 = time.perf_counter()
    corrected, motion_info = si.motion_chain(
        si.read_neuropixels(out.path, out.n_chan, out.fs, out.sy_index, has_meta=True),
        spec.motion,
        jobs,
    )
    t_built = time.perf_counter()
    path = si.write(corrected, folder / f"{out.path.stem}.motion.bin", jobs)
    t_written = time.perf_counter()
    stages += _motion_stages(spec.motion, motion_info, t_built - t0)
    stages.append(_stage(f"{spec.motion}: interpolate motion + write", t_written - t_built))
    stages.append(_save_motion(config, "neuropixels", folder, corrected, motion_info,
                               spec.motion))
    return {
        **record,
        **_record_of(corrected, path,
                     steps=record["steps"] + [f"correct_motion preset={spec.motion}"]),
        "catgt_binary": str(out.path),
        "stages": stages,
    }


#: What the CatGT stage's record says about its output, carried into the
#: binary's own ``preprocess.json`` (``extract_sync`` reads ``edge_dir`` and
#: ``extractors`` there).
_CATGT_STAGE_FIELDS = ("binary", "n_chan", "fs", "n_samples", "steps", "trimmed_s",
                       "extractors", "edge_dir")


def _catgt_stage_expected(config: SessionConfig) -> dict[str, Any]:
    """What CatGT's output has to have been made from: everything the binary
    depends on except ``motion``, which is applied after it."""
    expected = _preprocess_expected(config, "neuropixels")
    expected["settings"] = {k: v for k, v in expected["settings"].items() if k != "motion"}
    return expected


def _catgt_stage(
    config: SessionConfig, dest: Path, previous: dict | None
) -> tuple[dict[str, Any], str | None]:
    """CatGT's output for this session -- reused when its record still matches,
    else made. Returns its record and the timing note (None when made now:
    the row then reads like the rest of this preprocessing).

    The record (``catgt/preprocess.json``) is written only after CatGT
    finished and its output was checked; :func:`.catgt.fresh_dest` empties
    the folder, record included, before CatGT starts, so a run cut short is
    never reused. ``previous`` is the binary's record from before this
    change split the stages: CatGT output it vouches for is adopted once
    rather than made again.
    """
    import time
    from datetime import datetime

    expected = _catgt_stage_expected(config)
    stage = provenance.read(dest)
    reason = _catgt_stage_problem(stage, expected)
    if reason is None:
        made = stage.get("timing") or {}
        log.info("reusing CatGT's output %s (made %s on %s)", stage["binary"],
                 made.get("finished", "?"), made.get("machine", "?"))
        return stage, _reuse_note(made)

    adopted = _adopt_catgt_stage(previous, expected)
    if adopted is not None:
        provenance.write(dest, adopted)
        log.info("reusing CatGT's output %s, recorded by the previous preprocessing",
                 adopted["binary"])
        return adopted, _reuse_note(adopted.get("timing") or {})

    log.info("running CatGT (%s)", reason)
    started = time.perf_counter()
    stage = _run_catgt(config, dest)
    stage.update(expected)
    stage["timing"] = {
        "seconds": round(time.perf_counter() - started, 2),
        "machine": config.machine.name,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    provenance.write(dest, stage)
    return stage, None


def _reuse_note(made: dict[str, Any]) -> str:
    if made.get("finished"):
        return f"reused: made {made['finished']} on {made.get('machine', '?')}"
    return "reused: made before per-stage times were recorded"


def _catgt_stage_problem(stage: dict | None, expected: dict[str, Any]) -> str | None:
    """Why CatGT's output on disk cannot be reused, or None if it can."""
    reasons = provenance.stale_reasons(stage, expected)
    if reasons:
        return "; ".join(reasons)
    return _binary_size_problem(stage)


def _binary_size_problem(record: dict[str, Any]) -> str | None:
    """A recorded binary that is gone, or not the size its shape says."""
    try:
        path = Path(record["binary"])
        expected = int(record["n_chan"]) * int(record["n_samples"]) * 2
    except (KeyError, TypeError, ValueError):
        return "its record does not name a binary"
    if not path.exists():
        return f"{path} is gone"
    if path.stat().st_size != expected:
        return f"{path} is {path.stat().st_size} bytes, not {expected}"
    return None


def _adopt_catgt_stage(previous: dict | None, expected: dict[str, Any]) -> dict | None:
    """The CatGT stage record a pre-split ``preprocess.json`` vouches for, or None.

    That record was written only after CatGT *and* any motion correction
    finished, so its CatGT output is complete. Adopted only when everything
    but ``motion`` matches and the file is still the size of its shape. Its
    time is kept only when there was no motion correction to share it with.
    """
    from ._io import spikeglx

    if not previous or previous.get("engine") != "catgt" or not previous.get("edge_dir"):
        return None
    was = dict(previous)
    was["settings"] = {k: v for k, v in (previous.get("settings") or {}).items()
                       if k != "motion"}
    if provenance.stale_reasons(was, expected):
        return None
    binary = previous.get("catgt_binary") or previous.get("binary")
    try:
        out = spikeglx.stream_info(Path(binary))
    except (OSError, ValueError, TypeError):
        return None
    stage = {
        "binary": str(out.path), "n_chan": out.n_chan, "fs": out.fs,
        "n_samples": out.n_samples,
        "steps": [s for s in previous.get("steps") or [] if s.startswith("CatGT")],
        "trimmed_s": previous.get("trimmed_s"),
        "extractors": previous.get("extractors"),
        "edge_dir": previous["edge_dir"],
        **expected,
    }
    if _binary_size_problem(stage) is not None:
        return None
    made = previous.get("timing") or {}
    alone = not (previous.get("settings") or {}).get("motion")
    stage["timing"] = {**made, "seconds": made.get("seconds") if alone else None}
    if not alone or not made:
        stage["timing"].pop("finished", None)
    return stage


def _run_catgt(config: SessionConfig, dest: Path) -> dict[str, Any]:
    """Run CatGT into a fresh ``dest``. Returns what it made, for the record."""
    from ._io import spikeglx
    from ._preprocess import catgt as catgt_pre

    npx = config.neuropixels
    spec = npx.preprocess
    layout = spikeglx.run_layout(npx.bin_file)
    why_not = catgt_pre.catgt_unavailable(layout, config.machine.catgt_dir)
    if why_not is not None:
        raise RuntimeError(
            f"neuropixels.preprocess.method is catgt, but {why_not}.\n"
            "  Run it where CatGT is installed, or preprocess with SpikeInterface "
            "instead by changing the session:\n"
            "      neuropixels:\n"
            "        preprocess:\n"
            "          method: spikeinterface   # and gfix: null, if it is set"
        )

    extractors = extract.catgt_extractors(config, layout)
    excluded = _catgt_excluded(config)
    args = catgt_pre.build_preprocess_args(layout, spec, dest, extractors, excluded)
    launcher = catgt.catgt_script(config.machine.catgt_dir)
    log.info("CatGT: %s %s", launcher.name, " ".join(args))
    # CatGT prints nothing while it works; a long silence here is normal.
    log.info(
        "CatGT is silent while it works (expect minutes per hour of recording); "
        "progress: %s", Path(config.machine.catgt_dir) / "CatGT.log",
    )
    # Emptied only now that CatGT is about to run: a previous binary here was
    # made with other settings (its record is already gone), and must not be
    # found afterwards as though this run had written it.
    catgt.fresh_dest(dest)
    try:
        catgt.run_catgt(config.machine.catgt_dir, args)
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}\n  To preprocess without CatGT, set "
            "neuropixels.preprocess.method: spikeinterface (and gfix: null, if set)."
        ) from None

    raw = _neuropixels_stream(config)
    out = spikeglx.stream_info(catgt_pre.find_output(dest, layout))
    trimmed = catgt_pre.trimmed_seconds(raw.n_samples, out.n_samples, raw.fs)
    if trimmed:
        log.info("CatGT trimmed %.4f s off the end of %s", trimmed, raw.path.name)
    return {
        "binary": str(out.path),
        "n_chan": out.n_chan,
        "fs": out.fs,
        "n_samples": out.n_samples,
        "steps": ["CatGT " + " ".join(args)],
        "trimmed_s": trimmed,
        "extractors": [e.to_flag() for e in extractors],
        "edge_dir": str(dest),
    }


def _read_blackrock(config: SessionConfig) -> Any:
    """The Utah array's ``.ns6`` as a SpikeInterface recording, every channel.

    **Every channel is kept**, including any sync or analog inputs sharing the
    file: the probe's ``chanMap`` is what selects electrodes, and dropping them
    here would shift every contact after the removed one. (They are filtered and
    re-referenced along with the rest, which costs nothing -- Kilosort never
    reads them. The median itself is taken over the probe's channels minus
    ``bad_channels`` only: see :func:`_reference_rows`.)
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
    return recording


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


def read_nsp_time_map(directory: Path, source: Path | None = None) -> Any | None:
    """The map written by :func:`write_nsp_time_map`, or None if there is none.

    With ``source``, a map measured from a *different* file is refused (None):
    each ``.nsX`` is its own crystal, so the ``.ns5`` sync file's map applied to
    ``.ns6`` spike samples is wrong by tens of ms over a session. Sessions sorted
    before the sync map got its own name could hold exactly that.
    """
    from ._io.blackrock import NspTimeMap

    path = Path(directory) / "nsp_time_map.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    recorded = payload.get("source")
    if source is not None and recorded and Path(recorded).name != Path(source).name:
        log.warning(
            "%s was measured from %s, not the spike file %s; not using it",
            path, Path(recorded).name, Path(source).name,
        )
        return None
    return NspTimeMap.from_dict(payload)


def _input_path(config: SessionConfig, system: str) -> Path | None:
    """The path the session names for this system's spike data."""
    if system == "blackrock":
        return config.blackrock.spike_file
    return config.neuropixels.bin_file


def _export_stem(config: SessionConfig, system: str) -> str:
    """What the exported .mat files are named after: the recording they came from.

    ``<run>_t0.imec0.ap`` for a Neuropixels AP binary, ``HUB-<x>_001`` for a
    ``.ns6`` -- the same rule the LF band follows (``<lf stem>.lfp.mat``), so a
    file copied away from its folder still says which recording it is.
    """
    path = _input_path(config, system)
    return Path(path).stem if path is not None else config.session


def _check_input_reachable(config: SessionConfig, system: str) -> reachable.ReadProbe | None:
    """Fail fast, and loudly, when the recording's filesystem is not answering.

    Returns the read probe -- the file's size and measured read rate, which the
    timing table reports beside the steps that read it.
    """
    path = _input_path(config, system)
    if path is None:
        return None                        # nothing named; the caller reports that

    probe = reachable.check_reachable(path)
    log.info("%s: %s", path, probe.summary())
    if probe.total_bytes and probe.estimated_seconds() > 60:
        log.warning(
            "%s reads at %.0f MB/s -- sorting streams the whole file, so expect "
            "about %.0f min of I/O. Sort beside the data, or stage it to the "
            "machine's cache_dir first.",
            path.name, probe.bytes_per_s / 1e6, probe.estimated_seconds() / 60,
        )
    return probe


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
    #: Sync trains the CatGT preprocessing pass found missing or short.
    sync_problems: tuple[str, ...] = ()

    def summary(self) -> str:
        text = (
            f"{self.n_units} units, {self.n_spikes} spikes at {self.fs:g} Hz "
            f"-> {self.results_dir}"
        )
        for problem in self.sync_problems:
            text += f"\n       WARNING sync pulses: {problem}"
        return text


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
    "do_CAR": "<system>.preprocess.car_mode -- the common reference is taken "
    "before Kilosort, never inside it, so Kilosort always gets do_CAR=False",
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
    # The common reference is preprocess.car_mode's job, done before this.
    kwargs: dict[str, Any] = {"do_CAR": False}
    accepted = set(inspect.signature(run_kilosort).parameters)

    overridden = config.nblocks_override(system)
    if overridden:
        log.info(overridden)
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

    The whole of pipeline 1: resolve the channel map, take the preprocessed
    binary (making it first if :func:`preprocess` has not), hand both to
    Kilosort's own ``run_kilosort``, and put the results where Phy and the
    export stages look for them. Nothing is loaded into memory here and no
    recording object is passed around -- Kilosort streams the file itself.

    Returns ``None`` without sorting when ``kilosort_on_<system>`` is false, the
    "extract the pulses and the LFP but do not sort" case.

    The band-pass and the common reference are done before this, by the
    session's ``preprocess:`` block, so Kilosort is always given ``do_CAR=False``.
    What it still does on every batch -- its own highpass at
    ``highpass_cutoff``, whitening, drift correction at ``nblocks`` -- is set
    under ``kilosort:`` and resolved by :meth:`SessionConfig.kilosort_for`.

    Sorting runs in the machine's ``cache_dir`` and the results are copied out,
    because Kilosort writes beside its results and a network share is both slow
    and rude to everyone else on it. ``dry_run`` reports what would be run --
    map, binary, settings -- and sorts nothing.
    """
    _check(system)
    if not getattr(config, f"sorts_{system}"):
        return None

    import time
    from datetime import datetime

    started, t0 = datetime.now().astimezone(), time.perf_counter()
    read_probe = _check_input_reachable(config, system)
    t_reachable = time.perf_counter()
    probe = setup_probe(config, system)
    t_probe = time.perf_counter()
    final_dir = config.paths.sorted_for(system)

    if dry_run:
        return _dry_run(config, system, probe, final_dir)

    reused = not _stale_reasons(config, system)[0]
    # A rebuild replaces the binary the existing sorting (if any) was made
    # from; if this sort does not finish, that sorting is left beside new traces.
    left_behind = not reused and (final_dir / "spike_times.npy").exists()
    try:
        binary = _binary_for(config, system)
        t_preprocessed = time.perf_counter()

        settings, kwargs = _kilosort_arguments(config, system, binary)

        import torch
        from kilosort import run_kilosort

        work_dir = _work_dir(config, system) or final_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        log.info("sorting %s -> %s", binary.summary(), work_dir)
        if work_dir == final_dir:  # sorting in place: clear before Kilosort writes
            _clear_previous_curation(final_dir)

        try:
            t_kilosort = time.perf_counter()
            ops, spike_times, clusters, *_rest = run_kilosort(
                settings=settings,
                probe=probe,
                filename=binary.path,
                data_dtype="int16",
                results_dir=work_dir,
                device=torch.device(config.machine.device),
                **kwargs,
            )
            t_sorted = time.perf_counter()
            if work_dir != final_dir:
                _clear_previous_curation(final_dir)
                _publish(work_dir, final_dir)
                _repoint_params(final_dir, binary)
            _copy_drift_figures(binary.path.parent, final_dir)
        finally:
            if work_dir != final_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
    except BaseException:
        if left_behind:
            log.warning(
                "the preprocessed binary was rebuilt and this sort did not finish: "
                "the existing sorting in %s was made from the previous binary, so Phy "
                "now shows it the new traces, and the waveform export will skip it "
                "until it is re-sorted", final_dir,
            )
        raise
    t_published = time.perf_counter()

    sync_edges = _preprocessing_sync_check(config) if system == "neuropixels" else None
    result = SortSummary(
        results_dir=final_dir,
        n_units=int(np.unique(clusters).size),
        n_spikes=_saved_spike_count(final_dir),
        fs=binary.fs,
        binary=binary.path,
        settings=settings,
        sync_problems=tuple((sync_edges or {}).get("problems", ())),
    )

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
    t_done = time.perf_counter()

    timing = _sort_timing(
        config, system, read_probe, started,
        loading={"reachability check (read-rate probe)": t_reachable - t0,
                 "channel map": t_probe - t_reachable},
        reused=reused,
        reuse_check_s=t_preprocessed - t_probe,
        ops=ops if isinstance(ops, dict) else {},
        kilosort_wall_s=t_sorted - t_kilosort,
        postprocessing={
            "copy results out of the cache": t_published - t_sorted,
            **({"NSP clock map": t_done - t_published} if system == "blackrock" else {}),
        },
    )
    _write_run_info(config, system, result, kwargs, timing, sync_edges)
    return result


def _sort_timing(
    config: SessionConfig,
    system: str,
    read_probe: reachable.ReadProbe | None,
    started: Any,
    *,
    loading: dict[str, float],
    reused: bool,
    reuse_check_s: float,
    ops: dict[str, Any],
    kilosort_wall_s: float,
    postprocessing: dict[str, float],
) -> dict[str, Any]:
    """The sorting run's half of the timing table (see ``_export/timing.py``).

    Preprocessing is usually done by the time the sort starts -- the driver runs
    :func:`preprocess` first, or the binary was made on another machine and
    copied -- so its time comes from ``preprocess.json``, where
    :func:`_binary_for` recorded it, with the machine it was made on.
    """
    from datetime import datetime

    from ._export import timing as tm

    record = provenance.read(config.paths.preprocessed_for(system)) or {}
    made = record.get("timing") or {}
    steps: list[tm.Step] = [("loading", k, v) for k, v in loading.items()]
    notes: dict[str, str] = {}
    stages = record.get("stages")
    if stages:
        # One row per stage, as preprocessing recorded it. A stage reused from
        # an earlier run (CatGT, when only motion changed) keeps its own note.
        whole = (
            (f"made {made['finished']} on {made.get('machine', '?')}"
             if made.get("finished") else "made before preprocessing times were recorded")
            if reused else "made by this sort"
        )
        for row in stages:
            steps.append(("preprocessing", row["step"], row.get("seconds")))
            notes[row["step"]] = row.get("note") or whole
        if not reused:
            # What the stages do not cover of making the binary here: the
            # reuse checks, reading the .meta, writing the record.
            covered = sum(r.get("seconds") or 0.0 for r in stages
                          if not (r.get("note") or "").startswith("reused"))
            label = "preprocessing: checks & records"
            steps.append(("preprocessing", label, max(reuse_check_s - covered, 0.0)))
            notes[label] = "made by this sort"
    else:
        label = f"{record.get('engine', 'preprocessing')} (band-pass, CAR, ...)"
        if reused:
            seconds = made.get("seconds")
            note = (
                f"made {made.get('finished')} on {made.get('machine')}"
                if made else "made before preprocessing times were recorded"
            )
            if made:
                note += "; made before per-stage times were recorded"
        else:
            seconds, note = reuse_check_s, "made by this sort"
        steps.append(("preprocessing", label, seconds))
        notes[label] = note
    steps += tm.kilosort_steps(ops, kilosort_wall_s)
    steps += [("postprocessing", k, v) for k, v in postprocessing.items()]

    source = None
    if read_probe is not None:
        source = {
            "path": str(read_probe.path),
            "gb": round(read_probe.total_bytes / 1e9, 3),
            "read_mb_per_s": round(read_probe.bytes_per_s / 1e6, 1),
            "open_latency_s": round(read_probe.latency_s, 3),
        }
    run = {
        "machine": config.machine.name,
        "device": config.machine.device,
        "started": started.isoformat(timespec="seconds"),
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    return tm.build_timing(
        steps, run, notes=notes, extra={"input": source} if source else None
    )


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


_PREVIOUS_CURATION = (
    ".phy", "phy.log", "cluster_*.tsv",
    # drift figures of the old sort: Kilosort's own (nblocks > 0) and the ones
    # copied from preprocessed/motion_<preset>/ (a SpikeInterface preset) -- a re-sort
    # that makes neither must not leave the old ones describing it
    "drift_amount.png", "drift_scatter.png", "motion_drift_*.png",
    # time_remapping's per-spike arrays, one per spike of the *old* sort
    "*_spike_seconds.npy", "*_spike_seconds_blackrock.npy",
    "*_spike_seconds_blackrock.json",
)


def _clear_previous_curation(final_dir: Path) -> None:
    """Remove what was derived from a previous sort before a new one lands here.

    Kilosort overwrites its own arrays but knows nothing of Phy's: the ``.phy/``
    cache (spikes-per-cluster and cluster stats for the *old* cluster ids),
    ``cluster_info.tsv`` and any label column added during curation. A re-sort
    merged on top of them fails to open in Phy, and the labels would name
    clusters that no longer exist. Every ``cluster_*.tsv`` goes, including
    Kilosort's own, because the new sort writes its own set.

    So do the per-spike times ``time_remapping`` wrote: one entry per spike of
    the old sort, and ``export_results`` would take them as the new sort's
    Blackrock times whenever the counts happened to agree. The time map itself
    stays -- it describes the sync edges, not the spikes.
    """
    for pattern in _PREVIOUS_CURATION:
        for path in final_dir.glob(pattern):
            log.info("removing %s from the previous sort", path)
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _copy_drift_figures(binary_dir: Path, target: Path) -> None:
    """Put the motion-correction figures beside the sorting made from them.

    Where Kilosort's ``drift_*.png`` would be had it corrected the drift. They
    sit beside the binary that was sorted (``preprocessed/motion_<preset>/``);
    a binary without them (no motion correction, or made before they were
    kept) leaves none here.
    """
    for name in DRIFT_FIGURES:
        source = binary_dir / name
        stale = target / name
        if source.exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, stale)
        elif stale.exists():
            stale.unlink()


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
    probe's channels while the binary holds every channel in the file, and
    ``n_channels_dat`` is the stride Phy walks the file with -- leave it behind
    and every trace is silently sheared rather than erroring. ``hp_filtered``
    goes with them, and is set either way: the binary Kilosort read is the
    preprocessed one, already band-passed, so Phy must not filter it again.

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
    hp_filtered = {"hp_filtered": f"hp_filtered = {binary.filtered}\n"}
    if stated is not None and ((final_dir / stated).exists() or Path(stated).exists()):
        # The path resolves, so it stays as Kilosort wrote it. Only whether the
        # file is already filtered is ours to say: Kilosort writes False, and a
        # preprocessed binary would then be filtered again for display.
        if not binary.filtered:
            return
        replacements = hp_filtered
        message = "params.py: hp_filtered = %s" % binary.filtered
    else:
        replacements = {
            "dat_path": f"dat_path = '{binary.path.as_posix()}'\n",
            "n_channels_dat": f"n_channels_dat = {binary.n_chan_bin}\n",
            **hp_filtered,
        }
        message = "params.py: dat_path repointed at %s (%d channels)" % (
            binary.path, binary.n_chan_bin,
        )
    out = [replacements.pop(_assigned_name(line), line) for line in lines]
    out.extend(replacements.values())   # keys this params.py did not carry
    params.write_text("".join(out), encoding="utf-8")
    log.info(message)


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
    final_dir: Path,
) -> dict[str, Any]:
    """What a real run would do, without doing it. Needs no GPU, writes nothing.

    The preprocessed binary is reported as it stands: reused, or made by the
    real run and why. Kilosort's settings are split against the binary it
    *would* read, whose channel count is known without writing it.
    """
    from ._probes.common import probe_summary

    reasons, recorded = _stale_reasons(config, system)
    if reasons:
        binary = _planned_binary(config, system)
        preprocessing = {
            "status": "will be made: " + "; ".join(reasons),
            "steps": _planned_steps(config, system),
        }
        if system == "neuropixels" and config.preprocess_for(system).method == "catgt":
            directory = config.paths.preprocessed_for(system)
            problem = _catgt_stage_problem(provenance.read(directory / "catgt"),
                                           _catgt_stage_expected(config))
            preprocessing["catgt"] = (
                "reused (only what follows it is remade)" if problem is None
                else f"will run: {problem}"
            )
    else:
        binary = _binary_from_record(recorded)
        preprocessing = {"status": "reused", "steps": recorded.get("steps")}

    report: dict[str, Any] = {
        "probe": probe_summary(probe),
        "preprocessing": preprocessing,
        **({"drift correction": config.nblocks_override(system)}
           if config.nblocks_override(system) else {}),
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


def _planned_binary(config: SessionConfig, system: str) -> Binary:
    """The binary :func:`preprocess` would write, without writing it.

    CatGT keeps every channel; SpikeInterface leaves out the Neuropixels SY word.
    Blackrock is opened (its header only) to learn the channel count.
    """
    spec = config.preprocess_for(system)
    directory = config.paths.preprocessed_for(system)
    if system == "blackrock":
        recording = _read_blackrock(config)
        folder = motion_folder(directory, spec.motion) if spec.motion else directory
        return Binary(
            folder / f"{config.blackrock.spike_file.stem}.bin",
            int(recording.get_num_channels()),
            float(recording.get_sampling_frequency()),
            True,
        )
    info = _neuropixels_stream(config)
    if spec.method == "catgt":
        from ._io import spikeglx

        layout = spikeglx.run_layout(info.path)
        name = (
            f"{layout.run_name}_g{layout.gate}_tcat.imec{layout.probe}.ap.bin"
            if layout is not None else "(CatGT cannot read this binary)"
        )
        path = directory / "catgt" / name
        n_chan = info.n_chan
        if spec.motion:
            path = motion_folder(directory, spec.motion) / Path(name).with_suffix(".motion.bin").name
            n_chan -= info.sy_index is not None
    else:
        folder = motion_folder(directory, spec.motion) if spec.motion else directory
        path = folder / f"{info.path.stem}.bin"
        n_chan = info.n_chan - (info.sy_index is not None)
    return Binary(path, int(n_chan), float(info.fs), True)


def _planned_steps(config: SessionConfig, system: str) -> list[str]:
    """The preprocessing a real run would do, as the record will name it."""
    spec = config.preprocess_for(system)
    if spec.method != "catgt":
        return _si_steps(spec)
    steps = [
        f"CatGT -apfilter=butter,12,{spec.highpass_hz:g},{spec.lowpass_hz:g}"
        + (f" -{spec.car_mode}" if spec.car_mode else "")
        + (" -gfix=" + ",".join(f"{v:g}" for v in spec.gfix) if spec.gfix else "")
        + ("" if spec.phase_shift else " -no_tshift")
    ]
    if config.machine.catgt_dir is None:
        steps.append(
            "WILL FAIL: this machine has no catgt_dir -- run on the rig, or set "
            "neuropixels.preprocess.method: spikeinterface"
        )
    if spec.motion:
        steps.append(f"correct_motion preset={spec.motion}")
    return steps


def _measured_means(measured: dict | None) -> dict:
    """``build_unit_table``'s waveform arguments: each unit's measured mean.

    A unit with no measured spike (all of them near a file end, or in blanked
    batches) has a NaN mean and is left out, so its shape columns are NaN.
    """
    if measured is None:
        return {}
    result = measured["result"]
    means = {
        int(uid): np.asarray(mean)
        for uid, mean in zip(result.unit_ids, result.mean.T)   # mean: (width, n_units)
        if np.isfinite(mean).all()
    }
    return {"mean_waveforms": means,
            "waveform_units": measured.get("mean_units", "ADC")}


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


#: Phy's curation labels as single vs multi unit, for the figure headers.
_UNIT_CLASS = {"good": "SU", "mua": "MU"}

#: Spikes drawn in the overview's position plot. Spread evenly over the session;
#: past this many the dots only overlap, and reading the features costs more.
_OVERVIEW_MAX_SPIKES = 100_000


def _unit_page_data(
    phy,
    unit_ids: np.ndarray,
    spike_times_s: dict[int, np.ndarray] | None,
    table,
    measured: dict | None,
    window: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    """What ``plots.save_unit_pages`` draws, one dict per unit, ordered by channel.

    ``window`` is the sorted recording's ``(start_s, duration_s)`` -- the rate
    is binned over it, as ``units.csv`` divides by it; without one, over the
    span of the spikes.

    The waveform is exactly what ``sorted_spikes.mat`` holds: ``MeanWaveform``
    +/- ``StdWaveform`` from :func:`export_waveforms`, in uV or ADC as it says.
    A unit with no measured mean gets no waveform rather than Kilosort's
    template, so the figure can never show a shape the ``.mat`` does not.
    """
    from ._export import final, metrics

    rows = table.set_index("unit_id")
    mean = std = None
    if measured is not None:
        result = measured["result"]
        column = {int(u): i for i, u in enumerate(result.unit_ids)}
        mean, std, counts = result.mean, result.std, result.n_per_unit
        t_measured = (
            (np.arange(mean.shape[0]) - measured["samples_before"])
            / measured["fs"] * 1000.0
        )

    # On the NSP or Blackrock axis the recording starts nowhere near zero, so
    # the rate is binned over the recording's window on that axis.
    start, duration = window or final.recording_window(phy, unit_ids, spike_times_s)
    units = []
    for unit_id in unit_ids:
        unit_id = int(unit_id)
        times = spike_times_s[unit_id] if spike_times_s is not None else phy.times_for(unit_id)
        _template, channel = final.mean_template_waveform(phy, unit_id)
        isi_counts, isi_edges = metrics.compute_isi_histogram(metrics.compute_isi(times))
        centers, rate = metrics.compute_firing_rate(
            times, duration, bin_s=max(1.0, duration / 100), start_s=start
        )
        label = phy.labels.get(unit_id, "")
        unit = {
            "unit_id": unit_id,
            "label": label,
            "unit_class": _UNIT_CLASS.get(label, ""),
            "ks_label": phy.ks_labels.get(unit_id, ""),
            "channel": channel,
            "n_spikes": int(times.size),
            "isi_counts": isi_counts,
            "isi_edges_ms": isi_edges,
            # Elapsed since the recording window's start -- the same reference for
            # every unit, so a unit that only fires late still starts late. Binned
            # on the export's axis first, so the rates are exactly those.
            "rate_centers_s": centers - start,
            "rate_hz": rate,
            "amp_times_s": times - start,         # the rate panel's axis: since the start
            "amplitudes": phy.amplitudes_for(unit_id),
        }
        if unit_id in rows.index:
            row = rows.loc[unit_id]
            for key in ("firing_rate_hz", "isi_fraction", "presence_ratio", "amp_cv",
                        "peak_to_trough_ms"):
                if key in row:
                    unit[key] = float(row[key])

        if mean is not None and unit_id in column and counts[column[unit_id]] > 0:
            i = column[unit_id]
            unit["waveform"] = mean[:, i]
            unit["waveform_std"] = std[:, i]
            unit["waveform_t_ms"] = t_measured
            unit["waveform_units"] = measured.get("mean_units", "ADC")
        units.append(unit)

    units.sort(key=lambda u: (u["channel"] if u["channel"] is not None else np.inf,
                              u["unit_id"]))
    return units


def _records_for(config: SessionConfig, system: str) -> list[Path]:
    """The records copied into ``export/``, so the bundle explains itself once
    copied away. Missing ones are skipped by the caller."""
    sorted_dir = config.paths.sorted_for(system)
    records = [
        sorted_dir / "run_info.json",                    # Kilosort settings, timing
        sorted_dir / "kilosort4.log",                    # Kilosort's own log
        config.paths.preprocessed_for(system) / "preprocess.json",
    ]
    if system == "neuropixels":
        records += [config.paths.aligned / "time_map.json",
                    config.paths.aligned / "validation.json"]
    else:
        records.append(sorted_dir / "nsp_time_map.json")  # the .ns6 clock
    return records


def _copy_records(config: SessionConfig, system: str, out_dir: Path) -> dict[str, Path]:
    """Copy the run's records beside the products they describe.

    Copies, not moves: the originals stay where the pipeline reads them, and a
    re-export refreshes these. Small JSON and text only -- nothing here is data.
    """
    copied: dict[str, Path] = {}
    for source in _records_for(config, system):
        if source.exists():
            target = out_dir / source.name
            shutil.copy2(source, target)
            copied[source.stem] = target
    return copied


def _write_summary(
    config: SessionConfig,
    system: str,
    out_dir: Path,
    timebase: str,
    unit_ids: np.ndarray,
    measured: dict | None,
    curated: bool,
    time_map: Any = None,
    window: tuple[float, float] | None = None,
) -> Path:
    """The manifest that makes the bundle self-describing.

    What ran, on what, and where everything it used came from -- so a folder
    copied to another machine still says which recording produced it and which
    time map its spike times are on.
    """
    results_dir = config.paths.sorted_for(system)
    run_info = _corrected_run_info(results_dir)

    aligned_dir = config.paths.aligned
    payload = {
        "session": config.session,
        "system": system,
        "stream": stream_label(system),
        "sorter": config.sorter,
        # The folder it all went to: the sorter's name unless export_folder_name says otherwise.
        "output_folder": config.output_folder,
        "timebase": timebase,
        "time_units": "seconds",
        "curated": curated,
        "n_units_exported": int(np.asarray(unit_ids).size),
        # Whether the sync pulses were there when the sort's CatGT pass ran.
        **({"sync_edges_at_sorting": run_info["sync_edges"]}
           if (run_info or {}).get("sync_edges") else {}),
        # What units.csv's firing rates and presence ratios are taken over.
        "recording_window": (
            {"start_s": window[0], "duration_s": window[1], "from": "samples sorted"}
            if window else {"from": "span of the exported spikes"}
        ),
        **({"nsp_time_map": time_map.to_dict()} if time_map is not None else {}),
        "waveforms": (
            {
                "n_spikes": measured["n_spikes"],
                "n_available": measured.get("n_available"),
                "n_dropped": measured["n_dropped"],
                "snippets_kept": measured["snippets_kept"],
                "units": measured.get("mean_units"),
                "preprocessing": measured.get("chain", {}),
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


def _saved_spike_count(results_dir: Path) -> int:
    """Spikes in the sorting as saved: rows of ``spike_times.npy``.

    Not Kilosort's returned array, which has a row per spike *before* duplicate
    removal and three columns (time, template, amplitude) -- counting its
    entries once reported 32.5 M spikes for a 10.8 M-spike sort.
    """
    return int(np.load(Path(results_dir) / "spike_times.npy", mmap_mode="r").shape[0])


def _corrected_run_info(results_dir: Path) -> dict[str, Any]:
    """``run_info.json``, with ``n_spikes`` recounted from the saved sorting.

    A record written before the count was fixed is corrected on disk, keeping
    the old number as ``n_spikes_recorded``, so the copy in ``export/`` and the
    manifest agree with the arrays beside them.
    """
    path = Path(results_dir) / "run_info.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        run_info = json.load(handle)
    if not (Path(results_dir) / "spike_times.npy").exists():
        return run_info
    count = _saved_spike_count(results_dir)
    if run_info.get("n_spikes") != count:
        log.info("run_info.json: n_spikes %s -> %d, recounted from spike_times.npy",
                 run_info.get("n_spikes"), count)
        run_info["n_spikes_recorded"] = run_info.get("n_spikes")
        run_info["n_spikes"] = count
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(run_info, handle, indent=2, default=str)
    return run_info


def _waveform_chain(
    run_info: dict[str, Any], binary: Path, kilosort: dict[str, Any]
) -> dict[str, Any]:
    """Everything the waveform signal went through, in order.

    Two stages: whatever made the binary Kilosort sorted (CatGT or SpikeInterface,
    as its ``preprocess.json`` recorded -- or nothing, the raw recording), then
    Kilosort's reader, which this export cuts through.
    """
    record = run_info.get("preprocessing") or {}
    engine = record.get("engine") or "none (the raw recording)"
    reader = (
        f"kilosort reader: {kilosort['highpass_cutoff']:g} Hz FFT high-pass, "
        f"CAR {'on' if kilosort['do_CAR'] else 'off'}, whitening undone, "
        "no drift correction"
    )
    return {
        "summary": f"{engine}, then {reader}",
        "input": str(binary),
        "input_preprocessed_by": engine,
        "input_steps": record.get("steps"),
        "then": reader,
        **kilosort,
    }


def _write_run_info(
    config: SessionConfig,
    system: str,
    result: SortSummary,
    kwargs: dict[str, Any],
    timing: dict[str, Any] | None = None,
    sync_edges: dict[str, Any] | None = None,
) -> None:
    """Record what produced this sorting, beside the sorting itself."""
    payload = {
        "session": config.session,
        "machine": config.machine.name,
        "device": config.machine.device,
        "system": system,
        # Which stream, not just which system: a run can hold several probes.
        "stream": stream_label(system),
        # The recording as the session names it, and the binary Kilosort read.
        "source_recording": str(_input_path(config, system)),
        "binary": str(result.binary),
        "timing": timing,
        "n_units": result.n_units,
        "n_spikes": result.n_spikes,
        "fs": result.fs,
        "settings": result.settings,
        "run_kilosort_arguments": kwargs,
        # What was done to the signal before Kilosort read it, as recorded beside
        # the binary -- engine, settings, the steps and the source it came from.
        "preprocessing": provenance.read(config.paths.preprocessed_for(system)),
        # Whether the sync pulses were in the edges that pass extracted: kept
        # apart from "preprocessing", which must stay the record verbatim (the
        # waveform guard compares the two).
        "sync_edges": sync_edges,
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

    Edges land in the Neuropixels sorting's folder, ``<neuropixels_dir>/kilosort4/``
    -- Blackrock's too, since they exist only to align it -- which is computable
    before anything is sorted. Nothing is written under ``blackrock_dir``.

    New edges void the map fitted from the old ones, so in a session with both
    systems any acceptance is withdrawn first -- ``validation.json`` and the
    sync report removed, the LFP and OneBox exports put back on their own
    clocks -- for this probe, or for every probe when Blackrock's edges, which
    each probe's map uses, are re-extracted. Re-running only this stage then
    leaves nothing labelled Blackrock time from the old map.

    The source (the AP binary, or the ``.ns5``) is checked first: one that is
    missing or not answering raises an ``OSError`` before anything is withdrawn
    or deleted, so the edges already written stay usable.
    """
    _check(system)
    if not config.has_data(system):
        return None

    # Before anything is withdrawn or deleted: a recording that is not there or
    # not answering raises here and leaves the earlier edges -- the same
    # recording's -- and their acceptance as they were. A read that fails
    # partway still clears them below, since half a file's edges must not be
    # fitted from.
    source = (config.neuropixels.bin_file if system == "neuropixels"
              else config.blackrock.sync_file)
    if source is not None:
        log.info("%s: %s", source, reachable.check_reachable(source).summary())

    if config.aligns_systems:
        affected = ([config] if system == "neuropixels"
                    else [probe_config for _, probe_config in config.per_probe()])
        for probe_config in affected:
            _withdraw_acceptance(probe_config)

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

    # If a map was already accepted -- a re-export after alignment -- put it on
    # the Blackrock clock now. Otherwise validate_remapping stamps it once it
    # accepts one.
    mapping = _fitted_map(config) if _exports_accepted(config) else None
    if mapping is not None:
        stamp_lfp_timebase(config, mapping)
    return path


def extract_analog(config: SessionConfig, system: str) -> Path | None:
    """Export the OneBox's analog inputs. Returns the path written, or ``None``.

    Every XA input the OneBox saved goes out -- eye, pupil, photodiode, whatever
    else is wired, and the 14 s burst itself as ``sync_burst`` -- labelled from
    ``neuropixels.analog_channels``. Runs when the session says
    ``export_analog: true``, at the ``analog_decimate`` stride.

    The samples are written on the OneBox's own clock. Blackrock time comes from
    ``time_map.json["onebox_to_blackrock"]`` -- the OneBox -> probe fit composed
    with probe -> Blackrock, the same chain the held-out burst validates -- and
    is stamped in now if that map has been accepted, or by ``validate_remapping``
    when it accepts one.

    One OneBox per run, so a session covering two probes exports it once: the
    exporting driver calls this with the first probe's config, and it lands in
    that probe's ``export/``.
    """
    _check(system)
    if system == "blackrock" or not config.has_data("neuropixels"):
        return None
    if not config.export_analog:
        return None
    obx = _onebox_binary(config)
    if obx is None:
        return None

    from ._io import spikeglx

    probe = spikeglx.check_reachable(obx)
    log.info("%s: %s", obx.name, probe.summary())
    npx = config.neuropixels
    path = spikeglx.export_onebox_analog(
        obx,
        config.paths.export_for("neuropixels"),
        named=npx.analog_channels,
        burst_row=npx.burst_word,
        decimate=config.analog_decimate,
    )
    log.info(
        "wrote %s%s", path,
        f" (decimated {config.analog_decimate}x)" if config.analog_decimate > 1 else "",
    )

    onebox_map = _fitted_onebox_map(config) if _exports_accepted(config) else None
    if onebox_map is not None:
        stamp_analog_timebase(config, onebox_map)
    else:
        log.info("no accepted onebox_to_blackrock map yet; validate_remapping "
                 "will stamp it")
    return path


def _onebox_binary(config: SessionConfig) -> Path | None:
    """The run's OneBox file, found beside the AP binary's gate folder."""
    from ._io import spikeglx

    npx = config.neuropixels
    layout = spikeglx.run_layout(npx.bin_file) if npx.bin_file is not None else None
    return layout.obx if layout is not None else None


def _onebox_export_dir(config: SessionConfig) -> Path:
    """Where the session's one OneBox analog export is: the first probe's ``export/``.

    One OneBox per run, so the driver exports it once, with the first probe's
    config. Every probe *reads* it from there -- to pick its cycle and to check
    its map -- and applies its own ``onebox_to_blackrock``; only the first
    probe's map is stamped into it. A one-probe session is its own first probe.
    """
    if config.neuropixels.bin_files is None or not config.probes:
        return config.paths.export_for("neuropixels")
    return config.for_probe(config.probes[0]).paths.export_for("neuropixels")


def _fitted_onebox_map(config: SessionConfig) -> align.LinearMap | None:
    """``onebox_to_blackrock`` from this probe's ``time_map.json``, if there is one."""
    map_path = config.paths.aligned / "time_map.json"
    if not map_path.exists():
        return None
    with open(map_path, "r", encoding="utf-8") as handle:
        composed = json.load(handle).get("onebox_to_blackrock")
    if not composed:
        return None
    return align.LinearMap(
        slope=float(composed["slope"]),
        intercept=float(composed["intercept"]),
        n_points=0,
        residuals_s=np.empty(0),
    )


def stamp_analog_timebase(
    config: SessionConfig, onebox_map: align.LinearMap
) -> Path | None:
    """Put the OneBox -> Blackrock map into an analog export. Samples untouched.

    The same rewrite :func:`stamp_lfp_timebase` does, with the composed
    ``onebox_to_blackrock`` map: ``t_blackrock(k) = slope * (k / fs) +
    intercept``, a uniform axis at rate ``fs / slope`` starting at the intercept.
    Returns the file it stamped, or ``None`` when there is no analog export.
    """
    from ._io.matlab import update_mat

    export_dir = config.paths.export_for("neuropixels")
    exports = sorted(export_dir.glob("*.analog.mat")) if export_dir.exists() else []
    if not exports:
        return None

    import h5py

    stamped = None
    for path in exports:
        with h5py.File(path, "r") as handle:
            fs = float(np.asarray(handle["analog/fs"][()]).ravel()[0])
        update_mat(path, {"analog": _blackrock_axis(fs, onebox_map)})
        log.info("stamped the Blackrock timebase into %s", path.name)
        stamped = path
    return stamped


def _blackrock_axis(fs: float, mapping: align.LinearMap) -> dict[str, Any]:
    """The fields that put a uniformly sampled export on Blackrock time. Pure."""
    return {
        "timebase": "blackrock",
        "fs_blackrock": fs / mapping.slope,
        "t0_blackrock": mapping.intercept,
        "blackrock_slope": mapping.slope,
        "blackrock_intercept": mapping.intercept,
    }


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


def _burst_on_probe_clock(
    config: SessionConfig, npx_burst: np.ndarray | None
) -> tuple[np.ndarray | None, align.LinearMap | None]:
    """The burst times on the probe's clock, and the map that put them there.

    The burst is read off the OneBox, the 1 Hz train off the probe's headstage:
    two crystals. The OneBox's own SY word carries the same 1 Hz wave, so pairing
    its edges with the probe's maps one clock onto the other. Without them the
    burst stays on the OneBox clock and is returned unchanged with ``None``.
    """
    obx_map = _onebox_to_probe(config)
    if npx_burst is None or not npx_burst.size:
        return npx_burst, obx_map
    if obx_map is None:
        log.warning(
            "no OneBox 1 Hz edges (%s.txt), so the burst stays on the OneBox clock and "
            "its comparison with Blackrock includes the OneBox-to-probe drift; "
            "re-run extract_sync to map it",
            extract.NPX_OBX_1HZ,
        )
        return npx_burst, None
    return obx_map.apply(npx_burst), obx_map


def _onebox_to_probe(config: SessionConfig) -> align.LinearMap | None:
    """The OneBox -> probe fit from the two SY 1 Hz trains, or None without them.

    Fitted whether or not there is a burst: the OneBox analog channels need it
    to reach Blackrock time, and so does the photodiode/eye cycle fallback.
    """
    npx_1hz = _load_edges(config, extract.NPX_1HZ)
    obx_1hz = _load_edges(config, extract.NPX_OBX_1HZ)
    if npx_1hz is None or not npx_1hz.size or obx_1hz is None or not obx_1hz.size:
        return None
    obx_map = align.fit_same_run_map(obx_1hz, npx_1hz, config.sync_period_s)
    log.info(
        "OneBox -> probe clock from %d edges: %+.2f ppm, offset %.6f s, "
        "fit residual max %.1f us",
        obx_map.n_points, obx_map.drift_ppm, obx_map.intercept,
        np.abs(obx_map.residuals_s).max() * 1e6,
    )
    return obx_map


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
    one probe's fit says nothing about another's. Each map goes beside its own
    sorting (``paths.aligned``, the probe's ``<neuropixels_dir>/<sorter>/``), so a
    local sorting stays local; only Blackrock's edge files are read from its
    folder. Nothing here knows about probes: it is handed a config already
    projected onto one.

    Coarse offset from the 14 s coded bursts first: their onsets alone are
    periodic and ambiguous, so the full pulse trains are matched on the coded
    intra-burst pattern to pick the right cycle. Then both 1 Hz trains are
    trimmed to their overlap and a linear map is fitted -- by TPrime where it is
    installed, otherwise by least squares on the same matched edges. Both paths
    write the same output file, with a record of which sorting it mapped
    (``*_spike_seconds_blackrock.json``, a hash of ``spike_times.npy``).

    **Everything written here is provisional.** A new map withdraws any earlier
    acceptance (``validation.json`` is removed, the LFP and OneBox exports go
    back to their own clocks); :func:`validate_remapping` decides whether it is
    published.

    Returns ``None`` when only one system is declared, or when asked for Blackrock
    itself (there is nothing to map the reference onto). Missing 1 Hz edge files
    in a session with both systems **raise**: there is no map from this run, and
    that is a failure, not a stage with nothing to do. Any earlier acceptance
    is withdrawn first, before the edges are read, so no failure below --
    missing edges, too few pairs, burst codes that never agree -- leaves it
    standing.
    """
    _check(system)
    if not config.aligns_systems or system == REFERENCE_SYSTEM:
        return None

    from ._sync import tprime as tprime_mod

    # Before reading anything: whatever an earlier run accepted was fitted from
    # other edges, and a fit that raises below must not leave it standing.
    _withdraw_acceptance(config)

    br_1hz = _load_edges(config, extract.BR_1HZ)
    npx_1hz = _load_edges(config, extract.NPX_1HZ)
    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst, obx_map = _burst_on_probe_clock(
        config, _load_edges(config, extract.NPX_BURST)
    )

    missing = [
        name
        for name, values in ((extract.BR_1HZ, br_1hz), (extract.NPX_1HZ, npx_1hz))
        if values is None or values.size == 0
    ]
    if missing:
        # Both systems recorded, so there must be a map; the earlier acceptance
        # is already withdrawn above.
        raise FileNotFoundError(
            f"missing or empty edge files: {', '.join(missing)} in "
            f"{config.paths.sync_for('neuropixels')}; without both 1 Hz trains there "
            "is no map from this run, and any earlier acceptance has been withdrawn. "
            "Run extract_sync, and check the sync channels were recorded."
        )
    rate_problem = _rate_problem(config)
    if rate_problem:
        raise ValueError(rate_problem)
    edge_fs = _edge_rate(config)

    # 1. Coarse map from the 14 s coded bursts, paired by code so drift cannot
    #    pick the wrong cycle -- and a line, so it holds at both ends of hours.
    match = None
    vote = None
    tried = None
    if br_burst is not None and npx_burst is not None and br_burst.size and npx_burst.size:
        match = burst.match_bursts(br_burst, npx_burst, min_gap_s=config.burst_interval_s / 2)
        offset = match.offset_s
        coarse = match.apply
        log.info(
            "coarse map from %d code-matched bursts (of %d Blackrock / %d SpikeGLX; "
            "runner-up offset had %d): %+.1f ppm, offset %.6f s, max residual %.3f ms",
            match.n_matched, match.n_reference, match.n_other, match.runner_up_matches,
            (match.slope - 1.0) * 1e6, offset, match.max_residual_s * 1e3,
        )
        cycles_from = "burst"
    else:
        log.warning(
            "no burst edges on one or both systems; the 1 Hz alone cannot tell "
            "which cycle pairs with which -- trying the photodiode, then the eye"
        )
        offset = burst.estimate_offset(br_1hz, npx_1hz, tolerance_s=0.1)
        log.info("coarse offset %.6f s from the 1 Hz trains", offset)
        # The provisional line has the drift right and the cycle possibly wrong;
        # the correction is a whole number of periods added to it.
        provisional = _fit_1hz(config, br_1hz, npx_1hz, offset)[2]
        vote, cycles_from, tried = _cycle_from_behaviour(config, provisional, obx_map)
        shift = 0.0
        if vote is not None:
            shift = vote.shift_s
            offset += shift
            log.info("cycle picked by the %s: %s", cycles_from, vote.summary())
        else:
            log.warning(
                "%s; the map may be off by whole seconds and cannot be validated",
                cycles_from,
            )
            cycles_from = "1hz_only"

        def coarse(times):
            return provisional.apply(times) + shift

    br_trim, npx_trim, mapping = _fit_1hz(config, br_1hz, npx_1hz, coarse)
    aligned_dir = config.paths.aligned
    aligned_dir.mkdir(parents=True, exist_ok=True)
    br_trim_path = catgt.write_edge_file(aligned_dir / "blackrock_1hz_trimmed.txt", br_trim)
    npx_trim_path = catgt.write_edge_file(aligned_dir / "npx_1hz_trimmed.txt", npx_trim)
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
        _save_map(aligned_dir, mapping, offset, config, obx_map, method="map_only",
                  cycles_from=cycles_from, cycle_vote=vote, coarse_tried=tried,
                  inputs=_sync_inputs(config))
        return TimeMap(mapping, offset, "map_only", burst_match=match)

    # The rate the edges were converted with; _rate_problem has already checked
    # the sorting's own (params.py) agrees, so the map applies to these seconds.
    fs = edge_fs

    # Kilosort writes sample indices; TPrime needs seconds. Skipping this is the
    # easiest way to produce a confidently wrong alignment.
    # Both per-spike arrays sit beside the sorting they came from.
    spike_samples = np.load(spike_times_path)
    spike_seconds = tprime_mod.spike_times_to_seconds(spike_samples, fs)
    seconds_path = results_dir / f"{system}_spike_seconds.npy"
    np.save(seconds_path, spike_seconds)

    out_path = _aligned_spike_path(config, system)
    _aligned_identity_path(out_path).unlink(missing_ok=True)
    method, tprime_error = "linear_fit", None
    if config.machine.has_tprime:
        try:
            _run_tprime_shifted(config, br_trim, npx_trim, mapping,
                                seconds_path, out_path, spike_seconds.size)
        except RuntimeError as error:          # TPrimeTimeout included
            # The least-squares map is the same 1 Hz pairing fitted as a line;
            # for linear drift it agrees with TPrime to well under a millisecond.
            tprime_error = str(error).splitlines()[0]
            log.warning("TPrime did not finish; using the least-squares map instead\n%s",
                        error)
        else:
            log.info("TPrime mapped %d spike times -> %s", spike_seconds.size, out_path.name)
            method = "tprime"
    if method == "linear_fit":
        np.save(out_path, mapping.apply(spike_seconds))
        if tprime_error is None:
            log.info(
                "TPrime not available on machine '%s'; applied the least-squares map to "
                "%d spike times instead",
                config.machine.name, spike_seconds.size,
            )

    _write_aligned_identity(out_path, spike_samples, method)

    _save_map(aligned_dir, mapping, offset, config, obx_map, method=method,
              cycles_from=cycles_from, tprime_error=tprime_error, cycle_vote=vote,
              coarse_tried=tried, inputs=_sync_inputs(config))
    log.info("the map and %s are provisional until validate_remapping accepts them",
             out_path.name)
    return TimeMap(mapping, offset, method, aligned_path=out_path, burst_match=match)


def _run_tprime_shifted(
    config: SessionConfig,
    br_trim: np.ndarray,
    npx_trim: np.ndarray,
    mapping: align.LinearMap,
    seconds_path: Path,
    out_path: Path,
    n_events: int,
) -> None:
    """TPrime with the probe clock put onto Blackrock's by the fitted line first.

    **What TPrime does**, measured on a real session (fits its output to 0.1 us):
    for an event at ``t`` it takes the last *reference* edge ``to_k <= t`` and
    returns ``t + (to_k - from_k)`` -- one constant offset per sync interval,
    chosen by comparing a from-stream time against to-stream edges. That is only
    right when the two edge lists sit on top of each other.

    So both are brought together before TPrime sees them, by the 1 Hz line
    ``L(t) = slope*t + intercept`` already fitted from these same pairs:

    - **from**: ``slope*npx`` and ``slope*t`` -- the probe clock, drift removed;
    - **to**:   ``br - intercept`` -- Blackrock, its 1.8e9 s PTP origin removed.

    The two lists then differ only by the line's residual (< 1 ms), so TPrime's
    interval switches at the probe's own edge rather than up to drift x duration
    before it (7.6 ms by the end of a 2 h session at 1 ppm, when only the first
    difference was subtracted). Its output plus ``intercept`` is
    ``L(t) + (br_k - L(npx_k))``: the line, plus that interval's measured
    departure from it -- the slow wander TPrime exists to follow. Values stay
    near the run's own seconds, so nothing loses float precision.

    The offset still steps by whole samples at each edge (edge quantisation),
    so two spikes straddling an edge closer than the step can swap order. That
    is TPrime's model, not this shift.
    """
    from ._sync import catgt, tprime as tprime_mod

    slope, intercept = float(mapping.slope), float(mapping.intercept)
    work = out_path.parent
    to_path = catgt.write_edge_file(
        work / "blackrock_1hz_trimmed_tprime.txt", br_trim - intercept)
    from_path = catgt.write_edge_file(
        work / "npx_1hz_trimmed_tprime.txt", slope * npx_trim)
    events_path = work / f"{seconds_path.stem}_tprime.npy"
    np.save(events_path, slope * np.load(seconds_path).astype(np.float64))
    # TPrime writes to a fresh file, never to out_path: an exit that wrote
    # nothing must not leave the previous run's times to be read back.
    tprime_out = work / f"{seconds_path.stem}_tprime_out.npy"
    tprime_out.unlink(missing_ok=True)
    log.info("TPrime edges brought together by the 1 Hz line: max gap %.3f ms",
             1e3 * float(np.abs((br_trim - intercept) - slope * npx_trim).max()))
    args = tprime_mod.build_tprime_args(
        to_stream=to_path,
        from_streams={1: from_path},
        events=[tprime_mod.TPrimeEvent(1, events_path, tprime_out)],
        sync_period_s=config.sync_period_s,
    )
    log.info("TPrime command (probe x %.9f, Blackrock - %.6f s): %s",
             slope, intercept, " ".join(args))
    try:
        tprime_mod.run_tprime(config.machine.tprime_dir, args,
                              timeout_s=tprime_mod.tprime_timeout_s(n_events))
        mapped = _checked_tprime_output(tprime_out, n_events)
    finally:
        events_path.unlink(missing_ok=True)       # a scaled copy of every spike
        tprime_out.unlink(missing_ok=True)
    np.save(out_path, mapped + intercept)


def _checked_tprime_output(path: Path, n_events: int) -> np.ndarray:
    """TPrime's output, or ``RuntimeError`` -- which the linear fallback catches.

    It exited cleanly, but that is all the exit code says: the file must exist,
    load, hold one finite time per event.
    """
    try:
        mapped = np.load(path).astype(np.float64).ravel()
    except (OSError, ValueError) as error:
        raise RuntimeError(f"TPrime exited cleanly but its output is unreadable: {error}")
    if mapped.size != n_events:
        raise RuntimeError(f"TPrime wrote {mapped.size} times for {n_events} spikes")
    if not np.isfinite(mapped).all():
        raise RuntimeError(
            f"TPrime wrote {int((~np.isfinite(mapped)).sum())} non-finite times")
    return mapped


#: Every edge file a map depends on: the two 1 Hz trains it is fitted from, the
#: OneBox's 1 Hz that puts the OneBox on the probe clock (and so sets
#: ``onebox_to_blackrock``), and the bursts it is checked against.
_SYNC_INPUTS = (extract.BR_1HZ, extract.NPX_1HZ, extract.NPX_OBX_1HZ,
                extract.BR_BURST, extract.NPX_BURST)


def _edge_digest(edges: np.ndarray | None) -> str:
    """SHA-256 of one edge train as loaded; a missing file hashes as empty."""
    import hashlib

    flat = np.ascontiguousarray(
        np.asarray(edges if edges is not None else [], dtype=np.float64).ravel())
    return hashlib.sha256(flat.tobytes()).hexdigest()


_UNREADABLE = "unreadable"


def _sync_inputs(config: SessionConfig) -> dict[str, str]:
    """SHA-256 of every sync edge file a map depends on, as on disk now.

    A file that cannot be read is ``"unreadable"`` rather than an exception, so
    the acceptance check that asks says "changed" and the export falls back to
    its own clock instead of the run stopping.
    """
    digests = {}
    for name in _SYNC_INPUTS:
        try:
            digests[name] = _edge_digest(_load_edges(config, name))
        except (OSError, ValueError):
            digests[name] = _UNREADABLE
    return digests


def _changed_inputs(config: SessionConfig, recorded: dict | None) -> list[str]:
    """Which sync edge files differ from what ``recorded`` hashed (all, if nothing was).

    An unreadable file is always changed, whatever was recorded.
    """
    current = _sync_inputs(config)
    recorded = recorded or {}
    return [name for name in _SYNC_INPUTS
            if current[name] == _UNREADABLE or recorded.get(name) != current[name]]


def _map_digest(config: SessionConfig) -> str | None:
    """SHA-256 of this probe's ``time_map.json`` as it is on disk, or None."""
    import hashlib

    try:
        return hashlib.sha256((config.paths.aligned / "time_map.json").read_bytes()).hexdigest()
    except OSError:
        return None


def _edge_rate(config: SessionConfig) -> float | None:
    """The AP rate the Neuropixels edges were converted to seconds with, if recorded."""
    try:
        with open(config.paths.sync_for("neuropixels") / extract.NPX_EDGE_RATE,
                  "r", encoding="utf-8") as handle:
            return float(json.load(handle)["ap_fs"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _rate_problem(config: SessionConfig) -> str | None:
    """Why spike samples and the edges would be on different rates, or None.

    A rate error shared by both is absorbed by the clock fit; a *mix* -- the
    ``.meta`` edited after extraction, or a sorting made at another rate --
    shifts every spike by up to (rate difference x session length), and the
    burst check, which sees only edges, cannot notice. So the edges' rate
    (recorded at extraction) must equal the ``.meta``'s now and, when there is
    a sorting, the one in its ``params.py``.
    """
    recorded = _edge_rate(config)
    if recorded is None:
        return ("the Neuropixels edges were extracted before their AP rate was "
                "recorded; re-run extract_sync")
    try:
        current = float(_neuropixels_stream(config).fs)
    except (OSError, ValueError) as error:
        # Not readable here (no .meta, share away): an edit cannot be detected,
        # but that is no reason to void a map. The sorting's rate still is.
        log.info("AP rate not re-read (%s); checking the sorting's rate only", error)
        current = recorded
    if abs(current - recorded) > 1e-6:
        return (f"the AP rate is {current:g} Hz in the .meta now, but the edges were "
                f"extracted at {recorded:g} Hz; re-run extract_sync, then time_remapping")
    params_path = config.paths.sorted_for("neuropixels") / "params.py"
    if params_path.exists():
        from ._export.curated import parse_params_py

        try:
            sorted_fs = float(parse_params_py(params_path).get("sample_rate") or 0.0)
        except (OSError, ValueError, SyntaxError) as error:
            return f"cannot read the sorting's rate from {params_path} ({error})"
        if not sorted_fs:
            return f"no sample_rate in {params_path}; cannot convert spike times"
        if abs(sorted_fs - recorded) > 1e-6:
            return (f"the sorting used {sorted_fs:g} Hz (params.py) but the edges "
                    f"were extracted at {recorded:g} Hz; re-sort (the .meta changed "
                    "since the sort), then time_remapping")
    return None


def _acceptance_problem(config: SessionConfig, decision: str = "accepted") -> str | None:
    """Why this probe's map cannot be treated as accepted now, or None if it can.

    Three things, all checked on disk every time, so an export run on its own --
    after ``extract_sync`` was re-run, say -- cannot label anything Blackrock
    time with a map fitted from other edges:

    - ``validation.json["accepted"]`` (a burst PASS or ``CONSISTENT (no burst)``);
    - the ``time_map.json`` it accepted is the one on disk (its SHA-256);
    - that map was fitted from, and checked against, the sync edge files on
      disk now (its ``inputs``: both 1 Hz trains, the OneBox 1 Hz, both bursts).
    """
    aligned = config.paths.aligned
    try:
        with open(aligned / "validation.json", "r", encoding="utf-8") as handle:
            validation = json.load(handle)
    except (OSError, ValueError):
        return "no validation.json"
    if validation.get(decision) is not True:
        what = "it" if decision == "accepted" else "its line for the exports"
        return f"validate_remapping did not accept {what} ({validation.get('verdict')})"
    if validation.get("time_map_sha256") != _map_digest(config):
        return ("time_map.json is not the map validate_remapping accepted; "
                "re-run time_remapping and validate_remapping")
    try:
        with open(aligned / "time_map.json", "r", encoding="utf-8") as handle:
            inputs = json.load(handle).get("inputs")
    except (OSError, ValueError):
        return "time_map.json cannot be read"
    changed = _changed_inputs(config, inputs)
    if changed:
        return (f"the sync edge files changed since the map was fitted "
                f"({', '.join(changed)}); re-run time_remapping and validate_remapping")
    return _rate_problem(config)


def _alignment_accepted(config: SessionConfig) -> bool:
    """Whether this probe's map is accepted *and* still the map of these edges.

    See :func:`_acceptance_problem`. An acceptance that no longer holds is
    logged, since everything that asks then stays on its own clock.
    """
    problem = _acceptance_problem(config)
    if problem is not None and (config.paths.aligned / "validation.json").exists():
        log.warning("alignment not used: %s", problem)
    return problem is None


def _exports_accepted(config: SessionConfig) -> bool:
    """Whether the *line* may be stamped into the LFP and OneBox exports.

    They can only carry a line (``t0_blackrock``/``fs_blackrock``), so this is
    the line's own decision, ``validation.json["exports_accepted"]``, bound to
    the same map and edges as ``accepted``. Without TPrime the two are one
    decision; with it they can differ either way.
    """
    problem = _acceptance_problem(config, "exports_accepted")
    if problem is not None and (config.paths.aligned / "validation.json").exists():
        log.info("exports not stamped: %s", problem)
    return problem is None


def _withdraw_acceptance(config: SessionConfig) -> None:
    """Forget an earlier acceptance and take the exports back to their own clocks.

    The sync report goes too: it describes the map being withdrawn, and a PASS
    left in the bundle would read as this run's.
    """
    (config.paths.aligned / "validation.json").unlink(missing_ok=True)
    export_dir = config.paths.export_for("neuropixels")
    for name in ("sync_report.json", "sync_report.txt"):
        (export_dir / name).unlink(missing_ok=True)
    (config.paths.figures_for("neuropixels") / SYNC_FIGURE).unlink(missing_ok=True)
    _remove_copied_sync_records(config)
    _unstamp_exports(config)
    _remove_aligned_spike_exports(config)


def _remove_copied_sync_records(config: SessionConfig) -> None:
    """Delete the bundle's copies of ``validation.json`` and ``time_map.json``.

    They describe the decision they were copied after; once a new one is made
    or an old one withdrawn they are stale, and ``export_results`` copies the
    current records again.
    """
    export_dir = config.paths.export_for("neuropixels")
    for name in ("validation.json", "time_map.json"):
        (export_dir / name).unlink(missing_ok=True)


def _remove_aligned_spike_exports(config: SessionConfig) -> None:
    """Delete the spike exports stamped with a map that is being withdrawn.

    ``sorted_spikes.mat`` and ``waveforms.mat`` carry a time per spike; on
    Blackrock time those came from the withdrawn map, and next to LFP and OneBox
    exports back on their own clocks they would leave the bundle on two clocks.
    ``export_results`` writes all three again (waveforms included) on the map
    accepted then, or in sorter time. Exports in sorter time are kept: they
    never used a map. Without a readable ``sorting_summary_info.json`` saying
    which, they go too.
    """
    export_dir = config.paths.export_for("neuropixels")
    summary = export_dir / "sorting_summary_info.json"
    products = (sorted(export_dir.glob("*.sorted_spikes.mat"))
                + sorted(export_dir.glob("*.waveforms.mat")))
    if not products and not summary.exists():
        return
    try:
        with open(summary, "r", encoding="utf-8") as handle:
            timebase = json.load(handle).get("timebase")
    except (OSError, ValueError):
        timebase = None
    if timebase not in (None, "blackrock"):
        return
    for path in [*products, summary]:
        if path.exists():
            path.unlink()
            log.info("removed %s: its spike times were on the withdrawn map; "
                     "export_results writes it again", path.name)


def _stamp_accepted(config: SessionConfig, mapping: align.LinearMap) -> None:
    """Put Blackrock time into the LFP and OneBox exports. Only on acceptance."""
    stamp_lfp_timebase(config, mapping)
    onebox = _fitted_onebox_map(config)
    if onebox is not None:
        stamp_analog_timebase(config, onebox)
    elif list(config.paths.export_for("neuropixels").glob("*.analog.mat")):
        log.warning(
            "the OneBox analog export stays on the OneBox clock: no OneBox 1 Hz "
            "edges to map it with (re-run extract_sync)"
        )


def _unstamp_exports(config: SessionConfig) -> None:
    """Put the LFP and OneBox exports back on their own clocks, NaN for Blackrock's.

    What they were written with. Only the small fields are rewritten.
    """
    from ._io.matlab import update_mat

    export_dir = config.paths.export_for("neuropixels")
    if not export_dir.exists():
        return
    nan = float("nan")
    unset = {"fs_blackrock": nan, "t0_blackrock": nan,
             "blackrock_slope": nan, "blackrock_intercept": nan}
    for pattern, group, own in (("*.lfp.mat", "lfp", "stream"),
                                ("*.analog.mat", "analog", "onebox")):
        for path in sorted(export_dir.glob(pattern)):
            update_mat(path, {group: {"timebase": own, **unset}})
            log.info("%s back on the %s clock until the map is accepted", path.name, own)


def _aligned_identity_path(aligned_path: Path) -> Path:
    return aligned_path.with_suffix(".json")


def _spike_fingerprint(samples: np.ndarray) -> str:
    """SHA-256 of the spike samples as int64: which sorting a per-spike array is for."""
    import hashlib

    flat = np.ascontiguousarray(np.asarray(samples).ravel(), dtype=np.int64)
    return hashlib.sha256(flat.tobytes()).hexdigest()


def _write_aligned_identity(aligned_path: Path, samples: np.ndarray, method: str) -> None:
    """Beside the aligned array: the sorting it was mapped from, by content."""
    with open(_aligned_identity_path(aligned_path), "w", encoding="utf-8") as handle:
        json.dump({"spike_times_sha256": _spike_fingerprint(samples),
                   "n_spikes": int(np.asarray(samples).size), "method": method},
                  handle, indent=2)


def _checked_aligned_times(path: Path, n_spikes: int) -> tuple[np.ndarray | None, str | None]:
    """``(times, None)``, or ``(None, why)``: it loads, one finite time per spike."""
    try:
        times = np.asarray(np.load(path), dtype=np.float64).ravel()
    except (OSError, ValueError) as error:
        return None, f"{path.name} cannot be read ({error})"
    if times.size != n_spikes:
        return None, f"{path.name} holds {times.size} times for {n_spikes} spikes"
    if not np.isfinite(times).all():
        return None, (f"{path.name} holds {int((~np.isfinite(times)).sum())} "
                      "non-finite times")
    return times, None


def _aligned_mismatch(aligned_path: Path, samples: np.ndarray) -> str | None:
    """Why the aligned array is not this sorting's, or None when it is."""
    try:
        with open(_aligned_identity_path(aligned_path), "r", encoding="utf-8") as handle:
            recorded = json.load(handle).get("spike_times_sha256")
    except (OSError, ValueError):
        return f"{aligned_path.name} has no record of the sorting it was mapped from"
    if recorded != _spike_fingerprint(samples):
        return f"{aligned_path.name} was mapped from a different sorting"
    return None


def _fit_1hz(
    config: SessionConfig, br_1hz: np.ndarray, npx_1hz: np.ndarray, coarse
) -> tuple[np.ndarray, np.ndarray, align.LinearMap]:
    """Trim both 1 Hz trains to their overlap, pair them under ``coarse``, fit a line.

    Returns ``(blackrock, probe, line)`` where the two edge arrays are the
    **matched pairs** -- same length, element ``i`` of each the same pulse --
    not the trimmed trains. TPrime aligns edge *sequences*, so an edge present
    in only one recording (a pulse dropped mid-session, not just at the ends)
    would shift every later interval by a cycle; handing it the pairs the line
    was fitted through means a dropped pulse costs one longer interval instead.

    ``coarse`` maps probe seconds onto Blackrock time to within a quarter
    period, or is a constant offset.
    """
    if not callable(coarse):
        constant = float(coarse)

        def coarse(times, constant=constant):
            return np.asarray(times, dtype=np.float64) + constant

    br_trim, npx_trim, npx_coarse = align.trim_pair_under_map(
        br_1hz, npx_1hz, coarse(npx_1hz), margin_s=config.sync_period_s / 4
    )
    log.info(
        "1 Hz trains trimmed to overlap: Blackrock %d -> %d, SpikeGLX %d -> %d",
        br_1hz.size, br_trim.size, npx_1hz.size, npx_trim.size,
    )
    ref_idx, other_idx = burst.match_times(
        br_trim, npx_coarse, 0.0, tolerance_s=config.sync_period_s / 4
    )
    if ref_idx.size < 2:
        raise ValueError(f"only {ref_idx.size} matched 1 Hz edges; cannot fit a time map")
    unpaired = br_trim.size + npx_trim.size - 2 * ref_idx.size
    if unpaired:
        log.warning(
            "%d 1 Hz edge(s) inside the overlap had no partner on the other system "
            "(Blackrock %d, SpikeGLX %d, %d pairs); only the pairs are used",
            unpaired, br_trim.size, npx_trim.size, ref_idx.size,
        )
    br_pairs, npx_pairs = br_trim[ref_idx], npx_trim[other_idx]
    return br_pairs, npx_pairs, align.fit_linear_map(npx_pairs, br_pairs)


#: How far, in seconds, a 1 Hz-only offset may be from the truth. It is
#: ambiguous by whole periods within roughly the difference between when the two
#: recordings started; a minute covers any start order a rig produces.
CYCLE_SEARCH_S = 60.0


def _cycle_from_behaviour(
    config: SessionConfig,
    provisional: align.LinearMap,
    obx_map: align.LinearMap | None,
) -> tuple[Any, str, list[dict[str, str]]]:
    """``(CycleVote or None, source, tried)``: the whole-cycle correction, from
    the photodiode, else the eye. On ``None`` the source says why not.
    ``tried`` lists each channel asked, in order, and what it answered.

    Both signals are on the OneBox and on Blackrock's ``.ns2``. The OneBox copy
    reaches Blackrock time through ``onebox_to_probe`` and the provisional 1 Hz
    fit, which is right but for a whole number of periods; the vote finds it.
    Never a fit: the corrected 1 Hz pairing is refitted afterwards.
    """
    from dataclasses import replace as _replace

    from ._io import analog
    from ._sync import cycles

    brk, npx = config.blackrock, config.neuropixels
    if obx_map is None:
        return None, "no OneBox 1 Hz edges, so the OneBox copies cannot be mapped", []
    shared = [n for n in brk.analog_inputs if n in npx.analog_channels]
    if brk.analog_file is None or not shared:
        return None, ("no channel named in both blackrock.analog_inputs and "
                      "neuropixels.analog_channels to pick the cycle with"), []
    try:
        onebox = analog.load_onebox_analog(
            analog.find_onebox_analog(_onebox_export_dir(config)),
            require_blackrock=False,
        )
        br = analog.open_blackrock_analog(
            brk.analog_file, {n: brk.analog_inputs[n] for n in shared},
            brk.nsp_frontend_channels, gap_tolerance_ms=brk.gap_tolerance_ms,
            tolerance_s=config.alignment_tolerance_s,
        )
    except (OSError, ValueError, KeyError) as error:
        return None, f"no photodiode or eye to pick the cycle with ({error})", []
    composed = align.compose_linear_maps(provisional, obx_map)
    onebox = _replace(onebox, slope=composed.slope, intercept=composed.intercept)
    period = config.sync_period_s

    tried: list[dict[str, str]] = []
    if "photodiode" in shared:
        try:
            obx_t, _ = analog.detect_onebox_events(onebox, "photodiode")
            br_t, _ = analog.detect_blackrock_events(br, "photodiode")
            vote = cycles.cycle_shift_from_events(
                br_t, composed.apply(obx_t), period_s=period, search_s=CYCLE_SEARCH_S,
            )
        except (OSError, ValueError) as error:
            vote, note = None, str(error)
        else:
            note = f"{obx_t.size} OneBox / {br_t.size} Blackrock edges, no clear winner"
        if vote is not None:
            tried.append({"channel": "photodiode", "result": "picked"})
            return vote, "photodiode", tried
        tried.append({"channel": "photodiode", "result": note})
        log.info("the photodiode did not pick the cycle: %s", note)

    for name in [n for n in shared if n != "photodiode"]:
        try:
            _, lags, _ = analog.measure_lags(
                onebox, br, name, n_windows=20, window_s=2 * CYCLE_SEARCH_S,
                max_lag_s=CYCLE_SEARCH_S + period, min_corr=0.5, differentiate=True,
            )
        except (OSError, ValueError) as error:
            tried.append({"channel": name, "result": str(error)})
            log.info("the %s did not pick the cycle: %s", name, error)
            continue
        vote = cycles.cycle_shift_from_lags(lags, period_s=period)
        if vote is not None:
            tried.append({"channel": name, "result": "picked"})
            return vote, name, tried
        tried.append({"channel": name,
                      "result": f"{lags.size} windows locked, no clear winner"})
        log.info("the %s did not pick the cycle: %d windows locked, no clear winner",
                 name, lags.size)
    return None, "could not pick the cycle: " + "; ".join(
        f"{t['channel']}: {t['result']}" for t in tried), tried


def _save_map(
    aligned_dir: Path,
    mapping: align.LinearMap,
    offset: float,
    config: SessionConfig,
    obx_map: align.LinearMap | None = None,
    method: str | None = None,
    cycles_from: str | None = None,
    tprime_error: str | None = None,
    cycle_vote: Any = None,
    coarse_tried: list[dict[str, str]] | None = None,
    inputs: dict[str, str] | None = None,
) -> None:
    obx_to_br = None if obx_map is None else align.compose_linear_maps(mapping, obx_map)
    # Only a burst is held out of the fit; without one nothing validates the map
    # and there was no burst whose clock to report.
    had_burst = cycles_from == "burst"
    payload = {
        "session": config.session,
        "reference_timebase": "blackrock",
        # The 1 Hz edges this map was fitted from, by content: validate_remapping
        # refuses the map once the edge files no longer match.
        "inputs": inputs,
        "coarse_offset_s": offset,
        # What picked which 1 Hz cycle pairs with which: "burst", "photodiode",
        # an eye channel's label, or "1hz_only" (ambiguous by whole seconds).
        "cycles_from": cycles_from,
        # Without a burst: the vote that chose the whole-second offset in its
        # place. Photodiode votes count flashes, the others correlation windows.
        "sync_period_s": config.sync_period_s,
        "cycle_vote": None if cycle_vote is None else {
            "source": cycles_from,
            "shift_s": cycle_vote.shift_s,
            "n_periods": cycle_vote.n_periods,
            "n_support": cycle_vote.n_support,
            "n_runner_up": cycle_vote.n_runner_up,
            "n_total": cycle_vote.n_total,
            "counted": "flashes" if cycles_from == "photodiode" else "windows",
        },
        # Without a burst: every channel asked for the whole-second offset, in
        # order (photodiode, then the eye), and what each answered.
        "coarse_match_tried": coarse_tried,
        "slope": mapping.slope,
        "intercept": mapping.intercept,
        "drift_ppm": mapping.drift_ppm,
        "n_points": mapping.n_points,
        "fit_max_residual_s": float(np.abs(mapping.residuals_s).max()),
        "fit_rms_residual_s": float(np.sqrt(np.mean(mapping.residuals_s ** 2))),
        # tprime, linear_fit, or map_only when there was no sorting to map yet.
        "method": method,
        # Why TPrime was set aside for the least-squares map, when it was.
        "tprime_error": tprime_error,
        "formula": "blackrock_time = slope * spikeglx_time + intercept",
        # Which clock the burst was on when it picked the coarse offset; null
        # without a burst.
        "burst_clock": None if not had_burst else ("probe" if obx_map is not None else "onebox"),
        "onebox_to_probe": None if obx_map is None else {
            "slope": obx_map.slope,
            "intercept": obx_map.intercept,
            "drift_ppm": obx_map.drift_ppm,
            "n_points": obx_map.n_points,
            "fit_max_residual_s": float(np.abs(obx_map.residuals_s).max()),
            "fit_rms_residual_s": float(np.sqrt(np.mean(obx_map.residuals_s ** 2))),
            # spikeglx_time is the probe's clock: one name for it throughout.
            "formula": "spikeglx_time = slope * onebox_time + intercept",
        },
        # The two fits above composed: what a OneBox analog channel (eye, pupil,
        # photodiode) needs to land on Blackrock time. It is the exact chain the
        # held-out burst travels in validate_remapping, so validation.json is its
        # quality check.
        "onebox_to_blackrock": None if obx_to_br is None else {
            "slope": obx_to_br.slope,
            "intercept": obx_to_br.intercept,
            "drift_ppm": obx_to_br.drift_ppm,
            "formula": "blackrock_time = slope * onebox_time + intercept",
            # The held-out burst checks this chain; without one, nothing does.
            "validated_by": "validation.json" if had_burst else None,
        },
        "clocks": _clock_definitions(config),
    }
    with open(Path(aligned_dir) / "time_map.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _clock_definitions(config: SessionConfig) -> dict[str, Any]:
    """What each ``*_time`` in ``time_map.json`` means, so it can be applied by hand.

    A SpikeGLX time is ``sample / rate`` from the stream's first sample, with the
    rate the stream's own ``.meta`` states -- the one the edge extraction used.
    Converting samples at any other rate (the nominal 30 kHz, say) reintroduces a
    ppm-level drift in exactly the place the map exists to remove it, so the rate
    is written out. Reading it touches only the ``.meta``; an unreadable one
    leaves ``sample_rate_hz`` null rather than stopping the map being saved.
    """
    from ._io import spikeglx

    def stream(path: Path | None, n_chan=None, fs=None, has_sy=None) -> dict[str, Any] | None:
        if path is None:
            return None
        record: dict[str, Any] = {
            "definition": "seconds = sample_index / sample_rate_hz, from the file's first sample",
            "file": str(path),
            "sample_rate_hz": None,
        }
        try:
            record["sample_rate_hz"] = spikeglx.stream_info(path, n_chan, fs, has_sy).fs
        except (OSError, ValueError) as err:
            log.warning("time_map.json: no sample rate for %s (%s)", Path(path).name, err)
        return record

    npx = config.neuropixels
    layout = spikeglx.run_layout(npx.bin_file) if npx.bin_file is not None else None
    return {
        "blackrock_time": {
            "definition": "seconds on the NSP PTP clock -- the axis of the .nev "
            "markers and Blackrock's own analog channels (see nsp_time_map.json)",
        },
        "spikeglx_time": stream(npx.bin_file, npx.n_chan_bin, npx.sample_rate,
                                npx.has_sy_word),
        "onebox_time": stream(layout.obx if layout is not None else None),
    }


@dataclass(frozen=True)
class Unvalidated:
    """``validate_remapping`` ran, wrote the sync report, and had no burst to hold out.

    ``verdict`` is the report's: ``CONSISTENT (no burst)`` when the channels
    recorded on both systems agree, ``NOT VALIDATED`` otherwise.

    Only the first is accepted. Without a burst the 1 Hz fixes the drift and
    the offset only modulo whole periods, so with nothing independent agreeing
    the map may be whole seconds off however small its fit residuals are.
    ``passed`` is what the runner gates on.
    """

    report_path: Path | None
    reason: str
    verdict: str = "NOT VALIDATED"

    @property
    def passed(self) -> bool:
        return self.verdict.startswith("CONSISTENT")

    def summary(self) -> str:
        where = f"; sync report at {self.report_path}" if self.report_path else ""
        return f"{self.verdict} ({self.reason}){where}"


def validate_remapping(
    config: SessionConfig, figures: bool = True
) -> Any | None:
    """Check one probe's map against the 14 s burst onsets. Returns a ``ValidationReport``.

    A real check precisely because the bursts were **held out** of the 1 Hz fit.
    Uncorrected clock drift of tens of ppm is 72 ms/hour at 20 ppm and fails the
    1 ms tolerance, which is the point.

    The burst is recorded on the OneBox, not the probe, so it is first mapped onto
    the probe clock by the OneBox's own 1 Hz edges. ``validation.json`` says
    whether that happened (``burst_clock``); when it did not, the residuals
    include the OneBox-to-probe drift as well as the alignment's error.

    Returns ``None`` when only one system is declared. With both, a missing map
    -- or one not fitted from the sync edge files now on disk -- is
    :class:`Unvalidated` and not accepted, so a run that could not fit cannot
    publish an earlier run's map. With no burst on one side it still writes the sync report and returns an
    :class:`Unvalidated` saying so -- the 1 Hz fit residuals in ``time_map.json``
    are then the only quality measure, and they are not held-out.

    **This is what publishes the map.** ``time_remapping``'s outputs are
    provisional; only an accepted map -- a burst PASS, or ``CONSISTENT (no
    burst)`` -- is written into ``validation.json`` as ``accepted``, stamped into
    the LFP and OneBox exports, and lets ``export_results`` use the aligned
    spike times. Anything else leaves them on their own clocks, and the sync
    report says why.
    """
    from ._plots import summary as plots

    if not config.aligns_systems:
        return None

    aligned_dir = config.paths.aligned
    map_path = aligned_dir / "time_map.json"
    if not map_path.exists():
        return _refuse(config, f"no time map at {map_path}: time_remapping fitted none")

    with open(map_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    changed = _changed_inputs(config, payload.get("inputs"))
    if changed:
        return _refuse(config, (
            "time_map.json was not fitted from the sync edge files now on disk "
            f"(changed: {', '.join(changed)})"
            if payload.get("inputs") else
            "time_map.json predates maps recording their sync edges")
            + "; re-run time_remapping")
    rate_problem = _rate_problem(config)
    if rate_problem:
        return _refuse(config, rate_problem)
    mapping = align.LinearMap(
        slope=float(payload["slope"]),
        intercept=float(payload["intercept"]),
        n_points=int(payload["n_points"]),
        residuals_s=np.empty(0),
    )

    # The spikes went through TPrime: whichever path judges the map, it must be
    # able to reconstruct TPrime's, or it would accept one the spikes never got.
    # Read and checked once; the same arrays rebuild TPrime's map below.
    tprime_unchecked = None
    pairs = None
    if payload.get("method") == "tprime":
        pairs = _tprime_pairs(aligned_dir)
        if isinstance(pairs, str):
            tprime_unchecked = (
                f"the spikes went through TPrime, but its matched edge pairs are "
                f"unusable ({pairs}), so that map cannot be checked; re-run "
                "time_remapping")

    br_burst = _load_edges(config, extract.BR_BURST)
    npx_burst, obx_map = _burst_on_probe_clock(
        config, _load_edges(config, extract.NPX_BURST)
    )
    if br_burst is None or npx_burst is None or not br_burst.size or not npx_burst.size:
        log.warning(
            "no 14 s burst edges on one or both systems, so nothing held out checks "
            "the map; comparing the channels recorded on both systems instead"
        )
        path = _write_sync_report(config, None, refused=tprime_unchecked)
        verdict = "NOT VALIDATED"
        if path is not None:
            with open(path, "r", encoding="utf-8") as handle:
                verdict = json.load(handle)["verdict"]
        reason = "no 14 s burst on one or both systems"
        if tprime_unchecked:
            log.warning("%s", tprime_unchecked)
            reason += "; " + tprime_unchecked
        line_ok = verdict.startswith("CONSISTENT")
        # The channels were compared through the line. TPrime's map is the line
        # plus a constant per 1 Hz interval -- that interval's pair residual --
        # and every interval is reached by some time (it starts at a Blackrock
        # edge, not a probe edge, so sampling at the probe edges can skip one).
        # The exact gap is therefore the largest pair residual; the channels'
        # agreement speaks for TPrime only where it is inside the tolerance.
        tprime_gap = None
        if pairs is not None and not tprime_unchecked:
            tprime_gap = float(np.max(np.abs(pairs[0] - mapping.apply(pairs[1]))))
            if line_ok and tprime_gap > config.alignment_tolerance_s:
                too_far = (
                    f"the spikes went through TPrime, which is up to "
                    f"{tprime_gap * 1e3:.3f} ms from the line the channels were "
                    f"compared through (tolerance "
                    f"{config.alignment_tolerance_s * 1e3:g} ms), so their agreement "
                    "does not check it; the LFP and OneBox exports carry the line")
                log.warning("%s", too_far)
                reason += "; " + too_far
                verdict = "NOT VALIDATED"
                path = _write_sync_report(config, None, refused=too_far)
        result = Unvalidated(path, reason, verdict)
        settled = {"verdict": verdict, "n_checked": 0, "reason": result.reason}
        if tprime_unchecked:
            settled["map_checked"] = "tprime (could not be checked)"
        elif tprime_gap is not None:
            settled["map_checked"] = "tprime (against the line)"
            settled["tprime_line_gap_s"] = tprime_gap
        _settle_acceptance(config, mapping, result.passed, settled,
                           exports_accepted=line_ok)
        if figures:
            _draw_sync_figure(config)
        return result

    br_onsets = burst.group_burst_onsets(br_burst, config.burst_interval_s / 2)
    npx_onsets = burst.group_burst_onsets(npx_burst, config.burst_interval_s / 2)
    shared = _shared_interval(aligned_dir)
    line_report = align.validate_alignment(
        mapping.apply(npx_onsets), br_onsets, tolerance_s=config.alignment_tolerance_s,
        shared_interval_s=shared,
    )
    # Check the map the spikes actually went through: TPrime's per-interval
    # offsets when it ran, the line otherwise.
    checked_map = "linear_fit"
    report = line_report
    if payload.get("method") == "tprime":
        if tprime_unchecked:
            # The spikes got TPrime's map; accepting the line instead would
            # accept a map they never went through.
            from dataclasses import replace as _replace

            checked_map = "tprime (could not be checked)"
            report = _replace(line_report, passed=False, fail_reason=tprime_unchecked)
        else:
            tprime_mapped = _tprime_model(pairs, mapping, npx_onsets)
            checked_map = "tprime"
            report = align.validate_alignment(
                tprime_mapped, br_onsets, tolerance_s=config.alignment_tolerance_s,
                shared_interval_s=shared,
            )
            log.info("line: %s", line_report.summary())
    log.info("%s: %s", checked_map, report.summary())

    validation = {
        "map_checked": checked_map,
        "n_checked": report.n_checked,
        "n_expected": report.n_expected,
        "coverage": report.coverage,
        "span_s": report.span_s,
        "shared_s": report.shared_s,
        "max_residual_s": report.max_residual_s,
        "median_residual_s": report.median_residual_s,
        "tolerance_s": report.tolerance_s,
        "passed": report.passed,
        "fail_reason": report.fail_reason,
        "burst_clock": "probe" if obx_map is not None else "onebox",
    }
    if checked_map != "linear_fit":
        validation["line"] = {
            "n_checked": line_report.n_checked,
            "max_residual_s": line_report.max_residual_s,
            "median_residual_s": line_report.median_residual_s,
            "passed": line_report.passed,
            "fail_reason": line_report.fail_reason,
        }
    validation["verdict"] = "PASS" if report.passed else "FAIL"
    # The exports carry the line, so they need the line to have passed too.
    _settle_acceptance(config, mapping, report.passed, validation,
                       exports_accepted=line_report.passed)
    _write_sync_report(config, {**validation, "exports_accepted": line_report.passed})

    if figures and report.n_checked:
        axes = plots.plot_alignment_residuals(
            report.times_s, report.residuals_s, report.tolerance_s
        )
        path = (
            config.paths.figures_for("neuropixels") / "alignment_residuals.png"
        )
        plots.save_figure(axes.figure, path)
        log.info("wrote %s", path)
    if figures:
        _draw_sync_figure(config)

    return report


def _refuse(config: SessionConfig, reason: str) -> "Unvalidated":
    """No map from this run's edges to check: not accepted, and the runner stops."""
    log.warning("%s", reason)
    config.paths.aligned.mkdir(parents=True, exist_ok=True)
    (config.paths.figures_for("neuropixels") / SYNC_FIGURE).unlink(missing_ok=True)
    _settle_acceptance(config, None, False,
                       {"verdict": "NOT VALIDATED", "n_checked": 0, "reason": reason})
    return Unvalidated(_write_refusal_report(config, reason), reason, "NOT VALIDATED")


def _write_refusal_report(config: SessionConfig, reason: str) -> Path:
    """A sync report saying only that no map was validated, and why.

    Replaces whatever the last run wrote: with no map from this run's edges,
    nothing in ``time_map.json`` describes this run, so nothing is reported from it.
    """
    export_dir = config.paths.export_for("neuropixels")
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "sync_report.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"verdict": "NOT VALIDATED", "verdict_note": reason,
                   "accepted": False}, handle, indent=2)
    title = f"{config.session} | {stream_label('neuropixels')}"
    if config.probe_tag is not None:
        title += f" {config.probe_tag}"
    (export_dir / "sync_report.txt").write_text(
        f"{title}\n\nVerdict: NOT VALIDATED  ({reason})\n", encoding="utf-8")
    return path


def _settle_acceptance(
    config: SessionConfig, mapping: align.LinearMap | None, accepted: bool,
    validation: dict, exports_accepted: bool | None = None,
) -> None:
    """Write ``validation.json`` with the decision, and publish or withhold the map.

    Two decisions, the same unless the spikes went through TPrime: ``accepted``
    for the spike times (the map they went through), ``exports_accepted`` for
    the line stamped into the LFP and OneBox exports. Either can hold without
    the other; ``exports_accepted`` defaults to ``accepted``.
    """
    exports_ok = bool(accepted if exports_accepted is None else exports_accepted)
    # Bound to the exact map judged: a different time_map.json is not accepted.
    validation = {**validation, "accepted": bool(accepted),
                  "exports_accepted": exports_ok,
                  "time_map_sha256": _map_digest(config)}
    with open(config.paths.aligned / "validation.json", "w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2)
    # Every route to a decision, a standalone run included, leaves one clock in
    # the bundle: no stale copied records, and nothing on Blackrock time from a
    # map not accepted now.
    _remove_copied_sync_records(config)
    if not accepted:
        _remove_aligned_spike_exports(config)
    if exports_ok:
        _stamp_accepted(config, mapping)
    elif accepted:
        _unstamp_exports(config)
        log.warning(
            "the spike times' map (TPrime) passed but its line did not (%s): the "
            "aligned spike times are used, the LFP and OneBox exports stay on their "
            "own clocks", (validation.get("line") or {}).get("fail_reason") or "line FAIL",
        )
    else:
        _unstamp_exports(config)
        log.warning(
            "map not accepted (%s): the LFP and OneBox exports stay on their own "
            "clocks and export_results will not use the aligned spike times",
            validation.get("verdict"),
        )
    if exports_ok and not accepted:
        log.warning(
            "the line passed but the map the spikes went through did not (%s): the "
            "LFP and OneBox exports carry the line, export_results will not use the "
            "aligned spike times", validation.get("reason") or validation.get("verdict"),
        )


def _shared_interval(aligned_dir: Path) -> tuple[float, float] | None:
    """The stretch both systems recorded, on Blackrock time: the matched 1 Hz pairs.

    None when the Blackrock pair file is missing or unusable -- never a span
    estimated from whatever values survive. Coverage is then not computed, and
    nothing here raises ahead of the refusal being recorded.
    """
    br = _usable_edge_file(aligned_dir / "blackrock_1hz_trimmed.txt")
    return None if isinstance(br, str) else (float(br[0]), float(br[-1]))


def _usable_edge_file(path: Path) -> np.ndarray | str:
    """One paired-edge file: at least two finite, strictly increasing times, or why not."""
    try:
        times = catgt.read_edge_file(path)
    except FileNotFoundError:
        return f"{path.name} is missing"
    except (OSError, ValueError) as error:
        return f"{path.name} cannot be read ({error})"
    if times.size < 2:
        return f"{path.name} holds {times.size} time(s), fewer than two"
    if not np.all(np.isfinite(times)):
        return f"{path.name} holds non-finite times"
    if not np.all(np.diff(times) > 0):
        return f"{path.name} is not strictly increasing"
    return times


def _tprime_pairs(aligned_dir: Path) -> tuple[np.ndarray, np.ndarray] | str:
    """The matched 1 Hz pairs TPrime was handed, ``(blackrock, probe)``, or why not.

    Both files must parse, hold at least two finite, strictly increasing times,
    and pair one to one. The string names the file and the fault.
    """
    br = _usable_edge_file(aligned_dir / "blackrock_1hz_trimmed.txt")
    if isinstance(br, str):
        return br
    npx = _usable_edge_file(aligned_dir / "npx_1hz_trimmed.txt")
    if isinstance(npx, str):
        return npx
    if br.size != npx.size:
        return (f"blackrock_1hz_trimmed.txt has {br.size} times and "
                f"npx_1hz_trimmed.txt {npx.size}, so they do not pair")
    return br, npx


def _tprime_model(
    pairs: tuple[np.ndarray, np.ndarray], mapping: align.LinearMap,
    npx_times: np.ndarray,
) -> np.ndarray:
    """``npx_times`` through the map TPrime applied, from :func:`_tprime_pairs`.

    Exactly what :func:`_run_tprime_shifted` hands TPrime -- the probe edges and
    times scaled by the line's slope, Blackrock's with its intercept removed --
    put through TPrime's measured rule (:func:`align.interval_offset_map`), then
    the intercept added back.
    """
    br, npx = pairs
    slope, intercept = float(mapping.slope), float(mapping.intercept)
    mapped = align.interval_offset_map(
        slope * np.asarray(npx_times, dtype=np.float64), slope * npx, br - intercept
    )
    return mapped + intercept


def _write_sync_report(
    config: SessionConfig, validation: dict | None, refused: str | None = None,
) -> Path | None:
    """``sync_report.json`` and ``.txt`` in the probe's ``export/``.

    Built from ``time_map.json`` and the burst check, so it says nothing the
    pipeline did not measure -- it puts it in one place, per signal and per
    channel, beside the data it describes. ``refused`` overrides the verdict
    with NOT VALIDATED and that reason, keeping the measurements beside it.
    """
    from ._io import analog
    from ._sync.report import build_sync_report, render_sync_report

    map_path = config.paths.aligned / "time_map.json"
    if not map_path.exists():
        return None
    with open(map_path, "r", encoding="utf-8") as handle:
        time_map = json.load(handle)

    export_dir = config.paths.export_for("neuropixels")
    names: list[str] = []
    try:
        names = list(analog.load_onebox_analog(
            analog.find_onebox_analog(_onebox_export_dir(config)), require_blackrock=False
        ).names)
    except (OSError, ValueError):
        pass                               # no analog export for this probe

    checks = _independent_checks(config)
    report = build_sync_report(
        time_map, validation, names, stream_label("neuropixels"), checks=checks
    )
    report["time_map"] = str(map_path)
    if refused:
        report["verdict"] = "NOT VALIDATED"
        report["verdict_note"] = refused
    for check in checks:
        if check.get("status") == "DRIFT":
            log.warning(
                "%s drifts %.3f ms across the session against Blackrock's copy -- "
                "a rate error the burst check did not show", check["channel"],
                check["drift_ms"],
            )
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "sync_report.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    title = f"{config.session} | {stream_label('neuropixels')}"
    if config.probe_tag is not None:
        title += f" {config.probe_tag}"
    text = render_sync_report(report, title)
    (export_dir / "sync_report.txt").write_text(text, encoding="utf-8")
    log.info("sync %s -> %s", report["verdict"], path)
    return path


#: Pairing window for photodiode edges after mapping. Wider than the burst's:
#: Blackrock's .ns2 is filtered and 1 kHz, so its copy sits a few ms late.
PHOTODIODE_PAIRING_S = 0.02


def _independent_checks(config: SessionConfig) -> list[dict[str, Any]]:
    """Each behavioural channel recorded twice, compared after synchronization.

    Needs the OneBox analog export, the fitted ``onebox_to_blackrock`` map and ``blackrock.analog_file``
    with its ``analog_inputs``. These channels play no part in alignment --
    not even in picking cycles, as the burst does -- so their agreement is the
    most independent evidence there is. The photodiode is compared edge by edge;
    every other channel by cross-correlation in windows across the session.

    Judged on **drift**, not offset: Blackrock's 1 kHz stream is filtered before
    it is stored, which delays it by a constant, so a constant offset is the
    hardware and a changing one is a synchronization error.
    """
    from ._io import analog
    from ._sync import compare

    brk = config.blackrock
    if brk.analog_file is None or not brk.analog_inputs:
        return []

    def unmeasured(note: str) -> list[dict[str, Any]]:
        return [{"channel": n, "status": "not measured", "note": note}
                for n in brk.analog_inputs]

    # The export is still on the OneBox clock -- it is stamped only once this
    # check has helped accept the map -- so the fitted map is applied here.
    onebox_map = _fitted_onebox_map(config)
    if onebox_map is None:
        return unmeasured("no onebox_to_blackrock map (no OneBox 1 Hz edges), so the "
                          "OneBox copy cannot be put on Blackrock time")
    try:
        onebox = analog.load_onebox_analog(
            analog.find_onebox_analog(_onebox_export_dir(config)),
            require_blackrock=False,
        )
    except (OSError, ValueError) as error:
        return unmeasured(f"no OneBox analog export: {error}")
    from dataclasses import replace as _replace

    onebox = _replace(onebox, slope=onebox_map.slope, intercept=onebox_map.intercept)
    try:
        br = analog.open_blackrock_analog(
            brk.analog_file, brk.analog_inputs, brk.nsp_frontend_channels,
            gap_tolerance_ms=brk.gap_tolerance_ms, tolerance_s=config.alignment_tolerance_s,
        )
    except (OSError, ValueError, KeyError) as error:
        return unmeasured(f"could not read {brk.analog_file}: {error}")

    tolerance_ms = config.alignment_tolerance_s * 1e3
    checks = []
    for name, number in brk.analog_inputs.items():
        stream = br.streams[name]
        entry: dict[str, Any] = {
            "channel": name,
            "blackrock": f"{brk.analog_file.name} ainp{number} "
                         f"(id {stream.channel_id}, {stream.fs:g} Hz)",
        }
        try:
            if name == "photodiode":
                obx_t, _ = analog.detect_onebox_events(onebox, name)
                br_t, _ = analog.detect_blackrock_events(br, name)
                comp = compare.compare_event_times(
                    br_t, obx_t, onebox.slope, onebox.intercept,
                    label=name, tolerance_s=PHOTODIODE_PAIRING_S,
                )
                stats = compare.offset_and_drift(comp.blackrock_s, comp.after_s)
                entry["method"] = "event onsets (leaving baseline)"
            else:
                t, lag, _ = analog.measure_lags(onebox, br, name)
                stats = compare.offset_and_drift(t, lag)
                entry["method"] = "cross-correlation in windows"
        except (OSError, ValueError) as error:
            entry.update(status="not measured", note=str(error))
            checks.append(entry)
            continue
        entry.update(stats)
        entry["status"] = "consistent" if abs(stats["drift_ms"]) <= tolerance_ms else "DRIFT"
        checks.append(entry)
    return checks


#: ``validate_remapping``'s picture of what synchronization did, in ``export/figures/``.
SYNC_FIGURE = "sync_overview.png"


def _blackrock_start(config: SessionConfig, br: Any) -> float | None:
    """NSP seconds at Blackrock's first sample: the ``.ns2``'s, else the sync file's."""
    if br is not None:
        return float(br.time_map.segments[0].intercept)
    from ._io.blackrock import NspTimeMap

    path = config.paths.sync_for("blackrock") / extract.SYNC_NSP_MAP
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return float(NspTimeMap.from_dict(json.load(handle)).segments[0].intercept)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _sync_figure_data(config: SessionConfig) -> dict[str, Any] | None:
    """What ``sync_overview.png`` draws, read and computed; None with nothing to draw.

    The OneBox side is the analog export plus the OneBox 1 Hz edges, put on
    Blackrock time by the fitted ``onebox_to_blackrock`` (accepted or not -- the
    figure shows what the map does, and its title the verdict on it). The
    Blackrock side is the ``.ns2`` on its own NSP clock and the Blackrock 1 Hz
    edges. "Before" is the same OneBox samples on the OneBox's own clock and
    Blackrock's samples the same number of seconds into its own recording.
    Nothing here re-reads a raw ``.obx.bin`` or ``.ns5``.
    """
    from ._io import analog
    from ._sync import compare, overlay

    onebox_map = _fitted_onebox_map(config)
    if onebox_map is None:
        log.info("no sync figure: time_map.json has no onebox_to_blackrock map")
        return None
    slope, intercept = onebox_map.slope, onebox_map.intercept
    obx_edges = _load_edges(config, extract.NPX_OBX_1HZ)
    br_edges = _load_edges(config, extract.BR_1HZ)
    obx_edges = np.empty(0) if obx_edges is None else obx_edges
    br_edges = np.empty(0) if br_edges is None else br_edges

    onebox = None
    try:
        onebox = analog.load_onebox_analog(
            analog.find_onebox_analog(_onebox_export_dir(config)), require_blackrock=False)
        from dataclasses import replace as _replace

        onebox = _replace(onebox, slope=slope, intercept=intercept)
    except (OSError, ValueError):
        pass
    brk = config.blackrock
    br = None
    if onebox is not None and brk.analog_file is not None and brk.analog_inputs:
        try:
            br = analog.open_blackrock_analog(
                brk.analog_file, brk.analog_inputs, brk.nsp_frontend_channels,
                gap_tolerance_ms=brk.gap_tolerance_ms, tolerance_s=config.alignment_tolerance_s)
        except (OSError, ValueError, KeyError) as error:
            log.info("sync figure without the .ns2: %s", error)
    labels = [n for n in brk.analog_inputs if br is not None and n in onebox.names]

    # The window: the middle of what both systems recorded, on Blackrock time.
    if onebox is not None:
        start, stop = analog.get_recording_overlap(onebox, br)
    elif obx_edges.size and br_edges.size:
        mapped = slope * obx_edges + intercept
        start, stop = max(mapped[0], br_edges[0]), min(mapped[-1], br_edges[-1])
    else:
        log.info("no sync figure: no OneBox analog export and no 1 Hz edges")
        return None
    t0, t1 = overlay.centre_window(start, stop, overlay.SYNC_WINDOW_S)
    br_start = _blackrock_start(config, br)
    obx_t0 = (t0 - intercept) / slope                      # the window on the OneBox clock
    width = t1 - t0
    mapped_edges = slope * obx_edges + intercept
    centre = overlay.nearest_event(mapped_edges, t0 + overlay.SYNC_ZOOM_S,
                                   t1 - overlay.SYNC_ZOOM_S, (t0 + t1) / 2)

    def zoom(t: np.ndarray, y: np.ndarray, at: float | None) -> tuple[np.ndarray, np.ndarray]:
        if at is None:
            return np.empty(0), np.empty(0)
        keep = np.abs(t - at) <= overlay.SYNC_ZOOM_S
        return (t[keep] - at) * 1e3, y[keep]

    high = config.neuropixels.sync_pulse_ms / 1e3
    grid = np.linspace(0.0, width, int(width * 1000) + 1)          # 1 kHz: a pulse is 500 ms
    fine = (centre + np.linspace(-overlay.SYNC_ZOOM_S, overlay.SYNC_ZOOM_S, 4001)
            if centre is not None else np.empty(0))
    one_hz: dict[str, Any] = {"name": "1 Hz", "before": {}, "after": {}, "zoom": {}}
    if obx_edges.size:
        one_hz["before"]["onebox"] = (grid, overlay.square_wave(obx_edges, obx_t0 + grid, high))
        one_hz["after"]["onebox"] = (grid, overlay.square_wave(mapped_edges, t0 + grid, high))
        one_hz["zoom"]["onebox"] = zoom(fine, overlay.square_wave(mapped_edges, fine, high), centre)
    if br_edges.size:
        if br_start is not None:
            own = br_start + obx_t0                         # as far into Blackrock's recording
            one_hz["before"]["blackrock"] = (grid, overlay.square_wave(br_edges, own + grid, high))
        one_hz["after"]["blackrock"] = (grid, overlay.square_wave(br_edges, t0 + grid, high))
        one_hz["zoom"]["blackrock"] = zoom(fine, overlay.square_wave(br_edges, fine, high), centre)
    channels = [one_hz]

    for name in labels:
        k0, k1 = compare.onebox_sample_range(t0, t1, onebox.fs, slope, intercept, onebox.n_samples)
        obx_y = analog.read_onebox(onebox, name, k0, k1)
        obx_after_t = onebox.blackrock_time(k0, k1)
        s0, s1 = compare.blackrock_sample_range(t0, t1, br.segments)
        br_y = analog.read_blackrock(br, name, s0, s1)
        br_after_t = br.time_map.apply(np.arange(s0, s1))
        stream = br.streams[name]
        b0 = int(round(obx_t0 * stream.fs))                 # as far into Blackrock's recording
        b1 = min(stream.n_samples, b0 + int(round(width * stream.fs)))
        br_before = analog.read_blackrock(br, name, b0, b1) if b1 > b0 else np.empty(0)
        obx_n, br_n = overlay.normalise(obx_y), overlay.normalise(br_y)
        # Each channel zooms on its own transition nearest the window's middle:
        # a flash rarely falls within milliseconds of a 1 Hz edge.
        at = overlay.nearest_crossing(obx_after_t, obx_n, 0.5, (t0 + t1) / 2,
                                      margin_s=overlay.SYNC_ZOOM_S)
        channels.append({
            "name": name,
            "before": {
                "onebox": overlay.thin(np.arange(k0, k1) / onebox.fs - obx_t0, obx_n),
                "blackrock": overlay.thin(np.arange(b0, b1) / stream.fs - obx_t0,
                                          overlay.normalise(br_before, br_y)),
            },
            "after": {
                "onebox": overlay.thin(obx_after_t - t0, obx_n),
                "blackrock": overlay.thin(br_after_t - t0, br_n),
            },
            "zoom": {"onebox": zoom(obx_after_t, obx_n, at), "blackrock": zoom(br_after_t, br_n, at)},
        })

    edges = None
    if obx_edges.size and br_edges.size:
        try:
            comp = compare.compare_event_times(br_edges, obx_edges, slope, intercept,
                                               label="1 Hz", tolerance_s=overlay.EDGE_PAIRING_S)
            residual_ms = comp.after_s * 1e3
            median = float(np.median(residual_ms))
            edges = {
                "t_min": (comp.blackrock_s - comp.blackrock_s[0]) / 60.0,
                "residual_ms": residual_ms,
                "median_ms": median,
                "max_dev_ms": float(np.max(np.abs(residual_ms - median))),
                "n_paired": comp.n_pairs, "n_onebox": comp.n_onebox,
                "n_blackrock": comp.n_blackrock,
            }
        except ValueError as error:
            log.info("sync figure without the 1 Hz residuals: %s", error)

    report = {}
    try:
        with open(config.paths.export_for("neuropixels") / "sync_report.json",
                  "r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, ValueError):
        pass
    title = f"{config.session} | {stream_label('neuropixels')}"
    if config.probe_tag is not None:
        title += f" {config.probe_tag}"
    return {
        "title": title, "verdict": report.get("verdict", "?"), "note": report.get("verdict_note", ""),
        "window_s": (t0, t1), "zoom_ms": overlay.SYNC_ZOOM_S * 1e3,
        "channels": channels, "edges": edges,
    }


def _draw_sync_figure(config: SessionConfig) -> Path | None:
    """``export/figures/sync_overview.png``. A failure warns; it never fails validation."""
    path = config.paths.figures_for("neuropixels") / SYNC_FIGURE
    path.unlink(missing_ok=True)                 # never last run's picture under this verdict
    try:
        from ._plots import summary as plots
        from ._plots.sync import plot_sync_overview

        data = _sync_figure_data(config)
        if data is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        plots.save_figure(plot_sync_overview(data), path)
    except Exception as error:  # noqa: BLE001 -- a figure must not stop the run
        log.warning("sync figure not drawn: %s: %s", type(error).__name__, error)
        return None
    log.info("wrote %s", path)
    return path


# ----------------------------------------------------------------------------
# Step 6a: waveforms through Kilosort's own preprocessing
# ----------------------------------------------------------------------------


def _aligned_spike_path(config: SessionConfig, system: str) -> Path:
    """Spike times on Blackrock time, beside the sorting they belong to."""
    return config.paths.sorted_for(system) / f"{system}_spike_seconds_blackrock.npy"


def _sorted_window(
    config: SessionConfig, system: str, phy, timebase: str, nsp_map: Any = None
) -> tuple[float, float] | None:
    """``(start_s, duration_s)`` of the samples Kilosort sorted, on ``timebase``.

    The denominator of every firing rate and the span of the presence ratio.
    Taken from the sort's own record -- the binary's sample count
    (``preprocess.json``, or its size) cut to the ``tmin``/``tmax`` Kilosort was
    given -- not from the spikes, whose first and last fall wherever the units
    happened to fire. Mapped onto ``timebase`` the way the spike times were.
    A recording with a pause (segments) spans the pause on the NSP and
    Blackrock axes, as its spike times do.

    ``None`` when the record is missing, so the caller falls back to the span
    of the spikes.
    """
    path = config.paths.sorted_for(system) / "run_info.json"
    try:
        with open(path, "r", encoding="utf-8") as handle:
            run_info = json.load(handle)
    except (OSError, ValueError):
        return None
    settings = run_info.get("settings") or {}
    n_samples = (run_info.get("preprocessing") or {}).get("n_samples")
    if n_samples is None:
        try:
            n_samples = Path(run_info["binary"]).stat().st_size // (2 * int(settings["n_chan_bin"]))
        except (KeyError, TypeError, ValueError, OSError):
            return None
    fs = float(phy.fs)
    tmin = float(settings.get("tmin") or 0.0)
    tmax = settings.get("tmax")
    tmax = float("inf") if tmax is None else float(tmax)
    first = max(0, int(tmin * fs))
    stop = int(min(float(n_samples), tmax * fs))
    if stop <= first:
        return None
    edges = np.array([first, stop - 1])
    if timebase == "sorter":
        times = edges / fs
    elif timebase == "nsp" and nsp_map is not None:
        times = nsp_map.apply(edges)
    elif timebase == "blackrock":
        try:
            with open(config.paths.aligned / "time_map.json", "r", encoding="utf-8") as handle:
                line = json.load(handle)
        except (OSError, ValueError):
            return None
        times = float(line["slope"]) * (edges / fs) + float(line["intercept"])
    else:
        return None
    return float(times[0]), float(times[1] - times[0]) + 1.0 / fs


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
    aligned_path = _aligned_spike_path(config, system)
    if aligned_path.exists():
        why = _aligned_mismatch(aligned_path, phy.spike_samples)
        if why is None and not _alignment_accepted(config):
            why = "its map has not been accepted by validate_remapping"
        aligned = None
        if why is None:
            aligned, why = _checked_aligned_times(aligned_path, phy.spike_samples.size)
        if why is None:
            return aligned, "blackrock", None
        log.warning("%s; exporting in the sorter timebase instead", why)
    elif system == "blackrock":
        spike_file = config.blackrock.spike_file
        time_map = read_nsp_time_map(results_dir, spike_file)
        if time_map is None and spike_file is not None:
            # Missing, or measured from the wrong file: measure the right one.
            try:
                time_map = write_nsp_time_map(config, results_dir, spike_file)
            except Exception as error:
                log.warning(
                    "could not measure the NSP clock for %s (%s: %s)",
                    spike_file, type(error).__name__, error,
                )
        if time_map is not None:
            log.info("NSP clock: %s", time_map.summary())
            return time_map.apply(phy.spike_samples), "nsp", time_map
    return None, "sorter", None


_NEEDS_KILOSORT = (
    "the waveform export reads the recording through Kilosort's own "
    "preprocessing, so it needs kilosort installed -- run "
    "run_exporting_pipeline.py in the sorting env (kilosort4)"
)


def _kilosort_preprocessing(results_dir: Path) -> dict[str, Any]:
    """What Kilosort's reader will do to the signal, read from ``ops.npy``.

    Recorded in ``waveforms.mat`` so the file says which band its means are in.
    Also the first thing the waveform export touches that needs Kilosort, so a
    missing install fails here, with the fix, before any recording is opened.
    """
    try:
        import kilosort  # noqa: F401  (unpickling ops.npy needs it, and torch)
    except ImportError as error:
        raise ImportError(_NEEDS_KILOSORT) from error
    for name in ("ops.npy", "whitening_mat_inv.npy"):
        if not (results_dir / name).exists():
            raise FileNotFoundError(
                f"{results_dir / name} is missing; Kilosort writes it with every "
                "sort, and the waveform export rebuilds Kilosort's reader from it"
            )
    ops = np.load(results_dir / "ops.npy", allow_pickle=True).item()
    settings = ops.get("settings", {})
    return {
        "highpass_cutoff": float(settings.get("highpass_cutoff", np.nan)),
        "do_CAR": bool(ops.get("do_CAR", settings.get("do_CAR", True))),
        "invert_sign": bool(ops.get("invert_sign", settings.get("invert_sign", False))),
        "artifact_threshold": float(
            ops.get("artifact_threshold", settings.get("artifact_threshold", np.inf))
        ),
    }


#: Longest range handed to Kilosort's reader at once: one of its batches. A
#: merged range of dense spikes is split into pieces this long, so memory stays
#: bounded however many spikes are measured.
_KILOSORT_BLOCK_SAMPLES = 60_000
#: Snippet ranges closer than this are read as one block (up to a batch): a
#: read's fixed cost outweighs a quarter second of samples.
_READ_THROUGH_GAP_S = 0.25
#: How often the waveform export reports progress, as a fraction of its reads.
_PROGRESS_STEP = 0.1


def _kilosort_blocks(
    results_dir: Path,
    binary: Path,
    n_chan: int,
    regions: list[tuple[int, int]],
    device: str,
):
    """Read ``regions`` through Kilosort's own reader, as ``(start, samples)``.

    **This is Kilosort's preprocessing, not ours.** ``bfile_from_ops`` rebuilds
    the reader the sort used from ``ops.npy`` -- per-channel mean, the median
    across channels when ``do_CAR`` is set, its FFT high-pass at
    ``highpass_cutoff``, whitening -- and ``whitening_mat_inv.npy`` undoes the
    whitening, exactly as ``kilosort.data_tools.get_spike_waveforms`` does. What
    comes back is ADC counts of the signal Kilosort sorted. Drift correction is
    *not* in it: Kilosort applies that only batch by batch during sorting, and
    its reader leaves it out for any other slice.

    Three things differ from ``get_spike_waveforms``, all on purpose:

    - **absolute samples.** ``spike_times.npy`` counts from the start of the
      file, but the reader counts from ``tmin`` when the sort had a window. Each
      range is shifted by the reader's ``imin`` -- Kilosort's own helper does not,
      and cuts the wrong samples whenever ``tmin`` is set.
    - **binary-row columns.** The reader returns the probe's channels in
      ``chanMap`` order; they are put back at their rows of the binary, so a
      snippet is indexed by the same channel number the export reports and the
      gain is looked up by. Rows the probe does not use are NaN.
    - **blanked batches are skipped.** With ``artifact_threshold`` set the reader
      returns zeros for a batch it rejects; averaging those in would pull the
      mean toward zero, so the range is skipped and its spikes left uncut.

    Imports Kilosort and torch, which is why it is the one function here that
    does; the cutting and averaging in ``_export.waveforms`` stay pure.
    """
    import torch
    from kilosort import io as ks_io

    ops = ks_io.load_ops(results_dir / "ops.npy", device=torch.device(device))
    bfile = ks_io.bfile_from_ops(ops, filename=str(binary), device=torch.device(device))
    unwhiten = np.load(results_dir / "whitening_mat_inv.npy")
    # The reader opens a fresh memmap on every slice unless it is handed one --
    # one file open per range, which over a share is most of the cost.
    bfile.file_object = np.memmap(
        binary, mode="r", dtype=bfile.dtype, shape=(bfile.total_samples, bfile.n_chan_bin)
    )
    rows = np.asarray(ops["probe"]["chanMap"], dtype=np.int64)
    imin = int(bfile.imin)
    for start, stop in _with_progress(regions, "waveforms"):
        local = bfile[start - imin : stop - imin].cpu().numpy()   # (n_probe, n)
        if not local.any():
            log.warning(
                "samples %d-%d were blanked by Kilosort's artifact_threshold; "
                "their spikes are left out of the mean", start, stop,
            )
            continue
        block = np.full((local.shape[1], n_chan), np.nan, dtype=np.float32)
        block[:, rows] = (unwhiten @ local).T
        yield start, block


def _with_progress(items: list, label: str, step: float = _PROGRESS_STEP):
    """Yield ``items``, reporting progress through them (``_progress``).

    A stage that reads for half an hour without a line looks exactly like one
    that hangs. The drivers draw it as a bar; the log gets a line every ``step``.
    """
    from ._progress import Progress

    progress = Progress(log, label, len(items), unit="reads", step=step)
    for i, item in enumerate(items, 1):
        yield item
        progress.update(i)
    progress.finish()


def _uv_per_digit(
    config: SessionConfig, system: str, n_chan: int, binary: Path
) -> tuple[np.ndarray, str]:
    """Per-channel microvolts per ADC unit, and a note on where it came from.

    Blackrock records the gain per channel and neo reports it. SpikeGLX keeps it
    in the ``.meta`` beside the *recording* (``neuropixels.bin_file``), and
    :func:`spikeglx.uv_per_digit` reads it from there: the preprocessed binary
    the sorter read keeps the ADC scale, and its rows are the recording's, in
    order -- every one after CatGT, all but the trailing SY word after
    SpikeInterface -- so the first ``n_chan`` gains are its gains. Anything unknown comes
    back NaN rather than 1.0 -- a NaN propagates loudly through any scaling,
    where a fabricated unit gain would quietly produce plausible, wrong
    amplitudes.
    """
    if system == "neuropixels":
        from ._io import spikeglx

        source_bin = config.neuropixels.bin_file or binary
        meta_file = spikeglx.meta_path_for(source_bin)
        if meta_file.exists():
            scale, source = spikeglx.uv_per_digit(spikeglx.read_meta(meta_file))
            if scale.size == n_chan:
                return scale, f"{source}, from {meta_file.name}"
            if scale.size == n_chan + 1:            # the SY word, left out
                return scale[:n_chan], f"{source}, from {meta_file.name}"
            return (
                np.full(n_chan, np.nan),
                f"gain unknown: {meta_file.name} lists {scale.size} channels "
                f"but the sorting read {n_chan}; snippets stay in ADC",
            )
        return (
            np.full(n_chan, np.nan),
            f"gain unknown: no {meta_file.name} beside the recording; snippets stay in ADC",
        )

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
        "gain unknown: the .ns6 header did not open; snippets stay in ADC",
    )


def _inside(path: Path, directory: Path) -> bool:
    """Whether ``path`` lies under ``directory`` (neither has to exist)."""
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:
        return False
    return True


def _binary_changed_since_sort(
    config: SessionConfig, system: str, run_info: dict, binary: Path,
) -> str | None:
    """Why the preprocessed binary may not be the one this sorting read, or None.

    Preprocessing rewrites the binary in place, so a re-sort that fails after
    the rewrite -- or a rewrite cut short, which leaves no record at all, since
    the record is removed first -- leaves the old sorting beside new traces.
    The record the sort saw is in ``run_info.json``; the binary is the same only
    while the record **beside that binary** (``<binary>.json``) still exists
    and says the same. That record, not ``preprocess.json``, because two sorts of
    one recording -- other ``export_folder_name``, other ``motion`` -- read two
    binaries, and ``preprocess.json`` describes only the one made or reused
    last. It is removed before its binary is rewritten, so a torn rewrite has
    none. A binary from before those records falls back to ``preprocess.json``.
    A sorting of a binary outside ``preprocessed/`` (made before this pipeline
    preprocessed) has no record to compare and is not judged.
    """
    directory = config.paths.preprocessed_for(system)
    if not _inside(binary, directory):
        return None
    own = _binary_record_path(binary)
    current = _read_record(own) if own.exists() else provenance.read(directory)
    if current is None:
        return (f"{own if own.exists() else directory / provenance.SIDECAR} is missing "
                "or unreadable -- a rewrite was started and not finished")
    at_sort = run_info.get("preprocessing")
    if not at_sort:
        return "run_info.json does not record the preprocessing this sorting read"
    if provenance.normalise(current) != provenance.normalise(at_sort):
        return f"{binary.name} was preprocessed again after this sorting was made"
    return None


def export_waveforms(
    config: SessionConfig, system: str = "blackrock"
) -> dict | None:
    """Measure every spike's waveform through Kilosort's own preprocessing.

    Kilosort saves templates, not snippets: ``templates.npy`` is the shape it
    *fitted*, in whitened units, so nothing else in the export has a real
    amplitude. These snippets are the signal Kilosort sorted -- read with the
    reader the sort used, whitening undone (see :func:`_kilosort_blocks`) --
    centred on the spike sample and scaled to microvolts by each channel's gain.

    **The mean is always computed; only the snippets are optional.** Reading the
    recording is the expense and it is the same single pass either way, so
    ``waveforms.export_snippets`` decides whether the snippets are *kept*, not
    whether the recording is read.

    **How much of the recording is read is bounded by the spikes, not the file.**
    ``waveforms.max_spikes`` caps the spikes per unit -- chosen uniformly over the
    session, so the subset spans it rather than clustering where the unit was
    busiest -- and only the ranges holding those spikes are read, each padded by
    ``filter_pad_ms`` so Kilosort's filter transient is cut away before a
    snippet is taken.

    Needs Kilosort installed, and raises without it. Returns ``None`` when there
    is no sorting to read, or when the binary the sorting was made from is not
    reachable -- the rest of the export is still worth having, so that is a
    warning rather than a failure.
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
    stale = results_dir / "export" / f"{_export_stem(config, system)}.waveforms.mat"
    if not binary.exists():
        log.warning(
            "the binary this sorting read is gone (%s), so there is nothing to "
            "measure a waveform from; the rest of the export is unaffected",
            binary,
        )
        # A rewrite cut short empties preprocessed/ and leaves no record: the
        # folder is there, the record is not. Then an old waveforms.mat cannot
        # be checked against anything and goes. A folder that is missing or not
        # answering (the share is away) proves nothing, so the file is kept.
        directory = config.paths.preprocessed_for(system)
        if (stale.exists() and _inside(binary, directory) and directory.is_dir()
                and not (directory / provenance.SIDECAR).exists()):
            log.warning("removed %s: %s holds no %s, so a rewrite was started and "
                        "not finished; re-sort", stale.name, directory, provenance.SIDECAR)
            stale.unlink()
        return None
    changed = _binary_changed_since_sort(config, system, run_info, binary)
    if changed:
        log.warning(
            "not measuring waveforms: %s. The traces there are no longer the ones "
            "this sorting was made from; re-sort first.%s", changed,
            f" Removed {stale.name}, which belongs to another state of the "
            "binary." if stale.exists() else "",
        )
        stale.unlink(missing_ok=True)
        return None
    preprocessing = _kilosort_preprocessing(results_dir)
    chain = _waveform_chain(run_info, binary, preprocessing)

    phy = load_phy_results(results_dir)
    unit_ids = select_units(phy)
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
    # Read either side of each snippet and throw it away, so the samples that
    # are kept carry no transient from Kilosort's filter.
    pad = int(round(wf.filter_pad_ms * fs / 1000.0))

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
    regions = snippet_regions(plan, pad, max_samples=_KILOSORT_BLOCK_SAMPLES + 2 * pad,
                              max_gap=int(_READ_THROUGH_GAP_S * fs))
    scale, scale_note = _uv_per_digit(config, system, n_chan, binary)
    # An unknown gain must not destroy the mean: the shape and the relative
    # amplitude are still worth having. uv_per_digit stays NaN so nobody
    # mistakes ADC units for microvolts, and mean_units says which one the mean
    # is in. Every unit's channel must be known for "uV" -- a mean half in
    # microvolts and half in counts would be worse than either.
    used = np.unique(plan.channel) if plan.n_spikes else np.empty(0, dtype=np.int64)
    known = bool(used.size) and not np.isnan(scale[used]).any()
    mean_units = "uV" if known else "ADC"
    band = (
        f"Kilosort's preprocessing: {preprocessing['highpass_cutoff']:g} Hz FFT "
        f"high-pass, CAR {'on' if preprocessing['do_CAR'] else 'off'}, whitening "
        "undone, no drift correction"
    )
    log.info(
        "cutting %d of %d spikes (%d samples each) from %s in %d range(s); "
        "%s; %s, in %s",
        plan.n_spikes, plan.n_available, width, binary.name, len(regions),
        band, scale_note, mean_units,
    )
    result = accumulate(
        plan,
        _kilosort_blocks(results_dir, binary, n_chan, regions, config.machine.device),
        uv_per_digit=scale if known else 1.0,
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
    out_path = out_dir / f"{_export_stem(config, system)}.waveforms.mat"
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
                # What the signal went through: the sorted binary's own
                # preprocessing (CatGT, SpikeInterface or none), then Kilosort's
                # reader on top of it.
                "preprocessing": chain["summary"],
                "highpass_cutoff_hz": preprocessing["highpass_cutoff"],
                "do_CAR": float(preprocessing["do_CAR"]),
                "invert_sign": float(preprocessing["invert_sign"]),
                "filter_pad_ms": float(wf.filter_pad_ms),
                "source": str(binary),
                "kept_snippets": float(bool(wf.export_snippets)),
                "note": (
                    f"snippet is single-precision {mean_units}, nSamples x nSpikes "
                    "in MATLAB, and is present only when waveforms.export_snippets "
                    f"is true; mean and std are nSamples x nUnits in {mean_units} "
                    "and are exactly the mean and std of those snippets. Each "
                    "snippet is centred on its spike sample. Signal is "
                    f"{band}. "
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
        "samples_before": int(before),
        "fs": fs,
        "preprocessing": preprocessing,
        "chain": chain,
        "timebase": timebase,
        "result": result,
    }


# ----------------------------------------------------------------------------
# Step 6: export
# ----------------------------------------------------------------------------


def export_results(
    config: SessionConfig,
    system: str = "neuropixels",
    max_unit_figures: int | None = None,
) -> dict | None:
    """Export metrics, figures and the final bundle for a sorted folder.

    Every cluster not labelled ``noise`` in Phy is exported; whether to draw
    figures is the session's ``export_figures``.

    Uses aligned spike times when ``time_remapping`` has written them, and
    records which timebase it used. Reads Phy's ``cluster_group.tsv`` where it
    exists, so curated labels override Kilosort's own.

    Unit figures go into one ``figures/units.pdf``, a row per unit, ordered by
    channel; ``max_unit_figures`` caps how many, and ``None`` draws them all.

    Returns ``None`` when that system has no sorting output.
    """
    _check(system)
    from ._export import final
    from ._export.curated import load_phy_results, select_units
    from ._plots import summary as plots

    if not config.has_data(system):
        return None

    figures = config.export_figures

    results_dir = config.paths.sorted_for(system)
    if not (results_dir / "spike_times.npy").exists():
        log.info("no sorting results in %s", results_dir)
        return None

    import time

    clock = [time.perf_counter()]
    timing: dict[str, float] = {}

    def lap(step: str) -> None:
        now = time.perf_counter()
        timing[step] = round(now - clock[0], 2)
        clock[0] = now

    phy = load_phy_results(results_dir)
    unit_ids = select_units(phy)
    log.info(
        "%d units total, %d exported (%s; all but noise)",
        phy.unit_ids.size, unit_ids.size,
        "curated" if phy.curated else "not curated in Phy",
    )

    seconds, timebase, time_map = _resolve_spike_seconds(config, system, phy)
    spike_times_s: dict[int, np.ndarray] | None = None
    if seconds is not None:
        spike_times_s = {int(uid): seconds[phy.spike_clusters == uid] for uid in unit_ids}
    log.info("timebase: %s", timebase)
    lap("load sorting & spike times")

    # One folder per stream holding only what an analysis copies away: the .mat
    # files, units.csv, the figures, the manifest, and copies of the small
    # records that explain them (_copy_records). Kilosort's arrays stay in the
    # sorting folder above.
    out_dir = config.paths.export_for(system)
    # Measured first: units.csv's shape columns come from these means (in uV),
    # never from Kilosort's whitened templates.
    measured = export_waveforms(config, system)
    lap("measure waveforms (reads the binary)")
    window = _sorted_window(config, system, phy, timebase, time_map)
    if window is None:
        log.warning(
            "no run_info.json record of the samples sorted; firing rates and "
            "presence use the span of the spikes instead"
        )
    table = final.build_unit_table(
        phy, unit_ids, spike_times_s,
        **({"start_s": window[0], "duration_s": window[1]} if window else {}),
        **_measured_means(measured))
    # Where each unit sits on the probe: the same placement overview.pdf draws
    # its waveforms at, so units.csv can redraw it without the sorting folder.
    layout = _probe_layout(phy, unit_ids)
    table = final.with_unit_positions(table, layout["positions"])
    units_csv = final.export_unit_table(out_dir, table)
    channels_csv = (final.export_channel_positions(out_dir, layout["channel_map"], layout["site_xy"])
                    if layout["site_xy"] is not None else None)

    # The spikes in the format the lab's online-spike code already reads, plus
    # the measured waveform when the recording was reachable.
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
        filename=f"{_export_stem(config, system)}.sorted_spikes.mat",
    )
    log.info("exported %d units -> %s", unit_ids.size, mat_path)
    summary = _write_summary(
        config, system, out_dir, timebase, unit_ids, measured, phy.curated, time_map,
        window,
    )
    paths = {"units": units_csv, "sorted_spikes": mat_path, "summary": summary,
             **({"channel_positions": channels_csv} if channels_csv else {})}
    paths.update(_copy_records(config, system, out_dir))
    lap("write .mat, units.csv, manifest")

    if figures:
        # The sorting's drift figures, as published with it (not re-read from
        # preprocessed/, which a later rewrite may have changed).
        sorted_dir = config.paths.sorted_for(system)
        for name in DRIFT_FIGURES:
            shown, kept = sorted_dir / name, config.paths.figures_for(system) / name
            if shown.exists():
                kept.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shown, kept)
            elif kept.exists():
                kept.unlink()
    if figures and unit_ids.size:
        figure_dir = config.paths.figures_for(system)
        units = _unit_page_data(phy, unit_ids, spike_times_s, table, measured, window)
        if max_unit_figures is not None:
            units = units[:max_unit_figures]
        plots.save_unit_pages(
            units,
            figure_dir / "units.pdf",
            title=f"{config.session} | {stream_label(system)} | {timebase} timebase",
        )

        log.info("wrote units.pdf (%d units) -> %s", len(units), figure_dir)
        _write_overview(phy, unit_ids, figure_dir, measured, layout)
        if len(units) < unit_ids.size:
            log.warning(
                "only the first %d of %d units were plotted (max_unit_figures)",
                len(units), unit_ids.size,
            )
        lap("figures")

    # Not printed by the runner (a dict); record_timing puts it in the table.
    return {"paths": paths, "timebase": timebase, "n_units": int(unit_ids.size),
            "timing": timing}


def record_timing(
    config: SessionConfig, system: str, stages: list[dict[str, Any]]
) -> dict | None:
    """Add the exporting run's times to ``run_info.json`` and write ``timing.txt``.

    ``stages`` is one entry per exporting stage that ran for this stream, in
    order: ``{"stage", "seconds", "started"}`` and optionally ``"steps"``, a
    breakdown the stage returned (``export_results`` does). The runner measures
    them, because only it sees every stage.

    The sorting run's phases are already in ``run_info.json``; this adds the
    ``exporting`` phase (replacing an earlier export's) and re-copies the record
    into ``export/``, along with ``timing.txt`` -- the table, one row per step --
    and the ``sorting.timing`` of ``sorting_summary_info.json``, so the three
    agree.

    Returns ``None`` when there is no sorting with a ``run_info.json`` to add to.
    """
    _check(system)
    from datetime import datetime

    from ._export import timing as tm

    sorted_dir = config.paths.sorted_for(system)
    info_path = sorted_dir / "run_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as handle:
        run_info = json.load(handle)

    steps: list[tm.Step] = []
    for stage in stages:
        if stage.get("steps"):
            steps += [("exporting", f"{stage['stage']}: {name}", seconds)
                      for name, seconds in stage["steps"].items()]
        else:
            steps.append(("exporting", stage["stage"], stage.get("seconds")))
    started = [s["started"] for s in stages if s.get("started")]
    run = {
        "machine": config.machine.name,
        "started": min(started) if started else None,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    run_info["timing"] = tm.with_exporting(run_info.get("timing"), steps, run)
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2, default=str)

    out_dir = config.paths.export_for(system)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(info_path, out_dir / info_path.name)
    summary_path = out_dir / "sorting_summary_info.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summary.setdefault("sorting", {})["timing"] = run_info["timing"]
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=str)

    title = f"{config.session} | {stream_label(system)}"
    if system == "neuropixels" and config.probe_tag is not None:
        title += f" {config.probe_tag}"
    text = tm.render_timing(run_info["timing"], title)
    table_path = out_dir / "timing.txt"
    table_path.write_text(text, encoding="utf-8")
    return {
        "path": table_path,
        "total": tm.format_seconds(run_info["timing"]["total_s"]),
        "_text": text,                     # the table, for the driver to print
    }


def timing_table(config: SessionConfig, system: str) -> str | None:
    """The timing table of this stream's ``run_info.json``, as text; None without one.

    The same rendering :func:`record_timing` writes to ``timing.txt``, read back
    from whatever phases have run so far -- after sorting alone, four of five.
    """
    _check(system)
    from ._export import timing as tm

    info_path = config.paths.sorted_for(system) / "run_info.json"
    if not info_path.exists():
        return None
    with open(info_path, "r", encoding="utf-8") as handle:
        timing = json.load(handle).get("timing")
    if not timing:
        return None
    title = f"{config.session} | {stream_label(system)}"
    if system == "neuropixels" and config.probe_tag is not None:
        title += f" {config.probe_tag}"
    return tm.render_timing(timing, title)


def _probe_layout(phy, unit_ids: np.ndarray) -> dict[str, Any]:
    """The probe and where every exported unit sits on it, read once.

    ``xy``/``clusters``: Kilosort's own positions (``spike_positions.npy``) of
    an even subsample of spikes, and their clusters -- None without the file.
    ``positions``: each unit's place (:func:`probe_layout.place_units`: its
    sampled spikes' median, else its peak channel's site). ``site_xy`` and
    ``channel_map``: Kilosort's sites and the binary row of each, or None.
    ``units.csv`` and ``overview.pdf`` both use this, so they agree.
    """
    from ._export import final, probe_layout
    from ._export.curated import load_spike_positions
    from ._export.metrics import select_spikes_evenly

    results_dir = Path(phy.results_dir)
    n_total = int(phy.spike_clusters.size)
    index = select_spikes_evenly(n_total, _OVERVIEW_MAX_SPIKES)
    xy = load_spike_positions(results_dir, index)
    clusters = phy.spike_clusters[index]
    site_file = results_dir / "channel_positions.npy"
    site_xy = np.load(site_file) if site_file.exists() and phy.channel_map is not None else None
    channel_map = np.asarray(phy.channel_map).ravel() if site_xy is not None else None
    medians = probe_layout.compute_unit_positions(xy, clusters, unit_ids) if xy is not None else {}
    peak_rows = {int(u): final.best_channel(phy, int(u)) for u in unit_ids
                 if int(u) not in medians}
    positions = probe_layout.place_units(medians, unit_ids, peak_rows, site_xy, channel_map)
    return {"xy": xy, "clusters": clusters, "n_total": n_total, "positions": positions,
            "site_xy": site_xy, "channel_map": channel_map}


def _write_overview(
    phy, unit_ids: np.ndarray, figure_dir: Path, measured: dict | None = None,
    layout: dict[str, Any] | None = None,
) -> Path | None:
    """``overview.pdf``: the sorting on the probe, at a glance.

    Page 1 is every exported unit's spikes on the probe, noise in grey, at
    Kilosort's own positions -- skipped without its ``spike_positions.npy``. Pages 2 and 3 are every exported unit's measured mean waveform at its
    place on the probe (:func:`_probe_waveform_pages`): one colour per unit,
    then single units red and multi-units light blue. A page that cannot be
    drawn is left out, with a line in the log; the PDF is written when at least
    one can.
    """
    from ._plots import summary as plots

    layout = layout or _probe_layout(phy, unit_ids)
    xy, clusters, n_total = layout["xy"], layout["clusters"], layout["n_total"]
    pages = []
    # Exports before overview.pdf wrote this page as overview.png; it is the PDF's
    # page 1 now, so an old one would only describe an earlier sort.
    (Path(figure_dir) / "overview.png").unlink(missing_ok=True)
    if xy is None:
        log.info("no spike_positions.npy (Kilosort's own) in %s; no spike-position page",
                 phy.results_dir)
    else:
        exported = np.isin(clusters, unit_ids)
        pages.append(plots.plot_sorting_overview(
            xy[exported], clusters[exported], background_xy=xy[~exported], n_total=n_total
        ))

    pages.extend(_probe_waveform_pages(phy, unit_ids, layout, measured))
    pdf = Path(figure_dir) / "overview.pdf"
    if not pages:
        pdf.unlink(missing_ok=True)
        return None
    plots.save_pdf(pages, pdf)
    log.info("wrote %s (%d page%s)", pdf.name, len(pages), "" if len(pages) == 1 else "s")
    return pdf


def _probe_waveform_pages(
    phy, unit_ids: np.ndarray, layout: dict[str, Any], measured: dict | None,
) -> list[Any]:
    """Each exported unit's measured mean waveform where it sits on the probe.

    The waveform is the one ``units.pdf`` and ``sorted_spikes.mat`` hold -- a
    unit with none is counted in the title's total but not drawn. A unit sits
    where ``layout`` places it (:func:`_probe_layout`) -- the ``x_um``/``y_um``
    ``units.csv`` records. Two pages of the same glyphs: one colour per unit, then
    single units (``good``) red and multi-units (``mua``) light blue. Empty
    when nothing can be drawn.
    """
    from ._export import probe_layout
    from ._plots import summary as plots

    if measured is None:
        log.info("no measured waveforms, so no waveform pages in overview.pdf")
        return []
    site_xy, positions = layout["site_xy"], layout["positions"]
    if site_xy is None:
        log.info("no channel_positions.npy in %s; no waveform pages", phy.results_dir)
        return []

    result = measured["result"]
    wanted = np.isin(np.asarray(result.unit_ids), unit_ids)
    waves = np.asarray(result.mean)[:, wanted]
    ids = np.asarray(result.unit_ids)[wanted]
    scale = probe_layout.compute_glyph_scale(waves)
    glyphs = probe_layout.compute_waveform_glyphs(waves, ids, positions, scale)
    for glyph in glyphs:
        glyph["kind"] = _UNIT_CLASS.get(phy.labels.get(glyph["unit_id"], ""), "")

    units = measured.get("mean_units", "ADC")
    median_ptp = probe_layout.GLYPH_HEIGHT_UM / scale
    note = (f"waveform height: {median_ptp:.0f} {units} trough-to-peak "
            f"= {probe_layout.GLYPH_HEIGHT_UM:g} um")
    n_total = int(np.size(unit_ids))
    return [plots.plot_probe_waveforms(glyphs, site_xy, n_total, note, colour_by=by)
            for by in ("unit", "kind")]
