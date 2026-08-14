"""Drive sync-edge extraction for a session (pipeline steps 2 and 4).

Produces four canonical edge files in ``<output>/sync/``:

===================== ==============================================
``npx_1hz.txt``       SpikeGLX SY-word bit 6, the 1 Hz square wave
``npx_burst.txt``     OneBox XA1, the 14 s coded burst
``blackrock_1hz.txt`` NSP ns5 channel 1, the same 1 Hz square wave
``blackrock_burst.txt`` NSP ns5 channel 2, the same 14 s burst
===================== ==============================================

Every file uses the CatGT convention: leading-edge times in seconds from stream
start, one per line. Downstream (``burst``, ``align``, ``tprime``) reads only
these, so it does not care which extractor produced them.

SpikeGLX edges are produced twice where possible -- once by CatGT, once by the
NumPy fallback -- and compared. Agreement is the correctness check that lets the
fallback be trusted on machines without CatGT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import SessionConfig
from ..io import spikeglx
from . import catgt, edges

__all__ = [
    "EdgeSet",
    "ExtractionReport",
    "extract_neuropixels_edges",
    "extract_blackrock_edges",
    "extract_session_edges",
]

NPX_1HZ = "npx_1hz"
NPX_BURST = "npx_burst"
BR_1HZ = "blackrock_1hz"
BR_BURST = "blackrock_burst"


@dataclass(frozen=True)
class EdgeSet:
    """One extracted pulse train."""

    name: str
    times_s: np.ndarray
    fs: float
    #: "catgt" or "numpy" -- which extractor produced ``times_s``.
    source: str
    #: Where the canonical copy was written.
    path: Path | None = None
    #: Human-readable description of the stream it came from.
    stream: str = ""

    @property
    def n(self) -> int:
        return int(np.asarray(self.times_s).size)

    @property
    def span_s(self) -> float:
        times = np.asarray(self.times_s)
        return float(times[-1] - times[0]) if times.size > 1 else 0.0


@dataclass
class ExtractionReport:
    """What happened during extraction, including the parts that did not run."""

    edge_sets: dict[str, EdgeSet] = field(default_factory=dict)
    #: name -> result of comparing CatGT against the NumPy fallback.
    comparisons: dict[str, dict] = field(default_factory=dict)
    #: Human-readable notes: skipped steps, missing tools, empty results.
    notes: list[str] = field(default_factory=list)
    #: CatGT command lines that were run, or would have been run.
    catgt_commands: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _sy_bit_chunks(
    info: spikeglx.StreamInfo, bit: int, chunk_samples: int
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(start, bit_values)`` for the SY word, chunk by chunk."""
    for start, block in spikeglx.iter_channel(info, info.sy_index, chunk_samples):
        yield start, edges.digital_bit_signal(block.view(np.uint16), bit)


def _resolve_ap_stream(config: SessionConfig, probe: int) -> spikeglx.StreamInfo | None:
    npx = config.neuropixels
    if npx.bin_file is not None:
        return spikeglx.stream_info(npx.bin_file, npx.n_chan_bin, npx.sample_rate)
    if npx.run_dir is None or not npx.run_name:
        return None
    files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    if files["ap"] is None:
        return None
    return spikeglx.stream_info(files["ap"])


def _resolve_obx_stream(config: SessionConfig, probe: int) -> spikeglx.StreamInfo | None:
    npx = config.neuropixels
    if npx.run_dir is None or not npx.run_name:
        return None
    files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, probe)
    return spikeglx.stream_info(files["obx"]) if files["obx"] is not None else None


def _catgt_specs(config: SessionConfig) -> list[catgt.ExtractorSpec]:
    """The extractors this session needs, in CatGT terms."""
    npx = config.neuropixels
    specs = [
        catgt.digital_spec(
            catgt.JS_AP, 0, -1, npx.sync_bit, npx.sync_pulse_ms, label=NPX_1HZ
        )
    ]
    if npx.run_dir is not None:
        specs.append(
            catgt.analog_spec(
                catgt.JS_OB,
                0,
                npx.burst_word,
                npx.burst_threshold_v[0],
                npx.burst_threshold_v[1],
                npx.burst_pulse_ms,
                label=NPX_BURST,
            )
        )
    return specs


def _run_catgt_extraction(
    config: SessionConfig, report: ExtractionReport, probe: int
) -> dict[str, np.ndarray]:
    """Run CatGT if it is both available and applicable. Returns name -> times."""
    npx = config.neuropixels
    machine = config.machine

    if npx.run_dir is None or not npx.run_name:
        report.note(
            "CatGT skipped: this session points at a bare .bin, and CatGT needs a "
            "SpikeGLX run (run_dir + run_name + _g<n>_t<n> naming)."
        )
        return {}

    specs = _catgt_specs(config)
    args = catgt.build_extract_args(
        run_dir=npx.run_dir,
        run_name=npx.run_name,
        specs=specs,
        gate=npx.gate,
        trigger=npx.trigger,
        probes=(probe,),
        dest=config.paths.sync / "catgt",
    )
    report.catgt_commands.append("runit " + " ".join(args))

    if not machine.has_catgt:
        report.note(
            f"CatGT skipped: machine '{machine.name}' defines no catgt_dir. "
            "Using the NumPy fallback; the command that would have run is recorded."
        )
        return {}

    catgt.run_catgt(machine.catgt_dir, args)
    produced = catgt.find_edge_files(config.paths.sync / "catgt")

    result: dict[str, np.ndarray] = {}
    for spec in specs:
        matches = [key for key in produced if f".{spec.kind}_" in f".{key}"]
        if spec.kind == "xd":
            matches = [k for k in matches if ".ap." in k or ".imec" in k]
        elif spec.kind == "xa":
            matches = [k for k in matches if ".obx" in k or ".ob." in k]
        if matches:
            result[spec.label] = catgt.read_edge_file(produced[matches[0]])
        else:
            report.note(f"CatGT produced no edge file for {spec.label} ({spec.to_flag()})")
    return result


def extract_neuropixels_edges(
    config: SessionConfig,
    probe: int = 0,
    chunk_samples: int = 30_000_000,
) -> ExtractionReport:
    """Extract the SpikeGLX sync trains (pipeline step 2).

    Runs CatGT when available, always runs the NumPy fallback, compares the two,
    and writes the canonical edge files.
    """
    report = ExtractionReport()
    sync_dir = config.paths.sync
    sync_dir.mkdir(parents=True, exist_ok=True)

    catgt_times = _run_catgt_extraction(config, report, probe)

    ap_info = _resolve_ap_stream(config, probe)
    if ap_info is None:
        report.note("Neuropixels 1 Hz extraction skipped: no AP binary found.")
    elif ap_info.sy_index is None:
        report.note(f"Neuropixels 1 Hz extraction skipped: {ap_info.path.name} has no SY word.")
    else:
        npx = config.neuropixels
        times = edges.stream_pulse_times(
            _sy_bit_chunks(ap_info, npx.sync_bit, chunk_samples),
            threshold=0.5,
            fs=ap_info.fs,
            duration_ms=npx.sync_pulse_ms,
        )
        report.edge_sets[NPX_1HZ] = _finalize(
            NPX_1HZ,
            times,
            ap_info.fs,
            catgt_times.get(NPX_1HZ),
            sync_dir,
            report,
            stream=f"{ap_info.path.name} SY bit {npx.sync_bit}",
        )
        if times.size == 0:
            report.note(
                f"No 1 Hz sync pulses found in {ap_info.path.name} SY bit {npx.sync_bit}. "
                "For a recording made without SMA1 connected this is the correct result, "
                "not a failure -- but alignment cannot use this stream."
            )

    obx_info = _resolve_obx_stream(config, probe)
    if obx_info is None:
        report.note("Neuropixels 14 s burst extraction skipped: no OneBox (obx) stream found.")
    else:
        npx = config.neuropixels
        raw = spikeglx.read_channel(obx_info, npx.burst_word)
        volts = spikeglx.raw_to_volts(obx_info, raw)
        times = edges.detect_pulse_times(
            volts,
            threshold=npx.burst_threshold_v[0],
            fs=obx_info.fs,
            duration_ms=npx.burst_pulse_ms,
        )
        report.edge_sets[NPX_BURST] = _finalize(
            NPX_BURST,
            times,
            obx_info.fs,
            catgt_times.get(NPX_BURST),
            sync_dir,
            report,
            stream=f"{obx_info.path.name} XA{npx.burst_word}",
        )

    return report


def extract_blackrock_edges(
    config: SessionConfig,
    chunk_samples: int = 30_000_000,
) -> ExtractionReport:
    """Extract the Blackrock sync trains with neo (pipeline step 4).

    CatGT cannot read Blackrock files, so this path is always the NumPy detector.
    It writes the same edge-file format, which is what lets TPrime align the two
    systems later.
    """
    from ..io import blackrock  # lazy: neo is not installed everywhere

    report = ExtractionReport()
    spec = config.blackrock
    if spec.sync_file is None:
        report.note("Blackrock extraction skipped: no blackrock.sync_file configured.")
        return report

    sync_dir = config.paths.sync
    sync_dir.mkdir(parents=True, exist_ok=True)
    reader = blackrock.open_reader(spec.sync_file)

    jobs = [
        (BR_1HZ, spec.sync_1hz_channel, spec.sync_threshold, config.sync_period_s * 1000.0 / 2.0),
        (BR_BURST, spec.burst_channel, spec.burst_threshold, 0.0),
    ]
    for name, channel, threshold, duration_ms in jobs:
        stream = blackrock.stream_for_channel(reader, channel)
        if stream.n_segments > 1:
            report.note(
                f"{spec.sync_file.name} has {stream.n_segments} segments (paused recording); "
                "only segment 0 is extracted, and its times are relative to that segment."
            )
        times = edges.stream_pulse_times(
            blackrock.iter_channel(reader, stream, chunk_samples),
            threshold=threshold,
            fs=stream.fs,
            duration_ms=duration_ms,
        )
        report.edge_sets[name] = _finalize(
            name,
            times,
            stream.fs,
            None,
            sync_dir,
            report,
            stream=f"{spec.sync_file.name} ch {stream.channel_id}",
        )
        if times.size == 0:
            report.note(
                f"No pulses on {spec.sync_file.name} channel {channel} at threshold "
                f"{threshold}. Check blackrock.{'sync' if name == BR_1HZ else 'burst'}_threshold "
                "against the file's units."
            )

    return report


def _finalize(
    name: str,
    numpy_times: np.ndarray,
    fs: float,
    catgt_times: np.ndarray | None,
    sync_dir: Path,
    report: ExtractionReport,
    stream: str,
) -> EdgeSet:
    """Compare the two extractors, pick the authoritative one, write the file.

    CatGT wins when both ran: it is the reference implementation, and using it
    keeps results identical to a hand-run extraction. The comparison is recorded
    either way.
    """
    source = "numpy"
    times = numpy_times

    if catgt_times is not None:
        comparison = edges.compare_edge_sets(catgt_times, numpy_times)
        report.comparisons[name] = comparison
        if not comparison["agree"]:
            report.note(
                f"{name}: CatGT and the NumPy fallback disagree "
                f"(CatGT {comparison['n_a']} edges, NumPy {comparison['n_b']}, "
                f"max |diff| {comparison['max_abs_diff_s']:.6f} s). CatGT output is used."
            )
        times = catgt_times
        source = "catgt"

    path = catgt.write_edge_file(sync_dir / f"{name}.txt", times)
    return EdgeSet(name=name, times_s=times, fs=fs, source=source, path=path, stream=stream)


def extract_session_edges(config: SessionConfig, probe: int = 0) -> ExtractionReport:
    """Run both systems' extraction, honouring the session's skip flags."""
    report = ExtractionReport()

    if config.skip_neuropixels:
        report.note("Neuropixels extraction skipped (skip_neuropixels).")
    else:
        npx_report = extract_neuropixels_edges(config, probe)
        report.edge_sets.update(npx_report.edge_sets)
        report.comparisons.update(npx_report.comparisons)
        report.notes.extend(npx_report.notes)
        report.catgt_commands.extend(npx_report.catgt_commands)

    if config.skip_blackrock:
        report.note("Blackrock extraction skipped (skip_blackrock).")
    else:
        br_report = extract_blackrock_edges(config)
        report.edge_sets.update(br_report.edge_sets)
        report.notes.extend(br_report.notes)

    return report
