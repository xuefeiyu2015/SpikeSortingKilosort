"""CatGT as the preprocessor: one pass that filters the AP band and extracts edges.

Builds the command the workshop pipeline runs (``-apfilter=butter,12,lo,hi``,
``-gblcar``/``-gbldmx``, ``-gfix``, tshift unless ``-no_tshift``) from the run
the session's binary belongs to, and adds the sync extractors
``_sync.extract`` would otherwise run in a pass of its own -- the SY word is a
digital channel, which CatGT neither filters nor shifts, so the edges are the
same either way and the binary is read once instead of twice.

There is no fallback. When CatGT cannot run, :func:`catgt_unavailable` says why
and the caller stops, naming ``method: spikeinterface``: which preprocessing a
sorting had is stated in the session file, never decided by what was installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .._io.spikeglx import RunLayout
from .._sync.catgt import _FOLDER_HINTS, ExtractorSpec

__all__ = [
    "catgt_unavailable",
    "build_preprocess_args",
    "find_output",
    "trimmed_seconds",
    "MAX_TRIM_S",
]

#: CatGT pass-1 evens up stream lengths by trimming a ragged *end* (ReadMe,
#: v3.6+). More than this is not that, and the output no longer covers the
#: recording it claims to.
MAX_TRIM_S = 1.0


def catgt_unavailable(layout: RunLayout | None, catgt_dir: Path | None) -> str | None:
    """Why CatGT cannot preprocess this binary, or None when it can."""
    if catgt_dir is None:
        return "this machine's profile has no catgt_dir, so CatGT is not installed here"
    if layout is None:
        return (
            "the binary is not named like SpikeGLX output "
            "(<run>_g<n>_t<n>.imec<n>.ap.bin), so there is no run for CatGT to read"
        )
    return None


def build_preprocess_args(
    layout: RunLayout,
    spec: Any,
    dest: str | Path,
    extractors: list[ExtractorSpec] = (),
) -> list[str]:
    """The CatGT argument list for one probe's preprocessing pass.

    ``spec`` is a :class:`.._config.PreprocessSpec`. Never ``-startsecs``,
    ``-maxsecs`` or ``-save``: the output has to start at the raw file's first
    sample and keep every channel, or spike sample indices stop meaning the same
    thing as the sync edges from the raw file.
    """
    args = [
        f"-dir={Path(layout.directory)}",
        f"-run={layout.run_name}",
        f"-g={layout.gate}",
        f"-t={layout.trigger},{layout.trigger}",
        f"-prb={layout.probe}",
    ]
    hint = _FOLDER_HINTS[layout.folders]
    if hint:
        args.append(hint)
    # -ap always; -ob only when an extractor reads the OneBox.
    streams = ["-ap"] + [e.stream_flag for e in extractors if e.stream_flag != "-ap"]
    args.extend(dict.fromkeys(streams))
    args.append(f"-apfilter=butter,12,{_num(spec.highpass_hz)},{_num(spec.lowpass_hz)}")
    if spec.car_mode is not None:
        args.append(f"-{spec.car_mode}")
    if spec.gfix is not None:
        args.append("-gfix=" + ",".join(_num(v) for v in spec.gfix))
    if not spec.phase_shift:
        args.append("-no_tshift")
    args.extend(e.to_flag() for e in extractors)
    args.extend(["-out_prb_fld", "-no_catgt_fld", f"-dest={Path(dest)}"])
    return args


def find_output(dest: str | Path, layout: RunLayout) -> Path:
    """The filtered AP binary CatGT wrote for this probe, or raise."""
    name = f"{layout.run_name}_g{layout.gate}_tcat.imec{layout.probe}.ap.bin"
    found = sorted(Path(dest).rglob(name))
    if len(found) != 1:
        raise FileNotFoundError(
            f"CatGT finished but wrote {len(found)} files named {name} under {dest}; "
            "expected exactly one. Its CatGT.log says why."
        )
    return found[0]


def trimmed_seconds(raw_samples: int, out_samples: int, fs: float) -> float:
    """How much CatGT trimmed off the end, in seconds; raise if it was not that."""
    if out_samples > raw_samples:
        raise ValueError(
            f"CatGT's output has {out_samples} samples, more than the "
            f"{raw_samples} it read -- it is not this recording"
        )
    trimmed = (raw_samples - out_samples) / fs
    if trimmed > MAX_TRIM_S:
        raise ValueError(
            f"CatGT's output is {trimmed:.3f} s shorter than the recording. A pass-1 "
            f"run trims at most a ragged end (< {MAX_TRIM_S:g} s); this much means "
            "it skipped data, and spike times would no longer line up with the sync "
            "edges. Check CatGT.log."
        )
    return trimmed


def _num(value: float) -> str:
    """``300`` rather than ``300.0`` -- the form CatGT's own docs write."""
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)
