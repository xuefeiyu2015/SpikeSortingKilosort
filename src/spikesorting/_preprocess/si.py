"""SpikeInterface as the preprocessor: the same chain, built lazily.

``phase_shift -> bandpass_filter -> common_reference -> correct_motion``, each
only when the session asks for it, then cast back to int16 -- Kilosort reads a
flat int16 binary, and the filters keep the ADC scale, so a count still means
what the ``.meta`` / ``.ns6`` header says it means.

CatGT's names are kept: ``car_mode: gblcar`` is SpikeInterface's global
*median* reference, which is what ``-gblcar`` has done since CatGT 3.6.
``gbldmx`` and ``gfix`` have no SpikeInterface equivalent and are refused at
load (``_config._reject_bad_preprocess``), so they never reach here.

SpikeInterface is imported inside each function, so importing this module costs
nothing on a machine without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["build_chain", "motion_chain", "read_neuropixels", "write"]


def build_chain(recording: Any, spec: Any, job_kwargs: dict | None = None) -> Any:
    """The preprocessed recording, as a lazy SpikeInterface chain.

    ``spec`` is a :class:`.._config.PreprocessSpec`. Motion correction is the
    one step that computes on construction (it detects and localises peaks to
    estimate the motion), which is why it takes ``job_kwargs``.
    """
    import spikeinterface.preprocessing as spre

    rec = recording
    if spec.phase_shift:
        if "inter_sample_shift" not in rec.get_property_keys():
            raise ValueError(
                "phase_shift needs each channel's ADC sample offset, which "
                "SpikeInterface reads from a SpikeGLX .meta -- this recording has "
                "none. Set preprocess.phase_shift: false."
            )
        rec = spre.phase_shift(rec)
    rec = spre.bandpass_filter(rec, freq_min=spec.highpass_hz, freq_max=spec.lowpass_hz)
    if spec.car_mode == "gblcar":
        rec = spre.common_reference(rec, reference="global", operator="median")
    elif spec.car_mode is not None:
        raise ValueError(f"car_mode {spec.car_mode!r} has no SpikeInterface equivalent")
    if spec.motion is not None:
        return motion_chain(rec, spec.motion, job_kwargs)
    return spre.astype(rec, dtype="int16", round=True)


def motion_chain(recording: Any, preset: str, job_kwargs: dict | None = None) -> Any:
    """``recording`` drift-corrected with a ``correct_motion`` preset, as int16.

    Used on its own after CatGT, which filters but has no motion correction.
    """
    import spikeinterface.preprocessing as spre

    if not recording.has_probe():
        raise ValueError(
            "motion correction needs the probe's contact positions, which "
            "SpikeInterface reads from a SpikeGLX .meta -- this recording has none. "
            "Set preprocess.motion: null and let Kilosort's nblocks correct drift."
        )
    corrected = spre.correct_motion(recording, preset=preset, **(job_kwargs or {}))
    return spre.astype(corrected, dtype="int16", round=True)


def read_neuropixels(
    bin_file: Path, n_chan: int, fs: float, sy_index: int | None, has_meta: bool
) -> Any:
    """The AP band as a SpikeInterface recording, without the SY word.

    The sync word is a bit field, and filtering or re-referencing it would
    destroy it, so it is left out: the result is the AP channels alone, in the
    binary's own order.

    **Only the named binary is read.** SpikeInterface's ``read_spikeglx`` takes a
    *folder*, and neo walks it recursively, treating every ``.bin``/``.meta``
    pair below it as part of the recording -- including the CatGT output this
    pipeline writes to ``<neuropixels_dir>/preprocessed/``, whose ``tcat``
    trigger becomes a second segment with no LF band (``KeyError: (0,
    'imec0.lf')``), or any copy someone left beside the run. So the file is
    read flat, and what ``read_spikeglx`` would attach -- the probe and each
    channel's ADC sample offset, which ``phase_shift`` and motion correction
    need -- is attached from this binary's own ``.meta``, the way SpikeInterface
    itself does it. A bare binary with no ``.meta`` (the Kilosort demo) gets
    neither.
    """
    from spikeinterface.core import read_binary

    recording = read_binary(bin_file, sampling_frequency=fs, dtype="int16", num_channels=n_chan)
    if sy_index is not None:
        recording = recording.remove_channels([recording.channel_ids[sy_index]])
    if not has_meta:
        return recording

    import probeinterface

    probe = probeinterface.read_spikeglx(Path(bin_file).with_suffix(".meta"))
    if probe.shank_ids is not None:
        recording = recording.set_probe(probe, group_mode="by_shank")
    else:
        recording = recording.set_probe(probe)
    shifts = _sample_shifts(probe)
    if shifts is not None:
        recording.set_property("inter_sample_shift", shifts)
    return recording


def _sample_shifts(probe: Any) -> Any:
    """Each channel's ADC offset as a fraction of a sample, or None if unknown."""
    try:
        from spikeinterface.extractors.neuropixels_utils import (
            get_neuropixels_sample_shifts_from_probe,
        )
    except ImportError:          # an older SpikeInterface: phase_shift will say so
        return None
    return get_neuropixels_sample_shifts_from_probe(probe)


def write(recording: Any, path: Path, job_kwargs: dict | None = None) -> Path:
    """Write ``recording`` as one flat int16 binary at ``path``.

    Several segments -- a Blackrock file allowed to span a pause -- are written
    back to back, which is what a single Kilosort binary has to be.
    """
    from spikeinterface.core import concatenate_recordings, write_binary_recording

    if recording.get_num_segments() > 1:
        recording = concatenate_recordings(
            [recording.select_segments([i]) for i in range(recording.get_num_segments())]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_binary_recording(
        recording,
        file_paths=path,
        dtype="int16",
        add_file_extension=False,
        **(job_kwargs or {}),
    )
    return path
