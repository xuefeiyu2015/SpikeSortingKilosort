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
    binary's own order. With a ``.meta`` SpikeInterface's own reader is used,
    because it attaches the probe and each channel's ADC offset -- what
    ``phase_shift`` and motion correction need. A bare binary (the Kilosort demo)
    is read flat.
    """
    if has_meta:
        from spikeinterface.extractors import read_spikeglx

        stream = f"{bin_file.name.split('.')[-3]}.ap"          # "imec0.ap"
        return read_spikeglx(bin_file.parent, stream_id=stream)

    from spikeinterface.core import read_binary

    recording = read_binary(bin_file, sampling_frequency=fs, dtype="int16", num_channels=n_chan)
    if sy_index is not None:
        recording = recording.remove_channels([recording.channel_ids[sy_index]])
    return recording


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
