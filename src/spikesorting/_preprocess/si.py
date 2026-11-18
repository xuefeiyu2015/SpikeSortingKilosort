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
from typing import Any, Sequence

__all__ = ["build_chain", "motion_chain", "peak_drift", "read_neuropixels", "write"]


def build_chain(
    recording: Any,
    spec: Any,
    job_kwargs: dict | None = None,
    reference_rows: Sequence[int] | None = None,
    bad_rows: Sequence[int] | None = None,
) -> tuple[Any, dict | None]:
    """The preprocessed recording, as a lazy SpikeInterface chain, and what
    motion correction measured (None without it).

    ``spec`` is a :class:`.._config.PreprocessSpec`. Motion correction is the
    one step that computes on construction (it detects and localises peaks to
    estimate the motion), which is why it takes ``job_kwargs``, and why what it
    measured comes back: see :func:`motion_chain`.

    ``reference_rows`` are the rows the common median is taken over -- the
    probe's electrodes minus ``bad_channels``. It is still *subtracted* from
    every row, so the binary keeps its shape. ``None`` takes every row.

    ``bad_rows`` (``bad_channels``) are interpolated from their neighbours
    before motion correction; see :func:`motion_chain`.
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
        ref_ids = None
        if reference_rows is not None:
            ids = list(rec.channel_ids)
            outside = [r for r in reference_rows if not 0 <= r < len(ids)]
            if outside or not len(reference_rows):
                raise ValueError(
                    f"reference rows {outside or '(none)'} are not rows of this "
                    f"{len(ids)}-channel recording"
                )
            ref_ids = [ids[r] for r in reference_rows]
        rec = spre.common_reference(
            rec, reference="global", operator="median", ref_channel_ids=ref_ids
        )
    elif spec.car_mode is not None:
        raise ValueError(f"car_mode {spec.car_mode!r} has no SpikeInterface equivalent")
    if spec.motion is not None:
        return motion_chain(rec, spec.motion, job_kwargs, bad_rows=bad_rows)
    return spre.astype(rec, dtype="int16", round=True), None


def motion_chain(
    recording: Any, preset: str, job_kwargs: dict | None = None,
    bad_rows: Sequence[int] | None = None,
) -> tuple[Any, dict]:
    """``recording`` drift-corrected with a ``correct_motion`` preset, as int16,
    and SpikeInterface's ``motion_info``: the motion, every peak it was
    estimated from and their locations, and ``run_times`` per step. All of it
    is computed anyway; asking for it costs nothing.

    Used on its own after CatGT, which filters but has no motion correction.

    **Known bad channels are interpolated first** (``bad_rows``: binary rows,
    as ``bad_channels`` are), with SpikeInterface's ``interpolate_bad_channels``
    -- the IBL pipeline's kriging from neighbouring sites. Otherwise a noisy
    site adds peaks at a fixed depth to the drift estimate, and its signal is
    mixed into its good neighbours by the motion interpolation before Kilosort
    drops the row. Interpolated rather than removed: the binary keeps every row
    the ``.meta`` channel map indexes. Kilosort still ignores the row.
    """
    import spikeinterface.preprocessing as spre

    if not recording.has_probe():
        raise ValueError(
            "motion correction needs the probe's contact positions, read from the "
            "SpikeGLX .meta or, with none, from the session's probe_file -- this "
            "recording has neither. Name a probe_file, or set preprocess.motion: "
            "null and let Kilosort's nblocks correct drift."
        )
    if preset == "medicine":
        check_medicine()
    if bad_rows:
        ids = list(recording.channel_ids)
        outside = [r for r in bad_rows if not 0 <= r < len(ids)]
        if outside:
            raise ValueError(
                f"bad_channels {outside} are not rows of this {len(ids)}-channel "
                "recording"
            )
        recording = spre.interpolate_bad_channels(
            recording, [ids[r] for r in sorted(set(bad_rows))]
        )
    # Every channel is kept. The DREDge presets say so themselves; others
    # (medicine) leave it to interpolate_motion's default, "remove_channels",
    # which drops the sites drift carries off the probe's ends -- and the
    # binary Kilosort reads must keep the rows its channel map, read from the
    # .meta, indexes, or every unit lands on the wrong site.
    corrected, info = spre.correct_motion(
        recording,
        preset=preset,
        output_motion_info=True,
        interpolate_motion_kwargs={"border_mode": "force_extrapolate"},
        **(job_kwargs or {}),
    )
    if corrected.get_num_channels() != recording.get_num_channels():
        raise RuntimeError(
            f"motion correction ({preset}) returned {corrected.get_num_channels()} "
            f"channels from {recording.get_num_channels()}; the channel map would "
            "no longer match the binary"
        )
    return spre.astype(corrected, dtype="int16", round=True), info


def check_medicine() -> None:
    """Raise now, not after peak detection, when MEDiCINe is not installed.

    SpikeInterface imports it only at the motion-estimation step, after
    detecting and localising every peak in the recording.
    """
    try:
        import medicine  # noqa: F401
    except ImportError:
        raise ImportError(
            "preprocess.motion: medicine needs the MEDiCINe package, which is not "
            "installed in this environment. Install it where you sort:\n"
            "      pip install medicine-neuro"
        ) from None


def peak_drift(recording: Any, motion_info: dict, n_max: int) -> dict | None:
    """What the drift figures need, from ``correct_motion``'s ``motion_info``.

    A subsample of the peaks (:func:`.motion.subsample_indices`) with their
    depth before and after correction -- "after" is the same peaks moved by
    the estimated motion (``correct_motion_on_peaks``), so nothing is
    detected twice -- and the motion itself. Times are seconds from the
    recording's start, segments laid end to end. None when the motion could
    not be estimated.
    """
    import numpy as np
    from spikeinterface.sortingcomponents.motion import correct_motion_on_peaks

    from .motion import subsample_indices

    motion = motion_info.get("motion")
    peaks = motion_info.get("peaks")
    if motion is None or peaks is None or len(peaks) == 0:
        return None
    keep = subsample_indices(len(peaks), n_max)
    peaks, locations = peaks[keep], motion_info["peak_locations"][keep]
    corrected = correct_motion_on_peaks(peaks, locations, motion, recording)

    # Segment k starts where segment k-1 ended, measured in samples, so the
    # axis is seconds into the sorted binary whatever t_start each carries.
    fs = float(recording.get_sampling_frequency())
    n_seg = int(recording.get_num_segments())
    offsets = np.cumsum([0.0] + [
        recording.get_num_samples(segment_index=k) / fs for k in range(n_seg - 1)
    ])
    # get_start_time, not get_times()[0]: the latter builds the whole time
    # vector, a float per sample.
    starts = [float(recording.get_start_time(segment_index=k)) for k in range(n_seg)]
    seg = peaks["segment_index"].astype(int)
    times = np.empty(len(peaks))
    for k in range(n_seg):
        on = seg == k
        times[on] = (
            recording.sample_index_to_time(peaks["sample_index"][on], segment_index=k)
            - starts[k] + offsets[k]
        )
    bins = np.concatenate([
        np.asarray(motion.temporal_bins_s[k]) - starts[k] + offsets[k]
        for k in range(min(n_seg, motion.num_segments))
    ])
    displacement = np.concatenate(
        [np.asarray(d) for d in motion.displacement[: min(n_seg, motion.num_segments)]]
    )
    direction = motion.direction
    return {
        "times_s": times,
        "depth_raw_um": np.asarray(locations[direction], dtype=float),
        "depth_corrected_um": np.asarray(corrected[direction], dtype=float),
        "amplitude": np.asarray(peaks["amplitude"], dtype=float),
        "temporal_bins_s": bins,
        "spatial_bins_um": np.asarray(motion.spatial_bins_um, dtype=float),
        "displacement_um": displacement,
        "n_peaks": int(len(motion_info["peaks"])),
    }


def read_neuropixels(
    bin_file: Path, n_chan: int, fs: float, sy_index: int | None, has_meta: bool,
    locations: Any = None,
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
    neither, except contact ``locations`` -- ``(n AP rows, 2)`` in um, by binary
    row -- when the caller has them from the session's ``probe_file``, since
    motion correction cannot localise a peak without them.
    """
    from spikeinterface.core import read_binary

    recording = read_binary(bin_file, sampling_frequency=fs, dtype="int16", num_channels=n_chan)
    if sy_index is not None:
        recording = recording.remove_channels([recording.channel_ids[sy_index]])
    import numpy as np
    import probeinterface

    if not has_meta:
        if locations is not None:
            probe = probeinterface.Probe(ndim=2, si_units="um")
            probe.set_contacts(positions=np.asarray(locations, dtype=float),
                               shapes="square", shape_params={"width": 12})
            probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
            recording = recording.set_probe(probe)
        return recording

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
