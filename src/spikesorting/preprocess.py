"""Optional preprocessing before sorting (pipeline step 5).

SpikeInterface preprocessors are lazy: each call wraps the recording and computes
nothing until samples are pulled. So this module is side-effect free despite
looking like it does work -- it returns a new recording object and mutates
nothing.

Kilosort4 already high-pass filters and (by default) common-average-references
internally. Applying both here and there is not harmful but is wasted work, so
``apply: false`` in the session config is a reasonable default for Neuropixels;
Blackrock Utah data more often benefits from an explicit median reference.
"""

from __future__ import annotations

from typing import Any

__all__ = ["preprocess_recording", "describe_preprocessing"]


def preprocess_recording(
    recording: Any,
    bandpass: tuple[float, float] | None = (300.0, 6000.0),
    common_reference: str | None = "median",
    detect_bad_channels: bool = False,
    dtype: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(preprocessed_recording, info)``.

    ``info`` records what was applied, including any bad channels found, so the
    caller can save it beside the sorting results instead of it being lost.
    """
    import spikeinterface.preprocessing as spre  # lazy: heavy import

    info: dict[str, Any] = {
        "bandpass": list(bandpass) if bandpass else None,
        "common_reference": common_reference,
        "bad_channel_ids": [],
    }

    result = recording

    if detect_bad_channels:
        bad_ids, _ = spre.detect_bad_channels(result)
        info["bad_channel_ids"] = [str(c) for c in bad_ids]
        if len(bad_ids) > 0:
            keep = [c for c in result.channel_ids if c not in set(bad_ids)]
            result = result.select_channels(keep)

    if bandpass is not None:
        result = spre.bandpass_filter(result, freq_min=bandpass[0], freq_max=bandpass[1])

    if common_reference is not None:
        result = spre.common_reference(result, operator=common_reference, reference="global")

    if dtype is not None:
        result = spre.astype(result, dtype)

    info["n_channels_out"] = int(result.get_num_channels())
    return result, info


def describe_preprocessing(settings: dict[str, Any]) -> dict[str, Any]:
    """Normalise the ``preprocess:`` block of a session config into kwargs.

    Pure: config dict in, kwargs dict out.
    """
    if not settings or not settings.get("apply", False):
        return {"apply": False}
    bandpass = settings.get("bandpass")
    return {
        "apply": True,
        "bandpass": tuple(float(x) for x in bandpass) if bandpass else None,
        "common_reference": settings.get("common_reference"),
        "detect_bad_channels": bool(settings.get("detect_bad_channels", False)),
        "dtype": settings.get("dtype"),
    }
