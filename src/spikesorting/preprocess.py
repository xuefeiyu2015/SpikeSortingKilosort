"""Optional preprocessing before sorting (pipeline step 5).

SpikeInterface preprocessors are lazy: each call wraps the recording and computes
nothing until samples are pulled. So this module is side-effect free despite
looking like it does work -- it returns a new recording object and mutates
nothing.

Kilosort4 already high-pass filters and (by default) common-average-references
internally. Applying both here and there is not harmful but is wasted work, so
``apply: false`` is a reasonable default for Neuropixels; Blackrock Utah data more
often benefits from an explicit median reference. Because the right answer differs
per system, each declares its own block:

.. code-block:: yaml

    preprocess:                 # shared default for both
      bandpass: [300, 6000]
    neuropixels:
      preprocess: {apply: false}
    blackrock:
      preprocess: {apply: true, common_reference: median}

A system's block is merged *over* the shared one key by key, so overriding one
setting does not discard the rest. :meth:`SessionConfig.preprocess_for` resolves it.

Requesting preprocessing is always honoured. On the Blackrock route that is free
-- it already goes through SpikeInterface. On Neuropixels it switches the sort off
the direct ``run_kilosort(filename=...)`` path, which cannot preprocess because
Kilosort opens the file itself, onto the SpikeInterface path; that costs a second
copy of the recording in the cache. Leaving ``apply: false`` keeps the fast route.
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


#: What SpikeInterface's ``common_reference`` accepts, plus the spelling people
#: actually use. "mean subtraction" and "common average reference" are the same
#: operation; SpikeInterface only answers to ``"average"``.
_REFERENCE_OPERATORS = {"median": "median", "average": "average", "mean": "average"}


def _reference_operator(value: Any) -> str | None:
    """Normalise ``common_reference``. ``None`` means do not reference at all.

    Validated here rather than left to SpikeInterface so a typo fails at config
    load, not inside ``run_sorter`` after the recording has been copied to cache.
    """
    if value is None:
        return None
    key = str(value).strip().lower()
    if key not in _REFERENCE_OPERATORS:
        raise ValueError(
            f"common_reference must be 'median', 'average' (a.k.a. 'mean') or null, "
            f"got {value!r}"
        )
    return _REFERENCE_OPERATORS[key]


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
        "common_reference": _reference_operator(settings.get("common_reference")),
        "detect_bad_channels": bool(settings.get("detect_bad_channels", False)),
        "dtype": settings.get("dtype"),
    }
