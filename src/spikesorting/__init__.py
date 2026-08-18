"""Kilosort4 spike sorting for Blackrock Utah arrays and Neuropixels.

Two acquisition systems, one timebase. Blackrock is the reference: Neuropixels
times are mapped onto it, never the reverse.

Everything you run is a verb in :mod:`spikesorting.pipeline`, re-exported here::

    import spikesorting as ss

    config = ss.load_session_config("configs/athos.yaml", "windows_rig")
    ss.sort_with_kilosort(config, "blackrock")

Modules prefixed with an underscore are the machinery those verbs call:
``_config``, ``_io``, ``_probes``, ``_sync``, ``_export``, ``_plots``. Read
``pipeline.py`` first; go below it only when you need to know how a step works.

Environment checking lives in ``tools/`` (``doctor.py`` + ``check_env.py``): it
inspects the machine rather than running the pipeline, so it is not part of this
package.
"""

from .pipeline import (  # noqa: F401
    REFERENCE_SYSTEM,
    SYSTEMS,
    Binary,
    MachineProfile,
    OutputPaths,
    SessionConfig,
    SortSummary,
    TimeMap,
    export_results,
    extract_lfp,
    extract_sync,
    load_machine,
    load_session_config,
    setup_probe,
    skip_reason,
    sort_with_kilosort,
    stream_label,
    time_remapping,
    validate_remapping,
)

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
    "Binary",
    "SortSummary",
    "TimeMap",
    "SessionConfig",
    "MachineProfile",
    "OutputPaths",
    "load_machine",
    "stream_label",
]
