#!/usr/bin/env python
"""Pipeline part 2 of 2: everything that is not sorting.

    python run_exporting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

    extract_sync         the 1 Hz train and the 14 s coded burst, from each
                         stream, to its own sync/. CatGT where the machine has
                         it, always the NumPy detector, and the two compared.
    lfp                  the LF band, when the session says export_lfp: true.
                         Extraction only -- it goes out on the probe's own clock,
                         and time_remapping stamps the Blackrock axis into it
                         afterwards without re-reading the samples.
    time_remapping       coarse offset from the 14 s bursts, then a fine fit on
                         the 1 Hz train, onto Blackrock time. Blackrock is the
                         reference timebase; this maps onto it, never the reverse.
    validate_remapping   the same map checked against the burst onsets, which
                         were held out of the fit. Non-zero past tolerance, so it
                         can gate a batch job.
    export_results       aligned spike times, metrics, waveforms, figures.

Every stage runs, in that order, on every stream the session names. A stage
with nothing to do yet reports why and the run carries on -- so running this
straight after the recording, before sorting, extracts the pulses and the LFP
and puts the LFP on Blackrock time, and ``export_results`` simply finds no
sorting. Run it again once curation is done.

The export reads the labels Phy writes: ``cluster_group.tsv`` overrides
Kilosort's own ``cluster_KSLabel.tsv``, and every cluster not labelled noise is
exported.

Two flags, each overriding a session key for this run only:

    --export-lfp         export_lfp: true
    --export-waveforms   waveforms.export_snippets: true

**Needs no GPU and no sorter** -- only the recordings' edge channels and the
sorted output. It does use CatGT/TPrime where the machine has them, so this is
the half that wants to run on the rig even when sorting went to a cluster.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# The shared CLI helpers live in tools/; _cli adds src/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from _cli import Runner, build_parser, load  # noqa: E402

import spikesorting as ss  # noqa: E402


def _log_pipeline_messages() -> None:
    """Show this pipeline's own log lines, and only its own.

    ``logging.basicConfig`` would configure the *root* logger, which turns on
    INFO for every installed package too -- faiss announcing which build it
    loaded, matplotlib announcing a backend -- all wearing this project's indent.
    Everything here logs to one named logger (``pipeline.py``, ``_sync/``), so
    that is the one to attach a handler to.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("       %(message)s"))
    log = logging.getLogger("spikesorting")
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def main() -> int:
    parser = build_parser(__doc__)
    # Both override a session key for one run; the session file is where each
    # lives. See _cli.flag_overrides.
    parser.add_argument(
        "--export-lfp",
        action="store_true",
        help="export the LF band this run, whatever export_lfp says in the session",
    )
    parser.add_argument(
        "--export-waveforms",
        action="store_true",
        help="keep a snippet per spike this run, whatever the session says",
    )
    args = parser.parse_args()
    _log_pipeline_messages()

    config = load(args, require_inputs=False)
    run = Runner(config)
    print()

    def per_probe_config():
        """``(label, config)`` for each Neuropixels stream the session covers.

        One entry unless the session names several probes of a run; each config
        is an ordinary one-probe one, so the verbs below take no probe argument.
        """
        for tag, probe_config in config.per_probe():
            label = "neuropixels" if tag is None else f"neuropixels {tag}"
            yield label, probe_config

    def per_stream(verb, *extra):
        """Run a ``(config, system)`` verb on every stream.

        Each probe of the run, then Blackrock **once**. Blackrock is one
        recording however many probes there are, and re-reading a .ns5 per probe
        would be minutes of share I/O for a byte-identical result.
        """
        for label, probe_config in per_probe_config():
            run(verb, probe_config, "neuropixels", *extra,
                system="neuropixels", label=label, config=probe_config)
        run(verb, config, "blackrock", *extra, system="blackrock")

    # Extraction first: everything below reads the edge files it writes.
    per_stream(ss.extract_sync)
    per_stream(ss.extract_lfp)

    # Blackrock is the reference timebase, so it is what the other system is
    # mapped *onto* rather than a system to remap. Each probe gets its own fit:
    # separate oscillators, separate SY words, separate time_map.json. This is
    # also what stamps the Blackrock axis into the LFP extracted above.
    per_stream(ss.time_remapping)

    for label, probe_config in per_probe_config():
        run(ss.validate_remapping, probe_config, probe_config.export_figures,
            system="neuropixels", label=label, config=probe_config)

    # Measures the waveforms too (sorted_spikes.mat carries the mean), so there
    # is no separate waveform pass over the binary.
    per_stream(ss.export_results)

    return run.finish("exporting pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
