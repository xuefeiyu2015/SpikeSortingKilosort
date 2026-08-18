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

The alignment and export stages read the labels Phy writes, so run this once
curation is done: ``cluster_group.tsv`` overrides Kilosort's own
``cluster_KSLabel.tsv``. Running it before curating is not an error -- you get
Kilosort's labels instead -- but it is rarely what you want. ``--steps
extract_sync`` is the part that can be run straight after the recording.

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

from _cli import Runner, build_parser, load, run_probes  # noqa: E402

import spikesorting as ss  # noqa: E402

STAGES = [
    # Extraction first: everything below reads what it writes.
    "extract_sync",
    "lfp",
    "time_remapping",
    "validate_remapping",
    "export_results",
    # Last, because it is the one stage here that re-reads the recording.
    "waveforms",
]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=STAGES,
        choices=STAGES,
        help="stages to run (default: all)",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    # The four below override session keys for one run; the session file is where
    # each of them lives. See _cli.flag_overrides.
    parser.add_argument(
        "--export-lfp",
        action="store_true",
        help="export the LF band this run, whatever export_lfp says in the session",
    )
    parser.add_argument(
        "--lfp-decimate",
        type=int,
        default=None,
        help="override the session's lfp_decimate (and export the LF band). The LF "
        "band is hardware-limited to ~500 Hz at 2500 Hz sampling, so 2 is safe; "
        "higher aliases (no anti-alias filter)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="override the session's export_groups (Phy cluster_group labels)",
    )
    parser.add_argument(
        "--export-waveforms",
        action="store_true",
        help="cut a snippet per spike this run, whatever export_waveforms says",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="       %(message)s")

    # An LFP on the probe's own clock cannot be compared with anything else that
    # was recorded, so asking for it asks for the map that fixes that. Harmless
    # when it cannot run: with one system declared, or with no edge files yet,
    # time_remapping reports why and the LFP stays on stream time.
    steps = list(args.steps)
    if "lfp" in steps and "time_remapping" not in steps:
        steps.append("time_remapping")
        print("note: --steps lfp also runs time_remapping, which puts the LFP on "
              "Blackrock time; without it the export is on the probe's own clock")

    config = load(args, require_inputs=False)
    run = Runner(config, keep_going=args.keep_going)
    print()

    # Extraction first: everything below reads the edge files it writes. Each
    # Neuropixels probe is its own stream with its own clock, so it is extracted,
    # mapped and validated on its own; Blackrock yields one stream and is not
    # re-read per probe.
    for stage, verb in (("extract_sync", ss.extract_sync), ("lfp", ss.extract_lfp)):
        if stage not in steps:
            continue
        for system in ss.SYSTEMS:
            for probe_index in run_probes(config, system, args.probe):
                run(verb, config, system, probe_index, system=system, probe=probe_index)

    # Blackrock is the reference timebase, so it is what the other system is
    # mapped *onto* rather than a system to remap.
    if "time_remapping" in steps:
        for system in ss.SYSTEMS:
            for probe_index in run_probes(config, system, args.probe):
                run(
                    ss.time_remapping,
                    config,
                    system,
                    probe_index=probe_index,
                    system=system,
                    probe=probe_index,
                )

    if "validate_remapping" in steps:
        for probe_index in run_probes(config, "neuropixels", args.probe):
            run(
                ss.validate_remapping,
                config,
                config.export_figures,
                probe_index=probe_index,
                system="neuropixels",
                probe=probe_index,
            )

    if "export_results" in steps:
        for system in ss.SYSTEMS:
            for probe_index in run_probes(config, system, args.probe):
                run(
                    ss.export_results,
                    config,
                    system,
                    probe_index,
                    system=system,
                    probe=probe_index,
                )

    # Last: the only stage here that re-reads the recording itself.
    if "waveforms" in steps:
        for system in ss.SYSTEMS:
            for probe_index in run_probes(config, system, args.probe):
                run(
                    ss.export_waveforms,
                    config,
                    system,
                    probe_index,
                    system=system,
                    probe=probe_index,
                )

    return run.finish("exporting pipeline")


if __name__ == "__main__":
    raise SystemExit(main())
