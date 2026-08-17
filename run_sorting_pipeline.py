#!/usr/bin/env python
"""Pipeline part 1 of 2: everything up to manual curation.

    python run_sorting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

Extracts the sync edges for both systems, optionally exports the LFP, and sorts
whichever systems asked for it. Then it **stops**, because what comes next is
manual: curation in Phy is not scriptable, and the export stages read the labels
it writes. Run `run_exporting_pipeline.py` afterwards.

Which systems run is the session's business, not this script's. A system runs
when its block declares paths; remove the block and nothing for it runs at all.
The one thing paths cannot say is whether to sort:

    kilosort_on_<system>    false -> extract its pulses and LFP, but do not sort

Only the sorting needs a GPU. Nothing here reads an edge file, so the halves can
also be split across machines -- see --steps.

Exits non-zero if any stage fails. Stages that are skipped are reported and do
not fail the run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# The shared CLI helpers live in tools/; _cli adds src/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from _cli import Runner, build_parser, load, run_probes  # noqa: E402

import spikesorting as ss  # noqa: E402

STAGES = ["extract_sync", "sort"]

#: Selectable but not run by default: at decimate=1 the LFP export writes a copy
#: of the LF band the size of the recording (~7 GB/hour at 385 channels).
OPT_IN = ["lfp"]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=STAGES,
        choices=STAGES + OPT_IN,
        help="stages to run (default: all but %s)" % ", ".join(OPT_IN),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    parser.add_argument(
        "--lfp-decimate",
        type=int,
        default=1,
        help="decimation for --steps lfp. The LF band is hardware-limited to ~500 Hz "
        "at 2500 Hz sampling, so 2 is safe; higher aliases (no anti-alias filter)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="       %(message)s")

    config = load(args)
    run = Runner(config, keep_going=args.keep_going)
    print()

    # The pipeline, in order. Each verb returns what the next one takes, and
    # returns None when the session config says not to do that work.
    #
    # The inner loop is the probes of a SpikeGLX run: two probes are two streams
    # with two clocks, sorted and aligned separately. Blackrock yields one, so its
    # stages run once however many probes the Neuropixels run holds.
    for system in ss.SYSTEMS:
        for probe_index in run_probes(config, system, args.probe):
            # probe_index is passed to the verb; system= and probe= only name the
            # status line the Runner prints.
            where = dict(system=system, probe=probe_index)

            if "extract_sync" in args.steps:
                run(ss.extract_sync, config, system, probe_index=probe_index, **where)

            if "lfp" in args.steps:
                run(
                    ss.extract_lfp,
                    config,
                    system,
                    args.lfp_decimate,
                    probe_index=probe_index,
                    **where,
                )

            if "sort" in args.steps and not getattr(config, f"sorts_{system}"):
                # Nothing here is wanted: report the one reason and move on. The
                # steps below are the sort's own -- resolving a map and opening
                # the recording for a system nobody asked to sort fails on a
                # session that legitimately has neither.
                run(ss.sort_with_kilosort, None, config, system, **where)

            elif "sort" in args.steps:
                probe = run(ss.setup_probe, config, system, probe_index=probe_index, **where)
                if run.last_failed:
                    continue  # no map, so loading would fail the same way
                rec = run(
                    ss.load_spike_continuous,
                    config,
                    system,
                    probe,
                    probe_index=probe_index,
                    **where,
                )
                run(
                    ss.sort_with_kilosort,
                    rec,
                    config,
                    system,
                    probe_index=probe_index,
                    **where,
                )

    code = run.finish("sorting pipeline")
    if code == 0:
        print("\nNext, by hand:")
        print("    conda activate phy")
        for system in ss.SYSTEMS:
            if not getattr(config, f"sorts_{system}"):
                continue
            for probe_index in run_probes(config, system, args.probe):
                sorted_dir = config.paths.sorted_for(system, probe_index)
                print(f"    phy template-gui {sorted_dir}/params.py")
        print("then:")
        print(f"    python run_exporting_pipeline.py --config {args.config}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
