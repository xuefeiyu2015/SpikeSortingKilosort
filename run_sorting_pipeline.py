#!/usr/bin/env python
"""Pipeline part 1 of 2: everything up to manual curation.

    python run_sorting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

Extracts the sync edges for both systems, optionally exports the LFP, and sorts
whichever systems asked for it. Then it **stops**, because what comes next is
manual: curation in Phy is not scriptable, and the export stages read the labels
it writes. Run `run_exporting_pipeline.py` afterwards.

Which systems run is the session's business, not this script's:

    has_<system>_data       false -> nothing for that system runs at all
    kilosort_on_<system>    false -> extract its pulses and LFP, but do not sort

Only the sorting needs a GPU. Nothing here reads an edge file, so the whole thing
can also be split across machines -- see --steps.

Exits non-zero if any stage fails. Stages that are skipped are reported and do
not fail the run.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# The shared CLI helpers live with the numbered stages; _cli adds src/ itself.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

from _cli import build_parser, load, report  # noqa: E402

from spikesorting.pipeline import (  # noqa: E402
    step_extract_sync,
    step_lfp,
    step_sort_blackrock,
    step_sort_neuropixels,
)

ORDER = ["extract_sync", "sort_neuropixels", "sort_blackrock"]

#: Selectable but not run by default: at decimate=1 the LFP export writes a copy
#: of the LF band the size of the recording (~7 GB/hour at 385 channels).
OPT_IN = ["lfp"]


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--steps",
        nargs="+",
        default=ORDER,
        choices=ORDER + OPT_IN,
        help="stages to run, in the given order (default: all but %s)" % ", ".join(OPT_IN),
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
    config = load(args)

    runners = {
        "extract_sync": lambda: step_extract_sync(config, probe=args.probe),
        "lfp": lambda: step_lfp(config, probe=args.probe, decimate=args.lfp_decimate),
        "sort_neuropixels": lambda: step_sort_neuropixels(config, probe_index=args.probe),
        "sort_blackrock": lambda: step_sort_blackrock(config),
    }

    print()
    failures = 0
    for name in args.steps:
        try:
            result = runners[name]()
        except Exception as error:  # a stage that blows up should not hide the rest
            failures += 1
            print(f"[!!] {name}\n       {type(error).__name__}: {error}")
            traceback.print_exc()
            if not args.keep_going:
                break
            continue

        print(result.render())
        if result.status == "failed":
            failures += 1
            if not args.keep_going:
                break

    print()
    if failures:
        print(f"sorting pipeline finished with {failures} failure(s)")
        return 1

    print("sorting pipeline finished. Next, by hand:")
    print("    conda activate phy")
    for system in ("neuropixels", "blackrock"):
        if getattr(config, f"sorts_{system}"):
            print(f"    phy template-gui {config.paths.sorted_for(system)}/params.py")
    print("then:")
    print(f"    python run_exporting_pipeline.py --config {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
