#!/usr/bin/env python
"""Pipeline part 1 of 2: sorting, and nothing else.

    python run_sorting_pipeline.py --config configs/Athos_2026_08_13.yaml \
           --machine windows_rig

One stage per stream: resolve the channel map, then hand it and the recording's
binary to Kilosort4's own ``run_kilosort``. Nothing here reads a sync pulse or
writes an LFP -- those belong with the stages that consume them, in
``run_exporting_pipeline.py``. Then it **stops**, because what comes next is
manual: curation in Phy is not scriptable.

Which systems run is the session's business, not this script's. A system runs
when its block declares paths; remove the block and nothing for it runs at all.
The one thing paths cannot say is whether to sort:

    kilosort_on_<system>    false -> keep the recording, do not sort it

This is the half that needs a GPU, and the only half that does -- which is what
lets it go to a cluster while the rest stays on the rig.

Exits non-zero if any stage fails. Stages that are skipped are reported and do
not fail the run.
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
    parser.add_argument(
        "--system",
        choices=list(ss.SYSTEMS),
        default=None,
        help="sort one system this run, whatever kilosort_on_* says in the session",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failing stage instead of stopping",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report the map, the binary and the settings for each stream; sort nothing",
    )
    args = parser.parse_args()
    _log_pipeline_messages()

    config = load(args)
    run = Runner(config, keep_going=args.keep_going)
    print()

    # The whole pipeline: one sort per system the session declares.
    for system in ss.SYSTEMS:
        run(
            ss.sort_with_kilosort,
            config,
            system,
            dry_run=args.dry_run,
            system=system,
        )

    code = run.finish("sorting pipeline")
    if code == 0 and not args.dry_run:
        print("\nNext, by hand:")
        print("    conda activate phy")
        for system in ss.SYSTEMS:
            if not getattr(config, f"sorts_{system}"):
                continue
            sorted_dir = config.paths.sorted_for(system)
            print(f"    phy template-gui {sorted_dir}/params.py")
        print("then:")
        print(f"    python run_exporting_pipeline.py --config {args.config}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
