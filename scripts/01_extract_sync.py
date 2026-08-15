#!/usr/bin/env python
"""Pipeline steps 2 and 4: extract sync-pulse rising edges from both systems.

Writes ``<neuropixels_dir>/sync/{npx_1hz,npx_burst}.txt`` and
``<blackrock_dir>/sync/{blackrock_1hz,blackrock_burst}.txt``,
each one leading-edge time in seconds per line -- the format CatGT produces and
TPrime consumes.

Runs CatGT when the machine has it and the session points at a real SpikeGLX run,
always runs the pure-NumPy detector, and reports whether they agree. On the HPC,
where CatGT is absent, only the NumPy path runs and the CatGT command that would
have been used is printed.

    python scripts/01_extract_sync.py --config configs/demo.yaml
    python scripts/01_extract_sync.py --config configs/Athos_2026_08_13.yaml --machine windows_rig
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_extract_sync, step_lfp


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--lfp",
        action="store_true",
        help="also export the Neuropixels LF band to <neuropixels_dir>/lfp/ "
        "(same as run_sorting_pipeline.py --steps lfp)",
    )
    parser.add_argument(
        "--lfp-decimate",
        type=int,
        default=1,
        help="decimation for the LFP export. The LF band is hardware-limited to "
        "~500 Hz at 2500 Hz sampling, so 2 is safe; higher aliases",
    )
    args = parser.parse_args()
    config = load(args)

    result = step_extract_sync(config, probe=args.probe)
    code = report(result)

    if args.lfp:
        # Same stage run_sorting_pipeline.py drives, so they cannot drift.
        code = max(code, report(step_lfp(config, probe=args.probe, decimate=args.lfp_decimate)))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
