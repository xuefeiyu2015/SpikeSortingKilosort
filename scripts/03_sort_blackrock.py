#!/usr/bin/env python
"""Pipeline step 4: sort the Blackrock Utah array file with Kilosort4.

The array's geometry must be supplied. Without ``--cmp`` this falls back to a
placeholder 10x10 grid in channel order, which is very unlikely to match the real
wiring -- sorting still runs, but units are attributed to the wrong electrodes.
The script says so loudly when that happens.

    python scripts/03_sort_blackrock.py --config configs/Athos_2026_08_13.yaml \\
           --machine windows_rig --cmp /path/to/array.cmp
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_sort_blackrock
from spikesorting.probes import probe_from_cmp, utah_grid_probe


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--cmp", default=None, help="Blackrock .cmp channel map file")
    parser.add_argument(
        "--n-channels",
        type=int,
        default=96,
        help="channel count for the placeholder grid when --cmp is not given",
    )
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="let Kilosort template across electrodes instead of treating each as "
        "independent (not recommended at 400 um pitch)",
    )
    args = parser.parse_args()
    config = load(args)

    independent = not args.grouped
    if args.cmp:
        probe = probe_from_cmp(args.cmp, independent=independent)
        print(f"probe: {probe['n_chan']} channels from {args.cmp}")
    else:
        probe = utah_grid_probe(args.n_channels, independent=independent)
        print(
            "WARNING: no --cmp given, using a PLACEHOLDER 10x10 grid in channel order.\n"
            "         Channel positions are a guess; supply the array's .cmp file\n"
            "         before interpreting these results spatially."
        )

    return report(step_sort_blackrock(config, probe=probe))


if __name__ == "__main__":
    raise SystemExit(main())
