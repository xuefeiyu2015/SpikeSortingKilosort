#!/usr/bin/env python
"""Pipeline steps 7 and 10: export a sorted (and ideally curated) folder.

Run after curating in Phy. Writes ``units.csv``, aligned spike times, mean
waveforms and ISI histograms, plus per-unit summary figures under
``<blackrock_dir>/figures/`` (the Neuropixels dir for a single-system session).

Picks up aligned spike times automatically when
``<blackrock_dir>/aligned/<system>_spike_seconds_blackrock.npy`` exists, and reports
which timebase it exported in either way.

    python scripts/04_export_curated.py --config configs/demo.yaml
    python scripts/04_export_curated.py --config configs/Athos.yaml --groups good
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_export


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--system",
        choices=("neuropixels", "blackrock"),
        default="neuropixels",
        help="which sorting to export",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["good", "mua"],
        help="Phy labels to include (default: good mua)",
    )
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument(
        "--max-unit-figures",
        type=int,
        default=40,
        help="cap on per-unit figures (default: 40)",
    )
    args = parser.parse_args()
    config = load(args, require_inputs=False)

    return report(
        step_export(
            config,
            system=args.system,
            groups=tuple(args.groups),
            figures=not args.no_figures,
            max_unit_figures=args.max_unit_figures,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
