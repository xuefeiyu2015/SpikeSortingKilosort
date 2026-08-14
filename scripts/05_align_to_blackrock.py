#!/usr/bin/env python
"""Pipeline step 8: map sorted spike times onto the Blackrock timebase.

Three stages, all from the edge files written by ``01_extract_sync.py``:

1. coarse offset from the 14 s coded bursts (resolves whole-cycle ambiguity);
2. both 1 Hz trains trimmed to their overlapping window;
3. fine alignment -- TPrime when the machine has it, otherwise a least-squares
   fit of the same matched edges.

Kilosort spike times are sample indices; they are converted to seconds before
alignment and the result is written as seconds.

    python scripts/05_align_to_blackrock.py --config configs/Athos.yaml --machine windows_rig
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_align


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--system",
        choices=("neuropixels", "blackrock"),
        default="neuropixels",
        help="which sorting's spike times to align (default: neuropixels)",
    )
    args = parser.parse_args()
    config = load(args, require_inputs=False)
    return report(step_align(config, system=args.system))


if __name__ == "__main__":
    raise SystemExit(main())
