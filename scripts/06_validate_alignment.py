#!/usr/bin/env python
"""Pipeline step 9: check the alignment against the 14 s coded bursts.

The time map is fit on the 1 Hz train, so the bursts are held-out data -- which is
what makes this a real check rather than a restatement of the fit. Exits non-zero
when the maximum residual exceeds ``alignment_tolerance_s`` (default 1 ms), so it
can gate a batch job.

    python scripts/06_validate_alignment.py --config configs/Athos.yaml --machine windows_rig
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_validate


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument("--no-figures", action="store_true", help="skip the residual plot")
    args = parser.parse_args()
    config = load(args, require_inputs=False)
    return report(step_validate(config, figures=not args.no_figures))


if __name__ == "__main__":
    raise SystemExit(main())
