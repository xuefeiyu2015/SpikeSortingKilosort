#!/usr/bin/env python
"""Pipeline step 3: sort the Neuropixels recording with Kilosort4.

Needs an NVIDIA CUDA GPU and the ``kilosort4`` conda environment. Sorting runs in
the machine's SSD cache and the results are copied to
``<neuropixels_dir>/neuropixels/kilosort4/`` afterwards, so ``temp.dat`` never
touches the network share.

    python scripts/02_sort_neuropixels.py --config configs/demo.yaml
    python scripts/02_sort_neuropixels.py --config configs/Athos_2026_08_13.yaml --machine windows_rig
"""

from __future__ import annotations

from _cli import build_parser, load, report

from spikesorting.pipeline import step_sort_neuropixels


def main() -> int:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    config = load(args)
    return report(step_sort_neuropixels(config, probe_index=args.probe))


if __name__ == "__main__":
    raise SystemExit(main())
