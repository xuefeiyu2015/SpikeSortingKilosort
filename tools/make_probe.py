#!/usr/bin/env python
"""Build Kilosort probe configuration files. Desk work, run locally.

**Not a pipeline stage** -- deliberately outside the numbered ``NN_`` sequence,
like the demo setup notebook. Probe files are built once per array (Utah) or per
run (Neuropixels), checked by eye, then referenced from a session config. Nothing
here needs a GPU, Kilosort, torch or the HPC: a probe file is plain JSON.

    # Neuropixels, from the run's own .meta (recommended -- authoritative per run)
    python tools/make_probe.py neuropixels --meta /data/run_g0_t0.imec0.ap.meta \\
           --out configs/probes/np_athos_2026_08_13.json --plot

    # Utah array, from the array's .cmp map file
    python tools/make_probe.py utah --cmp /data/array.cmp \\
           --out configs/probes/utah_athos.json --plot

    # Convert an older Kilosort .mat channel map
    python tools/make_probe.py from-mat --mat ~/.kilosort/probes/NeuroPix1_default.mat \\
           --out configs/probes/np1_default.json

    # Inspect an existing probe file
    python tools/make_probe.py show --probe configs/probes/utah_athos.json --plot

Always pass ``--plot`` the first time. No code can verify that a channel map is
*correct*; a picture of the geometry usually can.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _cli import REPO_ROOT  # noqa: F401  (adds src/ to sys.path)

from spikesorting._io import spikeglx
from spikesorting._probes import (
    load_probe_json,
    probe_from_cmp,
    probe_from_mat,
    probe_from_meta,
    probe_summary,
    save_probe_json,
    save_probe_prb,
    utah_grid_probe,
    validate_probe,
)


def build_neuropixels(args: argparse.Namespace) -> dict:
    if not args.meta:
        raise SystemExit(
            "neuropixels needs --meta <run>.imec0.ap.meta (the .ap.bin beside it "
            "works too -- read_meta accepts either)"
        )
    meta = spikeglx.read_meta(args.meta)

    probe = probe_from_meta(meta)
    print(f"built from ~snsGeomMap: {probe['n_chan']} channels")
    return probe


def build_utah(args: argparse.Namespace) -> dict:
    if args.cmp:
        probe = probe_from_cmp(args.cmp, pitch_um=args.pitch, independent=not args.grouped)
        print(f"built from {args.cmp}: {probe['n_chan']} channels at {args.pitch} um pitch")
        return probe

    probe = utah_grid_probe(
        args.n_channels, pitch_um=args.pitch, n_cols=args.n_cols, independent=not args.grouped
    )
    print(
        "WARNING: no --cmp given, so this is a PLACEHOLDER grid in channel order.\n"
        "         Real Utah arrays are rarely wired that way. Units will be\n"
        "         attributed to the wrong electrodes. Supply the array's .cmp file."
    )
    return probe


def build_from_mat(args: argparse.Namespace) -> dict:
    probe = probe_from_mat(args.mat)
    print(f"read {args.mat}: {probe['n_chan']} channels")
    return probe


def show_probe(args: argparse.Namespace) -> dict:
    probe = load_probe_json(args.probe)
    print(f"read {args.probe}")
    return probe


def render(probe: dict, out_path: Path | None, show: bool) -> None:
    """Draw geometry and channel order side by side."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikesorting._plots import plot_probe_channel_order, plot_probe_geometry

    figure = plt.figure(figsize=(11, 7), dpi=110)
    left, right = figure.subplots(1, 2, width_ratios=[1, 1.4])
    plot_probe_geometry(probe, ax=left)
    plot_probe_channel_order(probe, ax=right)
    figure.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"figure -> {out_path}")
    if show:
        plt.show()
    else:
        plt.close(figure)


BUILDERS = {
    "neuropixels": build_neuropixels,
    "utah": build_utah,
    "from-mat": build_from_mat,
    "show": show_probe,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("source", choices=sorted(BUILDERS), help="where the geometry comes from")
    parser.add_argument("--out", default=None, help="probe .json to write")
    parser.add_argument("--prb", default=None, help="also write a probeinterface .prb")
    parser.add_argument("--plot", action="store_true", help="display the geometry")
    parser.add_argument("--plot-out", default=None, help="save the geometry figure to this path")

    # neuropixels
    parser.add_argument("--meta", default=None, help="SpikeGLX .meta (or .bin beside it)")
    parser.add_argument("--probe", default=None, help="probe index, or a .json path for 'show'")

    # utah
    parser.add_argument("--cmp", default=None, help="Blackrock .cmp channel map")
    parser.add_argument("--n-channels", type=int, default=96)
    parser.add_argument("--n-cols", type=int, default=10)
    parser.add_argument("--pitch", type=float, default=400.0, help="electrode spacing in um")
    parser.add_argument(
        "--grouped",
        action="store_true",
        help="one shared group instead of one group per channel "
        "(not recommended at 400 um pitch)",
    )

    # from-mat
    parser.add_argument("--mat", default=None, help="older Kilosort .mat channel map")

    args = parser.parse_args()

    # 'neuropixels' uses --probe as an index; 'show' uses it as a path.
    if args.source == "show":
        if not args.probe:
            raise SystemExit("show needs --probe <file.json>")
    else:
        args.probe = int(args.probe) if args.probe is not None else 0

    probe = BUILDERS[args.source](args)

    problems = validate_probe(probe)
    if problems:
        print("probe is not usable:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    summary = probe_summary(probe)
    print(
        f"  {summary['n_chan']} channels | x {summary['x_range_um']} um | "
        f"y {summary['y_range_um']} um | {summary['n_groups']} group(s)"
    )
    if summary["placeholder_geometry"]:
        print("  geometry is a PLACEHOLDER -- do not trust spatial claims from this sorting")

    if args.out:
        print(f"probe -> {save_probe_json(probe, args.out, indent=2)}")
    if args.prb:
        print(f"prb   -> {save_probe_prb(probe, args.prb)}")
    if args.plot or args.plot_out:
        render(probe, Path(args.plot_out) if args.plot_out else None, args.plot)

    if not (args.out or args.prb or args.plot or args.plot_out):
        print("(nothing written -- pass --out, --prb, --plot or --plot-out)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
