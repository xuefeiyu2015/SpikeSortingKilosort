#!/usr/bin/env python
"""Pipeline steps 2 and 4: extract sync-pulse rising edges from both systems.

Writes ``<output>/sync/{npx_1hz,npx_burst,blackrock_1hz,blackrock_burst}.txt``,
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

from spikesorting.pipeline import step_extract_sync


def main() -> int:
    parser = build_parser(__doc__)
    parser.add_argument(
        "--lfp",
        action="store_true",
        help="also export the Neuropixels LF band to <output>/lfp/",
    )
    parser.add_argument(
        "--lfp-decimate", type=int, default=1, help="decimation factor for the LFP export"
    )
    args = parser.parse_args()
    config = load(args)

    result = step_extract_sync(config, probe=args.probe)
    code = report(result)

    if args.lfp:
        code = max(code, _export_lfp(config, args))
    return code


def _export_lfp(config, args) -> int:
    from spikesorting.io import spikeglx

    npx = config.neuropixels
    if npx.run_dir is None or not npx.run_name:
        print("[--] lfp_export\n       session has no SpikeGLX run; nothing to export")
        return 0

    files = spikeglx.find_run_files(npx.run_dir, npx.run_name, npx.gate, npx.trigger, args.probe)
    if files["lf"] is None:
        print("[--] lfp_export\n       no .lf.bin stream in this run")
        return 0

    path = spikeglx.export_lfp(files["lf"], config.paths.lfp, decimate=args.lfp_decimate)
    print(f"[ok] lfp_export\n       {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
