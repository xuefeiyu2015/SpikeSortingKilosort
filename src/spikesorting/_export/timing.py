"""Where the processing time went: one table across both pipeline scripts.

A stream's time is spread over two runs, often on two machines -- sorting on
the HPC or the rig, exporting on the rig -- so ``run_info.json`` carries it as
five phases, each a list of named steps:

    loading         opening the recording: the reachability probe, the map
    preprocessing   making the binary Kilosort reads, then Kilosort's own
                    whitening/high-pass set-up
    sorting         Kilosort's drift correction, detection, clustering, merging
    postprocessing  Kilosort saving its arrays, the copy out of the cache, the
                    NSP clock map
    exporting       every stage of ``run_exporting_pipeline.py``

The same session sorted from a local disk and from a share then differs row by
row, which is the point: the read-bound steps move, the GPU-bound ones do not.

Pure: dicts in, dicts and text out. Nothing here reads a clock or a file.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

__all__ = [
    "PHASES",
    "KILOSORT_STEPS",
    "kilosort_steps",
    "build_timing",
    "with_exporting",
    "format_seconds",
    "render_timing",
]

#: The phases, in the order a stream goes through them.
PHASES = ("loading", "preprocessing", "sorting", "postprocessing", "exporting")

#: Kilosort's own per-stage runtimes (``ops['runtime_*']``, set by
#: ``run_kilosort``), and which phase each belongs to. A key a future Kilosort
#: adds is kept under ``sorting`` by its raw name rather than dropped.
KILOSORT_STEPS: dict[str, tuple[str, str]] = {
    "runtime_preproc": ("preprocessing", "kilosort: whitening & high-pass set-up"),
    "runtime_drift": ("sorting", "kilosort: drift correction"),
    "runtime_st0": ("sorting", "kilosort: detection (universal templates)"),
    "runtime_clu0": ("sorting", "kilosort: first clustering"),
    "runtime_st": ("sorting", "kilosort: detection (learned templates)"),
    "runtime_clu": ("sorting", "kilosort: final clustering"),
    "runtime_merge": ("sorting", "kilosort: merging"),
    "runtime_postproc": ("postprocessing", "kilosort: saving results"),
}

Step = tuple[str, str, "float | None"]          # (phase, label, seconds)


def kilosort_steps(ops: dict[str, Any], wall_s: float) -> list[Step]:
    """Kilosort's runtime split into phases, plus whatever its timers missed.

    ``wall_s`` is the time ``run_kilosort`` took from outside. The difference
    from its own timers -- settings, the drift and diagnostic plots, logging --
    is reported as one more ``sorting`` step, so the phases add up to the wall
    clock instead of quietly losing it.
    """
    unknown = sorted(k for k in ops if k.startswith("runtime_") and k not in KILOSORT_STEPS)
    steps: list[Step] = []
    counted = 0.0
    for key in [*KILOSORT_STEPS, *unknown]:
        try:
            seconds = float(ops[key])
        except (KeyError, TypeError, ValueError):
            continue
        phase, label = KILOSORT_STEPS.get(key, ("sorting", f"kilosort: {key}"))
        steps.append((phase, label, seconds))
        counted += seconds
    if steps:
        steps.append(("sorting", "kilosort: set-up, plots, other", max(0.0, wall_s - counted)))
    else:  # a Kilosort with no runtime_* keys: one row rather than none
        steps.append(("sorting", "kilosort (no per-stage times reported)", wall_s))
    return steps


def build_timing(
    steps: Iterable[Step],
    run: dict[str, Any],
    notes: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The ``timing`` block of ``run_info.json`` for the sorting run.

    ``run`` is where and when it ran (machine, device, started, finished);
    ``notes`` attaches a remark to a step by label (``"made on hpc, ..."``);
    ``extra`` is anything else worth keeping beside it (the input's size and
    measured read rate).
    """
    phases = _phases(steps, notes or {})
    timing: dict[str, Any] = {"sorting_run": dict(run), **(extra or {}), "phases": phases}
    timing["total_s"] = _total(phases)
    return timing


def with_exporting(
    timing: dict[str, Any] | None, steps: Iterable[Step], run: dict[str, Any]
) -> dict[str, Any]:
    """``timing`` with the exporting run's phase put in (replacing an earlier one).

    Returns a new dict; ``timing`` is not changed. A record from before phases
    existed (``preprocess_s``/``kilosort_s``/``publish_s``) is read as one step
    per phase, so an old sorting still gets a table.
    """
    out = _upgrade(deepcopy(timing) if timing else {})
    exporting = _phases(steps, {}).get("exporting")
    out["exporting_run"] = dict(run)
    phases = out.setdefault("phases", {})
    if exporting:
        phases["exporting"] = exporting
    else:
        phases.pop("exporting", None)
    out["phases"] = {p: phases[p] for p in PHASES if p in phases}
    out["total_s"] = _total(out["phases"])
    return out


def _upgrade(timing: dict[str, Any]) -> dict[str, Any]:
    """Read a pre-phase ``timing`` record as phases."""
    if "phases" in timing or not timing:
        return timing
    legacy = [
        ("preprocessing", f"preprocessing ({timing.get('preprocessing', '?')})", "preprocess_s"),
        ("sorting", "kilosort", "kilosort_s"),
        ("postprocessing", "copy out of the cache", "publish_s"),
    ]
    steps = [(phase, label, timing.get(key)) for phase, label, key in legacy if key in timing]
    run = {k: timing[k] for k in ("started", "finished") if k in timing}
    return {"sorting_run": run, "phases": _phases(steps, {})}


def _phases(steps: Iterable[Step], notes: dict[str, str]) -> dict[str, Any]:
    phases: dict[str, Any] = {}
    for phase, label, seconds in steps:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        entry = phases.setdefault(phase, {"seconds": 0.0, "steps": []})
        row: dict[str, Any] = {
            "step": label,
            "seconds": None if seconds is None else round(float(seconds), 2),
        }
        if label in notes:
            row["note"] = notes[label]
        entry["steps"].append(row)
        if seconds is not None:
            entry["seconds"] = round(entry["seconds"] + float(seconds), 2)
    return {p: phases[p] for p in PHASES if p in phases}


def _total(phases: dict[str, Any]) -> float:
    return round(sum(p["seconds"] for p in phases.values()), 2)


def format_seconds(seconds: float | None) -> str:
    """``0.4 s``, ``12.3 s``, ``4.2 min``, ``1.53 h``; ``--`` when unknown."""
    if seconds is None:
        return "--"
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def render_timing(timing: dict[str, Any], title: str) -> str:
    """The table: one line per phase with its share of the total, steps indented."""
    timing = _upgrade(dict(timing))
    phases = timing.get("phases", {})
    total = timing.get("total_s") or _total(phases)

    lines = [f"Processing time | {title}", ""]
    for key, verb in (("sorting_run", "sorted"), ("exporting_run", "exported")):
        run = timing.get(key)
        if run:
            where = run.get("machine", "?")
            if run.get("device"):
                where += f" ({run['device']})"
            lines.append(
                f"{verb:<9}on {where}, {run.get('started', '?')} -> {run.get('finished', '?')}"
            )
    source = timing.get("input")
    if source:
        rate = source.get("read_mb_per_s")
        lines.append(
            f"input    {source.get('path')}"
            + (f", {source['gb']:.2f} GB" if source.get("gb") is not None else "")
            + (f", reads at {rate:.0f} MB/s" if rate is not None else "")
        )

    width = max([46] + [len(r["step"]) + 4 for p in phases.values() for r in p["steps"]])
    rule = "-" * (width + 22)
    lines += ["", f"{'phase / step':<{width}}{'time':>11}{'share':>9}", rule]
    for name in PHASES:
        if name not in phases:
            lines.append(f"{name:<{width}}{'--':>11}{'':>9}   not run")
            continue
        phase = phases[name]
        share = f"{100 * phase['seconds'] / total:.1f}%" if total else ""
        lines.append(f"{name:<{width}}{format_seconds(phase['seconds']):>11}{share:>9}")
        for row in phase["steps"]:
            note = f"   {row['note']}" if row.get("note") else ""
            lines.append(
                f"{'  ' + row['step']:<{width}}{format_seconds(row['seconds']):>11}{'':>9}{note}"
            )
    lines += [rule, f"{'total':<{width}}{format_seconds(total):>11}"]
    return "\n".join(line.rstrip() for line in lines) + "\n"
