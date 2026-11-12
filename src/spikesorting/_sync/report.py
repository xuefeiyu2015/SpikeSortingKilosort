"""The synchronization report: how each signal was put on Blackrock time, and how well.

Pure. ``build_sync_report`` turns what ``time_remapping`` and ``validate_remapping``
already wrote (``time_map.json``, the burst check) into one record per signal;
``render_sync_report`` turns that into the text a person reads. The pipeline
writes both into the probe's ``export/``.

**How everything is aligned: the 1 Hz, always.** Spikes go probe -> Blackrock by
the 1 Hz fit (probe SY vs Blackrock .ns5 ch 1). OneBox analog channels go OneBox
-> probe by the OneBox's own 1 Hz (both SY words), then through that same probe
-> Blackrock fit. The 14 s burst only picks which 1 Hz cycle pairs with which;
it never sets a number.

**How the error is measured: the 14 s burst.** A *fit* residual is how far the
edges a map was fitted to sit from it -- a measure of fit, not of accuracy. The
burst's pulses never enter a fit, so how far they land from Blackrock's copies
is the real accuracy. It is sampled on the OneBox clock and travels exactly the
analog channels' two 1 Hz fits, so its error *is* theirs; spikes use one of
those two fits, so it bounds theirs too.

**Without a burst**, both of its jobs pass to channels recorded on both
systems. One of them (``time_map.json["cycle_vote"]``) picks the whole-second
offset in the burst's place; the 1 Hz still sets the map. The rest are the
independent checks: never fitted, and judged on drift. When at least one
channel *other than the one that picked the offset* agrees -- no drift past
tolerance, and an offset well under half a period, which a wrong whole-second
pick could not give -- the verdict is ``CONSISTENT (no burst)``. That is not a
``PASS``: no held-out pulse train was measured.
"""

from __future__ import annotations

from typing import Any

__all__ = ["build_sync_report", "render_sync_report", "CONSISTENT_NO_BURST"]

#: The verdict without a burst when independent channels agree.
CONSISTENT_NO_BURST = "CONSISTENT (no burst)"


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else float(seconds) * 1e3


def build_sync_report(
    time_map: dict[str, Any],
    validation: dict[str, Any] | None,
    analog_names: list[str] | None = None,
    stream: str = "neuropixels",
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One record per synchronized signal, and an overall verdict.

    ``validation`` is ``validation.json``'s content, or ``None`` when there were
    no bursts to check against. ``analog_names`` are the OneBox export's labels.
    """
    tolerance_ms = _ms(validation["tolerance_s"]) if validation else None
    obx = time_map.get("onebox_to_probe")
    burst_on_probe = time_map.get("burst_clock") == "probe"
    cycles_from = time_map.get("cycles_from")
    vote = time_map.get("cycle_vote")
    no_burst = cycles_from not in (None, "burst")
    checks = list(checks or [])
    measured = {c["channel"]: c for c in checks if c.get("status") in ("consistent", "DRIFT")}

    if validation is None or not validation.get("n_checked"):
        verdict, verdict_note = _verdict_without_burst(
            cycles_from, vote, measured, time_map, no_burst
        )
    elif not burst_on_probe:
        verdict = "FAIL" if not validation["passed"] else "PASS"
        verdict_note = (
            "the burst could not be put on the probe clock (no OneBox 1 Hz edges), "
            "so its residuals include the OneBox drift"
        )
    else:
        verdict = "PASS" if validation["passed"] else "FAIL"
        verdict_note = (
            f"held-out 14 s burst within {tolerance_ms:g} ms"
            if validation["passed"]
            else f"held-out 14 s burst exceeds {tolerance_ms:g} ms"
        )

    held_out = None
    if validation and validation.get("n_checked"):
        held_out = {
            "n": int(validation["n_checked"]),
            "median_ms": _ms(validation["median_residual_s"]),
            "max_ms": _ms(validation["max_residual_s"]),
            "passed": bool(validation["passed"]),
        }

    signals: list[dict[str, Any]] = [
        {
            "signal": "1 Hz square wave",
            "path": "probe SY bit 6 -> Blackrock .ns5 ch 1",
            "role": "fit (probe -> Blackrock)",
            "n": int(time_map["n_points"]),
            "max_ms": _ms(time_map["fit_max_residual_s"]),
            "rms_ms": _ms(time_map.get("fit_rms_residual_s")),
            "drift_ppm": float(time_map["drift_ppm"]),
        },
    ]
    if obx is not None:
        signals.append({
            "signal": "1 Hz square wave (OneBox copy)",
            "path": "OneBox SY bit 6 -> probe SY bit 6",
            "role": "fit (OneBox -> probe)",
            "n": int(obx["n_points"]),
            "max_ms": _ms(obx["fit_max_residual_s"]),
            "rms_ms": _ms(obx.get("fit_rms_residual_s")),
            "drift_ppm": float(obx["drift_ppm"]),
        })
    if no_burst:
        signals.append({
            "signal": "14 s coded burst",
            "path": "OneBox XA0 -> probe -> Blackrock .ns5 ch 2",
            "role": "not recorded on one or both systems",
            "n": 0, "max_ms": None, "median_ms": None, "passed": None,
            "absent": True,
        })
        if vote is not None:
            signals.append({
                "signal": f"coarse match ({vote['source']})",
                "path": f"OneBox {vote['source']} -> Blackrock .ns2",
                "role": "in place of the 14 s burst: picks the whole-second offset, "
                        "never fitted",
                "n": int(vote["n_total"]),
                "vote": vote,
            })
    else:
        signals.append({
            "signal": "14 s coded burst",
            "path": "OneBox XA0 -> probe -> Blackrock .ns5 ch 2",
            "role": "held-out check (picks the 1 Hz cycles, never fitted)",
            **(held_out or {"n": 0, "max_ms": None, "median_ms": None, "passed": None}),
        })

    def error_without_burst(name: str) -> tuple[str, dict[str, Any] | None]:
        check = measured.get(name)
        if check is not None:
            return "independent check of this channel (below)", check
        unmeasured = next((c for c in checks if c["channel"] == name), None)
        if unmeasured is not None:
            return f"not measured: {unmeasured.get('note', 'its check failed')}", None
        return "not measured: no burst, and no copy of it on Blackrock to compare", None

    channels: list[dict[str, Any]] = []
    for name in analog_names or []:
        if obx is None:
            channels.append({
                "channel": name, "max_ms": None,
                "aligned_by": "not synchronized: no OneBox 1 Hz edges",
                "error_from": None,
            })
        elif no_burst:
            error_from, check = error_without_burst(name)
            channels.append({
                "channel": name, "max_ms": None,
                "aligned_by": "1 Hz OneBox->probe, then 1 Hz probe->Blackrock",
                "error_from": error_from, "check": check,
            })
        else:
            channels.append({
                "channel": name,
                "max_ms": held_out["max_ms"] if held_out else None,
                "median_ms": held_out["median_ms"] if held_out else None,
                "aligned_by": "1 Hz OneBox->probe, then 1 Hz probe->Blackrock",
                "error_from": "14 s burst: same OneBox clock, same two fits",
            })
    if no_burst:
        channels.append({
            "channel": f"spikes ({stream})", "max_ms": None,
            "aligned_by": "1 Hz probe->Blackrock",
            "error_from": "no held-out measure; the independent checks pass through "
                          "this same fit, so their drift bounds it",
        })
    else:
        channels.append({
            "channel": f"spikes ({stream})",
            "max_ms": held_out["max_ms"] if held_out else None,
            "median_ms": held_out["median_ms"] if held_out else None,
            "aligned_by": "1 Hz probe->Blackrock",
            "error_from": "14 s burst: passes through this fit",
        })

    composed = time_map.get("onebox_to_blackrock")
    drifting = [c["channel"] for c in checks if c.get("status") == "DRIFT"]
    if drifting:
        verdict_note += f"; WARNING: {', '.join(drifting)} drift against Blackrock's copy"
    return {
        "verdict": verdict,
        "verdict_note": verdict_note,
        "tolerance_ms": tolerance_ms,
        "method": time_map.get("method"),
        "cycles_from": cycles_from,
        "probe_to_blackrock": {
            "slope": time_map["slope"], "intercept_s": time_map["intercept"],
            "drift_ppm": time_map["drift_ppm"],
        },
        "onebox_to_blackrock": None if composed is None else {
            "slope": composed["slope"], "intercept_s": composed["intercept"],
            "drift_ppm": composed["drift_ppm"],
        },
        "cycle_vote": vote,
        "no_burst": no_burst,
        "signals": signals,
        "channels": channels,
        # Behavioural channels recorded on both systems, compared after sync.
        # Not part of the verdict: they are judged on drift, and warn.
        "independent_checks": checks,
    }


#: An agreeing channel's offset must be under this fraction of a sync period:
#: a wrong whole-second pick would put it a full period off.
_MAX_OFFSET_PERIODS = 0.25


def _verdict_without_burst(
    cycles_from: str | None,
    vote: dict[str, Any] | None,
    measured: dict[str, dict[str, Any]],
    time_map: dict[str, Any],
    no_burst: bool,
) -> tuple[str, str]:
    """``CONSISTENT (no burst)`` or ``NOT VALIDATED``, and why."""
    if not no_burst:
        return "NOT VALIDATED", "no 14 s bursts to check against; only fit residuals are available"
    if cycles_from == "1hz_only" or vote is None:
        return "NOT VALIDATED", (
            "no 14 s burst; WARNING: no burst, photodiode or eye picked the whole-second "
            "offset, so the map may be off by whole seconds")
    coarse = (f"coarse match by {vote['source']} in place of the burst: shift "
              f"{vote['shift_s']:+.3f} s, {vote['n_support']} of {vote['n_total']} "
              f"{vote.get('counted', 'votes')} agree (runner-up {vote['n_runner_up']}); "
              "fine fit: 1 Hz")
    period_ms = 1e3 * float(time_map.get("sync_period_s", 1.0))
    agreeing = [
        name for name, c in measured.items()
        if name != vote["source"] and c["status"] == "consistent"
        and abs(c.get("median_offset_ms", 0.0)) < _MAX_OFFSET_PERIODS * period_ms
    ]
    if agreeing and not any(c["status"] == "DRIFT" for c in measured.values()):
        return CONSISTENT_NO_BURST, (
            f"no 14 s burst; {coarse}; {', '.join(agreeing)} agree with Blackrock's "
            "copies (never fitted)")
    return "NOT VALIDATED", (
        f"no 14 s burst; {coarse}; no channel besides {vote['source']} confirms it")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_sync_report(report: dict[str, Any], title: str = "") -> str:
    """The report as plain text, for reading in a file browser."""
    lines = [
        f"Synchronization report{f' -- {title}' if title else ''}",
        "=" * 60,
        f"Verdict: {report['verdict']}  ({report['verdict_note']})",
    ]
    p2b = report["probe_to_blackrock"]
    lines.append(
        f"probe -> Blackrock:  drift {p2b['drift_ppm']:+.2f} ppm, "
        f"intercept {p2b['intercept_s']:.6f} s"
        + (f", method {report['method']}" if report.get("method") else "")
    )
    if report["onebox_to_blackrock"]:
        o2b = report["onebox_to_blackrock"]
        lines.append(
            f"OneBox -> Blackrock: drift {o2b['drift_ppm']:+.2f} ppm, "
            f"intercept {o2b['intercept_s']:.6f} s"
        )

    lines += ["", "Sync signals (ms)", "-" * 60]
    for s in report["signals"]:
        if s.get("absent"):
            lines.append(f"{s['signal']:<32} not recorded")
        elif s.get("vote"):
            v = s["vote"]
            lines.append(
                f"{s['signal']:<32} shift {v['shift_s']:+.3f} s  {v['n_support']} of "
                f"{v['n_total']} {v.get('counted', 'votes')} agree, runner-up "
                f"{v['n_runner_up']}"
            )
        else:
            extra = (f"rms {_fmt(s.get('rms_ms'), 4)}" if "rms_ms" in s
                     else f"median {_fmt(s.get('median_ms'), 4)}")
            lines.append(
                f"{s['signal']:<32} n={s['n']:<6} max {_fmt(s['max_ms'], 4)}  {extra}"
            )
        lines.append(f"    {s['path']}  [{s['role']}]")

    if report.get("no_burst"):
        lines += [
            "", "Misalignment per channel (ms)", "-" * 60,
            "Aligned by the 1 Hz. With no 14 s burst there is no held-out maximum;",
            "a channel's error comes from its own independent check where there is one.",
            "",
        ]
    else:
        lines += [
            "", "Maximum misalignment per channel (ms)", "-" * 60,
            "Aligned by the 1 Hz; the 14 s burst only picks the cycles. The error",
            "is measured on the burst, which never enters a fit.",
            "",
        ]
    for c in report["channels"]:
        check = c.get("check")
        if check is not None:
            lines.append(
                f"{c['channel']:<32} drift {check['drift_ms']:+.3f}  spread(95%) "
                f"{check['spread_p95_ms']:.3f}  [{check['status']}]"
            )
        elif report.get("no_burst"):
            lines.append(f"{c['channel']:<32} no held-out measure")
        else:
            lines.append(f"{c['channel']:<32} max {_fmt(c['max_ms'], 4)}")
        lines.append(f"    aligned by: {c['aligned_by']}")
        if c.get("error_from"):
            lines.append(f"    error from: {c['error_from']}")

    checks = report.get("independent_checks") or []
    if checks:
        lines += [
            "", "Independent checks: channels recorded on both systems (ms)", "-" * 60,
            "Never fitted. Without a burst, one of them stands in for it to choose",
            "the whole-second offset; the 1 Hz still sets the map. Judged on DRIFT",
            "across the session; a constant offset is Blackrock's filter delay on",
            "its 1 kHz stream, not a sync error.",
            "",
        ]
        for c in checks:
            if c.get("status") == "not measured":
                lines.append(f"{c['channel']:<14} not measured: {c.get('note', '')}")
                continue
            lines.append(
                f"{c['channel']:<14} {c['status']:<10} drift {c['drift_ms']:+.3f} over "
                f"{c['span_min']:.0f} min   offset {c['median_offset_ms']:+.3f}   "
                f"spread(95%) {c['spread_p95_ms']:.3f}   n={int(c['n'])}"
            )
            lines.append(f"    {c['method']}; Blackrock {c['blackrock']}")
    return "\n".join(lines) + "\n"
