"""The synchronization report: per-signal and per-channel numbers, and the verdict."""

from __future__ import annotations

import pytest

from spikesorting._sync.report import build_sync_report, render_sync_report

TIME_MAP = {
    "slope": 1.00005, "intercept": 12.3456, "drift_ppm": 50.0, "n_points": 10000,
    "fit_max_residual_s": 4e-6, "fit_rms_residual_s": 1e-6, "method": "tprime",
    "burst_clock": "probe",
    "onebox_to_probe": {"slope": 1.00003, "intercept": 0.002, "drift_ppm": 30.0,
                        "n_points": 10000, "fit_max_residual_s": 3e-6,
                        "fit_rms_residual_s": 1e-6},
    "onebox_to_blackrock": {"slope": 1.00008, "intercept": 12.35, "drift_ppm": 80.0},
}
PASSED = {"n_checked": 770, "max_residual_s": 2e-5, "median_residual_s": 5e-6,
          "tolerance_s": 1e-3, "passed": True, "burst_clock": "probe"}


def test_every_analog_channel_carries_the_held_out_burst_error():
    report = build_sync_report(TIME_MAP, PASSED, ["sync_burst", "eye_x", "photodiode"])

    assert report["verdict"] == "PASS"
    rows = {c["channel"]: c for c in report["channels"]}
    assert rows["eye_x"]["max_ms"] == pytest.approx(0.02)
    # Aligned by the 1 Hz -- both fits -- and only *measured* by the burst.
    assert rows["photodiode"]["aligned_by"] == "1 Hz OneBox->probe, then 1 Hz probe->Blackrock"
    assert rows["photodiode"]["error_from"].startswith("14 s burst")
    assert rows["spikes (neuropixels)"]["aligned_by"] == "1 Hz probe->Blackrock"
    assert rows["spikes (neuropixels)"]["max_ms"] == pytest.approx(0.02)
    burst = report["signals"][-1]
    assert burst["role"].startswith("held-out") and burst["n"] == 770

    text = render_sync_report(report, "s1")
    assert "Verdict: PASS" in text and "eye_x" in text and "+50.00 ppm" in text


def test_a_failed_burst_check_fails_the_report():
    failed = dict(PASSED, max_residual_s=3e-3, passed=False)
    report = build_sync_report(TIME_MAP, failed, ["eye_x"])
    assert report["verdict"] == "FAIL"
    assert report["channels"][0]["max_ms"] == pytest.approx(3.0)


def test_no_bursts_means_not_validated_rather_than_a_number():
    report = build_sync_report(TIME_MAP, None, ["eye_x"])
    assert report["verdict"] == "NOT VALIDATED"
    assert report["channels"][0]["max_ms"] is None
    assert "n/a" in render_sync_report(report)


def test_without_the_onebox_fit_the_analog_channels_are_not_synchronized():
    no_obx = dict(TIME_MAP, onebox_to_probe=None, onebox_to_blackrock=None,
                  burst_clock="onebox")
    report = build_sync_report(no_obx, dict(PASSED, burst_clock="onebox"), ["eye_x"])
    assert report["channels"][0]["aligned_by"].startswith("not synchronized")
    assert len(report["signals"]) == 2              # no OneBox 1 Hz row
    assert "OneBox drift" in report["verdict_note"]


NO_BURST = dict(TIME_MAP, burst_clock=None, cycles_from="eye_x", sync_period_s=1.0,
                cycle_vote={"source": "eye_x", "shift_s": 4.0, "n_periods": 4,
                            "n_support": 20, "n_runner_up": 0, "n_total": 20,
                            "counted": "windows"})


def _check(name, drift=0.01, offset=0.0, status="consistent"):
    return {"channel": name, "status": status, "drift_ms": drift, "median_offset_ms": offset,
            "spread_p95_ms": 0.2, "span_min": 127, "n": 60, "method": "m", "blackrock": "b"}


def test_without_a_burst_an_independent_channel_makes_it_consistent():
    report = build_sync_report(NO_BURST, None, ["eye_x", "photodiode"],
                               checks=[_check("eye_x"), _check("photodiode", offset=-0.004)])
    assert report["verdict"] == "CONSISTENT (no burst)"
    assert "coarse match by eye_x in place of the burst: shift +4.000 s" in report["verdict_note"]
    assert "photodiode agree" in report["verdict_note"]
    text = render_sync_report(report)
    assert "14 s coded burst                 not recorded" in text
    assert "coarse match (eye_x)" in text and "20 of 20 windows agree, runner-up 0" in text
    assert "error from: independent check of this channel" in text
    assert "14 s burst: same OneBox clock" not in text


def test_the_channel_that_picked_the_offset_cannot_confirm_it_alone():
    report = build_sync_report(NO_BURST, None, ["eye_x"], checks=[_check("eye_x")])
    assert report["verdict"] == "NOT VALIDATED"
    assert "no channel besides eye_x" in report["verdict_note"]


@pytest.mark.parametrize("check", [
    _check("photodiode", offset=1000.0),             # a whole second off: wrong pick
    _check("photodiode", drift=5.0, status="DRIFT"),
])
def test_a_channel_that_disagrees_keeps_it_not_validated(check):
    report = build_sync_report(NO_BURST, None, [], checks=[_check("eye_y"), check]
                               if check["status"] == "DRIFT" else [check])
    assert report["verdict"] == "NOT VALIDATED"


def test_without_a_burst_the_report_names_the_coarse_match_channel_everywhere():
    tried = [{"channel": "photodiode", "result": "2460 OneBox / 2460 Blackrock edges, "
                                                  "no clear winner"},
             {"channel": "eye_x", "result": "picked"}]
    report = build_sync_report(dict(NO_BURST, coarse_match_tried=tried), None,
                               ["eye_x", "photodiode"],
                               checks=[_check("eye_x"), _check("photodiode")])
    text = render_sync_report(report)
    assert "Coarse match:        eye_x, shift +4.000 s (in place of the 14 s burst)" in text
    assert "tried photodiode" in text and "tried eye_x        picked" in text
    assert "whole seconds: eye_x coarse match (+4.000 s); fine: 1 Hz probe->Blackrock" in text
    assert "[consistent; chose the coarse shift]" in text
    assert "confirms the 1 Hz drift, not the whole-second shift" in text
    photodiode = next(c for c in report["channels"] if c["channel"] == "photodiode")
    assert photodiode["error_from"] == "independent check of this channel (below)"


def test_without_any_coarse_match_the_header_says_none_was_picked():
    no_vote = dict(NO_BURST, cycles_from="1hz_only", cycle_vote=None)
    text = render_sync_report(build_sync_report(no_vote, None, ["eye_x"]))
    assert "Coarse match:        none" in text
    assert "whole seconds: none picked" in text


def test_with_a_burst_there_is_no_coarse_match_line():
    text = render_sync_report(build_sync_report(TIME_MAP, PASSED, ["eye_x"]))
    assert "Coarse match" not in text and "whole seconds" not in text
