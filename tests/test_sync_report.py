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
