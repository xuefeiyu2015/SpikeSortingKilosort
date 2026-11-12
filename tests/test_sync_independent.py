"""Independent synchronization checks: behavioural channels recorded on both systems.

The .ns2 copy is filtered before Blackrock stores it, so it sits a constant few
ms late. That must read as an *offset* and pass; only a residual that changes
across the session -- a rate error -- may read as DRIFT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from spikesorting._io import analog, blackrock, matlab, spikeglx
from spikesorting._sync import compare

OBX_FS, BR_FS = 3000.0, 1000.0
DURATION = 300.0
T0 = 1_520_000_000.0                       # NSP clock origin
M, C = 1.0 + 40e-6, T0 + 0.5               # OneBox -> NSP, the truth
BR_PERIOD = 1.0 / 999.99                   # the .ns2's measured rate
FILTER_DELAY = 0.002                       # Blackrock's copy arrives 2 ms late


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------


def test_a_constant_offset_is_not_drift_and_a_rate_error_is():
    t = np.linspace(0.0, 3600.0, 200)
    still = compare.offset_and_drift(t, np.full(t.size, -0.002))
    assert still["median_offset_ms"] == pytest.approx(-2.0)
    assert still["drift_ms"] == pytest.approx(0.0, abs=1e-9)

    sliding = compare.offset_and_drift(t, -0.002 + 20e-6 * t)   # 20 ppm
    assert sliding["drift_ms"] == pytest.approx(72.0, rel=1e-6)
    assert sliding["drift_ppm"] == pytest.approx(20.0, rel=1e-6)
    with pytest.raises(ValueError):
        compare.offset_and_drift(t[:2], t[:2])


@pytest.mark.parametrize("true_ms", [0.0, 3.4, -7.1, 12.5])
def test_the_xcorr_lag_has_the_event_sign_and_sub_sample_accuracy(true_ms):
    t = np.arange(0.0, 10.0, 1 / BR_FS)
    sig = np.cumsum(np.random.default_rng(1).normal(size=t.size))
    later = np.interp(t - true_ms / 1e3, t, sig)            # "other" is later
    lag, corr = compare.xcorr_lag(sig, later, BR_FS, 0.02)
    assert lag * 1e3 == pytest.approx(true_ms, abs=0.15)
    assert corr > 0.99
    assert compare.xcorr_lag(sig, np.zeros_like(sig), BR_FS, 0.02)[1] == 0.0


# ---------------------------------------------------------------------------
# Position, never id
# ---------------------------------------------------------------------------


class _Reader:
    """The BlackrockRawIO calls a channel lookup and a chunk read make."""

    def __init__(self, columns: dict[str, np.ndarray], ids: list[str], fs: float, t0: float):
        self.columns, self.fs, self.t0 = columns, fs, t0
        self.header = {
            "signal_channels": np.array(
                [(i, f"ainp{k + 1}", "0") for k, i in enumerate(ids)],
                dtype=[("id", "U8"), ("name", "U16"), ("stream_id", "U4")],
            ),
            "signal_streams": np.array([("nsx2", "0")], dtype=[("name", "U8"), ("id", "U4")]),
        }
        self.n = next(iter(columns.values())).size

    def segment_count(self, block_index=0):
        return 1

    def get_signal_sampling_rate(self, stream_index=0):
        return self.fs

    def get_signal_size(self, block_index, seg_index, stream_index):
        return self.n

    def get_signal_t_start(self, block_index, seg_index, stream_index):
        return self.t0

    def get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                               stream_index, channel_ids):
        return self.columns[channel_ids[0]][i_start:i_stop].reshape(-1, 1)

    def rescale_signal_raw_to_float(self, raw, dtype, stream_index, channel_ids):
        return raw.astype(dtype)


def test_a_position_never_resolves_through_a_channel_id():
    # ids 1..4 at positions 0..3: an id lookup for "1" would take position 0.
    zeros = np.zeros(10)
    reader = _Reader({i: zeros for i in "1234"}, ["1", "2", "3", "4"], BR_FS, 0.0)
    assert blackrock.stream_for_position(reader, 1).channel_id == "2"
    assert blackrock.stream_for_channel(reader, 1).channel_id == "1"
    with pytest.raises(IndexError, match="outside"):
        blackrock.stream_for_position(reader, 4)


# ---------------------------------------------------------------------------
# End to end against synthetic recordings
# ---------------------------------------------------------------------------


def _world():
    """Photodiode and eye as functions of NSP time, shared by both systems."""
    rng = np.random.default_rng(3)
    onsets = T0 + 2.0 + np.cumsum(rng.uniform(1.0, 2.5, 400))
    onsets = onsets[onsets < T0 + DURATION - 5]
    knots = np.arange(T0 - 1.0, T0 + DURATION + 2.0, 0.005)      # 5 ms random walk
    walk = np.cumsum(rng.normal(size=knots.size))

    def signals(tau):
        j = np.searchsorted(onsets, tau) - 1
        pd = (j >= 0) & (tau - onsets[np.clip(j, 0, None)] < 0.3)
        return {"photodiode": pd.astype(float), "eye_x": np.interp(tau, knots, walk)}
    return signals


def _session(tmp_path, stamped_slope):
    """A OneBox analog export on Blackrock time, and Blackrock's .ns2 copy of it."""
    import spikesorting as ss

    signals = _world()
    npx_dir = tmp_path / "run_g0" / "run_g0_imec0"
    npx_dir.mkdir(parents=True)
    (npx_dir / "run_g0_t0.imec0.ap.bin").write_bytes(b"")
    session_file = tmp_path / "s.yaml"
    session_file.write_text(
        f"blackrock_dir: '{tmp_path / 'br'}'\nneuropixels_dir: '{npx_dir}'\n"
        "blackrock:\n  sync_file: '{blackrock_dir}/NSP_001.ns5'\n"
        "  analog_file: '{blackrock_dir}/NSP_001.ns2'\n"
        "  analog_channels: {eye_x: 0, photodiode: 3}\n"
        "neuropixels:\n  bin_file: '{neuropixels_dir}/run_g0_t0.imec0.ap.bin'\n"
        "  analog_channels: {eye_x: 1, photodiode: 4}\n",
        encoding="utf-8",
    )
    session = ss.load_session_config(session_file, "mac")

    # OneBox: XA0..XA4, XD, SY on its own clock.
    n = int((T0 + DURATION - C) / M * OBX_FS)
    world = signals(M * np.arange(n) / OBX_FS + C)
    data = np.zeros((n, 7), dtype=np.int16)
    data[:, 1] = np.round(world["eye_x"] * 300).astype(np.int16)
    data[:, 4] = np.round(world["photodiode"] * 20000).astype(np.int16)
    obx = tmp_path / "run_g0" / "run_g0_t0.obx0.obx.bin"
    obx.write_bytes(data.tobytes())
    obx.with_suffix(".meta").write_text(
        f"nSavedChans=7\nobSampRate={OBX_FS}\ntypeThis=obx\nsnsXaDwSy=5,1,1\n"
        "obAiRangeMax=5\nobMaxInt=32768\n", encoding="utf-8",
    )
    path = spikeglx.export_onebox_analog(
        obx, session.paths.export_for("neuropixels"),
        named={"eye_x": 1, "photodiode": 4}, burst_row=0,
    )
    matlab.update_mat(path, {"analog": {
        "timebase": "blackrock", "blackrock_slope": stamped_slope, "blackrock_intercept": C,
    }})

    # Blackrock .ns2: its own rate, and the stored copy FILTER_DELAY late.
    nbr = int(DURATION / BR_PERIOD)
    late = signals(T0 + np.arange(nbr) * BR_PERIOD - FILTER_DELAY)
    columns = {"129": late["eye_x"], "130": 0 * late["eye_x"],
               "131": 0 * late["eye_x"], "132": late["photodiode"]}
    reader = _Reader(columns, ["129", "130", "131", "132"], BR_FS, T0)
    return session, reader


def _open_with(reader):
    def fake_open(path, channels, gap_tolerance_ms=None, tolerance_s=None, by="id"):
        assert by == "position"                       # the session's keys are positions
        streams = {name: blackrock.stream_for_position(reader, pos)
                   for name, pos in channels.items()}
        time_map = blackrock.NspTimeMap(
            (blackrock.NspSegment(0, reader.n, BR_PERIOD, T0, 0.0),), BR_FS, "3.0", True
        )
        return analog.BlackrockAnalog(Path(path), reader, streams, time_map)
    return fake_open


def test_a_constant_filter_delay_reads_as_offset_and_passes(tmp_path, monkeypatch):
    from spikesorting import pipeline

    session, reader = _session(tmp_path, stamped_slope=M)
    monkeypatch.setattr(analog, "open_blackrock_analog", _open_with(reader))

    checks = {c["channel"]: c for c in pipeline._independent_checks(session)}

    pd, eye = checks["photodiode"], checks["eye_x"]
    assert pd["status"] == eye["status"] == "consistent"
    assert pd["method"].startswith("event onsets") and eye["method"].startswith("cross-corr")
    # OneBox minus Blackrock: Blackrock is late, so negative. The photodiode adds
    # up to one 1 kHz sample of rounding to Blackrock's edges.
    assert -3.1 < pd["median_offset_ms"] < -1.9
    assert eye["median_offset_ms"] == pytest.approx(-2.0, abs=0.3)
    assert abs(pd["drift_ms"]) < 0.3 and abs(eye["drift_ms"]) < 0.3
    assert "position 3 (id 132" in pd["blackrock"]     # what the position resolved to


def test_a_rate_error_reads_as_drift(tmp_path, monkeypatch):
    from spikesorting import pipeline
    from spikesorting._sync.report import build_sync_report, render_sync_report

    session, reader = _session(tmp_path, stamped_slope=M * (1 + 20e-6))
    monkeypatch.setattr(analog, "open_blackrock_analog", _open_with(reader))

    checks = pipeline._independent_checks(session)

    for check in checks:
        assert check["status"] == "DRIFT"
        # 20 ppm over the ~290 s the events span, OneBox running late by the end.
        assert check["drift_ms"] == pytest.approx(20e-6 * check["span_min"] * 60e3, rel=0.1)

    time_map = {"slope": 1.0, "intercept": 0.0, "drift_ppm": 0.0, "n_points": 10,
                "fit_max_residual_s": 0.0, "burst_clock": "probe",
                "onebox_to_probe": None}
    report = build_sync_report(time_map, None, [], checks=checks)
    assert "WARNING: eye_x, photodiode drift" in report["verdict_note"]
    assert "DRIFT" in render_sync_report(report)


def test_without_the_ns2_keys_there_is_nothing_to_check(tmp_path):
    from spikesorting import pipeline

    session, _ = _session(tmp_path, stamped_slope=M)
    from dataclasses import replace
    bare = replace(session, blackrock=replace(session.blackrock, analog_file=None,
                                              analog_channels={}))
    assert pipeline._independent_checks(bare) == []


# ---------------------------------------------------------------------------
# No burst: the same channels pick the 1 Hz cycle
# ---------------------------------------------------------------------------


def _plant_1hz_without_burst(session):
    """The OneBox's 1 Hz on every clock; the probe shares the OneBox's crystal."""
    from spikesorting._sync import catgt

    n_obx = int((T0 + DURATION - C) / M * OBX_FS)
    t = 0.25 + np.arange(int(n_obx / OBX_FS) - 1, dtype=np.float64)
    br = M * t + C
    keep = (br > T0 + 0.5) & (br < T0 + DURATION - 0.5)
    for system, files in (
        ("neuropixels", {"npx_1hz": t, "npx_obx_1hz": t, "npx_burst": np.empty(0)}),
        ("blackrock", {"blackrock_1hz": br[keep], "blackrock_burst": np.empty(0)}),
    ):
        folder = session.paths.sync_for(system)
        folder.mkdir(parents=True, exist_ok=True)
        for name, times in files.items():
            catgt.write_edge_file(folder / f"{name}.txt", times)


def _remap_with_a_planted_cycle_error(tmp_path, monkeypatch, drop=(), with_session=False):
    import json
    from dataclasses import replace

    import spikesorting as ss
    from spikesorting._sync import burst

    session, reader = _session(tmp_path, stamped_slope=M)
    if drop:
        kept = {k: v for k, v in session.blackrock.analog_channels.items() if k not in drop}
        session = replace(session, blackrock=replace(session.blackrock, analog_channels=kept))
    _plant_1hz_without_burst(session)
    monkeypatch.setattr(analog, "open_blackrock_analog", _open_with(reader))
    # The 1 Hz alone lands three cycles off -- what it does when one recording
    # started a few seconds before the other.
    true_offset = C + (M - 1.0) * DURATION / 2
    monkeypatch.setattr(burst, "estimate_offset", lambda *a, **k: true_offset + 3.0)

    result = ss.time_remapping(session, "neuropixels")
    payload = json.loads((session.paths.aligned / "time_map.json").read_text())
    return (result, payload, session) if with_session else (result, payload)


def test_without_a_burst_the_photodiode_picks_the_cycle(tmp_path, monkeypatch):
    result, payload = _remap_with_a_planted_cycle_error(tmp_path, monkeypatch)
    assert payload["cycles_from"] == "photodiode"
    # The fine map is still the 1 Hz fit: exact, not just within a cycle.
    assert result.mapping.intercept == pytest.approx(C, abs=1e-6)
    assert result.mapping.slope == pytest.approx(M, rel=1e-8)
    # ...and the OneBox analog export is stamped, burst or no burst.
    assert payload["onebox_to_blackrock"]["intercept"] == pytest.approx(C, abs=1e-6)


def test_without_a_burst_or_photodiode_the_eye_picks_the_cycle(tmp_path, monkeypatch):
    result, payload = _remap_with_a_planted_cycle_error(tmp_path, monkeypatch, drop=("photodiode",))
    assert payload["cycles_from"] == "eye_x"
    assert result.mapping.intercept == pytest.approx(C, abs=1e-6)


def test_with_nothing_to_pick_the_cycle_the_map_says_so(tmp_path, monkeypatch):
    from spikesorting._sync.report import build_sync_report

    result, payload = _remap_with_a_planted_cycle_error(
        tmp_path, monkeypatch, drop=("photodiode", "eye_x")
    )
    assert payload["cycles_from"] == "1hz_only"
    assert result.mapping.intercept == pytest.approx(C + 3.0, abs=1e-3)   # wrong, and flagged
    report = build_sync_report(payload, None, [])
    assert report["verdict"] == "NOT VALIDATED"
    assert "off by whole seconds" in report["verdict_note"]


def test_without_a_burst_agreeing_channels_make_the_map_consistent(tmp_path, monkeypatch):
    # The photodiode picks the offset; the eye, never used for it, agrees.
    import json

    import spikesorting as ss

    _, payload, session = _remap_with_a_planted_cycle_error(
        tmp_path, monkeypatch, with_session=True)
    vote = payload["cycle_vote"]
    assert vote["source"] == "photodiode" and vote["shift_s"] == pytest.approx(-3.0)
    assert vote["counted"] == "flashes" and vote["n_support"] > vote["n_runner_up"]

    outcome = ss.validate_remapping(session, figures=False)
    assert outcome.verdict == "CONSISTENT (no burst)"
    sync = json.loads((session.paths.export_for("neuropixels") / "sync_report.json").read_text())
    assert sync["verdict"] == "CONSISTENT (no burst)"
    assert "coarse match by photodiode in place of the burst" in sync["verdict_note"]
    text = (session.paths.export_for("neuropixels") / "sync_report.txt").read_text()
    assert "14 s coded burst                 not recorded" in text
    assert "coarse match (photodiode)" in text and "-3.000 s" in text
    assert "14 s burst: same OneBox clock" not in text
