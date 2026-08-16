"""End-to-end sync extraction and alignment against synthetic recordings.

Covers the seam the unit tests cannot: config -> file -> detector -> edge file,
and then edge files -> coarse offset -> trim -> map -> validate. Everything runs
without CatGT, TPrime, Kilosort or a GPU, which is exactly the HPC configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import coded_burst_times, square_wave

from spikesorting import _config as cfg
from spikesorting._io import spikeglx
from spikesorting import extract_sync
from spikesorting._sync import align, burst, catgt, extract

FS = 30000.0
CONFIG_DIR = None  # set per-test to the tmp config dir


def write_ap_with_sync(path, duration_s=20.0, phase_s=0.25, n_chan=385):
    """Synthetic AP binary whose SY word bit 6 carries a 1 Hz square wave."""
    n_samples = int(duration_s * FS)
    data = np.zeros((n_samples, n_chan), dtype=np.int16)
    wave = square_wave(duration_s, FS, period_s=1.0, phase_s=phase_s)
    data[:, -1] = (wave.astype(np.uint16) << 6).astype(np.int16)
    path.write_bytes(data.tobytes())
    return path


def make_session(tmp_path, bin_file, **overrides):
    """A SessionConfig pointing at a bare binary, resolved on a tool-free machine."""
    machines = tmp_path / "configs" / "machines"
    machines.mkdir(parents=True, exist_ok=True)
    (machines / "test.yaml").write_text(
        f"data_root: '{tmp_path}'\noutput_root: '{tmp_path / 'out'}'\n"
        f"cache_dir: '{tmp_path / 'cache'}'\ncatgt_dir: null\ntprime_dir: null\ndevice: cpu\n",
        encoding="utf-8",
    )
    lines = [
        "session: synthetic",
        "output_dir: '{output_root}/synthetic'",
        "neuropixels:",
        f"  bin_file: '{bin_file}'",
        "  n_chan_bin: 385",
        "  sample_rate: 30000",
        "  sync_bit: 6",
        "  sync_pulse_ms: 500",
    ]
    lines.extend(f"{k}: {v}" for k, v in overrides.items())
    session_path = tmp_path / "configs" / "synthetic.yaml"
    session_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    session = cfg.load_session_config(session_path, "test", tmp_path / "configs")
    session.paths.mkdirs()
    return session


def test_extract_writes_edge_file_with_correct_times(tmp_path):
    bin_file = write_ap_with_sync(tmp_path / "synthetic.imec0.ap.bin", duration_s=10.0)
    session = make_session(tmp_path, bin_file)

    report = extract.extract_neuropixels_edges(session)
    edge_set = report.edge_sets[extract.NPX_1HZ]

    assert edge_set.source == "numpy"
    assert edge_set.n == 10
    assert np.allclose(edge_set.times_s, 0.25 + np.arange(10))
    assert edge_set.path.exists()
    assert np.allclose(catgt.read_edge_file(edge_set.path), edge_set.times_s, atol=1e-6)


def test_extract_reports_a_recording_with_no_sync_pulses(tmp_path):
    # A file with a flat SY word must yield zero edges and say so -- this is the
    # correct result for a recording made without SMA1 connected, not an error.
    n_samples = int(2.0 * FS)
    path = tmp_path / "nosync.imec0.ap.bin"
    path.write_bytes(np.zeros((n_samples, 385), dtype=np.int16).tobytes())
    session = make_session(tmp_path, path)

    report = extract.extract_neuropixels_edges(session)
    assert report.edge_sets[extract.NPX_1HZ].n == 0
    assert any("No 1 Hz sync pulses" in note for note in report.notes)


def test_extract_records_the_catgt_command_it_would_have_run(tmp_path):
    # On a machine without CatGT the fallback runs, but the command is still
    # reported so a rig run can be reproduced by hand.
    run_dir = tmp_path / "rundir"
    probe_dir = run_dir / "run_g0" / "run_g0_imec0"
    probe_dir.mkdir(parents=True)
    write_ap_with_sync(probe_dir / "run_g0_t0.imec0.ap.bin", duration_s=3.0)
    spikeglx.meta_path_for(probe_dir / "run_g0_t0.imec0.ap.bin").write_text(
        "nSavedChans=385\nimSampRate=30000\ntypeThis=imec\nsnsApLfSy=384,0,1\n", encoding="utf-8"
    )

    machines = tmp_path / "configs" / "machines"
    machines.mkdir(parents=True, exist_ok=True)
    (machines / "test.yaml").write_text(
        f"output_root: '{tmp_path / 'out'}'\ncache_dir: '{tmp_path / 'c'}'\n"
        "catgt_dir: null\ntprime_dir: null\ndevice: cpu\n",
        encoding="utf-8",
    )
    session_path = tmp_path / "configs" / "run.yaml"
    session_path.write_text(
        "session: run\noutput_dir: '{output_root}/run'\n"
        f"neuropixels:\n  run_dir: '{run_dir}'\n  run_name: run\n  gate: 0\n  trigger: 0\n",
        encoding="utf-8",
    )
    session = cfg.load_session_config(session_path, "test", tmp_path / "configs")
    session.paths.mkdirs()

    report = extract.extract_neuropixels_edges(session)
    assert report.catgt_commands
    command = report.catgt_commands[0]
    assert "-xd=2,0,-1,6,500" in command
    assert "-ap" in command and "-prb=0" in command
    assert any("no catgt_dir" in note for note in report.notes)


def test_extract_sync_returns_none_for_a_system_that_never_recorded(tmp_path):
    # Each system is extracted on its own now, so "Blackrock did not record" is a
    # None from that call rather than a note buried in a combined report.
    bin_file = write_ap_with_sync(tmp_path / "s.imec0.ap.bin", duration_s=5.0)
    session = make_session(tmp_path, bin_file)

    assert extract_sync(session, "blackrock") is None

    report = extract_sync(session, "neuropixels")
    assert report is not None
    assert (session.paths.sync_for("neuropixels") / "npx_1hz.txt").exists()


def test_alignment_recovers_a_known_offset_and_drift(tmp_path):
    """The full step-8/9 chain on two synthetic systems.

    Blackrock is the reference. SpikeGLX starts 137.5 s later on its own clock and
    its crystal runs 30 ppm fast -- both of which the pipeline must undo.
    """
    offset, drift = 137.5, 30e-6
    duration = 1200.0

    br_1hz = np.arange(0.0, duration, 1.0)
    br_burst = coded_burst_times(int(duration // 14), interval_s=14.0, pulses_per_burst=8)

    def to_spikeglx(times):
        return (times - offset) / (1.0 + drift)

    # SpikeGLX started late and stopped early, so it saw fewer pulses.
    npx_1hz = to_spikeglx(br_1hz[br_1hz >= 200.0])
    npx_burst = to_spikeglx(br_burst[br_burst >= 200.0])

    match = burst.match_bursts(br_burst, npx_burst, min_gap_s=7.0, tolerance_s=0.05)
    assert match.n_matched > 50
    assert match.offset_s == pytest.approx(offset, abs=0.5)

    br_trim, npx_trim = align.trim_pair_to_overlap(br_1hz, npx_1hz, match.offset_s, margin_s=0.25)
    ref_idx, other_idx = burst.match_times(br_trim, npx_trim, match.offset_s, tolerance_s=0.25)
    assert ref_idx.size > 900

    mapping = align.fit_linear_map(npx_trim[other_idx], br_trim[ref_idx])
    assert mapping.slope == pytest.approx(1.0 + drift, rel=1e-9)
    assert mapping.intercept == pytest.approx(offset, abs=1e-6)
    assert mapping.drift_ppm == pytest.approx(30.0, abs=0.01)

    # Step 9: the bursts are held out from the fit, so this is a real check.
    npx_onsets = burst.group_burst_onsets(npx_burst, 7.0)
    br_onsets = burst.group_burst_onsets(br_burst, 7.0)
    report = align.validate_alignment(mapping.apply(npx_onsets), br_onsets, tolerance_s=1e-3)
    assert report.passed
    assert report.max_residual_s < 1e-6
    assert report.n_checked > 50


def test_ignoring_clock_drift_fails_validation(tmp_path):
    """Offset-only alignment must be caught by step 9, not quietly accepted."""
    drift, offset, duration = 30e-6, 100.0, 1200.0
    br_burst = coded_burst_times(int(duration // 14), interval_s=14.0, pulses_per_burst=8)
    npx_burst = (br_burst - offset) / (1.0 + drift)

    offset_only = align.LinearMap(slope=1.0, intercept=offset, n_points=2, residuals_s=np.empty(0))
    report = align.validate_alignment(
        offset_only.apply(burst.group_burst_onsets(npx_burst, 7.0)),
        burst.group_burst_onsets(br_burst, 7.0),
        tolerance_s=1e-3,
        match_tolerance_s=1.0,
    )
    assert not report.passed
    assert report.max_residual_s > 1e-3
