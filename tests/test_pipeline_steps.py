"""Stage selection: which stages exist, and what they do when asked alone.

The pipeline's stages are deliberately independent -- sorting needs no edge
files, extraction needs no sorter -- so any subset can be run on any machine.
These tests pin that, since the split is what lets sorting go to a cluster while
the CatGT/TPrime half stays on the rig.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting import config as cfg
from spikesorting.pipeline import STEPS, step_lfp


def _session(tmp_path, run_dir=None, run_name="", **overrides):
    machines = tmp_path / "configs" / "machines"
    machines.mkdir(parents=True, exist_ok=True)
    (machines / "test.yaml").write_text(
        f"cache_dir: '{tmp_path / 'cache'}'\ncatgt_dir: null\ntprime_dir: null\ndevice: cpu\n",
        encoding="utf-8",
    )
    lines = [
        "session: s",
        f"neuropixels_dir: '{tmp_path / 'np'}'",
        "has_blackrock_data: false",
        "neuropixels:",
        f"  run_dir: '{run_dir if run_dir else tmp_path / 'np'}'",
        f"  run_name: {run_name or 'r'}",
    ]
    lines.extend(f"{k}: {v}" for k, v in overrides.items())
    path = tmp_path / "configs" / "s.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return cfg.load_session_config(path, "test", tmp_path / "configs")


def test_lfp_is_a_stage_of_its_own(tmp_path):
    # It is the one stage whose cost is bulk I/O rather than compute, so it has
    # to be selectable without dragging sync extraction along.
    assert "lfp" in STEPS


def test_lfp_export_writes_the_array_beside_the_recording(tmp_path):
    run = tmp_path / "np" / "run_g0" / "run_g0_imec0"
    run.mkdir(parents=True)
    n_chan, n_samples = 4, 1000
    data = np.arange(n_samples * n_chan, dtype=np.int16).reshape(n_samples, n_chan)
    (run / "run_g0_t0.imec0.lf.bin").write_bytes(data.tobytes())
    (run / "run_g0_t0.imec0.lf.meta").write_text(
        f"nSavedChans={n_chan}\nimSampRate=2500\nfileSizeBytes={data.nbytes}\n"
        "typeThis=imec\nimAiRangeMax=0.6\nimMaxInt=512\n",
        encoding="utf-8",
    )
    session = _session(tmp_path, run_dir=tmp_path / "np", run_name="run")

    result = step_lfp(session)

    assert result.status == "ok", result.notes
    written = list(session.paths.lfp.glob("*.lfp.npy"))
    assert written, session.paths.lfp
    assert np.load(written[0]).shape == (n_samples, n_chan)


def test_lfp_export_decimates_when_asked(tmp_path):
    run = tmp_path / "np" / "run_g0" / "run_g0_imec0"
    run.mkdir(parents=True)
    n_chan, n_samples = 2, 1000
    data = np.arange(n_samples * n_chan, dtype=np.int16).reshape(n_samples, n_chan)
    (run / "run_g0_t0.imec0.lf.bin").write_bytes(data.tobytes())
    (run / "run_g0_t0.imec0.lf.meta").write_text(
        f"nSavedChans={n_chan}\nimSampRate=2500\nfileSizeBytes={data.nbytes}\n"
        "typeThis=imec\nimAiRangeMax=0.6\nimMaxInt=512\n",
        encoding="utf-8",
    )
    session = _session(tmp_path, run_dir=tmp_path / "np", run_name="run")

    result = step_lfp(session, decimate=4)

    assert result.status == "ok", result.notes
    written = list(session.paths.lfp.glob("*.lfp.npy"))
    assert np.load(written[0]).shape == (n_samples // 4, n_chan)


def test_lfp_export_skips_a_session_with_no_spikeglx_run(tmp_path):
    # A Blackrock-only session must report rather than fail: Central saves those
    # LFPs separately, so there is nothing here to export.
    session = _session(tmp_path)

    result = step_lfp(session)

    assert result.status == "skipped"
    assert any("lf" in note.lower() for note in result.notes)


def test_lfp_export_skips_when_the_system_did_not_record(tmp_path):
    session = _session(tmp_path, has_neuropixels_data="false")

    result = step_lfp(session)

    assert result.status == "skipped"
    assert any("has_neuropixels_data" in note for note in result.notes)
