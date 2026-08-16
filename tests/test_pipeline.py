"""The verb surface: one function per thing you do, per system.

Only the parts that need no sorter are covered -- probe resolution, the guards
that read the session config, the skip reasons the runners report. The
SpikeInterface and Kilosort calls are exercised on the rig; nothing here can run
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import spikesorting as ss
from spikesorting import _config as cfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _session(tmp_path, body=""):
    """A session declaring both systems, unless `body` declares them itself.

    Declaring paths is what says a system recorded, so a fixture that wants both
    systems present has to name files for both -- there is no flag any more.
    """
    lines = [
        "session: s",
        f"blackrock_dir: '{tmp_path / 'br'}'",
        f"neuropixels_dir: '{tmp_path / 'np'}'",
    ]
    if "neuropixels:" not in body:
        lines.append("neuropixels:\n  bin_file: '/a/x.bin'")
    if "blackrock:" not in body:
        lines.append("blackrock:\n  sync_file: '/b/y.ns5'")
    path = tmp_path / "s.yaml"
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
    return cfg.load_session_config(path, "mac", CONFIG_DIR)


def _probe_json(path: Path, n: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "chanMap": list(range(n)),
                "xc": [0.0] * n,
                "yc": [float(i * 40) for i in range(n)],
                "kcoords": [0] * n,
                "n_chan": n,
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


def test_the_package_exports_the_workflow_and_nothing_else():
    # Opening `import spikesorting` should read as the pipeline. A `step_*` or a
    # second surface reappearing here is the regression this guards.
    assert not [n for n in ss.__all__ if n.startswith("step_")]
    for verb in (
        "load_session_config",
        "setup_probe",
        "load_spike_continuous",
        "sort_with_kilosort",
        "extract_sync",
        "extract_lfp",
        "time_remapping",
        "validate_remapping",
        "export_results",
    ):
        assert callable(getattr(ss, verb)), verb


def test_every_per_system_verb_rejects_an_unknown_system(tmp_path):
    session = _session(tmp_path)
    for verb in (ss.setup_probe, ss.extract_sync, ss.extract_lfp, ss.load_spike_continuous):
        with pytest.raises(ValueError, match="system must be one of"):
            verb(session, "utah")


def test_importing_the_package_pulls_in_no_heavy_dependency():
    # config, sync, alignment and export must load on a machine with no sorter.
    import sys

    heavy = {"kilosort", "torch", "spikeinterface"}
    assert not (heavy & set(sys.modules)), sorted(heavy & set(sys.modules))


# ---------------------------------------------------------------------------
# The channel map
# ---------------------------------------------------------------------------


def _meta_run(tmp_path, n_chan=4, with_geom=True):
    """A minimal SpikeGLX run whose .meta may or may not carry ~snsGeomMap."""
    import numpy as np

    run = tmp_path / "np" / "r_g0" / "r_g0_imec0"
    run.mkdir(parents=True, exist_ok=True)
    data = np.zeros((100, n_chan), dtype=np.int16)
    (run / "r_g0_t0.imec0.ap.bin").write_bytes(data.tobytes())
    meta = [
        f"nSavedChans={n_chan}",
        "imSampRate=30000",
        f"fileSizeBytes={data.nbytes}",
        "typeThis=imec",
        "imAiRangeMax=0.6",
        "imMaxInt=512",
    ]
    if with_geom:
        # (shank, x, y, used) per channel -- the format probe_from_meta parses.
        meta.append("~snsGeomMap=(NP1000,1,0,70)" + "".join(
            f"(0:{11 + 16 * (i % 2)}:{20 * (i // 2)}:1)" for i in range(n_chan)
        ))
    (run / "r_g0_t0.imec0.ap.meta").write_text("\n".join(meta) + "\n", encoding="utf-8")
    return tmp_path / "np"


def test_the_runs_own_meta_wins_over_a_probe_file(tmp_path):
    # The .meta records which sites were actually active for *this* run, which a
    # file built from another run cannot know. So it is not an override target.
    _meta_run(tmp_path)
    other = _probe_json(tmp_path / "probes" / "other.json", n=99)
    session = _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{tmp_path / 'np'}'\n  run_name: r\n"
        f"  probe_file: '{other}'\n",
    )

    probe = ss.setup_probe(session, "neuropixels")

    assert probe["n_chan"] == 4          # from the .meta
    assert probe["n_chan"] != 99         # not the probe_file


def test_probe_file_is_used_when_the_run_has_no_meta(tmp_path):
    # A bare binary, which is the demo's situation.
    binary = tmp_path / "np" / "bare.imec0.ap.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00" * 800)
    named = _probe_json(tmp_path / "probes" / "np1.json", n=4)
    session = _session(
        tmp_path,
        f"neuropixels:\n  bin_file: '{binary}'\n  n_chan_bin: 4\n"
        f"  sample_rate: 30000\n  probe_file: '{named}'\n",
    )

    assert ss.setup_probe(session, "neuropixels")["n_chan"] == 4


def test_a_meta_without_geometry_falls_through_to_probe_file(tmp_path):
    # Runs predating ~snsGeomMap: the .meta exists but carries no geometry, so it
    # must fall through rather than raise past the configured fallback.
    _meta_run(tmp_path, with_geom=False)
    named = _probe_json(tmp_path / "probes" / "np1.json", n=4)
    session = _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{tmp_path / 'np'}'\n  run_name: r\n"
        f"  probe_file: '{named}'\n",
    )

    assert ss.setup_probe(session, "neuropixels")["n_chan"] == 4


def test_neuropixels_with_neither_raises_and_names_the_fix(tmp_path):
    # No honest default exists for a Neuropixels layout, so this must not guess.
    binary = tmp_path / "np" / "bare.imec0.ap.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00" * 800)
    session = _session(
        tmp_path,
        f"neuropixels:\n  bin_file: '{binary}'\n  n_chan_bin: 4\n  sample_rate: 30000\n",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        ss.setup_probe(session, "neuropixels")

    assert "make_probe.py" in str(excinfo.value)
    assert "probe_file" in str(excinfo.value)


def test_a_utah_cmp_wins_over_a_probe_file(tmp_path):
    # Same rule: the array's own wiring map beats a file built from another one.
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(
        tmp_path, f"blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '{probe}'\n"
        f"  cmp_file: '/nope/missing.cmp'\n"
    )

    # cmp_file is tried first, so a missing one surfaces rather than being skipped.
    with pytest.raises(Exception):
        ss.setup_probe(session, "blackrock")


def test_a_utah_probe_file_is_used_when_no_cmp_is_named(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(tmp_path, f"blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '{probe}'\n")

    assert ss.setup_probe(session, "blackrock")["n_chan"] == 4


def test_a_utah_array_with_no_map_at_all_falls_back_to_the_flagged_placeholder(tmp_path):
    # It must still sort -- and must still say the geometry is a guess.
    assert ss.setup_probe(_session(tmp_path), "blackrock")["_placeholder"] is True


# ---------------------------------------------------------------------------
# The config decides whether a verb does anything
# ---------------------------------------------------------------------------


def test_none_propagates_so_a_sequence_needs_no_branching(tmp_path):
    # This is what lets the runner be a flat list of calls.
    session = _session(tmp_path)

    assert ss.sort_with_kilosort(None, session, "neuropixels") is None


def test_blackrock_lfp_is_a_no_op_rather_than_an_error(tmp_path):
    # Central already saves those separately, so there is nothing to extract.
    session = _session(tmp_path)

    assert ss.extract_lfp(session, "blackrock") is None
    assert "Central" in ss.skip_reason(session, ss.extract_lfp, "blackrock")


def test_skip_reason_is_none_when_the_verb_will_actually_run(tmp_path):
    session = _session(tmp_path)

    assert ss.skip_reason(session, ss.extract_sync, "neuropixels") is None
    assert ss.skip_reason(session, ss.setup_probe, "blackrock") is None


# ---------------------------------------------------------------------------
# Time remapping
# ---------------------------------------------------------------------------


def test_time_remapping_reports_missing_edges_rather_than_guessing(tmp_path):
    assert ss.time_remapping(_session(tmp_path), "neuropixels") is None


def test_time_remapping_recovers_a_planted_offset_and_drift(tmp_path):
    from spikesorting._sync import catgt

    session = _session(tmp_path)
    for system in ("blackrock", "neuropixels"):
        session.paths.sync_for(system).mkdir(parents=True, exist_ok=True)

    offset, ppm = 12.3456, 20e-6
    br = np.arange(0.0, 600.0, 1.0) + 100.0
    npx = (br - offset) * (1 + ppm)
    catgt.write_edge_file(session.paths.sync_for("blackrock") / "blackrock_1hz.txt", br)
    catgt.write_edge_file(session.paths.sync_for("neuropixels") / "npx_1hz.txt", npx)

    result = ss.time_remapping(session, "neuropixels")

    assert isinstance(result, ss.TimeMap)
    # No sorting to map, so the map itself is the whole output.
    assert result.method == "map_only"
    assert result.mapping.intercept == pytest.approx(offset, abs=1e-6)
    assert result.mapping.drift_ppm == pytest.approx(-20.0, abs=0.1)
    assert (session.paths.aligned / "time_map.json").exists()
    assert "ppm" in result.summary()


def test_the_reference_timebase_is_not_remapped_onto_itself(tmp_path):
    # Blackrock is the reference. Asking to map it is meaningless, and the runner
    # loops over both systems, so the verb has to say so rather than the caller.
    session = _session(tmp_path)

    assert ss.REFERENCE_SYSTEM == "blackrock"
    assert ss.time_remapping(session, "blackrock") is None
    assert "reference timebase" in ss.skip_reason(session, ss.time_remapping, "blackrock")
    # ...while the other system has no such guard.
    assert ss.skip_reason(session, ss.time_remapping, "neuropixels") is None
