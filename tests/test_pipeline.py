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
    path = tmp_path / "s.yaml"
    path.write_text(
        f"session: s\nblackrock_dir: '{tmp_path / 'br'}'\n"
        f"neuropixels_dir: '{tmp_path / 'np'}'\n{body}",
        encoding="utf-8",
    )
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
        "preprocess",
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


def test_a_probe_file_is_used_for_either_system(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(tmp_path, f"blackrock:\n  probe_file: '{probe}'\n")

    loaded = ss.setup_probe(session, "blackrock")

    assert loaded["n_chan"] == 4
    assert list(loaded["chanMap"]) == [0, 1, 2, 3]


def test_a_probe_file_beats_the_cmp_it_was_built_from(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(
        tmp_path, f"blackrock:\n  probe_file: '{probe}'\n  cmp_file: '/nope/missing.cmp'\n"
    )

    # Resolves without touching the .cmp, which does not exist.
    assert ss.setup_probe(session, "blackrock")["n_chan"] == 4


def test_a_utah_array_without_a_map_falls_back_to_the_flagged_placeholder(tmp_path):
    # It must still sort -- and must still say the geometry is a guess.
    assert ss.setup_probe(_session(tmp_path), "blackrock")["_placeholder"] is True


# ---------------------------------------------------------------------------
# The config decides whether a verb does anything
# ---------------------------------------------------------------------------


def test_a_system_that_never_recorded_yields_none_everywhere(tmp_path):
    session = _session(tmp_path, "has_blackrock_data: false\n")

    assert ss.setup_probe(session, "blackrock") is None
    assert ss.load_spike_continuous(session, "blackrock") is None
    assert ss.extract_sync(session, "blackrock") is None
    assert ss.export_results(session, "blackrock") is None


def test_declining_to_sort_still_leaves_the_data_loadable(tmp_path):
    # has_data true + kilosort_on false: pulses and LFP, no sorter. The guard is
    # on sorting alone, so nothing upstream of it is turned off.
    session = _session(tmp_path, "kilosort_on_blackrock: false\n")

    assert session.has_blackrock_data is True
    assert ss.sort_with_kilosort(object(), session, "blackrock") is None
    assert ss.skip_reason(session, ss.sort_with_kilosort, "blackrock") == (
        "kilosort_on_blackrock is false"
    )


def test_none_propagates_so_a_sequence_needs_no_branching(tmp_path):
    # This is what lets the runner be a flat list of calls.
    session = _session(tmp_path)

    assert ss.preprocess(None, session, "neuropixels") is None
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


def test_skip_sync_turns_off_both_cross_system_verbs(tmp_path):
    session = _session(tmp_path, "skip_sync: true\n")

    assert ss.time_remapping(session, "neuropixels") is None
    assert ss.validate_remapping(session) is None
    assert "skip_sync" in ss.skip_reason(session, ss.time_remapping, "neuropixels")


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
