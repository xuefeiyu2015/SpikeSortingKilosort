"""The flat verb surface: one function per thing you do, per system.

Only the parts that need no sorter are covered -- dispatch, probe resolution,
which systems a pipeline touches. The SpikeInterface and Kilosort calls
themselves are exercised on the rig; nothing here can run them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spikesorting import api
from spikesorting import config as cfg

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


def test_every_verb_names_the_system_it_acts_on(tmp_path):
    # The point of the surface: both systems read the same way.
    session = _session(tmp_path)
    for verb in (api.load_probe, api.extract_sync, api.extract_lfp):
        with pytest.raises(ValueError, match="system must be one of"):
            verb(session, "utah")


def test_a_probe_file_is_used_for_either_system(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(tmp_path, f"blackrock:\n  probe_file: '{probe}'\n")

    loaded = api.load_probe(session, "blackrock")

    assert loaded["n_chan"] == 4
    assert list(loaded["chanMap"]) == [0, 1, 2, 3]


def test_a_utah_array_without_a_map_falls_back_to_the_flagged_placeholder(tmp_path):
    # It must still sort -- and must still say the geometry is a guess.
    session = _session(tmp_path)

    loaded = api.load_probe(session, "blackrock")

    assert loaded["_placeholder"] is True


def test_a_probe_file_beats_the_cmp_it_was_built_from(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(
        tmp_path,
        f"blackrock:\n  probe_file: '{probe}'\n  cmp_file: '/nope/missing.cmp'\n",
    )

    # Resolves without touching the .cmp, which does not exist.
    assert api.load_probe(session, "blackrock")["n_chan"] == 4


def test_blackrock_lfp_is_a_no_op_rather_than_an_error(tmp_path):
    # Central already saves those separately, so there is nothing to extract.
    assert api.extract_lfp(_session(tmp_path), "blackrock") is None


def test_neuropixels_lfp_reports_nothing_when_there_is_no_run(tmp_path):
    assert api.extract_lfp(_session(tmp_path), "neuropixels") is None


def test_the_sorting_pipeline_skips_a_system_that_never_recorded(tmp_path):
    session = _session(tmp_path, "has_blackrock_data: false\nhas_neuropixels_data: false\n")

    out = api.run_sorting_pipeline(session)

    assert out["skipped_blackrock"] == "has_blackrock_data is false"
    assert out["skipped_neuropixels"] == "has_neuropixels_data is false"
    assert not any(key.startswith("sort_") for key in out)


def test_the_sorting_pipeline_extracts_without_sorting_when_asked(tmp_path):
    # has_data true + kilosort_on false: pulses and LFP, no sorter.
    session = _session(
        tmp_path,
        "has_blackrock_data: false\nkilosort_on_neuropixels: false\n",
    )

    out = api.run_sorting_pipeline(session, systems=("neuropixels",))

    assert "extract_sync_neuropixels" in out
    assert out["skipped_sort_neuropixels"] == "kilosort_on_neuropixels is false"


def test_rough_align_says_what_is_missing_rather_than_guessing(tmp_path):
    session = _session(tmp_path)

    with pytest.raises(FileNotFoundError, match="run extract_sync"):
        api.rough_align(session)


def test_rough_align_recovers_a_planted_offset(tmp_path):
    from spikesorting.sync import catgt

    session = _session(tmp_path)
    for system in ("blackrock", "neuropixels"):
        session.paths.sync_for(system).mkdir(parents=True, exist_ok=True)

    offset = 12.3456
    br = np.arange(0.0, 600.0, 1.0) + 100.0
    catgt.write_edge_file(session.paths.sync_for("blackrock") / "blackrock_1hz.txt", br)
    catgt.write_edge_file(session.paths.sync_for("neuropixels") / "npx_1hz.txt", br - offset)

    # No burst files, so this is the 1 Hz-only estimate -- good to a whole cycle.
    assert abs(api.rough_align(session) - offset) < 0.5


def test_sort_requires_a_recording_rather_than_loading_one(tmp_path):
    # The three steps are separate on purpose: load, preprocess, sort. Hiding the
    # first two inside the third is what made the workflow hard to follow.
    import inspect

    params = list(inspect.signature(api.sort).parameters)
    assert params[0] == "recording"
    assert inspect.signature(api.sort).parameters["recording"].default is inspect.Parameter.empty


def test_the_convenience_wrapper_is_the_three_steps_in_order():
    import inspect

    body = inspect.getsource(api.load_preprocess_sort)
    assert body.index("load_data") < body.index("preprocess(") < body.index("return sort(")
