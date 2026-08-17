"""Probe file IO and validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spikesorting._io import spikeglx
from spikesorting._probes import io as probe_io
from spikesorting._probes import neuropixels, utah


def linear_probe(n=8):
    return {
        "chanMap": np.arange(n, dtype=np.int32),
        "xc": np.zeros(n, dtype=np.float32),
        "yc": (np.arange(n) * 20.0).astype(np.float32),
        "kcoords": np.zeros(n, dtype=np.float32),
        "n_chan": n,
    }


def test_probe_json_round_trips(tmp_path):
    probe = linear_probe()
    path = probe_io.save_probe_json(probe, tmp_path / "probe.json")
    restored = probe_io.load_probe_json(path)

    for key in ("chanMap", "xc", "yc", "kcoords"):
        assert np.array_equal(restored[key], probe[key])
    assert restored["n_chan"] == probe["n_chan"]


def test_probe_json_matches_kilosorts_format(tmp_path):
    # kilosort.io.save_probe writes json.dumps of a dict of plain lists, with
    # exactly these keys. Anything extra and Kilosort's loader is unhappy.
    path = probe_io.save_probe_json(linear_probe(), tmp_path / "probe.json")
    raw = json.loads(path.read_text())
    assert set(raw) == set(probe_io.REQUIRED_KEYS)
    assert isinstance(raw["chanMap"], list)
    assert isinstance(raw["n_chan"], int)


def test_bookkeeping_keys_are_not_written(tmp_path):
    probe = utah.utah_grid_probe(9, n_cols=3)
    assert probe["_placeholder"] is True
    raw = json.loads(probe_io.save_probe_json(probe, tmp_path / "p.json").read_text())
    assert "_placeholder" not in raw
    assert "labels" not in raw


def test_validate_accepts_a_good_probe():
    assert probe_io.validate_probe(linear_probe()) == []


def test_validate_reports_missing_keys():
    problems = probe_io.validate_probe({"chanMap": [0, 1]})
    assert len(problems) == 1
    assert "missing required key" in problems[0]


def test_validate_catches_length_mismatch():
    probe = linear_probe()
    probe["xc"] = probe["xc"][:4]
    assert any("lengths disagree" in p for p in probe_io.validate_probe(probe))


def test_validate_catches_wrong_n_chan():
    probe = linear_probe()
    probe["n_chan"] = 99
    assert any("n_chan" in p for p in probe_io.validate_probe(probe))


def test_validate_catches_duplicate_channels():
    probe = linear_probe()
    probe["chanMap"] = np.array([0, 0, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    assert any("duplicate" in p for p in probe_io.validate_probe(probe))


def test_validate_catches_overlapping_contacts():
    probe = linear_probe()
    probe["yc"] = np.zeros(8, dtype=np.float32)
    assert any("same" in p for p in probe_io.validate_probe(probe))


def test_validate_allows_shanks_to_reuse_coordinates():
    # SpikeGLX reports x/z per shank, so identical coordinates on different
    # shanks are correct, not a collision.
    probe = {
        "chanMap": np.arange(4, dtype=np.int32),
        "xc": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "yc": np.array([0.0, 20.0, 0.0, 20.0], dtype=np.float32),
        "kcoords": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        "n_chan": 4,
    }
    assert probe_io.validate_probe(probe) == []


def test_saving_an_invalid_probe_raises(tmp_path):
    probe = linear_probe()
    probe["n_chan"] = 99
    with pytest.raises(ValueError, match="refusing to save"):
        probe_io.save_probe_json(probe, tmp_path / "bad.json")


def test_load_missing_probe_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe_io.load_probe_json(tmp_path / "absent.json")


def test_probe_from_mat_converts_matlab_indexing(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "chanmap.mat"
    savemat(
        str(path),
        {
            "chanMap": np.arange(1, 5),  # 1-based, as MATLAB writes it
            "xcoords": np.zeros(4),
            "ycoords": np.arange(4) * 20.0,
            "shankInd": np.zeros(4),
        },
    )
    probe = probe_io.probe_from_mat(path)
    assert probe["chanMap"].tolist() == [0, 1, 2, 3]  # converted to 0-based
    assert probe["n_chan"] == 4


def test_probe_from_mat_prefers_the_zero_based_map(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "chanmap.mat"
    savemat(
        str(path),
        {
            "chanMap": np.arange(1, 5),
            "chanMap0ind": np.arange(0, 4),
            "xcoords": np.zeros(4),
            "ycoords": np.arange(4) * 20.0,
        },
    )
    assert probe_io.probe_from_mat(path)["chanMap"].tolist() == [0, 1, 2, 3]


def test_probe_from_mat_without_coordinates_raises(tmp_path):
    from scipy.io import savemat

    path = tmp_path / "bad.mat"
    savemat(str(path), {"chanMap0ind": np.arange(4)})
    with pytest.raises(ValueError, match="xcoords"):
        probe_io.probe_from_mat(path)


def test_geom_header_is_parsed():
    header = spikeglx.parse_geom_header({"~snsGeomMap": "(NP2004,4,250,70)(0:0:0:1)"})
    assert header["part_number"] == "NP2004"
    assert header["n_shank"] == 4
    assert header["shank_pitch_um"] == 250.0


def test_geom_header_absent_returns_none():
    assert spikeglx.parse_geom_header({}) is None


def test_shank_pitch_makes_multi_shank_coordinates_absolute():
    # Without the pitch, all four shanks stack on top of each other and Kilosort
    # sees one impossibly dense column.
    meta = {
        "~snsGeomMap": "(NP2004,4,250,70)"
        + "".join(f"({s}:0:{r * 15}:1)" for s in range(4) for r in range(2))
    }
    probe = neuropixels.probe_from_meta(meta)
    assert sorted(set(probe["xc"].tolist())) == [0.0, 250.0, 500.0, 750.0]
    assert probe_io.validate_probe(probe) == []


def test_single_shank_is_unaffected_by_shank_pitch():
    meta = {"~snsGeomMap": "(NP1000,1,0,70)(0:27:0:1)(0:59:20:1)"}
    probe = neuropixels.probe_from_meta(meta)
    assert probe["xc"].tolist() == [27.0, 59.0]
