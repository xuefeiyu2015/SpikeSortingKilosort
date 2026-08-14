"""Probe geometry construction."""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting.probes import common, neuropixels, utah


def test_probe_from_geom_keeps_every_site():
    # Unused sites are kept so chanMap stays contiguous; Kilosort's own
    # bad-channel handling is the right place to drop them.
    geom = np.array(
        [[0, 27, 0, 1], [0, 59, 0, 1], [0, 27, 20, 1], [0, 59, 20, 0]], dtype=float
    )
    probe = neuropixels.probe_from_geom(geom)
    assert probe["n_chan"] == 4
    assert probe["chanMap"].tolist() == [0, 1, 2, 3]
    assert probe["xc"].tolist() == [27.0, 59.0, 27.0, 59.0]
    assert probe["yc"].tolist() == [0.0, 0.0, 20.0, 20.0]
    assert np.all(probe["kcoords"] == 0)


def test_probe_from_geom_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        neuropixels.probe_from_geom(np.zeros((4, 2)))


def test_probe_from_meta_uses_the_geom_map():
    meta = {"~snsGeomMap": "(NP1000,1,0,70)(0:27:0:1)(0:59:0:1)"}
    probe = neuropixels.probe_from_meta(meta)
    assert probe["n_chan"] == 2


def test_probe_from_meta_without_geom_map_raises_helpfully():
    with pytest.raises(ValueError, match="snsShankMap"):
        neuropixels.probe_from_meta({"nSavedChans": "385"})


def test_multi_shank_geometry_maps_to_kcoords():
    geom = np.array([[0, 0, 0, 1], [0, 0, 20, 1], [1, 0, 0, 1], [1, 0, 20, 1]], dtype=float)
    probe = neuropixels.probe_from_geom(geom)
    assert np.unique(probe["kcoords"]).tolist() == [0.0, 1.0]


def test_utah_grid_is_flagged_as_a_placeholder():
    # This flag is what makes the sorting step warn; losing it would let a guessed
    # geometry through silently.
    probe = utah.utah_grid_probe(96)
    assert probe["_placeholder"] is True
    assert common.probe_summary(probe)["placeholder_geometry"] is True


def test_utah_grid_spacing_is_400_um():
    probe = utah.utah_grid_probe(100, n_cols=10)
    assert probe["xc"][:10].tolist() == [i * 400.0 for i in range(10)]
    assert probe["yc"][0] == 0.0 and probe["yc"][10] == 400.0


def test_utah_channels_are_independent_by_default():
    # At 400 um no spike reaches two electrodes, so each is its own group.
    probe = utah.utah_grid_probe(96)
    assert np.unique(probe["kcoords"]).size == 96

    grouped = utah.utah_grid_probe(96, independent=False)
    assert np.unique(grouped["kcoords"]).size == 1


def test_probe_from_cmp_parses_a_map_file(tmp_path):
    path = tmp_path / "array.cmp"
    path.write_text(
        "//Cerebus mapping file\n"
        "//Label\tBank\tElec\n"
        "0\t0\tA\t1\telec1-001\n"
        "1\t0\tA\t2\telec1-002\n"
        "0\t1\tB\t1\telec1-033\n"
        "\n",
        encoding="utf-8",
    )
    probe = utah.probe_from_cmp(path)
    assert probe["n_chan"] == 3
    # Channel order follows bank/elec: A1, A2, then B1 (bank offset 32).
    assert probe["labels"] == ["elec1-001", "elec1-002", "elec1-033"]
    assert probe["xc"].tolist() == [0.0, 400.0, 0.0]
    assert probe["yc"].tolist() == [0.0, 0.0, 400.0]
    assert "_placeholder" not in probe


def test_probe_from_cmp_rejects_an_unparseable_file(tmp_path):
    path = tmp_path / "bad.cmp"
    path.write_text("//only comments\n//nothing else\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no usable rows"):
        utah.probe_from_cmp(path)


def test_probe_summary_is_json_friendly():
    summary = common.probe_summary(utah.utah_grid_probe(96))
    assert summary["n_chan"] == 96
    assert summary["n_groups"] == 96
    assert summary["y_range_um"] == [0.0, 3600.0]
