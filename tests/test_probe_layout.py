"""The overview's waveform page: each unit's measured mean where it sits on the probe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spikesorting._export import probe_layout
from spikesorting._io import spikeglx


def _imro_meta(banks_by_chan, probe_type=1030):
    imro = f"({probe_type},{len(banks_by_chan)})" + "".join(
        f"({c} {b} 0 500 250 1)" for c, b in enumerate(banks_by_chan))
    return {"imDatPrb_type": str(probe_type), "~imroTbl": imro}


def test_the_bank_of_each_channel_comes_from_the_imro_table():
    banks = spikeglx.imro_banks(_imro_meta([1, 1, 0, 0, 2]))
    assert banks.tolist() == [1, 1, 0, 0, 2]
    # Another imro format names the bank elsewhere: no guess.
    assert spikeglx.imro_banks(_imro_meta([0, 0], probe_type=21)) is None
    assert spikeglx.imro_banks({"imDatPrb_type": "0"}) is None


def test_bank_spans_follow_the_recorded_sites_and_meet_between_banks():
    y = np.array([0, 20, 40, 60, 80, 100], float)
    spans = probe_layout.compute_bank_spans(y, np.array([0, 0, 0, 1, 1, -1]))
    assert spans == [(0, -10.0, 50.0), (1, 50.0, 90.0)]     # unknown bank skipped
    assert probe_layout.compute_bank_spans(y, np.full(6, -1)) == []


def test_a_unit_sits_at_the_median_of_its_spikes():
    xy = np.array([[10, 100], [12, 102], [11, 101], [500, 900], [40, 300]], float)
    clusters = np.array([1, 1, 1, 1, 2])
    positions = probe_layout.compute_unit_positions(xy, clusters, np.array([1, 2, 3]))
    assert positions[1] == (11.5, 101.5)     # one far-off spike does not move it
    assert positions[2] == (40.0, 300.0) and 3 not in positions


def test_glyphs_share_one_scale_and_sit_on_their_units():
    t = np.linspace(-1, 1, 21)
    waves = np.column_stack([-100 * np.exp(-(t / 0.2) ** 2), -50 * np.exp(-(t / 0.2) ** 2),
                             np.full(21, np.nan)])
    scale = probe_layout.compute_glyph_scale(waves[:, :2], height_um=40.0)
    assert scale == pytest.approx(40.0 / 75.0)            # median trough-to-peak -> 40 um
    glyphs = probe_layout.compute_waveform_glyphs(
        waves, np.array([7, 8, 9]), {7: (10.0, 500.0), 9: (20.0, 600.0)}, scale, width_um=30.0)
    assert [g["unit_id"] for g in glyphs] == [7]          # 8: no position; 9: no waveform
    g = glyphs[0]
    assert g["x"][0] == pytest.approx(-5.0) and g["x"][-1] == pytest.approx(25.0)
    assert g["y"].min() == pytest.approx(500.0 - 100 * scale)
    with pytest.raises(ValueError, match="one column per unit"):
        probe_layout.compute_waveform_glyphs(waves, np.array([1]), {}, scale)


def _probe_session(tmp_path):
    """Kilosort-style results on a 1030 probe whose channels span banks 0 and 1."""
    from spikesorting._export.curated import PhyResults

    n_chan = 8
    banks = [1, 1, 1, 1, 1, 0, 0, 0]
    site_xy = np.column_stack([np.tile([11.0, 43.0], 4), np.arange(n_chan) * 20.0])
    bin_file = tmp_path / "run_g0_t0.imec0.ap.bin"
    bin_file.write_bytes(b"")
    meta = _imro_meta(banks)
    bin_file.with_suffix(".meta").write_text(
        f"imDatPrb_type=1030\nnSavedChans={n_chan + 1}\nsnsApLfSy={n_chan},0,1\n"
        f"~imroTbl={meta['~imroTbl']}\n")
    results = tmp_path / "kilosort4"
    results.mkdir()
    np.save(results / "channel_positions.npy", site_xy)
    unit_ids = np.array([0, 1, 2, 3])
    clusters = np.repeat(unit_ids, 5)
    xy = np.repeat(np.array([[11, 20], [43, 60], [11, 100], [43, 140]], float), 5, axis=0)
    templates = np.zeros((4, 61, n_chan), np.float32)
    phy = PhyResults(results_dir=results, spike_samples=np.zeros(clusters.size, np.int64),
                     spike_clusters=clusters, fs=30000.0, templates=templates,
                     channel_map=np.arange(n_chan), spike_templates=clusters.copy(),
                     labels={0: "good", 1: "mua", 2: "good", 3: "noise-ish"})
    t = np.arange(40) - 15
    mean = np.stack([-(u + 1) * 30 * np.exp(-(t / 3.0) ** 2) for u in range(3)], axis=1)
    measured = {"result": SimpleNamespace(mean=mean, unit_ids=np.array([0, 1, 2])),
                "mean_units": "uV"}
    config = SimpleNamespace(neuropixels=SimpleNamespace(bin_file=bin_file))
    return phy, unit_ids, xy, clusters, measured, config


def test_the_waveform_page_draws_measured_units_by_kind_against_the_banks(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from spikesorting import pipeline
    from spikesorting._plots import summary as plots

    phy, unit_ids, xy, clusters, measured, config = _probe_session(tmp_path)
    drawn = {}
    original = plots.plot_probe_waveforms

    def spy(glyphs, site_xy, bank_spans, n_total, note, colour_by):
        drawn.setdefault("colour_by", []).append(colour_by)
        drawn.update(glyphs=glyphs, bank_spans=bank_spans, n_total=n_total, note=note)
        return original(glyphs, site_xy, bank_spans, n_total, note, colour_by=colour_by)

    monkeypatch.setattr(plots, "plot_probe_waveforms", spy)
    figs = pipeline._probe_waveform_pages(phy, unit_ids, xy, clusters, measured, config, "neuropixels")

    assert len(figs) == 2 and drawn["colour_by"] == ["unit", "kind"]   # colourful, then SU/MU
    assert drawn["n_total"] == 4                              # unit 3: no measured mean
    assert {g["unit_id"]: g["kind"] for g in drawn["glyphs"]} == {0: "SU", 1: "MU", 2: "SU"}
    assert [b for b, _, _ in drawn["bank_spans"]] == [1, 0]   # channels 0-4 bank 1, 5-7 bank 0
    assert drawn["bank_spans"][0][2] == drawn["bank_spans"][1][1] == pytest.approx(90.0)
    assert "uV trough-to-peak" in drawn["note"]
    # Each glyph sits on its unit's median position.
    assert np.median(drawn["glyphs"][1]["x"]) == pytest.approx(43.0)


def test_overview_pdf_holds_every_page_and_replaces_overview_png(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from spikesorting import pipeline
    from spikesorting._export import curated

    phy, unit_ids, xy, clusters, measured, config = _probe_session(tmp_path)
    monkeypatch.setattr(curated, "load_spike_positions", lambda results_dir, index: xy[index])
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "overview.png").write_bytes(b"from an earlier export")

    pdf = pipeline._write_overview(phy, unit_ids, figures, measured, config, "neuropixels")

    assert pdf == figures / "overview.pdf"
    assert not (figures / "overview.png").exists()          # the PDF's page 1 now
    data = pdf.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 3     # positions + 2 colourings
    # Without measured waveforms the PDF still has the spike-position page.
    pdf = pipeline._write_overview(phy, unit_ids, figures, None, config, "neuropixels")
    data = pdf.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 1
