"""The overview's waveform page: each unit's measured mean where it sits on the probe."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spikesorting._export import probe_layout

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


def test_a_unit_sits_at_its_median_else_its_peak_channel_else_nowhere():
    site_xy = np.array([[11.0, 0.0], [43.0, 20.0], [11.0, 40.0]])
    placed = probe_layout.place_units(
        {1: (12.0, 5.0)}, np.array([1, 2, 3, 4]),
        {2: 7, 3: 99, 4: None}, site_xy, np.array([5, 7, 9]))      # site rows 5, 7, 9
    assert placed == {1: (12.0, 5.0), 2: (43.0, 20.0)}       # 3: row not on the probe; 4: no peak
    assert probe_layout.place_units({}, np.array([2]), {2: 7}, None, None) == {}


def _probe_session(tmp_path):
    """Kilosort-style results: 8 sites in two columns, five units, three measured.

    Unit 4 has one spike, in the middle of the recording; its peak channel is
    site row 6.
    """
    from spikesorting._export.curated import PhyResults

    n_chan = 8
    site_xy = np.column_stack([np.tile([11.0, 43.0], 4), np.arange(n_chan) * 20.0])
    results = tmp_path / "kilosort4"
    results.mkdir()
    np.save(results / "channel_positions.npy", site_xy)
    np.save(results / "channel_map.npy", np.arange(n_chan))
    unit_ids = np.array([0, 1, 2, 3, 4])
    clusters = np.r_[np.repeat([0, 1, 2, 3], 5), 4, np.repeat(0, 5)]
    xy = np.r_[np.repeat(np.array([[11, 20], [43, 60], [11, 100], [43, 140]], float), 5, axis=0),
               [[99.0, 999.0]], np.repeat([[11.0, 20.0]], 5, axis=0)]
    np.save(results / "spike_positions.npy", xy)              # Kilosort's own, one per spike
    np.save(results / "spike_times.npy", np.arange(clusters.size))
    templates = np.zeros((5, 61, n_chan), np.float32)
    for unit, peak in enumerate([1, 3, 5, 7, 6]):
        templates[unit, 20, peak] = -1.0
    phy = PhyResults(results_dir=results, spike_samples=np.arange(clusters.size),
                     spike_clusters=clusters, fs=30000.0, templates=templates,
                     channel_map=np.arange(n_chan), spike_templates=clusters.copy(),
                     labels={0: "good", 1: "mua", 2: "good", 3: "noise-ish", 4: "mua"})
    t = np.arange(40) - 15
    mean = np.stack([-(u + 1) * 30 * np.exp(-(t / 3.0) ** 2) for u in (0, 1, 2, 4)], axis=1)
    measured = {"result": SimpleNamespace(mean=mean, unit_ids=np.array([0, 1, 2, 4])),
                "mean_units": "uV"}
    return phy, unit_ids, measured


def test_units_csv_and_the_waveform_pages_place_every_unit_the_same(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    from spikesorting import pipeline
    from spikesorting._export import final
    from spikesorting._plots import summary as plots

    phy, unit_ids, measured = _probe_session(tmp_path)
    # Two spikes sampled -- the first and last, both unit 0's -- so unit 0 sits at
    # its spikes' median and every other unit at its peak channel's site.
    monkeypatch.setattr(pipeline, "_OVERVIEW_MAX_SPIKES", 2)
    layout = pipeline._probe_layout(phy, unit_ids)
    assert layout["positions"][0] == (11.0, 20.0)             # its spikes' median
    assert layout["positions"][4] == (11.0, 120.0)            # its peak channel, site row 6
    assert layout["positions"][2] == (43.0, 100.0)            # peak channel 5, not its spikes' x

    drawn = {}
    original = plots.plot_probe_waveforms

    def spy(glyphs, site_xy, n_total, note, colour_by):
        drawn.setdefault("colour_by", []).append(colour_by)
        drawn.update(glyphs=glyphs, n_total=n_total, note=note)
        return original(glyphs, site_xy, n_total, note, colour_by=colour_by)

    monkeypatch.setattr(plots, "plot_probe_waveforms", spy)
    figs = pipeline._probe_waveform_pages(phy, unit_ids, layout, measured)
    assert len(figs) == 2 and drawn["colour_by"] == ["unit", "kind"]   # colourful, then SU/MU
    assert drawn["n_total"] == 5                              # unit 3: no measured mean
    assert {g["unit_id"]: g["kind"] for g in drawn["glyphs"]} == {0: "SU", 1: "MU", 2: "SU", 4: "MU"}
    assert "uV trough-to-peak" in drawn["note"]

    # units.csv records exactly the centre each waveform is drawn at.
    table = final.with_unit_positions(
        __import__("pandas").DataFrame({"unit_id": unit_ids, "channel": [1, 3, 5, 7, 6]}),
        layout["positions"])
    assert list(table.columns) == ["unit_id", "channel", "x_um", "y_um"]
    for glyph in drawn["glyphs"]:
        row = table.set_index("unit_id").loc[glyph["unit_id"]]
        assert np.median(glyph["x"]) == pytest.approx(row["x_um"])
        assert glyph["y"][0] == pytest.approx(row["y_um"])    # the trace starts at baseline


def test_channel_positions_csv_pairs_each_site_with_its_binary_row(tmp_path):
    from spikesorting._export import final
    import pandas as pd

    site_xy = np.array([[11.0, 40.0], [43.0, 0.0], [11.0, 20.0]])
    path = final.export_channel_positions(tmp_path, np.array([7, 2, 5]), site_xy)
    frame = pd.read_csv(path)
    assert path.name == "channel_positions.csv"
    assert frame.to_dict("list") == {"channel": [2, 5, 7], "x_um": [43.0, 11.0, 11.0],
                                     "y_um": [0.0, 20.0, 40.0]}


def test_overview_pdf_holds_every_page_and_replaces_overview_png(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from spikesorting import pipeline

    phy, unit_ids, measured = _probe_session(tmp_path)
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "overview.png").write_bytes(b"from an earlier export")

    pdf = pipeline._write_overview(phy, unit_ids, figures, measured)

    assert pdf == figures / "overview.pdf"
    assert not (figures / "overview.png").exists()          # the PDF's page 1 now
    data = pdf.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 3     # positions + 2 colourings
    # Without measured waveforms the PDF still has the spike-position page.
    pdf = pipeline._write_overview(phy, unit_ids, figures, None)
    data = pdf.read_bytes()
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 1
