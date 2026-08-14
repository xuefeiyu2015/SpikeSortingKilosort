"""Reading a results folder and writing the export bundle."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spikesorting.export import curated, final


def test_parse_params_py_reads_types(kilosort_results):
    params = curated.parse_params_py(kilosort_results / "params.py")
    assert params["sample_rate"] == 30000.0
    assert params["n_channels_dat"] == 8
    assert params["dtype"] == "int16"
    assert params["hp_filtered"] is False


def test_parse_params_py_missing_returns_empty(tmp_path):
    assert curated.parse_params_py(tmp_path / "nope.py") == {}


def test_load_phy_results_reads_arrays_and_labels(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert results.fs == 30000.0
    assert results.unit_ids.tolist() == [0, 1, 2]
    assert results.labels == {0: "good", 1: "mua", 2: "good"}
    assert results.curated is False  # only KSLabel present, no human curation
    assert results.spike_samples.size == 920


def test_curated_labels_override_kilosort_labels(kilosort_results):
    (kilosort_results / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n0\tnoise\n1\tgood\n", encoding="utf-8"
    )
    results = curated.load_phy_results(kilosort_results)
    assert results.curated is True
    assert results.labels[0] == "noise"  # human overrode "good"
    assert results.labels[1] == "good"
    assert results.labels[2] == "good"  # untouched, falls back to KSLabel


def test_select_units_filters_by_label(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert curated.select_units(results, ("good",)).tolist() == [0, 2]
    assert curated.select_units(results, ("good", "mua")).tolist() == [0, 1, 2]
    assert curated.select_units(results, ("noise",)).tolist() == []


def test_select_units_without_labels_returns_everything(kilosort_results):
    (kilosort_results / "cluster_KSLabel.tsv").unlink()
    results = curated.load_phy_results(kilosort_results)
    assert curated.select_units(results, ("good",)).tolist() == [0, 1, 2]


def test_times_are_converted_from_samples_to_seconds(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert np.allclose(results.spike_times_s, results.spike_samples / 30000.0)


def test_load_phy_results_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        curated.load_phy_results(tmp_path / "absent")


def test_best_channel_comes_from_the_templates(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert results.best_channels().tolist() == [2, 5, 7]


def test_mean_template_waveform_picks_the_peak_channel(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    for unit_id, expected_channel in [(0, 2), (1, 5), (2, 7)]:
        waveform, channel = final.mean_template_waveform(results, unit_id)
        assert channel == expected_channel
        assert waveform.size == 61
        assert waveform.min() < 0


def test_unit_table_has_one_row_per_unit(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    table = final.build_unit_table(results)
    assert len(table) == 3
    assert table["unit_id"].tolist() == [0, 1, 2]
    assert table["channel"].tolist() == [2, 5, 7]
    assert (table["n_spikes"] > 0).all()
    assert "peak_to_trough_ms" in table.columns


def test_unit_table_uses_supplied_aligned_times(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    shifted = {int(u): results.times_for(u) + 1000.0 for u in results.unit_ids}
    table = final.build_unit_table(results, spike_times_s=shifted)
    assert (table["first_spike_s"] > 1000.0).all()


def test_recording_window_follows_the_timebase(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert final.recording_window(results, results.unit_ids, None) == (0.0, results.duration_s)

    shifted = {int(u): results.times_for(u) + 1000.0 for u in results.unit_ids}
    start, duration = final.recording_window(results, results.unit_ids, shifted)
    assert start > 1000.0
    assert duration < results.duration_s + 1.0


def test_aligned_times_do_not_destroy_time_windowed_metrics(kilosort_results):
    """Blackrock time does not start at zero.

    A window assumed to start at t=0 would put every aligned spike outside it and
    report presence_ratio 0.0 for perfectly healthy units.
    """
    results = curated.load_phy_results(kilosort_results)
    sorter_table = final.build_unit_table(results)

    shifted = {int(u): results.times_for(u) + 1000.0 for u in results.unit_ids}
    aligned_table = final.build_unit_table(results, spike_times_s=shifted)

    assert (aligned_table["presence_ratio"] > 0.0).all()
    assert np.allclose(
        aligned_table["presence_ratio"], sorter_table["presence_ratio"], atol=0.1
    )


def test_export_bundle_is_self_consistent(tmp_path, kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    unit_ids = curated.select_units(results, ("good",))
    paths = final.export_units(tmp_path / "out", results, unit_ids=unit_ids, timebase="sorter")

    assert all(p.exists() for p in paths.values())

    times = np.load(paths["spike_times"])
    clusters = np.load(paths["spike_clusters"])
    assert times.size == clusters.size
    assert np.all(np.diff(times) >= 0), "exported spike times must be sorted"
    assert set(np.unique(clusters).tolist()) == {0, 2}

    info = json.loads(paths["info"].read_text())
    assert info["timebase"] == "sorter"
    assert info["n_units"] == 2
    assert info["n_spikes"] == times.size

    waveforms = np.load(paths["mean_waveforms"])
    assert waveforms.shape == (2, 61)

    with np.load(paths["isi_histograms"]) as bundle:
        assert bundle["counts"].shape[0] == 2
        assert bundle["unit_ids"].tolist() == [0, 2]


def test_export_records_the_blackrock_timebase(tmp_path, kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    aligned = {int(u): results.times_for(u) + 5.0 for u in results.unit_ids}
    paths = final.export_units(
        tmp_path / "out", results, spike_times_s=aligned, timebase="blackrock"
    )
    info = json.loads(paths["info"].read_text())
    assert info["timebase"] == "blackrock"
    assert np.load(paths["spike_times"]).min() >= 5.0


def test_template_for_unit_uses_the_modal_template(kilosort_results):
    # Simulate a Phy merge: unit 0's spikes come mostly from template 9.
    spike_templates = np.load(kilosort_results / "spike_templates.npy")
    clusters = np.load(kilosort_results / "spike_clusters.npy")
    spike_templates[clusters == 0] = 9
    spike_templates[np.flatnonzero(clusters == 0)[:10]] = 0
    np.save(kilosort_results / "spike_templates.npy", spike_templates)

    results = curated.load_phy_results(kilosort_results)
    assert final.template_for_unit(results, 0) == 9
