"""Reading a results folder and writing the export bundle."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spikesorting._export import curated, final


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


def test_kilosorts_own_copy_of_cluster_group_is_not_a_curation(kilosort_results):
    # A fresh Kilosort4 sort copies cluster_KSLabel.tsv to cluster_group.tsv,
    # header and all, so the file exists with a KSLabel column and no `group`.
    # That used to raise KeyError: 'group' on every export before Phy had saved,
    # and must not be read as a curation either. Written byte for byte as
    # Kilosort writes it on Windows, \r\n included.
    ks_label = b"cluster_id\tKSLabel\r\n0\tmua\r\n1\tgood\r\n2\tgood\r\n"
    (kilosort_results / "cluster_KSLabel.tsv").write_bytes(ks_label)
    (kilosort_results / "cluster_group.tsv").write_bytes(ks_label)

    results = curated.load_phy_results(kilosort_results)

    assert results.curated is False
    assert results.labels == {0: "mua", 1: "good", 2: "good"}
    assert curated.select_units(results).tolist() == [0, 1, 2]


def test_select_units_keeps_good_and_mua(kilosort_results):
    results = curated.load_phy_results(kilosort_results)
    assert curated.select_units(results).tolist() == [0, 1, 2]


def test_select_units_drops_only_noise(kilosort_results):
    # Phy merged 0 and 1 into a new cluster 2 and nobody labelled it; 0 is noise.
    (kilosort_results / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n", encoding="utf-8"
    )
    (kilosort_results / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n0\tnoise\n", encoding="utf-8"
    )
    results = curated.load_phy_results(kilosort_results)
    assert curated.select_units(results).tolist() == [1, 2]


def test_select_units_without_labels_returns_everything(kilosort_results):
    (kilosort_results / "cluster_KSLabel.tsv").unlink()
    results = curated.load_phy_results(kilosort_results)
    assert curated.select_units(results).tolist() == [0, 1, 2]


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


def test_the_unit_table_holds_every_exported_unit(tmp_path, kilosort_results):
    # Unit 1 marked noise, so the table is a strict subset of the sorting.
    import pandas as pd

    (kilosort_results / "cluster_group.tsv").write_text(
        "cluster_id\tgroup\n1\tnoise\n", encoding="utf-8"
    )
    results = curated.load_phy_results(kilosort_results)
    unit_ids = curated.select_units(results)
    table = final.build_unit_table(results, unit_ids)
    path = final.export_unit_table(tmp_path / "out", table)

    assert path.name == "units.csv"
    written = pd.read_csv(path)
    assert written["unit_id"].tolist() == [0, 2]
    assert written["n_spikes"].tolist() == [500, 120]
    assert {"label", "channel", "firing_rate_hz", "isi_fraction"} <= set(written.columns)


def test_the_sorted_spikes_mat_carries_the_times_it_was_given(tmp_path, kilosort_results):
    # The .mat is the only copy of the spike times in the export folder, so the
    # aligned seconds -- not the sorter's samples / fs -- must be what it holds.
    import h5py

    results = curated.load_phy_results(kilosort_results)
    aligned = {int(u): results.times_for(u) + 5.0 for u in results.unit_ids}
    table = final.build_unit_table(results, results.unit_ids, aligned)
    path = final.export_sorted_spikes_mat(
        tmp_path, results, results.unit_ids, aligned, "blackrock", table
    )
    with h5py.File(path, "r") as handle:
        s = handle["sorted_spikes"]
        times = s["TimeStamps"][()].ravel()
        samples = s["spike_sample"][()].ravel()
        timebase = bytes(s["info/timebase"][()].ravel().astype(np.uint8)).decode()
    assert timebase == "blackrock"
    assert times.min() >= 5.0
    assert times.size == samples.size == results.spike_samples.size


def test_template_for_unit_uses_the_modal_template(kilosort_results):
    # Simulate a Phy merge: unit 0's spikes come mostly from template 9.
    spike_templates = np.load(kilosort_results / "spike_templates.npy")
    clusters = np.load(kilosort_results / "spike_clusters.npy")
    spike_templates[clusters == 0] = 9
    spike_templates[np.flatnonzero(clusters == 0)[:10]] = 0
    np.save(kilosort_results / "spike_templates.npy", spike_templates)

    results = curated.load_phy_results(kilosort_results)
    assert final.template_for_unit(results, 0) == 9


def test_the_sorted_spikes_mat_mirrors_the_online_spike_container(tmp_path, kilosort_results):
    """The field names and orientations jlab_loader reads off a .nev.

    ``loader.py:1399-1423`` builds its online container as TimeStamps (seconds on
    the NSP clock), Channel, Unit and a ``(nSpikes, nSamp)`` int16 Waveforms. A
    sorted product that matches segments into trials with the same code, so these
    are pinned rather than left to drift. Only the snippets' type differs: they
    are single-precision microvolts here, not the .nev's int16 counts.
    """
    import h5py
    import numpy as np

    from spikesorting._export.curated import load_phy_results
    from spikesorting._export.final import build_unit_table, export_sorted_spikes_mat

    phy = load_phy_results(kilosort_results)
    unit_ids = phy.unit_ids
    table = build_unit_table(phy, unit_ids)
    width, n_units = 60, unit_ids.size
    n_spikes = int(phy.spike_samples.size)
    waveforms = {
        "mean": np.zeros((width, n_units), dtype=np.float32),
        "std": np.zeros((width, n_units), dtype=np.float32),
        "unit_ids": unit_ids,
        "units": "microVolts",
        "window_ms": 2.0,
        # (nSamp, nSpikes) -- the HDF5 shape, which MATLAB reverses
        "snippets": np.zeros((width, n_spikes), dtype=np.float32),
    }

    path = export_sorted_spikes_mat(
        tmp_path, phy, unit_ids, None, "nsp", table, waveforms=waveforms
    )

    with h5py.File(path, "r") as handle:
        s = handle["sorted_spikes"]
        assert s.attrs["MATLAB_class"] == b"struct"
        # MATLAB shape is the reverse of the stored one.
        assert s["TimeStamps"].shape == (n_spikes, 1)      # MATLAB 1 x nSpikes
        assert s["Channel"].shape == (n_spikes, 1)
        assert s["Unit"].shape == (n_spikes, 1)
        assert s["Waveforms"].shape == (width, n_spikes)   # MATLAB nSpikes x nSamp
        # single-precision microvolts, not int16 counts: the snippets are
        # Kilosort-preprocessed and scaled, so counts would be lossy and wrong
        assert s["Waveforms"].dtype == np.float32
        assert s["MeanWaveform"].shape == (width, n_units)  # MATLAB nUnits x nSamp
        assert s["TimeRes"][()].ravel()[0] == phy.fs
        assert s["info"].attrs["MATLAB_class"] == b"struct"
        assert set(unit_ids.tolist()) == set(np.unique(s["Unit"][()]).astype(int).tolist())
        # times come out in one train, in order, as the online container is
        assert np.all(np.diff(s["TimeStamps"][()].ravel()) >= 0)


def test_the_sorted_spikes_mat_omits_waveforms_when_none_were_measured(tmp_path, kilosort_results):
    # An unreachable recording costs the waveform fields and nothing else.
    import h5py

    from spikesorting._export.curated import load_phy_results
    from spikesorting._export.final import build_unit_table, export_sorted_spikes_mat

    phy = load_phy_results(kilosort_results)
    table = build_unit_table(phy, phy.unit_ids)

    path = export_sorted_spikes_mat(
        tmp_path, phy, phy.unit_ids, None, "sorter", table, waveforms=None
    )

    with h5py.File(path, "r") as handle:
        s = handle["sorted_spikes"]
        assert "TimeStamps" in s and "Unit" in s
        assert "Waveforms" not in s and "MeanWaveform" not in s


def test_an_unlabelled_unit_header_names_no_class():
    from spikesorting._plots.summary import _unit_header

    assert "SU (good)" in _unit_header({"unit_id": 3, "label": "good", "unit_class": "SU"})
    header = _unit_header({"unit_id": 7, "label": "", "unit_class": "", "channel": 4})
    assert header.startswith("unit 7  |  ch 4")
    assert "unsorted" not in header and "()" not in header
