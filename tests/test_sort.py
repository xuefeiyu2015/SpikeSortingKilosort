"""Publishing sorted results out of the SSD cache.

Only the cache/publish plumbing is covered here: it is plain ``shutil`` and runs
without Kilosort, torch or SpikeInterface, which is what makes it testable on a
laptop. The sorters themselves are exercised on the rig.
"""

from __future__ import annotations

from pathlib import Path

from spikesorting import _sort as sort


def _blackrock_body(work_dir: Path) -> str:
    """Mimic ``spikeinterface.run_sorter(folder=work_dir / "kilosort4")``.

    SpikeInterface puts Kilosort's own output under ``sorter_output/`` and keeps
    its bookkeeping beside it, so the Phy-format files land two levels down.
    """
    results = work_dir / "kilosort4" / "sorter_output"
    results.mkdir(parents=True)
    (results / "spike_times.npy").write_bytes(b"spikes")
    (results / "params.py").write_text("sample_rate = 30000.0\n", encoding="utf-8")
    (work_dir / "kilosort4" / "spikeinterface_log.json").write_text("{}", encoding="utf-8")
    return "sorted"


def test_publish_from_lifts_the_named_subdirectory_to_the_top(tmp_path):
    # pipeline.py reads <sorted_br>/spike_times.npy, and Phy expects the same
    # layout, so the published directory has to be the sorter's results dir.
    final = tmp_path / "out" / "blackrock" / "kilosort4"
    cache = tmp_path / "cache"
    cache.mkdir()

    result, _ = sort._run_in_cache(
        final, cache, "tag", _blackrock_body, publish_from="kilosort4/sorter_output"
    )

    assert result == "sorted"
    assert (final / "spike_times.npy").read_bytes() == b"spikes"
    assert (final / "params.py").exists()
    assert not (final / "kilosort4").exists()


def test_publish_from_produces_the_same_tree_without_a_cache(tmp_path):
    # The no-cache fallback sorts in place. It must not produce a different
    # layout, or a machine with no cache_dir would silently break every consumer.
    final = tmp_path / "out" / "blackrock" / "kilosort4"

    sort._run_in_cache(final, None, "tag", _blackrock_body, publish_from="kilosort4/sorter_output")

    assert (final / "spike_times.npy").read_bytes() == b"spikes"
    assert not (final / "kilosort4").exists()


def test_without_publish_from_the_whole_work_dir_is_copied(tmp_path):
    # The Neuropixels route: run_kilosort writes straight into the work dir.
    final = tmp_path / "out" / "neuropixels" / "kilosort4"
    cache = tmp_path / "cache"
    cache.mkdir()

    def body(work_dir: Path) -> str:
        (work_dir / "spike_times.npy").write_bytes(b"spikes")
        return "sorted"

    sort._run_in_cache(final, cache, "tag", body)

    assert (final / "spike_times.npy").exists()


def test_the_temp_binary_is_left_behind_in_the_cache(tmp_path):
    # temp.dat is a whitened copy of the whole recording; copying it to a network
    # share is both slow and rude. keep_dat=True is the opt-out.
    final = tmp_path / "out"
    cache = tmp_path / "cache"
    cache.mkdir()

    def body(work_dir: Path) -> None:
        (work_dir / "spike_times.npy").write_bytes(b"spikes")
        (work_dir / "temp.dat").write_bytes(b"huge")

    sort._run_in_cache(final, cache, "tag", body)

    assert (final / "spike_times.npy").exists()
    assert not (final / "temp.dat").exists()


class _FakeRecording:
    def get_sampling_frequency(self):
        return 30000.0


def _fake_spikeinterface(monkeypatch, seen):
    """Stand in for spikeinterface.sorters, recording what run_sorter was told.

    Both the package and the submodule are replaced: ``import a.b as x`` resolves
    ``b`` as an attribute of ``a``, so patching only ``sys.modules["a.b"]`` would
    still hand back the real one wherever SpikeInterface happens to be installed
    -- and this test has to behave the same on a laptop that has none of it.
    """
    import sys
    import types

    sorters = types.ModuleType("spikeinterface.sorters")

    def run_sorter(name, recording, folder, remove_existing_folder=True, **params):
        seen["folder"] = Path(folder)
        seen["params"] = params
        results = Path(folder) / "sorter_output"
        results.mkdir(parents=True)
        (results / "spike_times.npy").write_bytes(b"spikes")
        return types.SimpleNamespace(unit_ids=[1, 2, 3])

    sorters.run_sorter = run_sorter
    package = types.ModuleType("spikeinterface")
    package.sorters = sorters
    monkeypatch.setitem(sys.modules, "spikeinterface", package)
    monkeypatch.setitem(sys.modules, "spikeinterface.sorters", sorters)
    return sorters


def test_each_probe_is_sorted_into_its_own_directory(tmp_path, monkeypatch):
    # Publishing both probes to <npx>/neuropixels/kilosort4 would leave whichever
    # sorted last, with nothing to say the other ever ran.
    from spikesorting import _config as cfg

    session_path = tmp_path / "s.yaml"
    session_path.write_text(
        f"session: s\nneuropixels_dir: '{tmp_path / 'np'}'\n"
        "neuropixels:\n  run_dir: '/npx'\n  run_name: run\n  probes: [0, 1]\n",
        encoding="utf-8",
    )
    config = cfg.load_session_config(
        session_path, "mac", Path(__file__).resolve().parents[1] / "configs"
    )

    seen: dict = {}
    _fake_spikeinterface(monkeypatch, seen)

    result = sort.sort_recording(config, "neuropixels", _FakeRecording(), probe_index=1)

    assert result.results_dir == config.paths.sorted_for("neuropixels", 1)
    assert result.results_dir.parts[-2:] == ("imec1", "kilosort4")
    assert (result.results_dir / "spike_times.npy").exists()
    # The cache is shared by every sort on the machine, so the tag carries the
    # stream too -- two probes of one session are two different runs.
    assert "imec1" in str(seen["folder"])

    import json

    info = json.loads((result.results_dir / "run_info.json").read_text())
    assert info["probe_index"] == 1
    assert info["stream"] == "imec1"


def test_the_cache_is_deleted_afterwards(tmp_path):
    final = tmp_path / "out"
    cache = tmp_path / "cache"
    cache.mkdir()

    def body(work_dir: Path) -> None:
        (work_dir / "spike_times.npy").write_bytes(b"spikes")

    _, work_dir = sort._run_in_cache(final, cache, "tag", body)

    assert not work_dir.exists()
