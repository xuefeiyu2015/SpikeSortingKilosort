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


def test_the_cache_is_deleted_afterwards(tmp_path):
    final = tmp_path / "out"
    cache = tmp_path / "cache"
    cache.mkdir()

    def body(work_dir: Path) -> None:
        (work_dir / "spike_times.npy").write_bytes(b"spikes")

    _, work_dir = sort._run_in_cache(final, cache, "tag", body)

    assert not work_dir.exists()
