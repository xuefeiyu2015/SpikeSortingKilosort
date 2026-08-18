"""SpikeGLX meta parsing and binary access, against synthetic files."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from conftest import square_wave

from spikesorting._io import reachable, spikeglx
from spikesorting._sync import edges

GEOM_MAP = "(NP1000,1,0,70)(0:27:0:1)(0:59:0:1)(0:27:20:1)(0:59:20:0)"


def make_meta(n_chan: int) -> str:
    """A meta consistent with the binary being written.

    ``fileSizeBytes=0`` is deliberate: real metas can lag a truncated file, and
    the reader is supposed to trust the bytes on disk instead.
    """
    return (
        f"nSavedChans={n_chan}\n"
        "imSampRate=30000\n"
        "typeThis=imec\n"
        "fileSizeBytes=0\n"
        f"snsApLfSy={n_chan - 1},0,1\n"
        "imAiRangeMax=0.6\n"
        "imMaxInt=512\n"
        f"~snsGeomMap={GEOM_MAP}\n"
    )


def write_stream(tmp_path, name="run_g0_t0.imec0.ap.bin", n_chan=385, n_samples=6000, meta=True):
    """Write a synthetic SpikeGLX binary whose SY word carries a square wave."""
    data = np.zeros((n_samples, n_chan), dtype=np.int16)
    data[:, :-1] = 7  # arbitrary neural payload
    wave = square_wave(n_samples / 30000.0, 30000.0, period_s=0.1, phase_s=0.01)
    data[: wave.size, -1] = (wave.astype(np.uint16) << 6).astype(np.int16)

    bin_path = tmp_path / name
    bin_path.write_bytes(data.tobytes())
    if meta:
        spikeglx.meta_path_for(bin_path).write_text(make_meta(n_chan), encoding="utf-8")
    return bin_path


def test_read_meta_parses_keys_including_tilde(tmp_path):
    path = write_stream(tmp_path)
    meta = spikeglx.read_meta(path)
    assert meta["nSavedChans"] == "385"
    assert meta["imSampRate"] == "30000"
    assert meta["~snsGeomMap"].startswith("(NP1000")


def test_read_meta_accepts_the_bin_path(tmp_path):
    path = write_stream(tmp_path)
    assert spikeglx.read_meta(path) == spikeglx.read_meta(spikeglx.meta_path_for(path))


def test_read_meta_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        spikeglx.read_meta(tmp_path / "nope.meta")


def test_stream_info_derives_shape_from_the_file_on_disk(tmp_path):
    path = write_stream(tmp_path, n_samples=6000)
    info = spikeglx.stream_info(path)
    assert info.n_chan == 385
    assert info.fs == 30000.0
    assert info.n_samples == 6000  # not the stale fileSizeBytes=0 in the meta
    assert info.sy_index == 384
    assert info.duration_s == pytest.approx(0.2)


def test_stream_info_without_meta_requires_explicit_params(tmp_path):
    path = write_stream(tmp_path, name="bare.bin", meta=False)
    with pytest.raises(FileNotFoundError, match="pass n_chan and fs"):
        spikeglx.stream_info(path)

    info = spikeglx.stream_info(path, n_chan=385, fs=30000.0)
    assert info.n_chan == 385 and info.fs == 30000.0
    assert info.sy_index == 384


def test_read_channel_negative_index_is_the_sy_word(tmp_path):
    path = write_stream(tmp_path)
    info = spikeglx.stream_info(path)
    assert np.array_equal(
        spikeglx.read_channel(info, -1), spikeglx.read_channel(info, info.sy_index)
    )


def test_read_channel_out_of_range_raises(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path))
    with pytest.raises(IndexError):
        spikeglx.read_channel(info, 999)


def test_sy_word_bit_6_recovers_the_square_wave(tmp_path):
    path = write_stream(tmp_path)
    info = spikeglx.stream_info(path)
    words = spikeglx.read_sy_word(info)
    bit = edges.digital_bit_signal(words, 6)
    times = edges.detect_pulse_times(bit, 0.5, info.fs, duration_ms=50)
    assert np.allclose(times, 0.01 + np.arange(times.size) * 0.1)
    assert times.size == 2


def test_iter_channel_covers_every_sample(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path, n_samples=6000))
    whole = spikeglx.read_channel(info, 0)
    pieces = [block for _, block in spikeglx.iter_channel(info, 0, chunk_bytes=777)]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_iter_channel_rejects_zero_chunk(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path))
    with pytest.raises(ValueError):
        list(spikeglx.iter_channel(info, 0, chunk_bytes=0))


def test_reads_stream_the_file_rather_than_paging_a_memmap(tmp_path):
    # The SY word is interleaved with 384 neural channels, so reading it pulls
    # every byte of the file whatever we do. Doing that through a memmap column
    # slice makes it scattered 4 KB page faults, which over SMB is minutes for a
    # 1 GB file -- and uninterruptible, since one slice is a single C call.
    info = spikeglx.stream_info(write_stream(tmp_path, n_samples=6000))

    for channel in (0, info.n_chan - 1):
        expected = np.ascontiguousarray(spikeglx.memmap_stream(info)[:, channel])
        assert np.array_equal(spikeglx.read_channel(info, channel), expected)
        # ...and in pieces, at a chunk size that does not divide the file evenly.
        pieces = [block for _, block in spikeglx.iter_channel(info, channel, chunk_bytes=4321)]
        assert np.array_equal(np.concatenate(pieces), expected)


def test_iter_channel_yields_the_start_sample_of_each_block(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path, n_samples=6000))

    starts, sizes = zip(*[(s, b.size) for s, b in spikeglx.iter_channel(info, 0, chunk_bytes=8192)])

    assert starts[0] == 0
    assert list(starts[1:]) == list(np.cumsum(sizes)[:-1])
    assert sum(sizes) == info.n_samples


def test_a_probe_reports_the_read_speed_before_the_whole_file_is_touched(tmp_path):
    # The point: know in a second what an 8-minute read is going to cost, instead
    # of finding out eight minutes in.
    path = write_stream(tmp_path, n_samples=6000)
    probe = spikeglx.probe_read(path, sample_bytes=4096)

    assert probe.sample_bytes == 4096
    assert probe.total_bytes == path.stat().st_size
    assert probe.bytes_per_s > 0
    assert probe.estimated_seconds() > 0
    assert "MB/s" in probe.summary()


def test_the_probe_stops_at_its_time_budget(tmp_path):
    # The check has to stay cheap on the slow share it exists to warn about: it
    # reports how slow things are, it does not wait to measure it precisely.
    path = write_stream(tmp_path, n_samples=6000)
    probe = spikeglx.probe_read(path, sample_bytes=1 << 30, time_budget_s=0.0)

    assert 0 < probe.sample_bytes <= path.stat().st_size
    assert probe.bytes_per_s > 0


def test_an_unreachable_file_fails_at_the_probe_not_hours_later(tmp_path):
    # An unmounted share is the case this exists for: stat first, so the failure
    # is immediate and named rather than a stall inside the read loop.
    with pytest.raises(OSError):
        spikeglx.probe_read(tmp_path / "not_mounted" / "run_g0_t0.imec0.ap.bin")


def test_check_reachable_answers_like_a_direct_probe(tmp_path):
    path = write_stream(tmp_path, n_samples=6000)

    probe = reachable.check_reachable(path, sample_bytes=4096)

    assert probe.total_bytes == path.stat().st_size
    assert probe.sample_bytes == 4096


def test_check_reachable_gives_up_rather_than_waiting_out_a_stalled_mount(tmp_path, monkeypatch):
    # The case a bare probe cannot bound: a stat on a hung share blocks in the
    # kernel, so the check has to abandon it rather than wait for it.
    monkeypatch.setattr(reachable, "probe_read", lambda *a, **k: time.sleep(30))

    started = time.perf_counter()
    with pytest.raises(TimeoutError) as excinfo:
        reachable.check_reachable(tmp_path / "share" / "run.ap.bin", timeout_s=0.2)
    waited = time.perf_counter() - started

    assert waited < 5.0, "waited for the stall instead of the deadline"
    assert "run.ap.bin" in str(excinfo.value)


def test_the_stall_error_is_an_oserror(tmp_path, monkeypatch):
    # The notebooks catch OSError around these verbs. If this stops being one,
    # they go back to showing a traceback instead of "recording not reachable".
    monkeypatch.setattr(reachable, "probe_read", lambda *a, **k: time.sleep(30))

    with pytest.raises(OSError):
        reachable.check_reachable(tmp_path / "x.bin", timeout_s=0.1)


def test_a_directory_is_probed_by_looking_inside_it(tmp_path):
    # A SpikeGLX run is named by its directory, and find_run_files globs before
    # anything has a filename to check -- so the mount has to be answerable from
    # the directory alone, which cannot be probed by reading bytes from it.
    write_stream(tmp_path, n_samples=600)

    probe = reachable.check_reachable(tmp_path)

    assert probe.total_bytes == 0          # nothing to read, so nothing to estimate
    assert "answered" in probe.summary()


def test_a_real_failure_is_raised_as_itself_not_as_a_timeout(tmp_path):
    # An absent file answers immediately; saying "did not respond" would be a lie.
    with pytest.raises(FileNotFoundError):
        reachable.check_reachable(tmp_path / "nope.bin", timeout_s=5.0)


def test_the_probe_is_still_reachable_through_spikeglx(tmp_path):
    # It moved to _io/reachable.py; spikeglx re-exports it, and this pins that so
    # the re-export is not dropped as dead code later.
    assert spikeglx.probe_read is reachable.probe_read
    assert spikeglx.ReadProbe is reachable.ReadProbe


def test_the_estimate_scales_the_sample_to_the_whole_file():
    # Pure arithmetic, so it is checkable without pretending to know a disk speed.
    probe = spikeglx.ReadProbe(
        path=Path("x.bin"), total_bytes=1_000_000_000, sample_bytes=1_000_000, seconds=0.5
    )

    assert probe.bytes_per_s == pytest.approx(2e6)
    assert probe.estimated_seconds() == pytest.approx(500.0)
    assert probe.estimated_seconds(2_000_000) == pytest.approx(1.0)


def test_raw_to_volts_uses_the_meta_scaling(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path))
    volts = spikeglx.raw_to_volts(info, np.array([512, 256, 0]))
    assert np.allclose(volts, [0.6, 0.3, 0.0])


def test_parse_geom_map_skips_the_header_entry(tmp_path):
    meta = spikeglx.read_meta(write_stream(tmp_path))
    geom = spikeglx.parse_geom_map(meta)
    assert geom.shape == (4, 4)
    assert geom[:, 1].tolist() == [27.0, 59.0, 27.0, 59.0]  # x
    assert geom[:, 2].tolist() == [0.0, 0.0, 20.0, 20.0]  # z
    assert geom[-1, 3] == 0.0  # last site flagged unused


def test_parse_geom_map_returns_none_when_absent():
    assert spikeglx.parse_geom_map({"nSavedChans": "385"}) is None


def test_find_run_files_locates_ap_lf_and_obx(tmp_path):
    gate = tmp_path / "run_g0"
    probe = gate / "run_g0_imec0"
    probe.mkdir(parents=True)
    (probe / "run_g0_t0.imec0.ap.bin").write_bytes(b"")
    (probe / "run_g0_t0.imec0.lf.bin").write_bytes(b"")
    (gate / "run_g0_t0.obx0.obx.bin").write_bytes(b"")

    found = spikeglx.find_run_files(tmp_path, "run", gate=0, trigger=0, probe=0)
    assert found["ap"].name == "run_g0_t0.imec0.ap.bin"
    assert found["lf"].name == "run_g0_t0.imec0.lf.bin"
    assert found["obx"].name == "run_g0_t0.obx0.obx.bin"


def test_find_run_files_also_finds_catgt_output(tmp_path):
    gate = tmp_path / "run_g0"
    gate.mkdir(parents=True)
    (gate / "run_g0_tcat.imec0.ap.bin").write_bytes(b"")
    found = spikeglx.find_run_files(tmp_path, "run", trigger="cat")
    assert found["ap"].name == "run_g0_tcat.imec0.ap.bin"


def test_find_run_files_missing_run_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        spikeglx.find_run_files(tmp_path / "absent", "run")


def test_export_lfp_writes_a_matlab_struct(tmp_path):
    # The lab reads these from MATLAB, so the product is a v7.3 .mat struct
    # rather than a .npy plus a sidecar to reassemble by hand.
    import h5py

    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=4, n_samples=1000)
    out = spikeglx.export_lfp(path, tmp_path / "lfp", decimate=1)

    assert out.suffix == ".mat"
    assert not list(out.parent.glob("*.npy"))       # replaced, not written beside
    with h5py.File(out, "r") as handle:
        lfp = handle["lfp"]
        # MATLAB reverses dimensions, so nChan x nSamples on disk is (n, chan).
        assert lfp["data"].shape == (1000, 4)
        assert lfp["data"].dtype == np.int16
        assert lfp["data"].attrs["MATLAB_class"] == b"int16"
        assert lfp["fs"][()].ravel()[0] == 30000.0
        assert lfp["n_chan"][()].ravel()[0] == 4
        assert lfp["n_samples"][()].ravel()[0] == 1000
        # The gain that would turn these into microvolts is not in the meta this
        # module reads, and a fabricated 1.0 would scale silently wrong.
        assert np.isnan(lfp["uv_per_digit"][()]).all()


def test_export_lfp_decimation_keeps_the_grid_anchored(tmp_path):
    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=2, n_samples=1000)
    info = spikeglx.stream_info(path)
    reference = spikeglx.memmap_stream(info)[::10, :]

    import h5py

    out = spikeglx.export_lfp(path, tmp_path / "lfp10", decimate=10, chunk_samples=333)
    with h5py.File(out, "r") as handle:
        data = handle["lfp/data"][()]
        assert data.shape == (100, 2)
        assert np.array_equal(data, reference)
        assert handle["lfp/fs"][()].ravel()[0] == 3000.0
        assert handle["lfp/decimate"][()].ravel()[0] == 10


def test_export_lfp_rejects_bad_decimation(tmp_path):
    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=2, n_samples=100)
    with pytest.raises(ValueError):
        spikeglx.export_lfp(path, tmp_path / "lfp", decimate=0)
