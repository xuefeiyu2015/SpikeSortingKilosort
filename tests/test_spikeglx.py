"""SpikeGLX meta parsing and binary access, against synthetic files."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import square_wave

from spikesorting._io import spikeglx
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
    pieces = [block for _, block in spikeglx.iter_channel(info, 0, chunk_samples=777)]
    assert np.array_equal(np.concatenate(pieces), whole)


def test_iter_channel_rejects_zero_chunk(tmp_path):
    info = spikeglx.stream_info(write_stream(tmp_path))
    with pytest.raises(ValueError):
        list(spikeglx.iter_channel(info, 0, chunk_samples=0))


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


def test_export_lfp_writes_array_and_sidecar(tmp_path):
    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=4, n_samples=1000)
    out = spikeglx.export_lfp(path, tmp_path / "lfp", decimate=1)
    data = np.load(out)
    assert data.shape == (1000, 4)

    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["fs"] == 30000.0
    assert sidecar["n_chan"] == 4
    assert sidecar["n_samples"] == 1000


def test_export_lfp_decimation_keeps_the_grid_anchored(tmp_path):
    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=2, n_samples=1000)
    info = spikeglx.stream_info(path)
    reference = spikeglx.memmap_stream(info)[::10, :]

    out = spikeglx.export_lfp(path, tmp_path / "lfp10", decimate=10, chunk_samples=333)
    data = np.load(out)
    assert data.shape == (100, 2)
    assert np.array_equal(data, reference)

    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["fs"] == 3000.0


def test_export_lfp_rejects_bad_decimation(tmp_path):
    path = write_stream(tmp_path, name="run_g0_t0.imec0.lf.bin", n_chan=2, n_samples=100)
    with pytest.raises(ValueError):
        spikeglx.export_lfp(path, tmp_path / "lfp", decimate=0)
