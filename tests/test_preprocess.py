"""Preprocessing before Kilosort: the session block, the CatGT command, the record.

CatGT itself is a rig tool and is never run here. Its command is built by a pure
function and checked flag by flag; the SpikeInterface chain is run for real on a
small synthetic recording when SpikeInterface is installed.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from spikesorting import _config as cfg
from spikesorting._io import spikeglx
from spikesorting._preprocess import catgt as catgt_pre
from spikesorting._preprocess import provenance
from spikesorting._sync import catgt, extract

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(tmp_path, body: str, machine: str = "mac") -> cfg.SessionConfig:
    path = tmp_path / "s.yaml"
    path.write_text(
        f"blackrock_dir: '{tmp_path / 'br'}'\nneuropixels_dir: '{tmp_path / 'np'}'\n" + body,
        encoding="utf-8",
    )
    return cfg.load_session_config(path, machine, CONFIG_DIR)


_NPX = "neuropixels:\n  bin_file: '/a/x.bin'\n"
_UTAH = "blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '/b/z.ns6'\n"


# ---------------------------------------------------------------------------
# The session block
# ---------------------------------------------------------------------------


def test_each_system_starts_from_its_own_defaults(tmp_path):
    # Neuropixels follows the workshop pipeline; a Utah array gets the
    # SpikeInterface band and CAR, since CatGT cannot read a .ns6.
    session = _load(tmp_path, _NPX + _UTAH)

    npx = session.preprocess_for("neuropixels")
    assert npx.method == "catgt"
    assert (npx.highpass_hz, npx.lowpass_hz) == (300.0, 10000.0)
    assert npx.car_mode == "gblcar"
    assert npx.gfix is None                            # artifact removal is opt-in
    assert npx.phase_shift is True and npx.motion is None

    utah = session.preprocess_for("blackrock")
    assert utah.method == "spikeinterface"
    assert utah.gfix is None and utah.phase_shift is False and utah.motion is None


def test_a_stated_null_turns_a_step_off_rather_than_restoring_the_default(tmp_path):
    session = _load(
        tmp_path, _NPX + "  preprocess:\n    car_mode: null\n    gfix: null\n"
    )
    spec = session.preprocess_for("neuropixels")
    assert spec.car_mode is None and spec.gfix is None
    assert spec.method == "catgt"                      # the rest keeps its default


@pytest.mark.parametrize(
    "block, message",
    [
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    method: kilosort\n",
         "is not a method"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    method: catgtt\n",
         "is not one of"),
        ("blackrock:\n  spike_file: '/b/z.ns6'\n  preprocess:\n    method: catgt\n",
         "not a .ns6"),
        ("blackrock:\n  spike_file: '/b/z.ns6'\n  preprocess:\n    phase_shift: true\n",
         "Neuropixels only"),
        ("blackrock:\n  spike_file: '/b/z.ns6'\n  preprocess:\n    motion: dredge\n",
         "Neuropixels only"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    method: spikeinterface\n"
         "    gfix: [0.4, 0.1, 0.02]\n", "gfix is CatGT's"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    method: spikeinterface\n"
         "    gfix: null\n    car_mode: gbldmx\n", "demuxed"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    car_mode: loccar\n",
         "car_mode"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    gfix: [0.4, 0.1]\n",
         "three numbers"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    highpass_hz: 12000\n",
         "highpass_hz < lowpass_hz"),
        ("neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    motion: wobble\n",
         "correct_motion preset"),
    ],
)
def test_a_block_that_cannot_be_done_as_written_is_refused_at_load(tmp_path, block, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, block)


def test_motion_correction_with_kilosort_drift_correction_still_on_is_refused(tmp_path):
    # Drift corrected twice: once by DREDge, once by Kilosort's nblocks.
    with pytest.raises(ValueError, match="corrected twice"):
        _load(tmp_path, _NPX + "  preprocess:\n    motion: dredge\n")
    # ...nblocks: 0 is what lets it through.
    session = _load(
        tmp_path, _NPX + "  preprocess:\n    motion: dredge\n  kilosort:\n    nblocks: 0\n"
    )
    assert session.preprocess_for("neuropixels").motion == "dredge"


def test_motion_is_checked_for_every_probe_not_just_the_shared_block(tmp_path):
    # A by_probe: nblocks would otherwise turn drift correction back on for one.
    with pytest.raises(ValueError, match=r"nblocks \(imec1\) is 1"):
        _load(
            tmp_path,
            "neuropixels:\n"
            "  bin_files:\n    - '/a/r_g0_t0.imec0.ap.bin'\n    - '/a/r_g0_t0.imec1.ap.bin'\n"
            "  preprocess:\n    motion: dredge\n"
            "  kilosort:\n    nblocks: 0\n"
            "  by_probe:\n    1:\n      kilosort:\n        nblocks: 1\n",
        )


def test_a_typo_inside_the_preprocess_block_names_the_real_key(tmp_path):
    with pytest.raises(ValueError, match="did you mean: neuropixels.preprocess.car_mode"):
        _load(tmp_path, _NPX + "  preprocess:\n    carmode: gblcar\n")


def test_the_tracked_templates_state_every_preprocess_key():
    # Like the kilosort: blocks, every setting is written out so none runs on a
    # default nobody can see.
    keys = {f.name for f in fields(cfg.PreprocessSpec)}
    for name in (
        "session_template_utah_probe.yaml",
        "session_template_1probe.yaml",
        "session_template_2probes.yaml",
        "session_template_utah_only.yaml",
        "session_demo_1probe.yaml",
    ):
        data = cfg._read_yaml(CONFIG_DIR / name)
        for system in ("neuropixels", "blackrock"):
            block = data.get(system) or {}
            if system == "blackrock" and not block.get("spike_file"):
                continue                               # nothing sorted there
            if system in data:
                assert set(block.get("preprocess") or {}) == keys, (name, system)


# ---------------------------------------------------------------------------
# The CatGT command
# ---------------------------------------------------------------------------


def _layout(tmp_path) -> spikeglx.RunLayout:
    from conftest import spikeglx_run

    return spikeglx.run_layout(spikeglx_run(tmp_path / "npx")[0])


def test_the_catgt_command_is_the_workshops(tmp_path):
    layout = _layout(tmp_path)
    args = catgt_pre.build_preprocess_args(
        layout, cfg.NEUROPIXELS_PREPROCESS, tmp_path / "out"
    )
    assert not any(a.startswith("-gfix") for a in args)   # off unless asked for
    spec = cfg.replace(cfg.NEUROPIXELS_PREPROCESS, gfix=(0.40, 0.10, 0.02))
    args = catgt_pre.build_preprocess_args(layout, spec, tmp_path / "out")

    assert f"-dir={layout.directory}" in args
    assert "-run=run" in args and "-g=0" in args and "-t=0,0" in args
    assert "-prb=0" in args and "-prb_fld" in args
    assert "-ap" in args
    assert "-apfilter=butter,12,300,10000" in args
    assert "-gblcar" in args
    assert "-gfix=0.4,0.1,0.02" in args
    assert "-no_tshift" not in args                    # tshift is CatGT's default
    assert args[-3:] == ["-out_prb_fld", "-no_catgt_fld", f"-dest={tmp_path / 'out'}"]


def test_the_catgt_command_never_moves_the_first_sample_or_drops_a_channel(tmp_path):
    # Spike sample indices have to mean what the sync edges from the raw file
    # mean, so nothing may cut the start or the channel list.
    args = catgt_pre.build_preprocess_args(
        _layout(tmp_path), cfg.NEUROPIXELS_PREPROCESS, tmp_path / "out"
    )
    for forbidden in ("-startsecs", "-maxsecs", "-save", "-chnexcl"):
        assert not any(a.startswith(forbidden) for a in args), forbidden


def test_each_setting_maps_to_its_catgt_flag(tmp_path):
    layout = _layout(tmp_path)
    spec = cfg.replace(
        cfg.NEUROPIXELS_PREPROCESS,
        highpass_hz=250.0, lowpass_hz=9000.5, car_mode="gbldmx", gfix=None, phase_shift=False,
    )
    args = catgt_pre.build_preprocess_args(layout, spec, tmp_path / "out")
    assert "-apfilter=butter,12,250,9000.5" in args
    assert "-gbldmx" in args and "-gblcar" not in args
    assert not any(a.startswith("-gfix") for a in args)
    assert "-no_tshift" in args

    none = cfg.replace(spec, car_mode=None)
    args = catgt_pre.build_preprocess_args(layout, none, tmp_path / "out")
    assert not any(a in ("-gblcar", "-gbldmx") for a in args)


def test_the_sync_extractors_ride_along_in_the_same_pass(tmp_path):
    # Exactly the extractors extract_sync would run, so its edges can stand in.
    from conftest import spikeglx_run

    binary = spikeglx_run(tmp_path / "npx")[0]
    layout = spikeglx.run_layout(binary)
    obx = layout.gate_dir / "run_g0_t0.obx0.obx.bin"
    obx.write_bytes(b"\x00" * 16)
    session = _load(tmp_path, f"neuropixels:\n  bin_file: '{binary}'\n")

    extractors = extract.catgt_extractors(session, layout)
    args = catgt_pre.build_preprocess_args(
        layout, session.preprocess_for("neuropixels"), tmp_path / "out", extractors
    )

    assert "-xd=2,0,-1,6,500" in args
    assert any(a.startswith("-xa=1,0,1,") for a in args)
    assert "-ob" in args                               # the OneBox stream, for -xa
    assert "-obx=0" in args                            # ...and which OneBox
    assert args.count("-ap") == 1


def test_catgt_says_why_it_cannot_run(tmp_path):
    layout = _layout(tmp_path)
    assert catgt_pre.catgt_unavailable(layout, tmp_path) is None
    assert "no catgt_dir" in catgt_pre.catgt_unavailable(layout, None)
    assert "not named like SpikeGLX" in catgt_pre.catgt_unavailable(None, tmp_path)


def test_catgt_may_trim_a_ragged_end_but_nothing_more(tmp_path):
    assert catgt_pre.trimmed_seconds(30_000, 30_000, 30_000.0) == 0.0
    assert catgt_pre.trimmed_seconds(30_000, 29_970, 30_000.0) == pytest.approx(0.001)
    with pytest.raises(ValueError, match="more than"):
        catgt_pre.trimmed_seconds(30_000, 30_001, 30_000.0)
    with pytest.raises(ValueError, match="skipped data"):
        catgt_pre.trimmed_seconds(300_000, 30_000, 30_000.0)


def test_catgts_output_is_found_by_name_and_only_once(tmp_path):
    layout = _layout(tmp_path)
    with pytest.raises(FileNotFoundError, match="wrote 0 files"):
        catgt_pre.find_output(tmp_path / "out", layout)
    out = tmp_path / "out" / "run_g0_imec0" / "run_g0_tcat.imec0.ap.bin"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"")
    assert catgt_pre.find_output(tmp_path / "out", layout) == out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def test_a_record_is_stale_when_anything_that_made_the_binary_changed(tmp_path):
    source = tmp_path / "x.bin"
    source.write_bytes(b"\x00" * 8)
    expected = {
        "engine": "catgt",
        "settings": {"gfix": (0.4, 0.1, 0.02)},
        "source": provenance.source_fingerprint(source),
    }
    provenance.write(tmp_path, dict(expected, binary="out.bin"))
    recorded = provenance.read(tmp_path)

    assert provenance.stale_reasons(recorded, expected) == []   # a tuple reads back as a list
    assert provenance.stale_reasons(None, expected) == ["no preprocessed binary has been made yet"]
    for key, value in (
        ("engine", "spikeinterface"),
        ("settings", {"gfix": None}),
        ("source", {"path": str(source), "size": 9, "mtime_ns": 0}),
    ):
        reasons = provenance.stale_reasons(recorded, dict(expected, **{key: value}))
        assert len(reasons) == 1 and reasons[0].startswith(f"{key} changed")


def test_a_discarded_record_reads_as_none(tmp_path):
    provenance.write(tmp_path, {"engine": "catgt"})
    provenance.discard(tmp_path)
    assert provenance.read(tmp_path) is None
    (tmp_path / provenance.SIDECAR).write_text("{torn", encoding="utf-8")
    assert provenance.read(tmp_path) is None


# ---------------------------------------------------------------------------
# Edges from the preprocessing pass
# ---------------------------------------------------------------------------


def _catgt_edges(edge_dir: Path, times: np.ndarray) -> None:
    """Edge files laid out the way a CatGT pass writes them."""
    folder = edge_dir / "run_g0_imec0"
    folder.mkdir(parents=True, exist_ok=True)
    catgt.write_edge_file(folder / "run_g0_tcat.imec0.ap.xd_8_6_500.txt", times)


def test_extract_sync_reuses_the_edges_catgt_wrote_while_preprocessing(tmp_path):
    from conftest import spikeglx_run

    binary = spikeglx_run(tmp_path / "npx")[0]
    session = _load(tmp_path, f"neuropixels:\n  bin_file: '{binary}'\n")
    layout = spikeglx.run_layout(binary)
    directory = session.paths.preprocessed_for("neuropixels")
    edge_dir = directory / "catgt"
    _catgt_edges(edge_dir, 0.25 + np.arange(5.0))
    provenance.write(directory, {
        "engine": "catgt",
        "source": provenance.source_fingerprint(binary),
        "extractors": [e.to_flag() for e in extract.catgt_extractors(session, layout)],
        "edge_dir": str(edge_dir),
    })

    report = extract.extract_neuropixels_edges(session)

    assert report.edge_sets[extract.NPX_1HZ].source == "catgt"
    assert report.comparisons[extract.NPX_1HZ]["agree"]   # NumPy still checks them
    assert any("from the preprocessing pass" in note for note in report.notes)


def test_edges_from_a_pass_with_other_extractors_are_not_reused(tmp_path):
    from conftest import spikeglx_run

    binary = spikeglx_run(tmp_path / "npx")[0]
    session = _load(tmp_path, f"neuropixels:\n  bin_file: '{binary}'\n")
    directory = session.paths.preprocessed_for("neuropixels")
    _catgt_edges(directory / "catgt", 0.25 + np.arange(5.0))
    provenance.write(directory, {
        "engine": "catgt",
        "source": provenance.source_fingerprint(binary),
        "extractors": ["-xd=2,0,-1,5,500"],          # another bit
        "edge_dir": str(directory / "catgt"),
    })

    report = extract.extract_neuropixels_edges(session)

    assert report.edge_sets[extract.NPX_1HZ].source == "numpy"
    assert any("not reused" in note for note in report.notes)


# ---------------------------------------------------------------------------
# The SpikeInterface chain, run for real
# ---------------------------------------------------------------------------


def _recording(n_chan: int = 6, seconds: float = 1.0):
    si = pytest.importorskip("spikeinterface.core")
    rng = np.random.default_rng(0)
    traces = (rng.normal(0, 20, (int(30000 * seconds), n_chan))).astype(np.int16)
    traces += np.int16(300)                            # a DC offset the band-pass removes
    recording = si.NumpyRecording([traces], sampling_frequency=30000.0)
    recording.set_property("inter_sample_shift", np.linspace(0, 0.5, n_chan))
    return recording


def test_the_spikeinterface_chain_keeps_length_scale_and_type():
    from spikesorting._preprocess import si

    raw = _recording()
    out = si.build_chain(raw, cfg.replace(cfg.NEUROPIXELS_PREPROCESS, gfix=None))

    assert out.get_dtype() == np.int16
    assert out.get_num_samples() == raw.get_num_samples()   # sample for sample
    assert out.get_num_channels() == raw.get_num_channels()
    traces = out.get_traces()
    assert abs(float(np.median(traces))) < 5               # DC gone, CAR applied


def test_phase_shift_without_the_adc_offsets_says_what_to_turn_off():
    from spikesorting._preprocess import si

    pytest.importorskip("spikeinterface.core")
    import spikeinterface.core as sc

    bare = sc.NumpyRecording([np.zeros((3000, 4), dtype=np.int16)], sampling_frequency=30000.0)
    with pytest.raises(ValueError, match="phase_shift: false"):
        si.build_chain(bare, cfg.replace(cfg.NEUROPIXELS_PREPROCESS, gfix=None))


def test_a_bare_binary_is_read_without_its_sy_word_and_written_flat(tmp_path):
    from spikesorting._preprocess import si

    pytest.importorskip("spikeinterface.core")
    data = np.arange(3000 * 5, dtype=np.int16).reshape(3000, 5)
    source = tmp_path / "demo.ap.bin"
    source.write_bytes(data.tobytes())

    recording = si.read_neuropixels(source, 5, 30000.0, sy_index=4, has_meta=False)
    assert recording.get_num_channels() == 4               # the SY word left out

    out = si.write(recording, tmp_path / "out" / "demo.bin")
    written = np.fromfile(out, dtype=np.int16).reshape(-1, 4)
    assert np.array_equal(written, data[:, :4])


def test_only_the_named_binary_is_read_whatever_else_is_in_its_folder(tmp_path, monkeypatch):
    # neo's SpikeGLX reader walks a folder recursively, so a CatGT tcat output in
    # preprocessed/ below the run became a second segment with no LF band:
    # KeyError: (0, 'imec0.lf'). The binary is read by name instead, and the
    # probe and ADC offsets come from its own .meta.
    from spikesorting._preprocess import si

    pytest.importorskip("spikeinterface.core")
    probeinterface = pytest.importorskip("probeinterface")
    n_chan, n_ap = 9, 8
    data = np.arange(3000 * n_chan, dtype=np.int16).reshape(3000, n_chan)
    run = tmp_path / "run_g0" / "run_g0_imec0"
    run.mkdir(parents=True)
    source = run / "run_g0_t0.imec0.ap.bin"
    source.write_bytes(data.tobytes())
    source.with_suffix(".meta").write_text("nSavedChans=9\n", encoding="utf-8")
    (run / "run_g0_t0.imec0.lf.bin").write_bytes(b"")
    (run / "run_g0_t0.imec0.lf.meta").write_text("nSavedChans=9\n", encoding="utf-8")
    stray = run / "preprocessed" / "catgt" / "run_g0_imec0"
    stray.mkdir(parents=True)
    (stray / "run_g0_tcat.imec0.ap.bin").write_bytes(b"")
    (stray / "run_g0_tcat.imec0.ap.meta").write_text("nSavedChans=9\n", encoding="utf-8")

    probe = probeinterface.generate_linear_probe(num_elec=n_ap, ypitch=20)
    probe.set_device_channel_indices(np.arange(n_ap))
    read = []
    monkeypatch.setattr(
        probeinterface, "read_spikeglx", lambda path: read.append(Path(path)) or probe
    )
    monkeypatch.setattr(si, "_sample_shifts", lambda probe: np.linspace(0, 0.5, n_ap))

    recording = si.read_neuropixels(source, n_chan, 30000.0, sy_index=8, has_meta=True)

    assert read == [source.with_suffix(".meta")]          # this binary's own .meta
    assert recording.get_num_channels() == n_ap            # SY left out
    assert recording.has_probe()
    assert "inter_sample_shift" in recording.get_property_keys()
    assert np.array_equal(recording.get_traces(), data[:, :n_ap])
