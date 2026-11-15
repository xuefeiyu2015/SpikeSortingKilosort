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


def test_motion_correction_turns_kilosorts_drift_correction_off(tmp_path):
    # The two are alternatives: a preset corrects drift before Kilosort, so
    # Kilosort sorts with nblocks: 0 whatever the block states, and the stated
    # value is the one used once motion is null again.
    session = _load(
        tmp_path, _NPX + "  preprocess:\n    motion: dredge\n  kilosort:\n    nblocks: 5\n")
    assert session.kilosort_for("neuropixels")["nblocks"] == 0
    assert "nblocks: 5 is not used" in session.nblocks_override("neuropixels")

    plain = _load(tmp_path, _NPX + "  kilosort:\n    nblocks: 5\n")
    assert plain.kilosort_for("neuropixels")["nblocks"] == 5
    assert plain.nblocks_override("neuropixels") is None

    # nblocks stated 0, or not stated with motion: nothing to report but still 0.
    stated_zero = _load(
        tmp_path, _NPX + "  preprocess:\n    motion: dredge\n  kilosort:\n    nblocks: 0\n")
    assert stated_zero.nblocks_override("neuropixels") is None
    unstated = _load(tmp_path, _NPX + "  preprocess:\n    motion: dredge\n")
    assert unstated.kilosort_for("neuropixels")["nblocks"] == 0
    assert unstated.nblocks_override("neuropixels") is not None   # Kilosort's default is 1


def test_motion_turns_drift_correction_off_for_every_probe(tmp_path):
    # A by_probe: nblocks is overridden too, so no probe is corrected twice.
    session = _load(
        tmp_path,
        "neuropixels:\n"
        "  bin_files:\n    - '/a/r_g0_t0.imec0.ap.bin'\n    - '/a/r_g0_t0.imec1.ap.bin'\n"
        "  preprocess:\n    motion: dredge\n"
        "  kilosort:\n    nblocks: 5\n"
        "  by_probe:\n    1:\n      kilosort:\n        nblocks: 1\n",
    )
    assert [stream.kilosort_for("neuropixels")["nblocks"]
            for _, stream in session.per_probe()] == [0, 0]


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
    assert any(a.startswith("-xa=1,0,0,") for a in args)
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
    out, motion_info = si.build_chain(raw, cfg.replace(cfg.NEUROPIXELS_PREPROCESS, gfix=None))

    assert motion_info is None                              # no motion asked for

    assert out.get_dtype() == np.int16
    assert out.get_num_samples() == raw.get_num_samples()   # sample for sample
    assert out.get_num_channels() == raw.get_num_channels()
    traces = out.get_traces()
    assert abs(float(np.median(traces))) < 5               # DC gone, CAR applied


def test_the_median_is_taken_over_the_reference_rows_only():
    # Two electrodes carrying one signal, two rows that are not electrodes. The
    # median over all four is half the signal; over the electrodes, all of it.
    pytest.importorskip("spikeinterface.core")
    import spikeinterface.core as sc

    from spikesorting._preprocess import si

    t = np.arange(30000) / 30000.0
    shared = 1000.0 * np.sin(2 * np.pi * 1000.0 * t)
    traces = np.zeros((t.size, 4), dtype=np.int16)
    traces[:, 0] = traces[:, 1] = shared.astype(np.int16)
    traces[:, 2:] = 5000                                    # an analog input's DC
    raw = sc.NumpyRecording([traces], sampling_frequency=30000.0)
    spec = cfg.replace(cfg.BLACKROCK_PREPROCESS, gfix=None)
    middle = slice(3000, 27000)                             # away from filter edges

    every_row = si.build_chain(raw, spec)[0].get_traces()[middle, 0].astype(float)
    electrodes = si.build_chain(raw, spec, reference_rows=[0, 1])[0].get_traces()[middle, 0]

    assert np.abs(every_row).max() > 300                    # half the signal left in
    assert np.abs(electrodes).max() <= 1                    # the shared signal removed
    with pytest.raises(ValueError, match="not rows"):
        si.build_chain(raw, spec, reference_rows=[0, 7])


def test_the_reference_rows_are_recorded_so_bad_channels_remake_the_binary(tmp_path):
    from spikesorting import pipeline

    ns6 = tmp_path / "b" / "z.ns6"
    ns6.parent.mkdir()
    ns6.write_bytes(b"")
    probe = tmp_path / "utah.json"
    probe.write_text(
        '{"chanMap": [0, 1, 2, 3, 5], "xc": [0, 0, 0, 0, 0], "yc": [0, 1, 2, 3, 4],'
        ' "kcoords": [0, 1, 2, 3, 4], "n_chan": 5}', encoding="utf-8")
    body = (f"blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '{ns6}'\n"
            f"  probe_file: '{probe}'\n  kilosort:\n    bad_channels: [{{}}]\n")

    def expected(bad):
        config = _load(tmp_path, body.replace("{}", bad))
        return pipeline._preprocess_expected(config, "blackrock")

    # row 4 is not on the probe (an analog input), row 2 is bad
    assert expected("2")["settings"]["reference_rows"] == [0, 1, 3, 5]
    assert provenance.stale_reasons(expected("2"), expected("3"))


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


@pytest.mark.parametrize("has_sy_word, kept", [(True, 31), (False, 32)])
def test_the_session_says_whether_a_bare_binary_ends_in_a_sy_word(tmp_path, has_sy_word, kept):
    # 32 rows, no .meta. Which are neural is the session's statement; unstated,
    # the stream refuses rather than guess (the guess dropped channel 32).
    from spikesorting import pipeline
    from spikesorting._preprocess import si

    pytest.importorskip("spikeinterface.core")
    source = tmp_path / "np" / "bare.imec0.ap.bin"
    source.parent.mkdir()
    source.write_bytes(np.zeros((300, 32), dtype=np.int16).tobytes())
    body = f"neuropixels:\n  bin_file: '{source}'\n  n_chan_bin: 32\n  sample_rate: 30000\n"

    with pytest.raises(ValueError, match="has_sy_word"):
        pipeline._neuropixels_stream(_load(tmp_path, body))

    config = _load(tmp_path, body + f"  has_sy_word: {str(has_sy_word).lower()}\n")
    info = pipeline._neuropixels_stream(config)
    recording = si.read_neuropixels(info.path, info.n_chan, info.fs, info.sy_index,
                                    has_meta=False)
    assert recording.get_num_channels() == kept


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


# ---------------------------------------------------------------------------
# A CatGT run is judged by what *it* wrote (CatGT mocked: never run here)
# ---------------------------------------------------------------------------


def _with_catgt(tmp_path, session):
    """The session on a machine that has CatGT -- a stand-in launcher only."""
    from dataclasses import replace

    tool = tmp_path / "CatGT"
    tool.mkdir(exist_ok=True)
    (tool / "CatGT.exe").write_text("")
    return replace(session, machine=replace(session.machine, catgt_dir=tool))


def _fake_catgt(monkeypatch, writes: bool):
    """Replace the CatGT launch; ``writes`` decides whether it leaves output."""
    calls = []

    def run(catgt_dir, args):
        calls.append(args)
        if not writes:
            return None                        # exit 0, nothing written
        dest = Path(next(a for a in args if a.startswith("-dest="))[len("-dest="):])
        src = Path(next(a for a in args if a.startswith("-dir="))[len("-dir="):])
        raw = next(src.rglob("*.imec0.ap.bin"))
        out = dest / "run_g0_imec0" / "run_g0_tcat.imec0.ap.bin"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(raw.read_bytes())
        out.with_suffix(".meta").write_text(raw.with_suffix(".meta").read_text())
        return None

    monkeypatch.setattr(catgt, "run_catgt", run)
    return calls


def _catgt_session(tmp_path, highpass=300, extra="", preprocess=""):
    from conftest import spikeglx_run

    binary = next((tmp_path / "npx").rglob("*.imec0.ap.bin"), None) \
        or spikeglx_run(tmp_path / "npx")[0]
    body = (f"neuropixels:\n  bin_file: '{binary}'\n"
            f"  preprocess:\n    method: catgt\n    highpass_hz: {highpass}\n"
            + preprocess + extra)
    return _with_catgt(tmp_path, _load(tmp_path, body))


def test_a_preprocessed_binary_still_matching_is_reused_without_catgt(tmp_path, monkeypatch):
    from spikesorting import pipeline

    calls = _fake_catgt(monkeypatch, writes=True)
    first = pipeline._binary_for(_catgt_session(tmp_path), "neuropixels")
    # A Kilosort setting is not a preprocessing one: the binary still stands.
    again = pipeline._binary_for(
        _catgt_session(tmp_path, extra="  kilosort:\n    nblocks: 0\n"), "neuropixels")

    assert len(calls) == 1
    assert again.path == first.path and again.path.exists()


def test_a_required_rerun_that_writes_nothing_does_not_pass_off_the_old_binary(
        tmp_path, monkeypatch):
    # Made at 300 Hz, then the session asks for 500 Hz; CatGT "succeeds" but
    # writes nothing. The 300 Hz file must not come back labelled 500 Hz.
    from spikesorting import pipeline

    _fake_catgt(monkeypatch, writes=True)
    old = pipeline._binary_for(_catgt_session(tmp_path, highpass=300), "neuropixels")
    assert old.path.exists()

    _fake_catgt(monkeypatch, writes=False)
    session = _catgt_session(tmp_path, highpass=500)
    with pytest.raises(FileNotFoundError, match="wrote 0 files"):
        pipeline._binary_for(session, "neuropixels")

    assert not old.path.exists()
    assert provenance.read(session.paths.preprocessed_for("neuropixels")) is None


def test_extraction_never_reads_a_previous_catgt_runs_edges(tmp_path, monkeypatch):
    # Stale CatGT edges at a phase the recording does not have; CatGT exits 0
    # writing nothing. Without clearing, CatGT's "output" would beat NumPy's.
    from conftest import spikeglx_run

    binary = spikeglx_run(tmp_path / "npx")[0]
    session = _with_catgt(tmp_path, _load(tmp_path, f"neuropixels:\n  bin_file: '{binary}'\n"))
    _catgt_edges(session.paths.sync_for("neuropixels") / "catgt", 0.7 + np.arange(5.0))
    calls = _fake_catgt(monkeypatch, writes=False)

    report = extract.extract_neuropixels_edges(session)

    assert len(calls) == 1
    edge_set = report.edge_sets[extract.NPX_1HZ]
    assert edge_set.source == "numpy"
    assert np.allclose(edge_set.times_s, 0.25 + np.arange(5), atol=1e-6)
    assert any("CatGT produced no edge file" in n for n in report.notes)


# ---------------------------------------------------------------------------
# bad_channels reach CatGT as -chnexcl, in acquisition channel IDs
# ---------------------------------------------------------------------------


def test_chnexcl_names_the_probe_and_its_excluded_channels(tmp_path):
    layout = _layout(tmp_path)
    spec = cfg.replace(cfg.NEUROPIXELS_PREPROCESS, gfix=(0.40, 0.10, 0.02))
    args = catgt_pre.build_preprocess_args(layout, spec, tmp_path / "out", [], [5, 17])
    assert "-chnexcl={0;5,17}" in args
    assert not any(a.startswith("-chnexcl") for a in
                   catgt_pre.build_preprocess_args(layout, spec, tmp_path / "out"))
    # Nothing to exclude from without a reference or gfix: not passed at all.
    bare = cfg.replace(cfg.NEUROPIXELS_PREPROCESS, car_mode=None, gfix=None)
    assert catgt_pre.excluded_channels(bare, [5]) == []
    assert catgt_pre.excluded_channels(spec, [17, 5, 5]) == [5, 17]


def test_binary_rows_are_acquisition_channels_when_every_channel_was_saved():
    meta = {"nSavedChans": "385", "snsApLfSy": "384,0,1", "snsSaveChanSubset": "all"}
    assert spikeglx.ap_acquisition_channels(meta, [0, 17, 383]) == [0, 17, 383]


def test_a_saved_subset_translates_rows_to_the_channels_acquired():
    # Rows 0-1 are channels 0-1, row 2 is channel 202; the SY word is row 184.
    meta = {"nSavedChans": "185", "snsApLfSy": "184,0,1",
            "snsSaveChanSubset": "0:1,202:383,768"}
    assert spikeglx.ap_acquisition_channels(meta, [2, 0, 183]) == [202, 0, 383]
    with pytest.raises(ValueError, match="not an AP row"):
        spikeglx.ap_acquisition_channels(meta, [184])


def test_the_catgt_exclusions_are_recorded_so_changing_them_remakes_the_binary(tmp_path):
    from spikesorting import pipeline

    def expected(bad, preprocess=""):
        session = _catgt_session(tmp_path, preprocess=preprocess,
                                 extra=f"  kilosort:\n    bad_channels: {bad}\n")
        return pipeline._preprocess_expected(session, "neuropixels")

    assert expected("[2]")["settings"]["catgt_excluded"] == [2]
    assert provenance.stale_reasons(expected("[2]"), expected("[3]"))
    assert not provenance.stale_reasons(expected("[2]"), expected("[2]"))
    # No reference and no gfix: CatGT's output does not depend on them, so a
    # changed list does not remake an identical binary.
    off = "    car_mode: null\n    gfix: null\n"
    assert "catgt_excluded" not in expected("[2]", off)["settings"]
    assert not provenance.stale_reasons(expected("[2]", off), expected("[3]", off))


def test_gfix_alone_still_gets_the_exclusions(tmp_path):
    from spikesorting import pipeline

    session = _catgt_session(
        tmp_path, preprocess="    car_mode: null\n    gfix: [0.40, 0.10, 0.02]\n",
        extra="  kilosort:\n    bad_channels: [4, 1]\n")
    assert pipeline._preprocess_expected(session, "neuropixels")[
        "settings"]["catgt_excluded"] == [1, 4]


def test_the_catgt_command_carries_the_exclusions(tmp_path, monkeypatch):
    from spikesorting import pipeline

    calls = _fake_catgt(monkeypatch, writes=True)
    session = _catgt_session(tmp_path, extra="  kilosort:\n    bad_channels: [2]\n")
    pipeline._binary_for(session, "neuropixels")
    assert "-chnexcl={0;2}" in calls[0]
    record = provenance.read(session.paths.preprocessed_for("neuropixels"))
    assert record["settings"]["catgt_excluded"] == [2]


def test_a_catgt_binary_made_before_exclusions_were_recorded_is_still_reused(tmp_path):
    # No bad_channels: the record a binary was made with before this change
    # matches what the session asks for now, so the rig does not remake it.
    from spikesorting import pipeline

    expected = pipeline._preprocess_expected(_catgt_session(tmp_path), "neuropixels")
    assert "catgt_excluded" not in expected["settings"]
    with_key = _catgt_session(tmp_path, extra="  kilosort:\n    bad_channels: [2]\n")
    assert provenance.stale_reasons(
        expected, pipeline._preprocess_expected(with_key, "neuropixels"))


# ---------------------------------------------------------------------------
# CatGT and motion correction are cached as two stages
# ---------------------------------------------------------------------------


_MOTION = ""                       # nblocks needs no change: motion turns it off


class _FakeRecording:
    """Just enough of a recording for the record (``_record_of``)."""

    def __init__(self, n_chan, n_samples, fs=30000.0):
        self.n_chan, self.n_samples, self.fs = n_chan, n_samples, fs

    def get_num_segments(self):
        return 1

    def get_num_frames(self, segment_index=0):
        return self.n_samples

    def get_num_channels(self):
        return self.n_chan

    def get_sampling_frequency(self):
        return self.fs


def _synthetic_drift(n=4000, duration_s=5.0, shift_um=60.0):
    """Peaks on three units whose depth ramps by ``shift_um`` over the session."""
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, duration_s, n))
    home = rng.choice([200.0, 800.0, 1500.0], n)
    moved = shift_um * times / duration_s
    bins = np.linspace(0, duration_s, 6)
    return {
        "times_s": times,
        "depth_raw_um": home + moved,
        "depth_corrected_um": home + rng.normal(0, 1, n),
        "amplitude": -rng.uniform(50, 150, n),
        "temporal_bins_s": bins,
        "spatial_bins_um": np.array([400.0, 1200.0]),
        "displacement_um": np.stack([shift_um * bins / duration_s] * 2, axis=1),
        "n_peaks": 10 * n,
    }


def _fake_motion(monkeypatch):
    """Replace SpikeInterface's motion step; record which presets ran."""
    from spikesorting._preprocess import si

    ran = []

    def motion_chain(recording, preset, jobs=None):
        ran.append(preset)
        out = spikeglx.stream_info(recording)
        info = {"run_times": {"detect_and_localize": 2.0, "estimate_motion": 1.0},
                "peaks": [], "peak_locations": [], "motion": object()}
        return _FakeRecording(out.n_chan - 1, out.n_samples, out.fs), info

    def write(recording, path, jobs=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * (recording.n_chan * recording.n_samples * 2))
        return path

    monkeypatch.setattr(si, "read_neuropixels", lambda path, *a, **k: Path(path))
    monkeypatch.setattr(si, "motion_chain", motion_chain)
    monkeypatch.setattr(si, "write", write)
    monkeypatch.setattr(si, "peak_drift", lambda rec, info, n_max: _synthetic_drift())
    monkeypatch.setattr(si, "check_medicine", lambda: None)
    return ran


def test_changing_only_the_motion_preset_does_not_run_catgt_again(tmp_path, monkeypatch):
    # CatGT's output does not depend on motion: switching DREDge on, to
    # another preset, or off must not cost a full CatGT pass over the raw file.
    from spikesorting import pipeline

    calls = _fake_catgt(monkeypatch, writes=True)
    ran = _fake_motion(monkeypatch)
    plain = pipeline._binary_for(_catgt_session(tmp_path), "neuropixels")
    dredge = pipeline._binary_for(
        _catgt_session(tmp_path, preprocess="    motion: dredge\n", extra=_MOTION), "neuropixels")
    medicine = pipeline._binary_for(
        _catgt_session(tmp_path, preprocess="    motion: medicine\n", extra=_MOTION), "neuropixels")
    session = _catgt_session(tmp_path)
    off = pipeline._binary_for(session, "neuropixels")

    assert len(calls) == 1                                  # CatGT ran once
    assert ran == ["dredge", "medicine"]
    assert dredge.path.name.endswith(".motion.bin")
    assert dredge.path.parent.name == "motion_dredge"
    assert medicine.path.parent.name == "motion_medicine"
    assert dredge.path.exists() and medicine.path.exists()  # neither overwrote the other
    assert off.path == plain.path and off.path.exists()
    directory = session.paths.preprocessed_for("neuropixels")
    record = provenance.read(directory)
    assert record["edge_dir"] and record["extractors"]     # extract_sync still finds them

    # A setting CatGT does depend on still runs it.
    pipeline._binary_for(_catgt_session(tmp_path, highpass=500), "neuropixels")
    assert len(calls) == 2


def test_catgt_output_a_pre_split_record_vouches_for_is_adopted(tmp_path, monkeypatch):
    # A binary preprocessed before the stages were split has no catgt/ record;
    # its preprocess.json was written only after CatGT finished, so the first
    # motion change after the update does not run CatGT again either.
    from spikesorting import pipeline

    calls = _fake_catgt(monkeypatch, writes=True)
    _fake_motion(monkeypatch)
    session = _catgt_session(tmp_path, preprocess="    motion: dredge\n", extra=_MOTION)
    pipeline._binary_for(session, "neuropixels")
    directory = session.paths.preprocessed_for("neuropixels")
    (directory / "catgt" / provenance.SIDECAR).unlink()    # as before this change

    pipeline._binary_for(_catgt_session(tmp_path), "neuropixels")

    assert len(calls) == 1
    stage = provenance.read(directory / "catgt")
    assert stage is not None
    assert stage["timing"]["seconds"] is None              # shared with DREDge: unknown


def test_catgt_output_is_not_adopted_from_a_record_with_other_filters(tmp_path, monkeypatch):
    from spikesorting import pipeline

    calls = _fake_catgt(monkeypatch, writes=True)
    session = _catgt_session(tmp_path)
    pipeline._binary_for(session, "neuropixels")
    (session.paths.preprocessed_for("neuropixels") / "catgt" / provenance.SIDECAR).unlink()

    pipeline._binary_for(_catgt_session(tmp_path, highpass=500), "neuropixels")
    assert len(calls) == 2


def test_the_drift_record_and_figures_are_kept_and_follow_the_sorting(tmp_path, monkeypatch):
    from spikesorting import pipeline
    from spikesorting._preprocess import motion as motion_record

    _fake_catgt(monkeypatch, writes=True)
    _fake_motion(monkeypatch)
    session = _catgt_session(tmp_path, preprocess="    motion: dredge\n", extra=_MOTION)
    binary = pipeline._binary_for(session, "neuropixels")
    folder = binary.path.parent
    assert folder.name == "motion_dredge"

    drift = motion_record.load_drift(folder)
    assert drift["preset"] == "dredge" and drift["n_peaks"] == 40000
    for name in pipeline.DRIFT_FIGURES:
        assert (folder / name).stat().st_size > 0

    sorted_dir = tmp_path / "sorted"
    pipeline._copy_drift_figures(binary.path.parent, sorted_dir)
    assert all((sorted_dir / n).exists() for n in pipeline.DRIFT_FIGURES)

    # Motion turned off: the binary has no figures, so the sorting keeps none.
    plain = pipeline._binary_for(_catgt_session(tmp_path), "neuropixels")
    pipeline._copy_drift_figures(plain.path.parent, sorted_dir)
    assert not any((sorted_dir / n).exists() for n in pipeline.DRIFT_FIGURES)


def test_a_failed_figure_does_not_fail_the_preprocessing(tmp_path, monkeypatch):
    from spikesorting import pipeline
    from spikesorting._preprocess import si

    _fake_catgt(monkeypatch, writes=True)
    _fake_motion(monkeypatch)

    def broken(*a, **k):
        raise RuntimeError("no display")

    monkeypatch.setattr(si, "peak_drift", broken)
    session = _catgt_session(tmp_path, preprocess="    motion: dredge\n", extra=_MOTION)
    binary = pipeline._binary_for(session, "neuropixels")

    assert binary.path.exists()
    record = provenance.read(session.paths.preprocessed_for("neuropixels"))
    assert record["stages"][-1] == {"step": "drift record + figures",
                                    "seconds": record["stages"][-1]["seconds"],
                                    "note": "failed"}


def test_each_preprocessing_stage_is_its_own_row_in_the_timing(tmp_path, monkeypatch):
    from datetime import datetime

    from spikesorting import pipeline

    _fake_catgt(monkeypatch, writes=True)
    _fake_motion(monkeypatch)
    pipeline._binary_for(_catgt_session(tmp_path), "neuropixels")
    session = _catgt_session(tmp_path, preprocess="    motion: dredge\n", extra=_MOTION)
    pipeline._binary_for(session, "neuropixels")

    timing = pipeline._sort_timing(
        session, "neuropixels", None, datetime.now().astimezone(),
        loading={}, reused=True, reuse_check_s=0.0, ops={}, kilosort_wall_s=0.0,
        postprocessing={},
    )
    rows = {r["step"]: r for r in timing["phases"]["preprocessing"]["steps"]}
    catgt_row = rows["CatGT: filter, reference, gfix, sync edges"]
    assert catgt_row["note"].startswith("reused: made ")   # from the first run
    assert rows["dredge: detect and localize"]["seconds"] == 2.0
    assert rows["dredge: estimate motion"]["seconds"] == 1.0
    assert "dredge: noise levels & set-up" in rows
    assert "dredge: interpolate motion + write" in rows
    assert "drift record + figures" in rows
    assert rows["dredge: estimate motion"]["note"].startswith("made ")


def test_medicine_is_checked_before_any_work(monkeypatch):
    import sys

    from spikesorting._preprocess import si

    monkeypatch.setitem(sys.modules, "medicine", None)      # import fails
    with pytest.raises(ImportError, match="pip install medicine-neuro"):
        si.check_medicine()


# ---------------------------------------------------------------------------
# The drift record itself (pure)
# ---------------------------------------------------------------------------


def test_the_peaks_kept_are_a_deterministic_stride_over_all_of_them():
    from spikesorting._preprocess import motion as motion_record

    assert np.array_equal(motion_record.subsample_indices(5, 10), np.arange(5))
    kept = motion_record.subsample_indices(1_000_000, 1000)
    assert kept.size == 1000 and kept[0] == 0 and kept[-1] == 999_999
    assert np.ptp(np.diff(kept)) <= 1                       # evenly spread
    assert np.array_equal(kept, motion_record.subsample_indices(1_000_000, 1000))


def test_the_raster_shows_drift_before_and_none_after():
    from spikesorting._preprocess import motion as motion_record

    drift = _synthetic_drift(n=20000, shift_um=60.0)
    t_edges, d_edges = motion_record.drift_raster_edges(
        drift["times_s"], [drift["depth_raw_um"], drift["depth_corrected_um"]],
        time_bin_s=1.0, depth_bin_um=5.0)
    before = motion_record.compute_drift_raster(
        drift["times_s"], drift["depth_raw_um"], drift["amplitude"], t_edges, d_edges)
    after = motion_record.compute_drift_raster(
        drift["times_s"], drift["depth_corrected_um"], drift["amplitude"], t_edges, d_edges)

    centres = (d_edges[:-1] + d_edges[1:]) / 2
    near = (centres > 150) & (centres < 300)                # the 200 um unit
    first = centres[near][np.argmax(before[near, 0])]
    last = centres[near][np.argmax(before[near, -1])]
    assert last - first > 40                                # moved by ~60 um
    first = centres[near][np.argmax(after[near, 0])]
    last = centres[near][np.argmax(after[near, -1])]
    assert abs(last - first) <= 5                           # held still
    assert np.isclose(before.sum(), np.abs(drift["amplitude"]).sum())


def test_the_drift_record_reads_back_and_a_missing_one_is_none(tmp_path):
    from spikesorting._preprocess import motion as motion_record

    drift = dict(_synthetic_drift(), preset="dredge")
    motion_record.save_drift(tmp_path, drift)
    back = motion_record.load_drift(tmp_path)
    assert back["preset"] == "dredge" and back["n_peaks"] == drift["n_peaks"]
    assert np.array_equal(back["depth_raw_um"], drift["depth_raw_um"])
    assert motion_record.load_drift(tmp_path / "nowhere") is None


def test_peak_drift_moves_the_same_peaks_by_the_estimated_motion():
    # "After" is the peaks the motion was estimated from, shifted by it --
    # nothing is detected twice.
    pytest.importorskip("spikeinterface.sortingcomponents.motion")
    import spikeinterface.core as sc
    from spikeinterface.core.motion import Motion

    from spikesorting._preprocess import si

    fs, n = 30000.0, 300
    recording = sc.NumpyRecording([np.zeros((int(10 * fs), 4), dtype=np.int16)],
                                  sampling_frequency=fs)
    recording.set_channel_locations(np.c_[np.zeros(4), np.linspace(440.0, 560.0, 4)])
    peaks = np.zeros(n, dtype=[("sample_index", "int64"), ("channel_index", "int64"),
                               ("amplitude", "float64"), ("segment_index", "int64")])
    peaks["sample_index"] = np.linspace(0, 10 * fs - 1, n).astype(int)
    peaks["amplitude"] = -100.0
    locations = np.zeros(n, dtype=[("x", "float64"), ("y", "float64")])
    locations["y"] = 500.0 + 3.0 * peaks["sample_index"] / fs      # 3 um/s drift
    bins = np.arange(0.5, 10, 1.0)
    motion = Motion(3.0 * bins[:, None], bins, np.array([500.0]))

    drift = si.peak_drift(recording, {"motion": motion, "peaks": peaks,
                                      "peak_locations": locations}, n_max=100)

    assert len(drift["times_s"]) == 100 and drift["n_peaks"] == n
    inside = (drift["times_s"] > 0.5) & (drift["times_s"] < 9.5)
    assert np.allclose(drift["depth_corrected_um"][inside], 500.0, atol=1e-6)
    assert np.ptp(drift["depth_raw_um"]) > 25
    assert drift["displacement_um"].shape == (len(bins), 1)


def test_motion_correction_keeps_every_channel_whatever_the_preset(monkeypatch):
    # medicine's preset leaves interpolate_motion at "remove_channels", which
    # drops sites drift carries off the probe; the channel map read from the
    # .meta would then index the wrong rows.
    pytest.importorskip("spikeinterface.preprocessing")
    import spikeinterface.core as sc
    import spikeinterface.preprocessing as spre

    from spikesorting._preprocess import si

    recording = sc.NumpyRecording([np.zeros((3000, 4), dtype=np.int16)],
                                  sampling_frequency=30000.0)
    probeinterface = pytest.importorskip("probeinterface")
    probe = probeinterface.generate_linear_probe(num_elec=4, ypitch=20)
    probe.set_device_channel_indices(np.arange(4))
    recording = recording.set_probe(probe)
    asked = {}

    def correct_motion(rec, preset, output_motion_info, interpolate_motion_kwargs, **jobs):
        asked.update(interpolate_motion_kwargs)
        return rec, {"run_times": {}}

    monkeypatch.setattr(spre, "correct_motion", correct_motion)
    monkeypatch.setattr(si, "check_medicine", lambda: None)
    out, _ = si.motion_chain(recording, "medicine")
    assert asked == {"border_mode": "force_extrapolate"}
    assert out.get_num_channels() == 4

    monkeypatch.setattr(spre, "correct_motion", lambda rec, **k: (
        rec.select_channels(rec.channel_ids[:3]), {"run_times": {}}))
    with pytest.raises(RuntimeError, match="returned 3 channels from 4"):
        si.motion_chain(recording, "medicine")


def test_switching_back_to_a_preset_reuses_its_binary_and_record(tmp_path, monkeypatch):
    # DREDge, then MEDiCINe, then DREDge again: the DREDge binary is still in
    # motion_dredge/, so it is not made a third time -- and its record comes
    # back verbatim, so a sorting made from it still recognises its binary.
    from spikesorting import pipeline

    _fake_catgt(monkeypatch, writes=True)
    ran = _fake_motion(monkeypatch)
    dredge = _catgt_session(tmp_path, preprocess="    motion: dredge\n")
    medicine = _catgt_session(tmp_path, preprocess="    motion: medicine\n")
    directory = dredge.paths.preprocessed_for("neuropixels")

    first = pipeline._binary_for(dredge, "neuropixels")
    at_sort = provenance.read(directory)                     # what run_info.json keeps
    pipeline._binary_for(medicine, "neuropixels")
    again = pipeline._binary_for(dredge, "neuropixels")

    assert ran == ["dredge", "medicine"]
    assert again.path == first.path
    assert provenance.read(directory) == at_sort
    run_info = {"preprocessing": at_sort}
    assert pipeline._binary_changed_since_sort(dredge, "neuropixels", run_info,
                                               again.path) is None


def test_a_presets_binary_made_with_other_settings_is_remade_in_its_folder(tmp_path, monkeypatch):
    from spikesorting import pipeline

    _fake_catgt(monkeypatch, writes=True)
    ran = _fake_motion(monkeypatch)
    pipeline._binary_for(_catgt_session(tmp_path, preprocess="    motion: dredge\n"),
                         "neuropixels")
    remade = pipeline._binary_for(
        _catgt_session(tmp_path, highpass=500, preprocess="    motion: dredge\n"), "neuropixels")

    assert ran == ["dredge", "dredge"]
    assert remade.path.parent.name == "motion_dredge"
    folders = [p.name for p in remade.path.parents[1].iterdir() if p.name.startswith("motion_")]
    assert folders == ["motion_dredge"]                      # one folder per preset


def test_a_binary_made_before_records_sat_beside_it_is_kept_for_reuse(tmp_path, monkeypatch):
    # The rig's DREDge binary was written beside preprocess.json, before
    # motion_<preset>/ existed. Switching to MEDiCINe and back must not make
    # it again: preprocess.json vouches for it, so it gets a record in place.
    from spikesorting import pipeline

    _fake_catgt(monkeypatch, writes=True)
    ran = _fake_motion(monkeypatch)
    dredge = _catgt_session(tmp_path, preprocess="    motion: dredge\n")
    made = pipeline._binary_for(dredge, "neuropixels")
    directory = dredge.paths.preprocessed_for("neuropixels")
    legacy = directory / made.path.name                      # the old layout
    made.path.rename(legacy)
    for record in directory.rglob("*.bin.json"):
        record.unlink()
    shutil_rmtree = __import__("shutil").rmtree
    shutil_rmtree(made.path.parent)
    old = provenance.read(directory)
    old["binary"] = str(legacy)
    provenance.write(directory, old)

    pipeline._binary_for(_catgt_session(tmp_path, preprocess="    motion: medicine\n"),
                         "neuropixels")
    back = pipeline._binary_for(dredge, "neuropixels")

    assert ran == ["dredge", "medicine"]
    assert back.path == legacy and legacy.exists()
