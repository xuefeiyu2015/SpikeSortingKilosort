"""The verb surface: one function per thing you do, per system.

Only the parts that need no sorter are covered -- probe resolution, the guards
that read the session config, the skip reasons the runners report. The
SpikeInterface and Kilosort calls are exercised on the rig; nothing here can run
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import spikesorting as ss
from spikesorting import _config as cfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _session(tmp_path, body=""):
    """A session declaring both systems, unless `body` declares them itself.

    Declaring paths is what says a system recorded, so a fixture that wants both
    systems present has to name files for both -- there is no flag any more.
    """
    lines = [
        "session: s",
        f"blackrock_dir: '{tmp_path / 'br'}'",
        f"neuropixels_dir: '{tmp_path / 'np'}'",
    ]
    if "neuropixels:" not in body:
        lines.append("neuropixels:\n  bin_file: '/a/x.bin'")
    if "blackrock:" not in body:
        lines.append("blackrock:\n  sync_file: '/b/y.ns5'")
    path = tmp_path / "s.yaml"
    path.write_text("\n".join(lines) + "\n" + body, encoding="utf-8")
    return cfg.load_session_config(path, "mac", CONFIG_DIR)


def _probe_json(path: Path, n: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "chanMap": list(range(n)),
                "xc": [0.0] * n,
                "yc": [float(i * 40) for i in range(n)],
                "kcoords": [0] * n,
                "n_chan": n,
            }
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# The surface
# ---------------------------------------------------------------------------


def test_the_package_exports_the_workflow_and_nothing_else():
    # Opening `import spikesorting` should read as the pipeline. A `step_*` or a
    # second surface reappearing here is the regression this guards.
    assert not [n for n in ss.__all__ if n.startswith("step_")]
    for verb in (
        "load_session_config",
        "setup_probe",
        "sort_with_kilosort",
        "extract_sync",
        "extract_lfp",
        "time_remapping",
        "validate_remapping",
        "export_results",
    ):
        assert callable(getattr(ss, verb)), verb
    # Sorting is one verb now: Kilosort reads the binary itself, so there is no
    # recording to load and pass along.
    assert not hasattr(ss, "load_spike_continuous")


def test_every_per_system_verb_rejects_an_unknown_system(tmp_path):
    session = _session(tmp_path)
    for verb in (ss.setup_probe, ss.extract_sync, ss.extract_lfp, ss.sort_with_kilosort):
        with pytest.raises(ValueError, match="system must be one of"):
            verb(session, "utah")


def test_importing_the_package_pulls_in_no_heavy_dependency():
    # config, sync, alignment and export must load on a machine with no sorter.
    import sys

    heavy = {"kilosort", "torch", "spikeinterface"}
    assert not (heavy & set(sys.modules)), sorted(heavy & set(sys.modules))


# ---------------------------------------------------------------------------
# The channel map
# ---------------------------------------------------------------------------


def _meta_run(tmp_path, n_chan=4, with_geom=True):
    """A minimal SpikeGLX run whose .meta may or may not carry ~snsGeomMap."""
    import numpy as np

    run = tmp_path / "np" / "r_g0" / "r_g0_imec0"
    run.mkdir(parents=True, exist_ok=True)
    data = np.zeros((100, n_chan), dtype=np.int16)
    (run / "r_g0_t0.imec0.ap.bin").write_bytes(data.tobytes())
    meta = [
        f"nSavedChans={n_chan}",
        "imSampRate=30000",
        f"fileSizeBytes={data.nbytes}",
        "typeThis=imec",
        "imAiRangeMax=0.6",
        "imMaxInt=512",
    ]
    if with_geom:
        # (shank, x, y, used) per channel -- the format probe_from_meta parses.
        meta.append("~snsGeomMap=(NP1000,1,0,70)" + "".join(
            f"(0:{11 + 16 * (i % 2)}:{20 * (i // 2)}:1)" for i in range(n_chan)
        ))
    (run / "r_g0_t0.imec0.ap.meta").write_text("\n".join(meta) + "\n", encoding="utf-8")
    return tmp_path / "np"


def test_the_runs_own_meta_wins_over_a_probe_file(tmp_path):
    # The .meta records which sites were actually active for *this* run, which a
    # file built from another run cannot know. So it is not an override target.
    _meta_run(tmp_path)
    other = _probe_json(tmp_path / "probes" / "other.json", n=99)
    session = _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{tmp_path / 'np'}'\n  run_name: r\n"
        f"  probe_file: '{other}'\n",
    )

    probe = ss.setup_probe(session, "neuropixels")

    assert probe["n_chan"] == 4          # from the .meta
    assert probe["n_chan"] != 99         # not the probe_file


def test_probe_file_is_used_when_the_run_has_no_meta(tmp_path):
    # A bare binary, which is the demo's situation.
    binary = tmp_path / "np" / "bare.imec0.ap.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00" * 800)
    named = _probe_json(tmp_path / "probes" / "np1.json", n=4)
    session = _session(
        tmp_path,
        f"neuropixels:\n  bin_file: '{binary}'\n  n_chan_bin: 4\n"
        f"  sample_rate: 30000\n  probe_file: '{named}'\n",
    )

    assert ss.setup_probe(session, "neuropixels")["n_chan"] == 4


def test_a_meta_without_geometry_falls_through_to_probe_file(tmp_path):
    # Runs predating ~snsGeomMap: the .meta exists but carries no geometry, so it
    # must fall through rather than raise past the configured fallback.
    _meta_run(tmp_path, with_geom=False)
    named = _probe_json(tmp_path / "probes" / "np1.json", n=4)
    session = _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{tmp_path / 'np'}'\n  run_name: r\n"
        f"  probe_file: '{named}'\n",
    )

    assert ss.setup_probe(session, "neuropixels")["n_chan"] == 4


def test_neuropixels_with_neither_raises_and_names_the_fix(tmp_path):
    # No honest default exists for a Neuropixels layout, so this must not guess.
    binary = tmp_path / "np" / "bare.imec0.ap.bin"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"\x00" * 800)
    session = _session(
        tmp_path,
        f"neuropixels:\n  bin_file: '{binary}'\n  n_chan_bin: 4\n  sample_rate: 30000\n",
    )

    with pytest.raises(FileNotFoundError) as excinfo:
        ss.setup_probe(session, "neuropixels")

    assert "make_probe.py" in str(excinfo.value)
    assert "probe_file" in str(excinfo.value)


def test_a_utah_cmp_wins_over_a_probe_file(tmp_path):
    # Same rule: the array's own wiring map beats a file built from another one.
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(
        tmp_path, f"blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '{probe}'\n"
        f"  cmp_file: '/nope/missing.cmp'\n"
    )

    # cmp_file is tried first, so a missing one surfaces rather than being skipped.
    with pytest.raises(Exception):
        ss.setup_probe(session, "blackrock")


def test_each_probe_gets_the_geometry_from_its_own_meta(tmp_path):
    # Two probes in one run need not share an imro table: which sites are active
    # is chosen per probe. Reading imec0's .meta for imec1 would put imec1's units
    # on imec0's sites, and nothing downstream could tell.
    from conftest import spikeglx_run

    run_dir = spikeglx_run(tmp_path / "npx", probes=(0, 1), sites={0: 4, 1: 6})
    session = _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{run_dir}'\n  run_name: run\n  probes: [0, 1]\n",
    )

    assert ss.setup_probe(session, "neuropixels")["n_chan"] == 4
    assert ss.setup_probe(session, "neuropixels", probe_index=1)["n_chan"] == 6


def test_sorting_checks_the_recording_answers_before_importing_a_sorter(tmp_path, monkeypatch):
    # Opening the recording is metadata calls and no data, so on a share that has
    # stopped answering it hangs in stat() with nothing printed. The check goes
    # first -- and before the several-second Kilosort/torch import, which is
    # wasted work if the file is not there.
    import sys

    from spikesorting._io import reachable

    session = _session(tmp_path, "neuropixels:\n  bin_file: '/mnt/gone/x.bin'\n")

    def stalled(path, *args, **kwargs):
        raise TimeoutError(f"{path} did not respond within 10 s")

    monkeypatch.setattr(reachable, "check_reachable", stalled)
    heavy = set(sys.modules)

    with pytest.raises(OSError, match="did not respond"):
        ss.sort_with_kilosort(session, "neuropixels")

    assert not {"kilosort", "torch"} & (set(sys.modules) - heavy)


def test_a_utah_probe_file_is_used_when_no_cmp_is_named(tmp_path):
    probe = _probe_json(tmp_path / "probes" / "utah_A.json")
    session = _session(tmp_path, f"blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '{probe}'\n")

    assert ss.setup_probe(session, "blackrock")["n_chan"] == 4


def test_a_utah_array_with_no_map_at_all_refuses_to_sort(tmp_path):
    # There is no honest default. A grid in channel order attributes units to the
    # wrong electrodes, and its channel count silently drops every electrode past
    # it -- 96 of a 128-channel array, with nothing in the output to say so.
    # Neuropixels already raises here; this is the same answer for the same
    # question. A deliberate grid is make_probe.py's job, named in the session.
    with pytest.raises(FileNotFoundError) as excinfo:
        ss.setup_probe(_session(tmp_path), "blackrock")

    message = str(excinfo.value)
    for pointer in ("cmp_file", "probe_file", "make_probe.py"):
        assert pointer in message, message


# ---------------------------------------------------------------------------
# The config decides whether a verb does anything
# ---------------------------------------------------------------------------


def test_none_propagates_so_a_sequence_needs_no_branching(tmp_path):
    # This is what lets the runner be a flat list of calls.
    session = _session(tmp_path, "kilosort_on_neuropixels: false\n")

    assert ss.sort_with_kilosort(session, "neuropixels") is None


def test_blackrock_lfp_is_a_no_op_rather_than_an_error(tmp_path):
    # Central already saves those separately, so there is nothing to extract.
    session = _session(tmp_path)

    assert ss.extract_lfp(session, "blackrock") is None
    assert "Central" in ss.skip_reason(session, ss.extract_lfp, "blackrock")


def test_the_lfp_export_is_the_sessions_call_not_a_flags(tmp_path):
    # It writes an output the size of the recording, so it is off unless the
    # session asks -- and the session, not the command line, is where that is
    # written down. A flag can override it for one run; nothing else can.
    off = _session(tmp_path)
    assert off.export_lfp is False
    assert ss.extract_lfp(off, "neuropixels") is None
    assert "export_lfp is false" in ss.skip_reason(off, ss.extract_lfp, "neuropixels")

    on = _session(tmp_path, "export_lfp: true\nlfp_decimate: 2\n")
    assert on.export_lfp is True and on.lfp_decimate == 2
    assert ss.skip_reason(on, ss.extract_lfp, "neuropixels") is None


def test_the_export_settings_come_from_the_session(tmp_path):
    # Same rule for what export_results keeps and draws: session keys, so a run
    # from a bare --config exports what the session says it exports.
    default = _session(tmp_path)
    assert default.export_groups == ("good", "mua")
    assert default.export_figures is True

    stated = _session(
        tmp_path, "export_groups: [good]\nexport_figures: false\n"
    )
    assert stated.export_groups == ("good",)
    assert stated.export_figures is False


def test_export_results_reads_those_settings_rather_than_arguments(
    tmp_path, kilosort_results
):
    # The end of the chain: the keys above have to reach the export itself, which
    # the driver now calls with the config alone. The fixture's three units are
    # labelled good / mua / good, so the selection is visible in the count.
    import shutil

    session = _session(tmp_path, "export_groups: [good]\nexport_figures: false\n")
    sorted_dir = session.paths.sorted_for("neuropixels", 0)
    sorted_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(kilosort_results, sorted_dir, dirs_exist_ok=True)

    good_only = ss.export_results(session, "neuropixels", 0)
    assert good_only["n_units"] == 2                       # units 0 and 2
    assert not list(session.paths.figures_for("neuropixels", 0).glob("*.png"))

    with_mua = ss.export_results(
        cfg.with_overrides(session, export_groups=("good", "mua"), export_figures=True),
        "neuropixels",
        0,
    )
    assert with_mua["n_units"] == 3
    figures = sorted(p.name for p in session.paths.figures_for("neuropixels", 0).glob("*.png"))
    assert figures == ["overview.png", "unit_0000.png", "unit_0001.png", "unit_0002.png"]


def test_a_system_that_is_not_sorted_needs_no_probe_and_no_recording(tmp_path):
    # "Extract the pulses, do not sort" is a whole session shape -- the 2-probe
    # template is one. The driver used to resolve a probe and open the recording
    # for it anyway, which fails on a session that has neither: no map to build,
    # and no spike_file to open. What runs is the sort step, and this is what it
    # asks before running any of it.
    session = _session(
        tmp_path,
        "kilosort_on_blackrock: false\nblackrock:\n  sync_file: '/b/y.ns5'\n",
    )

    assert session.has_data("blackrock") is True        # it recorded
    assert session.sorts_blackrock is False             # ...but is not sorted
    assert "kilosort_on_blackrock is false" in ss.skip_reason(
        session, ss.sort_with_kilosort, "blackrock"
    )
    # ...and the map that setup_probe would have refused to invent is never needed.
    with pytest.raises(FileNotFoundError):
        ss.setup_probe(session, "blackrock")


def test_blackrock_recording_only_the_sync_channels_is_not_sorted(tmp_path):
    # An ordinary session shape: the spikes come from a Neuropixels probe and
    # Blackrock carries only the pulse train, so there is no Utah array to sort.
    # It must skip rather than fail -- this used to reach setup_probe and raise
    # about a channel map for an array that was never in the animal.
    session = _session(tmp_path, "blackrock:\n  sync_file: '/b/NSP.ns5'\n")

    assert session.has_data("blackrock") is True      # it recorded: pulses
    assert session.sorts_blackrock is False           # ...but nothing to sort
    assert "no blackrock.spike_file" in ss.skip_reason(
        session, ss.sort_with_kilosort, "blackrock"
    )
    assert ss.sort_with_kilosort(session, "blackrock") is None
    # ...and its pulses are still extracted, which is the whole point of it.
    assert ss.skip_reason(session, ss.extract_sync, "blackrock") is None


def test_a_utah_export_lands_on_the_nsp_clock(tmp_path, kilosort_results):
    # The end of the chain for a Blackrock-only session. Kilosort writes sample
    # indices; the .nev markers and eye traces recorded beside them are on the
    # NSP's PTP clock, whose origin is ~1.5e9 s and whose rate is not the nominal
    # one. nsp_time_map.json is what carries both, and the export must use it.
    import shutil

    from spikesorting._io.blackrock import NspSegment, NspTimeMap

    session = _session(
        tmp_path,
        "blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '/b/HUB.ns6'\n",
    )
    sorted_dir = session.paths.sorted_for("blackrock", 0)
    sorted_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(kilosort_results, sorted_dir, dirs_exist_ok=True)

    t0, rate = 1_521_182_029.171230, 29_999.8646
    time_map = NspTimeMap(
        segments=(NspSegment(0, 10_000_000, 1 / rate, t0, 1.2e-4),),
        fs_nominal=30_000.0,
        spec="3.0",
        per_sample_timestamps=True,
    )
    (sorted_dir / "nsp_time_map.json").write_text(
        json.dumps(time_map.to_dict()), encoding="utf-8"
    )

    out = ss.export_results(session, "blackrock", 0)

    assert out["timebase"] == "nsp"
    exported = np.load(sorted_dir / "export" / "spike_times.npy")
    samples = np.load(sorted_dir / "export" / "spike_samples.npy")
    assert exported.min() > t0                       # on the PTP clock, not near zero
    assert exported[0] == pytest.approx(t0 + samples[0] / rate, abs=1e-6)
    # ...and the sorter's own index survives beside it, so nothing is one-way.
    assert samples.dtype == np.int64
    assert samples.size == exported.size

    info = json.loads((sorted_dir / "export" / "export_info.json").read_text())
    assert info["timebase"] == "nsp"
    assert info["nsp_time_map"]["measured_rate_hz"] == pytest.approx(rate, abs=1e-3)
    assert info["nsp_time_map"]["drift_ppm"] == pytest.approx(-4.5, abs=0.2)


def test_skip_reason_is_none_when_the_verb_will_actually_run(tmp_path):
    session = _session(tmp_path)

    assert ss.skip_reason(session, ss.extract_sync, "neuropixels") is None
    assert ss.skip_reason(session, ss.setup_probe, "blackrock") is None


# ---------------------------------------------------------------------------
# Sorting: Kilosort4's own run_kilosort, against a stand-in for it
# ---------------------------------------------------------------------------


def _fake_kilosort(monkeypatch, seen: list[dict]):
    """Stand in for Kilosort4, recording what ``run_kilosort`` was told.

    The two-module shape matters and is Kilosort's own: ``from kilosort import
    run_kilosort`` is the function, ``from kilosort.run_kilosort import
    DEFAULT_SETTINGS`` is the module beneath it. The split between settings and
    arguments is read from exactly those two, so a stand-in that got the shape
    wrong would test nothing.
    """
    import sys
    import types

    def run_kilosort(
        settings=None,
        probe=None,
        filename=None,
        results_dir=None,
        data_dtype=None,
        device=None,
        do_CAR=True,
        invert_sign=False,
        bad_channels=None,
        save_preprocessed_copy=False,
    ):
        seen.append(dict(locals()))
        results = Path(results_dir)
        results.mkdir(parents=True, exist_ok=True)
        spike_times = np.arange(12, dtype=np.int64)
        clusters = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
        np.save(results / "spike_times.npy", spike_times)
        np.save(results / "spike_clusters.npy", clusters)
        # Kilosort has written this relative to the directory it sorted in, which
        # is the cache and about to be deleted. Phy reads the raw traces through
        # it, so the stand-in writes the awkward form on purpose.
        (results / "params.py").write_text(
            f"dat_path = '{Path(filename).name}'\nn_channels_dat = 8\n", encoding="utf-8"
        )
        # The whitened copy of the recording: written beside the results, the size
        # of the input, and not a result. It must not be published.
        (results / "temp_wh.dat").write_bytes(b"\x00" * 64)
        return {"ops": True}, spike_times, clusters, None, None, None, None, None, None

    module = types.ModuleType("kilosort.run_kilosort")
    module.DEFAULT_SETTINGS = {
        "n_chan_bin": 385,
        "fs": 30000.0,
        "nblocks": 1,
        "highpass_cutoff": 300.0,
        "whitening_range": 32,
        "batch_size": 60000,
    }
    module.run_kilosort = run_kilosort

    package = types.ModuleType("kilosort")
    package.run_kilosort = run_kilosort

    torch = types.ModuleType("torch")
    torch.device = lambda name: f"torch.device({name})"

    monkeypatch.setitem(sys.modules, "kilosort", package)
    monkeypatch.setitem(sys.modules, "kilosort.run_kilosort", module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return package


def _npx_session(tmp_path, body_extra="", probes="[0]"):
    """A Neuropixels session pointing at a real (tiny) SpikeGLX run."""
    from conftest import spikeglx_run

    run_dir = spikeglx_run(tmp_path / "npx", probes=(0, 1), sites={0: 4, 1: 6})
    return _session(
        tmp_path,
        f"neuropixels:\n  run_dir: '{run_dir}'\n  run_name: run\n"
        f"  probes: {probes}\n{body_extra}",
    )


def test_the_kilosort_block_is_split_into_settings_and_arguments(tmp_path, monkeypatch):
    # Kilosort takes its parameters in two places and which one a parameter lives
    # in has moved between releases, so the split is read from the installed
    # package. n_chan_bin and fs come from the binary itself, never from the file.
    seen: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    session = _npx_session(
        tmp_path,
        "  kilosort:\n    nblocks: 0\n    do_CAR: false\n    bad_channels: [3]\n",
    )

    ss.sort_with_kilosort(session, "neuropixels")

    call = seen[0]
    assert call["settings"] == {"n_chan_bin": 8, "fs": 30000.0, "nblocks": 0}
    assert call["do_CAR"] is False
    assert call["bad_channels"] == [3]
    assert call["data_dtype"] == "int16"


def test_a_parameter_kilosort_does_not_have_is_refused_by_name(tmp_path, monkeypatch):
    # A typo'd key used to reach the sorter and fail there, hours in. Both sets
    # are printed because which one a parameter belongs to is not obvious.
    _fake_kilosort(monkeypatch, [])
    session = _npx_session(tmp_path, "  kilosort:\n    highpas_cutof: 300\n")

    with pytest.raises(ValueError) as excinfo:
        ss.sort_with_kilosort(session, "neuropixels")

    message = str(excinfo.value)
    assert "highpas_cutof" in message
    assert "highpass_cutoff" in message      # the settings it could have meant
    assert "do_CAR" in message               # ...and the arguments


def test_the_session_cannot_set_what_the_pipeline_passes(tmp_path, monkeypatch):
    # filename, probe, results_dir and friends are decided here. Silently ignoring
    # a session that sets one would sort a different file than the file says.
    _fake_kilosort(monkeypatch, [])
    session = _npx_session(tmp_path, "  kilosort:\n    filename: /somewhere/else.bin\n")

    with pytest.raises(ValueError, match="set by the pipeline"):
        ss.sort_with_kilosort(session, "neuropixels")


def test_each_probe_is_sorted_into_its_own_directory(tmp_path, monkeypatch):
    # Two probes are two streams with two clocks. Sorting both into one directory
    # would leave the second on top of the first with nothing to say so.
    seen: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    session = _npx_session(tmp_path, probes="[0, 1]")

    first = ss.sort_with_kilosort(session, "neuropixels", probe_index=0)
    second = ss.sort_with_kilosort(session, "neuropixels", probe_index=1)

    assert first.results_dir != second.results_dir
    assert first.results_dir.name == "kilosort4"
    assert first.results_dir.parent.name == "imec0"
    assert second.results_dir.parent.name == "imec1"
    # ...and each read its own probe's binary, not the other's.
    assert "imec0" in str(seen[0]["filename"]) and "imec1" in str(seen[1]["filename"])


def test_a_probes_own_bad_channels_reach_the_sorter(tmp_path, monkeypatch):
    # A dead site is a fact about one probe, not about the run.
    seen: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    session = _npx_session(
        tmp_path,
        "  kilosort:\n    nblocks: 1\n"
        "  by_probe:\n    1:\n      kilosort:\n        bad_channels: [2, 5]\n",
        probes="[0, 1]",
    )

    ss.sort_with_kilosort(session, "neuropixels", probe_index=0)
    ss.sort_with_kilosort(session, "neuropixels", probe_index=1)

    assert seen[0]["bad_channels"] is None          # imec0 said nothing
    assert seen[1]["bad_channels"] == [2, 5]
    assert seen[0]["settings"]["nblocks"] == seen[1]["settings"]["nblocks"] == 1


def test_the_results_are_published_out_of_the_cache_without_the_dat(tmp_path, monkeypatch):
    # Kilosort writes beside its results, so it runs on local scratch and the
    # results are copied out. The whitened copy of the recording is the size of
    # the input and is not a result, so it stays behind with the cache.
    seen: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    cache = tmp_path / "cache"
    session = _npx_session(tmp_path)
    session = cfg.with_overrides(
        session, machine=cfg.MachineProfile(name="test", cache_dir=cache, device="cpu")
    )

    result = ss.sort_with_kilosort(session, "neuropixels")

    work_dir = Path(seen[0]["results_dir"])
    assert cache in work_dir.parents        # sorted on local scratch
    assert not work_dir.exists()            # ...whose working copy is then gone
    assert cache.exists()                   # the scratch dir itself is the machine's
    assert (result.results_dir / "spike_times.npy").exists()
    assert not list(result.results_dir.glob("*.dat"))
    assert result.n_units == 4 and result.n_spikes == 12

    # Phy reads the raw traces through params.py, and the path Kilosort wrote was
    # relative to the cache it sorted in. It has to resolve from where the results
    # actually landed, or the traces panel is empty with no explanation.
    params = (result.results_dir / "params.py").read_text(encoding="utf-8")
    dat_path = Path(params.splitlines()[0].partition("=")[2].strip().strip("'"))
    assert dat_path.is_absolute() and dat_path.exists()
    assert "n_channels_dat = 8" in params        # the rest of the file is untouched

    info = json.loads((result.results_dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["stream"] == "imec0" and info["device"] == "cpu"
    assert info["settings"]["n_chan_bin"] == 8


def test_a_dry_run_reports_what_it_would_do_and_sorts_nothing(tmp_path, monkeypatch):
    seen: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    session = _npx_session(tmp_path)

    report = ss.sort_with_kilosort(session, "neuropixels", dry_run=True)

    assert not seen
    assert report["settings"]["n_chan_bin"] == 8
    assert report["probe"]["n_chan"] == 4
    assert "imec0" in report["binary"]


def _fake_blackrock(
    monkeypatch,
    transformed: list,
    n_chan=96,
    n_frames=1000,
    segments: int = 1,
    opened: list | None = None,
):
    """Stand in for SpikeInterface's reader and Kilosort's binary transform."""
    import sys
    import types

    opened = [] if opened is None else opened

    class Recording:
        def __init__(self, n_segments=1):
            self._n_segments = n_segments

        def get_num_channels(self):
            return n_chan

        def get_sampling_frequency(self):
            return 30000.0

        def get_num_segments(self):
            return self._n_segments

        def get_num_frames(self, segment_index=0):
            return n_frames

    extractors = types.ModuleType("spikeinterface.extractors")

    def read_blackrock(path, **kwargs):
        opened.append(dict(kwargs, path=path))
        return Recording(n_segments=segments)

    extractors.read_blackrock = read_blackrock
    package = types.ModuleType("spikeinterface")
    package.extractors = extractors

    def spikeinterface_to_binary(recording, data_dir, data_name=None, dtype=None, **kwargs):
        transformed.append({"data_dir": Path(data_dir), "data_name": data_name, "dtype": dtype})
        path = Path(data_dir) / data_name
        path.write_bytes(b"\x00" * (n_chan * n_frames * 2))
        return path, n_frames, n_chan, 1, 30000.0, None

    io = types.ModuleType("kilosort.io")
    io.spikeinterface_to_binary = spikeinterface_to_binary

    monkeypatch.setitem(sys.modules, "spikeinterface", package)
    monkeypatch.setitem(sys.modules, "spikeinterface.extractors", extractors)
    monkeypatch.setitem(sys.modules, "kilosort.io", io)


def _utah_session(tmp_path, extra=""):
    """A Blackrock session with a spike file and a map, ready to sort."""
    probe = _probe_json(tmp_path / "probes" / "utah.json", n=96)
    session = _session(
        tmp_path,
        "blackrock:\n  sync_file: '/b/y.ns5'\n"
        f"  spike_file: '{tmp_path / 'br' / 'HUB-A_001.ns6'}'\n"
        f"  probe_file: '{probe}'\n{extra}",
    )
    (tmp_path / "br").mkdir(parents=True, exist_ok=True)
    (tmp_path / "br" / "HUB-A_001.ns6").write_bytes(b"\x00" * 16)
    return session


def test_the_gap_tolerance_reaches_the_reader(tmp_path, monkeypatch):
    # Without it neo raises on any timestamp jump over two sampling periods, and
    # PTP files carry occasional corrupted packet timestamps -- so most real
    # recordings do not open at all. The session names the tolerance.
    seen, transformed, opened = [], [], []
    _fake_kilosort(monkeypatch, seen)
    _fake_blackrock(monkeypatch, transformed, opened=opened)
    session = _utah_session(tmp_path, "  gap_tolerance_ms: 4.5\n")

    ss.sort_with_kilosort(session, "blackrock")

    assert opened[0]["gap_tolerance_ms"] == 4.5


def test_a_paused_recording_stops_rather_than_being_concatenated(tmp_path, monkeypatch):
    # A jump larger than the tolerance is a pause, not a glitch. Sorting across
    # one silently would splice two recordings and map them with a single line.
    seen, transformed = [], []
    _fake_kilosort(monkeypatch, seen)
    _fake_blackrock(monkeypatch, transformed, segments=3)
    session = _utah_session(tmp_path)

    with pytest.raises(ValueError) as excinfo:
        ss.sort_with_kilosort(session, "blackrock")

    message = str(excinfo.value)
    assert "3 segments" in message
    assert "allow_segments" in message and "gap_tolerance_ms" in message
    assert not seen                      # nothing was sorted

    # ...and saying so in the session is what lets it through.
    allowed = _utah_session(tmp_path, "  allow_segments: true\n")
    assert ss.sort_with_kilosort(allowed, "blackrock") is not None


def test_the_ns6_is_transformed_once_beside_the_recording(tmp_path, monkeypatch):
    # A .ns6 is not a flat binary, so it is transformed into one -- and that file
    # has to persist: params.py points at it, so Phy shows raw traces only while
    # it exists, and re-transforming an unchanged .ns6 is minutes of I/O for
    # nothing.
    seen: list[dict] = []
    transformed: list[dict] = []
    _fake_kilosort(monkeypatch, seen)
    _fake_blackrock(monkeypatch, transformed)

    probe = _probe_json(tmp_path / "probes" / "utah.json", n=96)
    session = _session(
        tmp_path,
        "blackrock:\n  sync_file: '/b/y.ns5'\n"
        f"  spike_file: '{tmp_path / 'br' / 'HUB-A_001.ns6'}'\n  probe_file: '{probe}'\n",
    )
    (tmp_path / "br").mkdir(parents=True, exist_ok=True)
    (tmp_path / "br" / "HUB-A_001.ns6").write_bytes(b"\x00" * 16)

    first = ss.sort_with_kilosort(session, "blackrock")

    assert transformed[0]["dtype"] == "int16"
    assert transformed[0]["data_name"] == "HUB-A_001.bin"
    binary = tmp_path / "br" / "blackrock" / "HUB-A_001.bin"
    assert binary.exists()                       # beside the recording, not in the cache
    assert Path(seen[0]["filename"]) == binary
    assert seen[0]["settings"]["n_chan_bin"] == 96
    assert first.results_dir.name == "kilosort4"

    ss.sort_with_kilosort(session, "blackrock")

    assert len(transformed) == 1                 # reused, not written again
    assert binary.exists()


# ---------------------------------------------------------------------------
# Time remapping
# ---------------------------------------------------------------------------


def test_time_remapping_reports_missing_edges_rather_than_guessing(tmp_path):
    assert ss.time_remapping(_session(tmp_path), "neuropixels") is None


def test_time_remapping_recovers_a_planted_offset_and_drift(tmp_path):
    from spikesorting._sync import catgt

    session = _session(tmp_path)
    for system in ("blackrock", "neuropixels"):
        session.paths.sync_for(system).mkdir(parents=True, exist_ok=True)

    offset, ppm = 12.3456, 20e-6
    br = np.arange(0.0, 600.0, 1.0) + 100.0
    npx = (br - offset) * (1 + ppm)
    catgt.write_edge_file(session.paths.sync_for("blackrock") / "blackrock_1hz.txt", br)
    catgt.write_edge_file(session.paths.sync_for("neuropixels") / "npx_1hz.txt", npx)

    result = ss.time_remapping(session, "neuropixels")

    assert isinstance(result, ss.TimeMap)
    # No sorting to map, so the map itself is the whole output.
    assert result.method == "map_only"
    assert result.mapping.intercept == pytest.approx(offset, abs=1e-6)
    assert result.mapping.drift_ppm == pytest.approx(-20.0, abs=0.1)
    assert (session.paths.aligned_for(0) / "time_map.json").exists()
    assert "ppm" in result.summary()


def test_each_probe_is_fitted_onto_blackrock_time_on_its_own(tmp_path):
    # Two probes in one run are two oscillators. One fit cannot serve both, so
    # each gets its own map in its own directory -- and planting *different*
    # drifts is what proves the second probe is not reading the first's edges.
    from spikesorting._sync import catgt

    session = _session(
        tmp_path,
        "neuropixels:\n  run_dir: '/npx'\n  run_name: run\n  probes: [0, 1]\n",
    )
    session.paths.sync_for("blackrock").mkdir(parents=True, exist_ok=True)

    br = np.arange(0.0, 600.0, 1.0) + 100.0
    catgt.write_edge_file(session.paths.sync_for("blackrock") / "blackrock_1hz.txt", br)

    planted = {0: (12.3456, 20e-6), 1: (34.5678, -5e-6)}
    for probe, (offset, ppm) in planted.items():
        sync_dir = session.paths.sync_for("neuropixels", probe)
        sync_dir.mkdir(parents=True, exist_ok=True)
        catgt.write_edge_file(sync_dir / "npx_1hz.txt", (br - offset) * (1 + ppm))

    for probe, (offset, ppm) in planted.items():
        result = ss.time_remapping(session, "neuropixels", probe)

        assert result.mapping.intercept == pytest.approx(offset, abs=1e-6)
        assert result.mapping.drift_ppm == pytest.approx(-ppm * 1e6, abs=0.1)
        assert (session.paths.aligned_for(probe) / "time_map.json").exists()

    # ...and the two maps are files of their own, not one overwritten twice.
    maps = sorted(p.parent.name for p in session.paths.aligned.rglob("time_map.json"))
    assert maps == ["imec0", "imec1"]


def test_the_reference_timebase_is_not_remapped_onto_itself(tmp_path):
    # Blackrock is the reference. Asking to map it is meaningless, and the runner
    # loops over both systems, so the verb has to say so rather than the caller.
    session = _session(tmp_path)

    assert ss.REFERENCE_SYSTEM == "blackrock"
    assert ss.time_remapping(session, "blackrock") is None
    assert "reference timebase" in ss.skip_reason(session, ss.time_remapping, "blackrock")
    # ...while the other system has no such guard.
    assert ss.skip_reason(session, ss.time_remapping, "neuropixels") is None


def test_the_probe_is_what_drops_non_neural_channels(tmp_path):
    """A .ns6 carries sync and analog inputs beside the neural channels.

    Nothing has to name them: attaching the probe slices the recording to exactly
    the channels its chanMap covers. There used to be a blackrock-only
    `exclude_channels` for this, applied *before* the probe -- which shifted every
    contact after the excluded one and silently mis-attributed units.
    """
    import numpy as np
    import spikeinterface.full as si

    from spikesorting._probes.common import to_probeinterface

    recording = si.generate_recording(num_channels=8, durations=[1.0])
    probe = {
        "chanMap": np.arange(6),
        "xc": np.tile([0.0, 25.0], 3),
        "yc": np.repeat(np.arange(3) * 20.0, 2),
        "kcoords": np.zeros(6),
        "n_chan": 6,
    }

    attached = recording.set_probe(to_probeinterface(probe))

    assert recording.get_num_channels() == 8
    assert attached.get_num_channels() == 6
    assert [str(c) for c in attached.channel_ids] == ["0", "1", "2", "3", "4", "5"]
