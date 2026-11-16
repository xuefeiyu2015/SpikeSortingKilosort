"""Are the sync pulses there? At sorting time, and on demand (--sync-exam).

A missing 1 Hz or 14 s burst used to surface only at synchronization, hours
later, as "no burst edges on one or both systems". The CatGT pass that
preprocesses already extracted them, so the sort now says so; and the exam
extracts them on their own, without touching anything the pipeline reads.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from spikesorting import _config as cfg
from spikesorting._sync import catgt, edge_check, extract

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DURATION = 30.0


def _one_hz(duration=DURATION, phase=0.25):
    return phase + np.arange(int(duration))


def _bursts(duration=DURATION, interval=14.0, pulses=5):
    starts = 1.0 + np.arange(0.0, duration - 1.0, interval)
    return np.sort(np.concatenate([s + 0.02 * np.arange(pulses) for s in starts]))


# ---------------------------------------------------------------------------
# The count, pure
# ---------------------------------------------------------------------------


def test_each_train_is_counted_against_the_recordings_length():
    ok_1hz, ok_burst = _one_hz(600), _bursts(600)
    checks = {c["name"]: c for c in edge_check.check_sync_edges({
        "a": (edge_check.ONE_HZ, ok_1hz, "AP SY"),
        "b": (edge_check.ONE_HZ, ok_1hz[:100], "AP SY"),
        "c": (edge_check.ONE_HZ, np.empty(0), "OneBox SY"),
        "d": (edge_check.BURST, ok_burst, "XA0"),
        "e": (edge_check.BURST, ok_burst[:15], "XA0"),       # 3 bursts of ~43
        "f": (edge_check.BURST, None, "no OneBox file"),
    }, duration_s=600.0)}

    assert checks["a"]["status"] == "ok" and checks["a"]["found"] == 600
    assert checks["b"]["status"] == "too few" and checks["b"]["expected"] == 600
    assert checks["c"]["status"] == "missing"
    # A burst is counted as one, however many pulses it codes.
    assert checks["d"]["status"] == "ok" and checks["d"]["found"] == 43
    assert checks["d"]["edges"] == 43 * 5
    assert checks["e"]["status"] == "too few" and checks["e"]["found"] == 3
    assert checks["f"]["status"] == "not extracted" and checks["f"]["found"] is None

    problems = edge_check.sync_problems(list(checks.values()))
    assert len(problems) == 4
    assert any("c (OneBox SY): no edges at all (expected ~600 pulses)" in p for p in problems)
    assert any("only 3 bursts (expected ~43)" in p for p in problems)


# ---------------------------------------------------------------------------
# A session with a SpikeGLX run, a OneBox and CatGT
# ---------------------------------------------------------------------------


def _session(tmp_path, catgt_dir=True):
    from conftest import spikeglx_run

    binary = spikeglx_run(tmp_path / "npx", obx_phase=0.25, duration_s=DURATION)[0]
    path = tmp_path / "s.yaml"
    path.write_text(
        f"blackrock_dir: '{tmp_path / 'br'}'\nneuropixels_dir: '{binary.parent}'\n"
        "blackrock:\n  sync_file: '{blackrock_dir}/NSP_001.ns5'\n"
        f"neuropixels:\n  bin_file: '{binary}'\n", encoding="utf-8")
    (tmp_path / "br").mkdir()
    session = cfg.load_session_config(path, "mac", CONFIG_DIR)
    if catgt_dir:
        tool = tmp_path / "CatGT"
        tool.mkdir()
        (tool / "CatGT.exe").write_text("")
        session = replace(session, machine=replace(session.machine, catgt_dir=tool))
    return session


def _write_catgt_edges(dest: Path, trains: dict[str, np.ndarray]) -> None:
    """Edge files named the way CatGT names them, nested the way it nests them."""
    names = {
        extract.NPX_1HZ: "run_g0_tcat.imec0.ap.xd_8_6_500.txt",
        extract.NPX_OBX_1HZ: "run_g0_tcat.obx0.obx.xd_3_6_500.txt",
        extract.NPX_BURST: "run_g0_tcat.obx0.obx.xa_0_0.txt",
    }
    folder = Path(dest) / "catgt_run_g0"
    folder.mkdir(parents=True, exist_ok=True)
    for name, times in trains.items():
        catgt.write_edge_file(folder / names[name], times)


def _fake_catgt(monkeypatch, trains):
    calls = []

    def run(catgt_dir, args):
        calls.append(args)
        dest = Path(next(a for a in args if a.startswith("-dest="))[len("-dest="):])
        _write_catgt_edges(dest, trains)

    monkeypatch.setattr(catgt, "run_catgt", run)
    return calls


def test_the_exam_finds_a_missing_burst_and_says_where(tmp_path, monkeypatch):
    import spikesorting as ss
    from spikesorting import pipeline

    session = _session(tmp_path)
    # CatGT finds both 1 Hz trains and writes nothing for the burst (XA0 silent).
    calls = _fake_catgt(monkeypatch, {extract.NPX_1HZ: _one_hz(),
                                      extract.NPX_OBX_1HZ: _one_hz()})

    exam = ss.exam_sync(session, "neuropixels")

    status = {c["name"]: c["status"] for c in exam.checks}
    assert status == {extract.NPX_1HZ: "ok", extract.NPX_OBX_1HZ: "ok",
                      extract.NPX_BURST: "missing"}
    assert not exam.ok
    burst = next(c for c in exam.checks if c["name"] == extract.NPX_BURST)
    assert burst["stream"].endswith("XA0") and "obx" in burst["stream"]
    assert "MISSING" in exam.summary() and "report:" in exam.summary()
    # Extraction only, with the session's own extractors, into the exam's folder.
    folder = session.paths.sorted_for("neuropixels") / pipeline.SYNC_EXAM_DIR
    assert any(a == f"-dest={folder / 'catgt'}" for a in calls[0])
    assert any(a.startswith("-xa=") for a in calls[0])
    assert not any(a.startswith(("-apfilter", "-gblcar", "-gfix")) for a in calls[0])
    saved = json.loads((folder / "sync_exam_neuropixels.json").read_text())
    assert saved["ok"] is False and saved["duration_s"] == pytest.approx(DURATION)


def test_the_exam_changes_nothing_the_pipeline_reads(tmp_path, monkeypatch):
    import spikesorting as ss

    session = _session(tmp_path)
    _fake_catgt(monkeypatch, {extract.NPX_1HZ: _one_hz()})
    sync_dir = session.paths.sync_for("neuropixels")
    sync_dir.mkdir(parents=True, exist_ok=True)
    edges = catgt.write_edge_file(sync_dir / f"{extract.NPX_1HZ}.txt", np.array([1.5, 2.5]))
    accepted = session.paths.aligned / "validation.json"
    accepted.write_text('{"accepted": true}')
    before = edges.read_text()

    ss.exam_sync(session, "neuropixels")

    assert edges.read_text() == before and accepted.exists()


def test_with_every_pulse_there_the_exam_is_ok(tmp_path, monkeypatch):
    import spikesorting as ss

    session = _session(tmp_path)
    _fake_catgt(monkeypatch, {extract.NPX_1HZ: _one_hz(), extract.NPX_OBX_1HZ: _one_hz(),
                              extract.NPX_BURST: _bursts()})
    assert ss.exam_sync(session, "neuropixels").ok


def test_without_catgt_the_exam_says_where_to_run_it(tmp_path):
    import spikesorting as ss

    with pytest.raises(RuntimeError, match="no catgt_dir.*extract_sync falls back"):
        ss.exam_sync(_session(tmp_path, catgt_dir=False), "neuropixels")


def test_the_blackrock_exam_reads_its_pulses_once_and_reports_beside_each_probe(
    tmp_path, monkeypatch
):
    import spikesorting as ss
    from spikesorting import pipeline
    from spikesorting._io import reachable

    session = _session(tmp_path)
    reads = []

    def trains(config, chunk_samples=0):
        reads.append(1)
        return ({extract.BR_1HZ: (edge_check.ONE_HZ, _one_hz(), "NSP_001.ns5 ainp7"),
                 extract.BR_BURST: (edge_check.BURST, np.array([3.0]), "NSP_001.ns5 ainp8")},
                DURATION, [])

    monkeypatch.setattr(extract, "blackrock_sync_trains", trains)
    monkeypatch.setattr(reachable, "check_reachable",
                        lambda p, **k: type("P", (), {"summary": lambda self: "ok"})())

    exam = ss.exam_sync(session, "blackrock")

    assert reads == [1]
    status = {c["name"]: c["status"] for c in exam.checks}
    # One burst in 30 s: two expected, so the generator stopped or the threshold is off.
    assert status == {extract.BR_1HZ: "ok", extract.BR_BURST: "too few"}
    folder = session.paths.sorted_for("neuropixels") / pipeline.SYNC_EXAM_DIR
    assert json.loads((folder / "sync_exam_blackrock.json").read_text())["system"] == "blackrock"


def test_a_session_with_nothing_to_examine_skips_with_the_reason(tmp_path):
    import spikesorting as ss

    session = _session(tmp_path)
    bare = replace(session, blackrock=replace(session.blackrock, sync_file=None))
    assert ss.skip_reason(bare, ss.exam_sync, "blackrock") == "no blackrock paths declared in the session"
    assert ss.exam_sync(bare, "blackrock") is None
    # A Utah array recorded, but no sync pulses with it.
    spikes_only = replace(session, blackrock=replace(
        session.blackrock, sync_file=None, spike_file=tmp_path / "br" / "HUB_001.ns6"))
    assert "no blackrock.sync_file" in ss.skip_reason(spikes_only, ss.exam_sync, "blackrock")


def test_the_driver_examines_every_system_and_fails_on_a_short_train(tmp_path, monkeypatch):
    import importlib.util
    import sys

    import spikesorting as ss
    from spikesorting import pipeline

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_sorting_pipeline",
                                                  root / "run_sorting_pipeline.py")
    driver = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "tools"))
    spec.loader.exec_module(driver)

    session = _session(tmp_path)
    seen = []

    def exam(config, system):
        seen.append(system)
        bad = {"status": "missing", "found": 0, "expected": 2, "unit": "bursts",
               "kind": "14 s burst", "name": "npx_burst", "stream": "XA0"}
        return pipeline.SyncExam(system, [bad] if system == "neuropixels" else [], [])

    monkeypatch.setattr(ss, "exam_sync", exam)
    exam.__name__ = "exam_sync"
    assert driver._exam(session) == 1
    assert seen == ["neuropixels", "blackrock"]         # a missing burst stops nothing


# ---------------------------------------------------------------------------
# At sorting time: the CatGT pass's own edges
# ---------------------------------------------------------------------------


def _catgt_record(session, edge_dir):
    from spikesorting._preprocess import provenance

    provenance.write(session.paths.preprocessed_for("neuropixels"), {
        "engine": "catgt", "edge_dir": str(edge_dir),
        "n_samples": int(DURATION * 30000), "fs": 30000.0,
    })


def test_the_sort_reports_a_pulse_the_catgt_pass_did_not_find(tmp_path):
    from spikesorting import pipeline

    session = _session(tmp_path)
    edge_dir = session.paths.preprocessed_for("neuropixels") / "catgt"
    _write_catgt_edges(edge_dir, {extract.NPX_1HZ: _one_hz(),
                                  extract.NPX_OBX_1HZ: _one_hz()})
    _catgt_record(session, edge_dir)

    checked = pipeline._preprocessing_sync_check(session)

    assert checked["source"] == "CatGT preprocessing pass"
    assert {c["name"]: c["status"] for c in checked["checks"]}[extract.NPX_BURST] == "missing"
    assert len(checked["problems"]) == 1 and "npx_burst" in checked["problems"][0]
    summary = pipeline.SortSummary(Path("."), 3, 10, 30000.0, Path("x"), {},
                                   sync_problems=tuple(checked["problems"])).summary()
    assert "WARNING sync pulses: 14 s burst npx_burst" in summary


def test_preprocessing_warns_when_a_pulse_is_missing(tmp_path, monkeypatch, caplog):
    import spikesorting as ss
    from spikesorting import pipeline

    session = _session(tmp_path)
    edge_dir = session.paths.preprocessed_for("neuropixels") / "catgt"
    _write_catgt_edges(edge_dir, {extract.NPX_OBX_1HZ: _one_hz(),
                                  extract.NPX_BURST: _bursts()})
    _catgt_record(session, edge_dir)
    monkeypatch.setattr(pipeline, "_binary_for", lambda config, system: "binary")
    monkeypatch.setattr(pipeline, "_check_input_reachable", lambda config, system: None)

    with caplog.at_level("WARNING", logger="spikesorting"):
        assert ss.preprocess(session, "neuropixels") == "binary"

    warned = [r.getMessage() for r in caplog.records if "sync pulses" in r.getMessage()]
    assert len(warned) == 1 and "1 Hz npx_1hz" in warned[0] and "--sync-exam" in warned[0]


def test_a_spikeinterface_binary_has_no_catgt_edges_to_check(tmp_path):
    from spikesorting import pipeline
    from spikesorting._preprocess import provenance

    session = _session(tmp_path)
    provenance.write(session.paths.preprocessed_for("neuropixels"),
                     {"engine": "spikeinterface", "n_samples": 10, "fs": 30000.0})
    assert pipeline._preprocessing_sync_check(session) is None
