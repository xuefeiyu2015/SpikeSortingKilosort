"""The processing-time table: phases from steps, Kilosort's timers, rendering."""

from __future__ import annotations

import pytest

from spikesorting._export import timing as tm


def test_kilosort_timers_land_in_their_phases_and_the_rest_is_kept():
    ops = {"runtime_preproc": 10.0, "runtime_drift": 20.0, "runtime_clu": 30.0,
           "runtime_postproc": 5.0, "runtime": 999.0, "runtime_new_stage": 1.0}
    steps = tm.kilosort_steps(ops, wall_s=70.0)

    by_label = {label: (phase, s) for phase, label, s in steps}
    assert by_label["kilosort: whitening & high-pass set-up"] == ("preprocessing", 10.0)
    assert by_label["kilosort: drift correction"] == ("sorting", 20.0)
    assert by_label["kilosort: saving results"] == ("postprocessing", 5.0)
    assert by_label["kilosort: runtime_new_stage"] == ("sorting", 1.0)
    # 70 s from outside, 66 s on Kilosort's timers: the 4 s is not lost.
    assert by_label["kilosort: set-up, plots, other"] == ("sorting", pytest.approx(4.0))
    assert sum(s for _p, _l, s in steps) == pytest.approx(70.0)


def test_a_kilosort_without_timers_is_one_row():
    assert tm.kilosort_steps({}, 12.0) == [
        ("sorting", "kilosort (no per-stage times reported)", 12.0)]


def test_phases_are_ordered_and_summed():
    timing = tm.build_timing(
        [("sorting", "a", 3.0), ("loading", "b", 1.0), ("sorting", "c", 2.0),
         ("preprocessing", "d", None)],
        run={"machine": "rig"},
        notes={"d": "made elsewhere"},
    )
    assert list(timing["phases"]) == ["loading", "preprocessing", "sorting"]
    assert timing["phases"]["sorting"]["seconds"] == 5.0
    assert timing["phases"]["preprocessing"]["steps"][0]["note"] == "made elsewhere"
    assert timing["total_s"] == 6.0


def test_an_unknown_phase_raises():
    with pytest.raises(ValueError, match="unknown phase"):
        tm.build_timing([("decoding", "x", 1.0)], run={})


def test_exporting_replaces_an_earlier_export_and_leaves_the_input_alone():
    sort = tm.build_timing([("sorting", "k", 10.0)], run={"machine": "hpc"})
    once = tm.with_exporting(sort, [("exporting", "a", 5.0)], {"machine": "rig"})
    twice = tm.with_exporting(once, [("exporting", "b", 1.0)], {"machine": "mac"})

    assert "exporting" not in sort["phases"]
    assert [r["step"] for r in twice["phases"]["exporting"]["steps"]] == ["b"]
    assert twice["total_s"] == 11.0
    assert twice["exporting_run"]["machine"] == "mac"


def test_a_record_from_before_phases_still_gets_a_table():
    legacy = {"started": "t0", "finished": "t1", "preprocessing": "reused",
              "preprocess_s": 2.0, "kilosort_s": 50.0, "publish_s": 3.0, "total_s": 55.0}
    timing = tm.with_exporting(legacy, [("exporting", "x", 5.0)], {})
    assert list(timing["phases"]) == ["preprocessing", "sorting", "postprocessing", "exporting"]
    assert timing["total_s"] == 60.0


def test_the_table_shows_every_phase_with_its_share():
    timing = tm.build_timing(
        [("loading", "probe", 1.0), ("sorting", "kilosort", 3.0)],
        run={"machine": "rig", "device": "cuda", "started": "s", "finished": "f"},
        extra={"input": {"path": "/x.bin", "gb": 1.5, "read_mb_per_s": 200.0}},
    )
    text = tm.render_timing(timing, "session | imec0")

    assert "sorted   on rig (cuda)" in text
    assert "1.50 GB, reads at 200 MB/s" in text
    assert "postprocessing" in text and "not run" in text
    sorting_line = next(line for line in text.splitlines() if line.startswith("sorting"))
    assert "75.0%" in sorting_line
    assert text.rstrip().splitlines()[-1].split()[-2:] == ["4.0", "s"]


@pytest.mark.parametrize("seconds, shown", [
    (None, "--"), (0.4, "0.4 s"), (90, "1.5 min"), (5400, "1.50 h")])
def test_format_seconds(seconds, shown):
    assert tm.format_seconds(seconds) == shown
