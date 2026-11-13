"""The drivers' shared argument handling.

The rule this file exists to pin: **every setting has a home in the session
file, and a flag only overrides it for one run.** A setting reachable only from
a command line is one nobody can read back off the session six months later, so
``flag_overrides`` maps each flag onto the session key it replaces rather than
being read directly by the loops.
"""

from __future__ import annotations

import pytest

import argparse

from _cli import flag_overrides


def _args(**kwargs) -> argparse.Namespace:
    """The flags a driver would parse, defaulting to "not passed"."""
    defaults = {
        "system": None,
        "export_lfp": False,
        "export_waveforms": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_no_flags_means_the_session_decides_everything():
    assert flag_overrides(_args()) == {}


def test_each_flag_names_the_session_key_it_replaces():
    assert flag_overrides(_args(export_lfp=True)) == {"export_lfp": True}
    assert flag_overrides(_args(export_waveforms=True)) == {
        "waveforms": {"export_snippets": True}
    }


def test_one_system_this_run_is_expressed_as_the_session_keys():
    # --system blackrock is "sort this one, not the other", which is exactly what
    # the two kilosort_on_* keys say -- so it is written as those, and the run
    # reports the reason the skipped one skipped.
    assert flag_overrides(_args(system="blackrock")) == {
        "kilosort_on_neuropixels": False,
        "kilosort_on_blackrock": True,
    }


def test_overrides_reach_the_loaded_session_without_touching_the_file(tmp_path):
    from _cli import load

    session = tmp_path / "s.yaml"
    session.write_text(
        f"neuropixels_dir: '{tmp_path / 'np'}'\n"
        "export_lfp: false\nneuropixels:\n  bin_file: '/a/x.bin'\n",
        encoding="utf-8",
    )
    common = {"config": str(session), "machine": "mac", "probe": None}
    (tmp_path / "np").mkdir()

    assert load(_args(**common), require_inputs=False).export_lfp is False

    overridden = load(_args(**common, export_lfp=True), require_inputs=False)
    assert overridden.export_lfp is True

    # ...and the session file still says what the session wants.
    assert "export_lfp: false" in session.read_text(encoding="utf-8")


def test_a_printed_path_survives_being_pasted_into_a_shell():
    # The shares are named like "Z:/Monkey Athos/..."; unquoted, the space splits
    # the phy command's one argument into two. Checked with a POSIX shell's own
    # splitting -- cmd and PowerShell read double quotes the same way.
    import shlex
    from pathlib import Path

    from _cli import shell_arg

    path = Path("Z:/Monkey Athos/2026-08-13/kilosort4/params.py")
    assert shlex.split(f"phy template-gui {shell_arg(path)}") == [
        "phy", "template-gui", str(path),
    ]
    # ...and a path with nothing to protect is printed as it was typed.
    assert shell_arg("session_x") == "session_x"


def test_each_stage_reports_how_long_it_took():
    from _cli import _elapsed

    assert _elapsed(0.42) == "0.4 s"
    assert _elapsed(252) == "4.2 min"
    assert _elapsed(5508) == "1.53 h"


@pytest.mark.parametrize("require_inputs", [True, False])     # sorting / exporting driver
def test_a_mistyped_directory_is_never_created(tmp_path, require_inputs):
    # The bug: load() made <typo>/kilosort4/export with mkdir(parents=True) before
    # anything checked the path, leaving a fake recording folder on disk.
    from _cli import load

    typo = tmp_path / "SGL_DATA" / "Athos_g1_typo"
    session = tmp_path / "s.yaml"
    session.write_text(
        f"neuropixels_dir: '{typo}'\nneuropixels:\n  bin_file: '{{neuropixels_dir}}/x.ap.bin'\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="neuropixels_dir does not exist"):
        load(_args(config=str(session), machine="mac", probe=None),
             require_inputs=require_inputs)
    assert not (tmp_path / "SGL_DATA").exists()          # not even the parent


def test_the_runner_keeps_each_stages_time_and_breakdown(monkeypatch):
    from types import SimpleNamespace

    import _cli
    from _cli import Runner

    def export_results(config, system):
        return {"n_units": 3, "timing": {"figures": 1.0}}

    def extract_sync(config, system):
        return None

    monkeypatch.setattr(_cli, "skip_reason", lambda *a, **k: None)
    run = Runner(SimpleNamespace())
    run(extract_sync, None, "blackrock", system="blackrock")
    run(export_results, None, "neuropixels", system="neuropixels", label="neuropixels imec1")

    (stage,) = run.stages_for("neuropixels imec1")
    assert stage["stage"] == "export_results" and stage["status"] == "ok"
    assert stage["steps"] == {"figures": 1.0}
    assert run.stages_for("blackrock")[0]["status"] == "nothing to do"


def test_the_run_log_keeps_what_was_printed_and_logged_and_saves_it_per_stream(tmp_path):
    import logging
    import sys

    from _cli import RunLog

    terminal = sys.stdout
    run_log = RunLog("exporting")
    try:
        print("[ok] extract_sync(neuropixels)")
        logging.getLogger("spikesorting").warning("a pipeline message")
    finally:
        run_log.close()
    assert sys.stdout is terminal                    # capture is undone

    saved = run_log.save([tmp_path / "np", tmp_path / "br", tmp_path / "np"], "rig")

    assert [p.parent for p in saved] == [tmp_path / "np" / "logs", tmp_path / "br" / "logs"]
    name = saved[0].name
    assert name.startswith("exporting_") and name.endswith("_rig.log")
    text = saved[0].read_text()
    assert "[ok] extract_sync(neuropixels)" in text and "a pipeline message" in text
    assert "command:" in text.splitlines()[0]


def test_the_stage_table_lists_every_stage_and_the_total(monkeypatch):
    from types import SimpleNamespace

    import _cli
    from _cli import Runner

    monkeypatch.setattr(_cli, "skip_reason", lambda *a, **k: None)
    run = Runner(SimpleNamespace())
    run.stages = [
        {"label": "neuropixels", "stage": "preprocess", "status": "ok", "seconds": 90.0},
        {"label": "blackrock", "stage": "sort_with_kilosort", "status": "ok", "seconds": 30.0},
    ]
    lines = run.stage_table().splitlines()

    assert lines[0] == "stage times"
    assert "preprocess" in lines[1] and "1.5 min" in lines[1]
    assert lines[-1].split()[-3:] == ["total", "2.0", "min"]
