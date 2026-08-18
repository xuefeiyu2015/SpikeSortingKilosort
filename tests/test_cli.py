"""The drivers' shared argument handling.

The rule this file exists to pin: **every setting has a home in the session
file, and a flag only overrides it for one run.** A setting reachable only from
a command line is one nobody can read back off the session six months later, so
``flag_overrides`` maps each flag onto the session key it replaces rather than
being read directly by the loops.
"""

from __future__ import annotations

import argparse

from _cli import flag_overrides


def _args(**kwargs) -> argparse.Namespace:
    """The flags a driver would parse, defaulting to "not passed"."""
    defaults = {
        "system": None,
        "export_lfp": False,
        "lfp_decimate": None,
        "groups": None,
        "no_figures": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_no_flags_means_the_session_decides_everything():
    assert flag_overrides(_args()) == {}


def test_each_flag_names_the_session_key_it_replaces():
    assert flag_overrides(_args(export_lfp=True)) == {"export_lfp": True}
    assert flag_overrides(_args(groups=["good"])) == {"export_groups": ("good",)}
    assert flag_overrides(_args(no_figures=True)) == {"export_figures": False}


def test_asking_for_a_stride_asks_for_the_export():
    # --lfp-decimate 2 with export_lfp still false would otherwise set the stride
    # for an export that does not happen, and report nothing.
    assert flag_overrides(_args(lfp_decimate=2)) == {"export_lfp": True, "lfp_decimate": 2}


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
        f"session: s\nneuropixels_dir: '{tmp_path / 'np'}'\n"
        "export_lfp: false\nneuropixels:\n  bin_file: '/a/x.bin'\n",
        encoding="utf-8",
    )
    common = {"config": str(session), "machine": "mac", "probe": None, "skip_checks": True}

    assert load(_args(**common), require_inputs=False).export_lfp is False

    overridden = load(_args(**common, lfp_decimate=2), require_inputs=False)
    assert overridden.export_lfp is True
    assert overridden.lfp_decimate == 2

    # ...and the session file still says what the session wants.
    assert "export_lfp: false" in session.read_text(encoding="utf-8")
