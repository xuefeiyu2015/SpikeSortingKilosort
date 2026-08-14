"""Session/machine config: placeholder resolution and input validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from spikesorting import config as cfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_demo_config_is_an_ordinary_session():
    # The demo is just a config: a binary path plus skip flags. Nothing in the
    # code branches on it being "the demo".
    session = cfg.load_session(CONFIG_DIR / "demo.yaml", "mac", CONFIG_DIR)
    assert session.session == "demo"
    assert session.skip_sync is True
    assert session.skip_blackrock is True
    assert session.neuropixels.n_chan_bin == 385
    assert session.neuropixels.sample_rate == 30000.0


def test_demo_paths_resolve_under_the_downloads_dir():
    session = cfg.load_session(CONFIG_DIR / "demo.yaml", "mac", CONFIG_DIR)
    downloads = cfg.downloads_dir()
    assert str(session.neuropixels.bin_file).startswith(str(downloads))
    assert "{downloads}" not in str(session.output_root)


def test_template_resolves_machine_placeholders():
    session = cfg.load_session(CONFIG_DIR / "session_template.yaml", "windows_rig", CONFIG_DIR)
    machine = cfg.load_machine("windows_rig", CONFIG_DIR)
    assert str(session.blackrock.sync_file).startswith(str(machine.data_root))
    assert str(session.output_root).endswith(session.session)
    assert "{data_root}" not in str(session.blackrock.spike_file)


def test_same_session_resolves_differently_per_machine():
    windows = cfg.load_session(CONFIG_DIR / "session_template.yaml", "windows_rig", CONFIG_DIR)
    hpc = cfg.load_session(CONFIG_DIR / "session_template.yaml", "hpc", CONFIG_DIR)
    assert windows.blackrock.sync_file != hpc.blackrock.sync_file
    assert windows.session == hpc.session


def test_hpc_profile_has_no_command_line_tools():
    # The sorting path must not depend on CatGT/TPrime; the HPC profile is what
    # proves the code path is exercised without them.
    machine = cfg.load_machine("hpc", CONFIG_DIR)
    assert machine.has_catgt is False
    assert machine.has_tprime is False
    assert machine.cache_dir is not None


def test_windows_profile_has_both_tools():
    machine = cfg.load_machine("windows_rig", CONFIG_DIR)
    assert machine.has_catgt and machine.has_tprime


def test_output_paths_are_derived_from_the_root(tmp_path):
    paths = cfg.OutputPaths(tmp_path / "session")
    assert paths.sync.name == "sync"
    assert paths.figures.parent == tmp_path / "session"
    paths.mkdirs()
    assert all(p.exists() for p in paths.all())


def test_missing_inputs_reports_absent_files():
    session = cfg.load_session(CONFIG_DIR / "session_template.yaml", "mac", CONFIG_DIR)
    problems = session.missing_inputs()
    assert problems
    assert any("does not exist" in p or "required" in p for p in problems)


def test_require_inputs_raises_with_every_problem_at_once():
    session = cfg.load_session(CONFIG_DIR / "session_template.yaml", "mac", CONFIG_DIR)
    with pytest.raises(FileNotFoundError) as excinfo:
        session.require_inputs()
    assert "not runnable" in str(excinfo.value)


def test_skip_blackrock_suppresses_blackrock_checks(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "session: s\nskip_blackrock: true\noutput_dir: '{output_root}/s'\n"
        "neuropixels:\n  bin_file: '/nope/missing.bin'\n",
        encoding="utf-8",
    )
    session = cfg.load_session(path, "mac", CONFIG_DIR)
    problems = session.missing_inputs()
    assert not any("blackrock" in p for p in problems)
    assert any("bin_file" in p for p in problems)


def test_unknown_machine_raises():
    with pytest.raises(FileNotFoundError):
        cfg.load_machine("no_such_machine", CONFIG_DIR)


def test_tilde_in_machine_paths_is_expanded():
    machine = cfg.load_machine("mac", CONFIG_DIR)
    assert "~" not in str(machine.cache_dir)
    assert machine.cache_dir.is_absolute()
