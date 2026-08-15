"""Session/machine config: path resolution and input validation.

Session files carry the data paths; machine profiles carry only system facts.
Several tests below read the real tracked configs, so they fail if that split
is broken rather than merely if the loader is.
"""

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


def test_demo_paths_are_written_out_in_the_session_config():
    session = cfg.load_session(CONFIG_DIR / "demo.yaml", "mac", CONFIG_DIR)
    binary = session.neuropixels.bin_file
    assert binary.name == "ZFM-02370_mini.imec0.ap.short.bin"
    assert binary.parent.name == "demo_data"
    # "~/..." went through _as_path().expanduser() rather than a machine root.
    assert binary.is_absolute()
    assert session.output_root.is_absolute()


def _template_paths(session):
    return (
        session.blackrock.sync_file,
        session.blackrock.spike_file,
        session.neuropixels.run_dir,
        session.output_root,
    )


def test_template_leaves_no_unresolved_placeholders():
    # A stale {data_root} would not raise: _substitute only knows five names and
    # an unknown one is left alone, while a known-but-unset one becomes "" --
    # silently turning "{data_root}/Monkey Athos" into "/Monkey Athos".
    session = cfg.load_session(CONFIG_DIR / "session_template.yaml", "windows_rig", CONFIG_DIR)
    for path in _template_paths(session):
        assert "{" not in str(path)
        assert str(path).startswith("Z:")
    assert str(session.output_root).endswith(session.session)


def test_session_paths_no_longer_vary_by_machine():
    # The inverse of the old two-layer behaviour, and the point of the split:
    # data paths live in the session file, so one file resolves identically
    # everywhere. The machine profile only decides where temp.dat goes and
    # whether CatGT/TPrime exist.
    loaded = [
        cfg.load_session(CONFIG_DIR / "session_template.yaml", m, CONFIG_DIR)
        for m in ("mac", "hpc", "windows_rig")
    ]
    assert len({str(s.blackrock.sync_file) for s in loaded}) == 1
    assert len({str(s.output_root) for s in loaded}) == 1
    assert len({str(s.machine.cache_dir) for s in loaded}) == 3


def test_machine_profiles_carry_no_recording_paths():
    # The invariant the split establishes: machine profile = system facts only.
    # cache_dir stays -- a local SSD for temp.dat really is a property of the box.
    for name in ("mac", "hpc", "windows_rig"):
        profile = cfg.load_machine(name, CONFIG_DIR)
        assert profile.data_root is None, name
        assert profile.output_root is None, name
        assert profile.cache_dir is not None, name


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


def test_env_names_default_when_the_profile_declares_none():
    # hpc.yaml has no conda_envs block; the defaults must survive that untouched.
    machine = cfg.load_machine("hpc", CONFIG_DIR)
    assert machine.conda_envs == {}
    assert machine.env_location("sorting", "kilosort4") == cfg.EnvLocation("kilosort4")


def test_env_names_come_from_the_profile_when_declared():
    machine = cfg.load_machine("windows_rig", CONFIG_DIR)
    assert machine.env_location("sorting", "kilosort4").name == "kilosort4"
    assert machine.env_location("curation", "phy").name == "phy"


def test_shorthand_and_full_form_mean_the_same_thing(tmp_path):
    # Naming only the name is the common case, so `sorting: ks5` must work as a
    # shorthand for the mapping form -- otherwise the simple case reads badly.
    machines = tmp_path / "machines"
    machines.mkdir()
    (machines / "short.yaml").write_text("conda_envs:\n  sorting: ks5\n", encoding="utf-8")
    (machines / "full.yaml").write_text(
        "conda_envs:\n  sorting:\n    name: ks5\n", encoding="utf-8"
    )

    short = cfg.load_machine("short", tmp_path)
    full = cfg.load_machine("full", tmp_path)

    assert short.conda_envs == full.conda_envs == {"sorting": cfg.EnvLocation("ks5")}
    assert short.conda_envs["sorting"].path is None


def test_env_path_is_the_directory_holding_the_env(tmp_path):
    machines = tmp_path / "machines"
    machines.mkdir()
    (machines / "p.yaml").write_text(
        'conda_envs:\n  sorting:\n    name: ks5\n    path: "/shared/envs"\n',
        encoding="utf-8",
    )

    location = cfg.load_machine("p", tmp_path).env_location("sorting", "kilosort4")

    assert location.path == Path("/shared/envs")
    assert location.prefix == Path("/shared/envs/ks5")  # joined, not used as-is


def test_malformed_conda_envs_degrades_to_empty(tmp_path):
    # A scalar where a mapping belongs must not crash the load: the profile still
    # works with defaults, and doctor reports the problem instead.
    machines = tmp_path / "machines"
    machines.mkdir()
    (machines / "broken.yaml").write_text("conda_envs: kilosort4\n", encoding="utf-8")

    machine = cfg.load_machine("broken", tmp_path)

    assert machine.conda_envs == {}
    assert machine.env_location("sorting", "kilosort4").name == "kilosort4"


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
        f"session: s\nskip_blackrock: true\noutput_dir: '{tmp_path / 'out'}'\n"
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
