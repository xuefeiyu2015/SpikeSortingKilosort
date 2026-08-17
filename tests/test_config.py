"""Session/machine config: path resolution and input validation.

Session files carry the data paths; machine profiles carry only system facts.
Several tests below read the real tracked configs, so they fail if that split
is broken rather than merely if the loader is.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spikesorting import _config as cfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_demo_config_is_an_ordinary_session():
    # The demo is just a config: a binary path plus a couple of flags. Nothing in
    # the code branches on it being "the demo".
    session = cfg.load_session_config(CONFIG_DIR / "demo.yaml", "mac", CONFIG_DIR)
    assert session.session == "demo"
    assert session.aligns_systems is False   # no blackrock: block
    assert session.has_data("blackrock") is False
    assert session.sorts_blackrock is False
    assert session.sorts_neuropixels is True
    assert session.neuropixels.n_chan_bin == 385
    assert session.neuropixels.sample_rate == 30000.0


def test_demo_paths_are_derived_the_same_way_a_real_session_is():
    # The demo uses roots + monkey + session like anything else, rather than
    # stating its directory outright -- so it exercises the derivation instead of
    # side-stepping it, and the path it produces is the one the file references
    # as {neuropixels_dir}.
    session = cfg.load_session_config(CONFIG_DIR / "demo.yaml", "mac", CONFIG_DIR)

    assert session.neuropixels_dir == session.paths.dir_for("neuropixels")
    assert session.neuropixels_dir.name == session.session
    assert session.neuropixels_dir.parent.name == session.monkey

    binary = session.neuropixels.bin_file
    assert binary.name == "ZFM-02370_mini.imec0.ap.short.bin"
    assert binary.parent == session.neuropixels_dir
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
    session = cfg.load_session_config(CONFIG_DIR / "session_template.yaml", "windows_rig", CONFIG_DIR)
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
        cfg.load_session_config(CONFIG_DIR / "session_template.yaml", m, CONFIG_DIR)
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


def test_output_paths_are_derived_from_each_systems_directory(tmp_path):
    paths = cfg.OutputPaths(tmp_path / "br", tmp_path / "np")
    assert paths.sync_for("blackrock").parent == tmp_path / "br"
    assert paths.figures.parent == tmp_path / "br"
    paths.mkdirs()
    assert all(p.exists() for p in paths.all())


def test_only_the_reachable_directories_are_created(tmp_path):
    # A single-system session must not be asked for the other system's paths.
    paths = cfg.OutputPaths(neuropixels_dir=tmp_path / "np")
    paths.mkdirs()

    assert all(p.exists() for p in paths.all())
    assert not (tmp_path / "np" / "blackrock").exists()
    with pytest.raises(ValueError, match="blackrock_dir"):
        _ = paths.sorted_br


def _write_session(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "s.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_roots_declared_in_the_session_file_expand_in_its_paths(tmp_path):
    # The point of the block: two systems whose data lives on different shares,
    # each named once instead of being spelled out on every line below.
    path = _write_session(
        tmp_path,
        """
session: Athos_2026_08_13
roots:
  blackrock: "Z:/server/Monkey Athos/2026-08-13"
  neuropixels: "Y:/npx/Monkey Athos/2026-08-13"
  output: "Z:/server/sorted"
output_dir: "{output}/{session}"
blackrock:
  sync_file: "{blackrock}/NSP-Athos_001.ns5"
  spike_file: "{blackrock}/HUB-Athos_001.ns6"
neuropixels:
  run_dir: "{neuropixels}"
  run_name: Athos_2026_08_13
""",
    )

    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert str(session.blackrock.sync_file).startswith("Z:/server/Monkey Athos/2026-08-13")
    assert session.blackrock.sync_file.name == "NSP-Athos_001.ns5"
    # The two roots stay independent -- this is why one data_root is not enough.
    assert str(session.neuropixels.run_dir) == "Y:/npx/Monkey Athos/2026-08-13"
    assert str(session.output_root) == "Z:/server/sorted/Athos_2026_08_13"


def test_a_root_may_itself_contain_a_placeholder(tmp_path):
    # Roots are resolved before they are used, so {monkey} and {session} work
    # inside one -- for a share that is already organised by subject.
    path = _write_session(
        tmp_path,
        """
monkey: Monkey Athos
session: "2026-08-13"
roots:
  neuropixels: "Y:/npx/{monkey}/raw"
""",
    )

    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert str(session.neuropixels_dir) == "Y:/npx/Monkey Athos/raw/Monkey Athos/2026-08-13"


def test_a_misspelled_root_raises_instead_of_resolving_to_nothing(tmp_path):
    # The trap this closes: an unknown placeholder used to be left as a literal
    # and a known-but-unset one became "", turning "{root}/Monkey Athos" into
    # "/Monkey Athos" -- a wrong path that only fails much later, if at all.
    path = _write_session(
        tmp_path,
        """
session: s
roots:
  blackrock: "Z:/server"
output_dir: "{blackrock}/out"
blackrock:
  sync_file: "{blackrok}/NSP.ns5"
""",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "{blackrok}" in str(excinfo.value)


def test_a_stale_data_root_raises_now_that_no_profile_supplies_one(tmp_path):
    # The old two-layer placeholder, left behind in a session file. No machine
    # profile defines data_root any more, so this must fail loudly.
    path = _write_session(
        tmp_path,
        """
session: s
output_dir: "{data_root}/Monkey Athos"
neuropixels:
  bin_file: "/a/x.bin"
""",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "{data_root}" in str(excinfo.value)


def _flags(tmp_path: Path, body: str):
    """Both directories stated, and both systems declared unless `body` does."""
    lines = [
        "session: s",
        f"blackrock_dir: '{tmp_path / 'br'}'",
        f"neuropixels_dir: '{tmp_path / 'np'}'",
    ]
    if "neuropixels:" not in body:
        lines.append("neuropixels:\n  bin_file: '/a/x.bin'")
    if "blackrock:" not in body:
        lines.append("blackrock:\n  sync_file: '/b/y.ns5'")
    path = _write_session(tmp_path, "\n".join(lines) + "\n" + body)
    return cfg.load_session_config(path, "windows_rig", CONFIG_DIR)


_TWO_SYSTEMS = """
monkey: Monkey Athos
session: "2026-08-13"
roots:
  blackrock: "Z:/server"
  neuropixels: "Y:/npx"
blackrock:
  sync_file: "{blackrock_dir}/NSP-Athos_001.ns5"
neuropixels:
  run_dir: "{neuropixels_dir}"
  run_name: Athos_2026_08_13
"""


def test_a_data_dir_is_root_then_monkey_then_session(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.blackrock_dir) == "Z:/server/Monkey Athos/2026-08-13"
    # Each system resolves against its own root: the two recordings are written
    # by different machines and need not share a share.
    assert str(session.neuropixels_dir) == "Y:/npx/Monkey Athos/2026-08-13"


def test_the_derived_data_dirs_are_available_as_placeholders(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.blackrock.sync_file) == "Z:/server/Monkey Athos/2026-08-13/NSP-Athos_001.ns5"
    assert str(session.neuropixels.run_dir) == "Y:/npx/Monkey Athos/2026-08-13"


def test_sorted_results_land_beside_each_systems_own_recording(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.paths.sorted_br) == "Z:/server/Monkey Athos/2026-08-13/blackrock/kilosort4"
    assert str(session.paths.sorted_np) == "Y:/npx/Monkey Athos/2026-08-13/neuropixels/kilosort4"


def test_sync_and_lfp_follow_the_system_they_came_from(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)
    paths = session.paths

    assert str(paths.sync_for("blackrock")) == "Z:/server/Monkey Athos/2026-08-13/sync"
    assert str(paths.sync_for("neuropixels")) == "Y:/npx/Monkey Athos/2026-08-13/sync"
    # LFP is a Neuropixels product: Blackrock LFPs are saved separately by Central.
    assert str(paths.lfp) == "Y:/npx/Monkey Athos/2026-08-13/lfp"


def test_cross_system_output_goes_under_the_reference_timebase(tmp_path):
    # Blackrock is the reference timebase, so the alignment that maps everything
    # onto it belongs beside the Blackrock recording.
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.paths.aligned) == "Z:/server/Monkey Athos/2026-08-13/aligned"
    assert str(session.paths.figures) == "Z:/server/Monkey Athos/2026-08-13/figures"


def test_cross_system_output_falls_back_when_blackrock_never_recorded(tmp_path):
    # A Neuropixels-only session has no Blackrock dir to put figures in.
    session = cfg.load_session_config(
        _write_session(
            tmp_path,
            """
monkey: Monkey Demo
session: "2026-08-13"
roots:
  neuropixels: "Y:/npx"
neuropixels:
  run_dir: "{neuropixels_dir}"
  run_name: demo
""",
        ),
        "windows_rig",
        CONFIG_DIR,
    )

    assert session.blackrock_dir is None
    assert str(session.paths.figures) == "Y:/npx/Monkey Demo/2026-08-13/figures"


def test_a_data_dir_may_be_stated_outright(tmp_path):
    # Recordings that predate the monkey/date convention, and the demo, name the
    # directory instead of having it built.
    session = cfg.load_session_config(
        _write_session(
            tmp_path,
            """
session: demo
neuropixels_dir: "~/ephys/demo_data"
neuropixels:
  bin_file: "{neuropixels_dir}/x.bin"
""",
        ),
        "windows_rig",
        CONFIG_DIR,
    )

    assert session.neuropixels_dir == Path.home() / "ephys" / "demo_data"
    assert session.paths.sorted_np == Path.home() / "ephys/demo_data/neuropixels/kilosort4"


def test_each_system_can_name_its_own_probe_file(tmp_path):
    # Which array is in which monkey is a property of the session, so the map
    # belongs here rather than in a --cmp flag on one script.
    session = _flags(
        tmp_path,
        "blackrock:\n  probe_file: 'configs/probes/utah_athos_A.json'\n"
        "neuropixels:\n  probe_file: 'configs/probes/np1_nhp_long.json'\n",
    )

    # Written relative, resolved against the repo so it works from any cwd.
    assert session.blackrock.probe_file == cfg.REPO_ROOT / "configs/probes/utah_athos_A.json"
    assert session.neuropixels.probe_file == cfg.REPO_ROOT / "configs/probes/np1_nhp_long.json"


def test_a_utah_array_can_name_its_cmp_instead(tmp_path):
    session = _flags(tmp_path, "blackrock:\n  cmp_file: '/arrays/athos_A.cmp'\n")

    assert session.blackrock.cmp_file == Path("/arrays/athos_A.cmp")
    assert session.blackrock.probe_file is None


def test_probe_paths_expand_placeholders_like_any_other_path(tmp_path):
    session = _flags(tmp_path, "blackrock:\n  cmp_file: '{blackrock_dir}/arrayA.cmp'\n")

    assert str(session.blackrock.cmp_file) == str(tmp_path / "br" / "arrayA.cmp")


def test_no_probe_named_is_the_default(tmp_path):
    session = _flags(tmp_path, "")

    assert session.blackrock.probe_file is None
    assert session.blackrock.cmp_file is None
    assert session.neuropixels.probe_file is None


def test_a_named_probe_file_that_is_absent_is_reported(tmp_path):
    # Named but not there is a hard error: the alternative is discovering it after
    # the recording has been copied to the cache.
    session = _flags(
        tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '/nope/utah_A.json'\n"
    )

    assert any("probe_file" in p for p in session.missing_inputs())


def test_a_named_cmp_file_that_is_absent_is_reported(tmp_path):
    session = _flags(
        tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n  cmp_file: '/nope/array.cmp'\n"
    )

    assert any("cmp_file" in p for p in session.missing_inputs())


def test_a_probe_file_that_exists_is_not_a_problem(tmp_path):
    probe = tmp_path / "utah_A.json"
    probe.write_text("{}", encoding="utf-8")
    session = _flags(tmp_path, f"blackrock:\n  probe_file: '{probe}'\n")

    assert not any("probe" in p for p in session.missing_inputs())


def test_naming_no_map_at_all_is_not_a_blocking_problem(tmp_path):
    # It means placeholder geometry, which must warn loudly but still sort --
    # doctor reports it; missing_inputs is for files that are genuinely absent.
    session = _flags(tmp_path, "")

    assert not any("probe" in p or "cmp" in p for p in session.missing_inputs())


def test_missing_inputs_reports_absent_files():
    session = cfg.load_session_config(CONFIG_DIR / "session_template.yaml", "mac", CONFIG_DIR)
    problems = session.missing_inputs()
    assert problems
    assert any("does not exist" in p or "required" in p for p in problems)


def test_require_inputs_raises_with_every_problem_at_once():
    session = cfg.load_session_config(CONFIG_DIR / "session_template.yaml", "mac", CONFIG_DIR)
    with pytest.raises(FileNotFoundError) as excinfo:
        session.require_inputs()
    assert "not runnable" in str(excinfo.value)


def test_unknown_machine_raises():
    with pytest.raises(FileNotFoundError):
        cfg.load_machine("no_such_machine", CONFIG_DIR)


def test_tilde_in_machine_paths_is_expanded():
    machine = cfg.load_machine("mac", CONFIG_DIR)
    assert "~" not in str(machine.cache_dir)
    assert machine.cache_dir.is_absolute()


def test_a_duplicate_key_is_rejected_rather_than_silently_dropped(tmp_path):
    # PyYAML keeps the last occurrence and says nothing, so pasting a template on
    # top of an existing file silently discards half of it -- the whole first
    # `neuropixels:` block, say. That is a wrong sort, not a wrong path.
    path = tmp_path / "s.yaml"
    path.write_text(
        "session: s\n"
        f"neuropixels_dir: '{tmp_path}'\n"
        ""
        "neuropixels:\n  run_dir: '/a'\n  run_name: r\n"
        "neuropixels:\n  bin_file: '/b/x.bin'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "mac", CONFIG_DIR)

    assert "neuropixels" in str(excinfo.value)
    assert "duplicate" in str(excinfo.value).lower()


def test_a_duplicate_key_nested_in_a_block_is_also_rejected(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text(
        "session: s\n"
        f"neuropixels_dir: '{tmp_path}'\n"
        ""
        "neuropixels:\n  bin_file: '/a/x.bin'\n  bin_file: '/b/y.bin'\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="[Dd]uplicate"):
        cfg.load_session_config(path, "mac", CONFIG_DIR)


def test_probe_name_is_gone(tmp_path):
    # Kilosort ships no probe files -- its own API takes a path too -- and the
    # three it can download do not include NP 1.0 NHP. Naming a "probe type"
    # could only ever hand a real NHP session the standard 384-site layout:
    # plausible, wrong, and silent.
    assert not hasattr(cfg.NeuropixelsSpec(), "probe_name")

    session = _flags(tmp_path, "neuropixels:\n  probe_name: 'NeuroPix1_default.mat'\n")
    assert not hasattr(session.neuropixels, "probe_name")


def test_a_relative_probe_file_resolves_against_the_repo_not_the_cwd(tmp_path, monkeypatch):
    # Every example writes "configs/probes/x.json". Left cwd-relative that works
    # from the repo root and nowhere else, which is not "consistently used".
    session = _flags(tmp_path, "blackrock:\n  probe_file: 'configs/probes/np1_default.json'\n")

    assert session.blackrock.probe_file.is_absolute()
    assert session.blackrock.probe_file == cfg.REPO_ROOT / "configs/probes/np1_default.json"

    monkeypatch.chdir(tmp_path)  # a different cwd must not change the answer
    again = _flags(tmp_path, "blackrock:\n  probe_file: 'configs/probes/np1_default.json'\n")
    assert again.blackrock.probe_file == session.blackrock.probe_file


def test_an_absolute_probe_file_is_left_alone(tmp_path):
    probe = tmp_path / "utah_A.json"
    probe.write_text("{}", encoding="utf-8")
    session = _flags(tmp_path, f"blackrock:\n  probe_file: '{probe}'\n")

    assert session.blackrock.probe_file == probe


# ---------------------------------------------------------------------------
# has_data / aligns_systems are derived, not declared
# ---------------------------------------------------------------------------


def _declared(tmp_path, body):
    path = _write_session(
        tmp_path,
        f"session: s\nblackrock_dir: '{tmp_path / 'br'}'\n"
        f"neuropixels_dir: '{tmp_path / 'np'}'\n{body}",
    )
    return cfg.load_session_config(path, "windows_rig", CONFIG_DIR)


def test_declaring_a_systems_paths_is_what_says_it_recorded(tmp_path):
    both = _declared(
        tmp_path,
        "neuropixels:\n  bin_file: '/a/x.bin'\nblackrock:\n  sync_file: '/b/y.ns5'\n",
    )
    assert both.has_data("neuropixels") and both.has_data("blackrock")

    npx_only = _declared(tmp_path, "neuropixels:\n  bin_file: '/a/x.bin'\n")
    assert npx_only.has_data("neuropixels")
    assert not npx_only.has_data("blackrock")


def test_a_spikeglx_run_counts_as_declared_only_with_its_name(tmp_path):
    # run_dir alone cannot locate anything; find_run_files needs the run name.
    assert not _declared(tmp_path, "neuropixels:\n  run_dir: '/a'\n").has_data("neuropixels")
    assert _declared(
        tmp_path, "neuropixels:\n  run_dir: '/a'\n  run_name: r\n"
    ).has_data("neuropixels")


def test_blackrock_counts_as_declared_from_either_file(tmp_path):
    assert _declared(tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n").has_data("blackrock")
    assert _declared(tmp_path, "blackrock:\n  spike_file: '/b/y.ns6'\n").has_data("blackrock")


def test_an_unreachable_share_is_a_loud_error_not_a_silent_skip(tmp_path):
    # The case the whole design turns on. Files declared on a share that is not
    # mounted must still count as data, so the run fails saying what is missing
    # rather than reporting the system as absent and quietly skipping it.
    session = _declared(
        tmp_path,
        "neuropixels:\n  bin_file: '/Volumes/nope/x.bin'\n"
        "blackrock:\n  sync_file: '/Volumes/nope/y.ns5'\n",
    )

    assert session.has_data("neuropixels") and session.has_data("blackrock")

    problems = session.missing_inputs()
    assert any("bin_file does not exist" in p for p in problems)
    assert any("sync_file does not exist" in p for p in problems)


def test_alignment_is_derived_from_having_both_systems(tmp_path):
    both = _declared(
        tmp_path,
        "neuropixels:\n  bin_file: '/a/x.bin'\nblackrock:\n  sync_file: '/b/y.ns5'\n",
    )
    assert both.aligns_systems is True

    one = _declared(tmp_path, "neuropixels:\n  bin_file: '/a/x.bin'\n")
    assert one.aligns_systems is False


def test_a_blackrock_only_session_needs_no_sync_file(tmp_path):
    # With nothing to align to, the 1 Hz train is not needed. This was impossible
    # to express before: sync_file was demanded whenever Blackrock had data.
    spike = tmp_path / "HUB.ns6"
    spike.write_bytes(b"")
    session = _declared(tmp_path, f"blackrock:\n  spike_file: '{spike}'\n")

    assert session.has_data("blackrock")
    assert session.aligns_systems is False
    assert not any("sync_file" in p for p in session.missing_inputs())


def test_sorting_still_needs_asking_for_even_when_data_is_declared(tmp_path):
    session = _declared(
        tmp_path, "neuropixels:\n  bin_file: '/a/x.bin'\nkilosort_on_neuropixels: false\n"
    )

    assert session.has_data("neuropixels")
    assert session.sorts_neuropixels is False


@pytest.mark.parametrize(
    "key, value",
    [
        ("has_neuropixels_data", "true"),
        ("has_blackrock_data", "false"),
        ("skip_sync", "true"),
        ("skip_neuropixels", "true"),
        ("skip_blackrock", "true"),
        ("sorter", "kilosort4"),
    ],
)
def test_a_removed_key_raises_rather_than_being_ignored(tmp_path, key, value):
    # Session copies live on the rig and still set these. Ignoring them silently
    # would change what a run does without a word.
    path = _write_session(
        tmp_path,
        f"session: s\nneuropixels_dir: '{tmp_path}'\n"
        f"neuropixels:\n  bin_file: '/a/x.bin'\n{key}: {value}\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert key in str(excinfo.value)


# ---------------------------------------------------------------------------
# Kilosort settings, per system
# ---------------------------------------------------------------------------


def test_kilosort_settings_default_to_the_shared_block(tmp_path):
    session = _flags(tmp_path, "kilosort:\n  nblocks: 1\n  highpass_cutoff: 300\n")

    for system in ("neuropixels", "blackrock"):
        assert session.kilosort_for(system)["nblocks"] == 1
        assert session.kilosort_for(system)["highpass_cutoff"] == 300


def test_a_system_overrides_the_shared_kilosort_block_key_by_key(tmp_path):
    # The two systems want different answers -- a 400 um Utah array and a dense
    # probe are not the same problem -- but they share most of the settings.
    session = _flags(
        tmp_path,
        "kilosort:\n  nblocks: 1\n  highpass_cutoff: 300\n"
        "blackrock:\n  sync_file: '/b/y.ns5'\n  kilosort:\n    nblocks: 0\n",
    )

    assert session.kilosort_for("neuropixels")["nblocks"] == 1
    assert session.kilosort_for("blackrock")["nblocks"] == 0
    # overriding one key keeps the rest of the shared block
    assert session.kilosort_for("blackrock")["highpass_cutoff"] == 300


def test_a_system_can_carry_settings_of_its_own(tmp_path):
    # bad_channels is per-array by nature: Kilosort removes a list you supply and
    # has no detection of its own.
    session = _flags(
        tmp_path,
        "blackrock:\n  sync_file: '/b/y.ns5'\n  kilosort:\n    bad_channels: [191, 192]\n",
    )

    assert session.kilosort_for("blackrock")["bad_channels"] == [191, 192]
    assert session.kilosort_for("neuropixels") == {}


def test_a_system_replaces_a_shared_list_rather_than_extending_it(tmp_path):
    # The merge is one level deep, which session_template.yaml now states: a
    # per-system bad_channels is the whole list for that system, not an addition
    # to the shared one. Deep-merging would make [7] read as [7, 191, 192] and
    # quietly keep sorting channels the session said to drop.
    session = _flags(
        tmp_path,
        "kilosort:\n  bad_channels: [191, 192]\n"
        "blackrock:\n  sync_file: '/b/y.ns5'\n  kilosort:\n    bad_channels: [7]\n",
    )

    assert session.kilosort_for("blackrock")["bad_channels"] == [7]
    assert session.kilosort_for("neuropixels")["bad_channels"] == [191, 192]


def test_no_kilosort_block_means_kilosorts_own_defaults(tmp_path):
    session = _flags(tmp_path, "")

    assert session.kilosort_for("neuropixels") == {}
    assert session.kilosort_for("blackrock") == {}


def test_an_external_preprocess_block_is_refused(tmp_path):
    # Kilosort highpasses and subtracts the median across channels itself, on
    # every batch. Doing it again outside was work done twice, and unsaid.
    path = _write_session(
        tmp_path,
        f"session: s\nneuropixels_dir: '{tmp_path}'\n"
        "neuropixels:\n  bin_file: '/a/x.bin'\n"
        "preprocess:\n  apply: true\n  bandpass: [300, 6000]\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "preprocess" in str(excinfo.value)
    assert "kilosort" in str(excinfo.value)


def test_a_removed_key_nested_in_a_system_block_is_also_refused(tmp_path):
    # Per-system preprocess: blocks were a real thing. Checking only top-level
    # keys would drop one silently -- the failure this rejection exists to stop.
    path = _write_session(
        tmp_path,
        f"session: s\nneuropixels_dir: '{tmp_path}'\n"
        "neuropixels:\n  bin_file: '/a/x.bin'\n  preprocess:\n    apply: true\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "preprocess" in str(excinfo.value)
    assert "neuropixels" in str(excinfo.value)


def test_exclude_channels_is_refused(tmp_path):
    # Blackrock-only, and applied before the probe -- so it shifted every contact
    # after the excluded one. The probe already drops what it does not map, and
    # a broken electrode is kilosort.bad_channels, the same on both systems.
    path = _write_session(
        tmp_path,
        f"session: s\nblackrock_dir: '{tmp_path}'\n"
        "blackrock:\n  spike_file: '/b/y.ns6'\n  exclude_channels: ['129']\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "exclude_channels" in str(excinfo.value)
    assert "bad_channels" in str(excinfo.value)


def test_drift_correction_is_off_for_the_utah_array():
    """Kilosort's drift correction interpolates between neighbouring contacts.

    Its kernel is Gaussian with sig_interp = 20 um, so at the Utah array's 400 um
    pitch the weight to the nearest neighbour is ~1e-87: there is nothing to
    interpolate from. A nonzero shift estimate then scales the data toward zero
    rather than moving it, so the correction can only subtract signal. On a dense
    probe the same machinery is what makes drift tractable.
    """
    session = cfg.load_session_config(CONFIG_DIR / "session_template.yaml", "windows_rig", CONFIG_DIR)

    assert session.kilosort_for("blackrock")["nblocks"] == 0
    assert session.kilosort_for("neuropixels")["nblocks"] == 1
