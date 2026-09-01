"""Session/machine config: path resolution and input validation.

Session files carry the data paths; machine profiles carry only system facts.
Several tests below read the real tracked configs, so they fail if that split
is broken rather than merely if the loader is.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from spikesorting import _config as cfg

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_demo_config_is_an_ordinary_session():
    # The demo is just a config: a binary path plus a couple of flags. Nothing in
    # the code branches on it being "the demo".
    session = cfg.load_session_config(CONFIG_DIR / "session_demo_1probe.yaml", "mac", CONFIG_DIR)
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
    session = cfg.load_session_config(CONFIG_DIR / "session_demo_1probe.yaml", "mac", CONFIG_DIR)

    assert session.neuropixels_dir == session.paths.dir_for("neuropixels")
    assert session.neuropixels_dir.name == session.session
    assert session.neuropixels_dir.parent.name == session.monkey

    binary = session.neuropixels.bin_file
    assert binary.name == "ZFM-02370_mini.imec0.ap.short.bin"
    assert binary.parent == session.neuropixels_dir
    # Rooted at the share, with nothing left unexpanded. Not `is_absolute()`:
    # the roots are Windows drive letters, which POSIX reads as relative, and
    # these tests run wherever the pure-compute suite runs.
    assert "{" not in str(binary)
    assert str(binary).startswith("Z:/")
    assert str(session.output_root).startswith("Z:/")


def _template_paths(session):
    return (
        session.blackrock.sync_file,
        session.blackrock.spike_file,
        session.neuropixels.bin_file,
        session.output_root,
    )


def test_template_leaves_no_unresolved_placeholders():
    # A stale {data_root} would not raise: _substitute only knows five names and
    # an unknown one is left alone, while a known-but-unset one becomes "" --
    # silently turning "{data_root}/Monkey Athos" into "/Monkey Athos".
    session = cfg.load_session_config(CONFIG_DIR / "session_template_utah_probe.yaml", "windows_rig", CONFIG_DIR)
    for path in _template_paths(session):
        assert "{" not in str(path)
        assert str(path).startswith("Z:")
    assert str(session.output_root).endswith(session.session)


_FENCE = "---------- defaults"


def _keys_below_the_fence(text: str, block: str) -> list[str]:
    """Keys inside ``block:`` that sit after its defaults fence, in file order."""
    lines = text.splitlines()
    fenced, keys = False, []
    for line in lines[lines.index(f"{block}:") + 1 :]:
        if line and not line.startswith((" ", "#")):
            break                                    # the next top-level key
        if _FENCE in line:
            fenced = True
        elif fenced:
            match = re.match(r"  (\w+):", line)      # two spaces: not a nested key
            if match:
                keys.append(match.group(1))
    return keys


def test_the_tracked_configs_state_the_code_defaults_below_their_fences():
    # The fence claims everything under it can be left alone. That claim rots the
    # moment someone tunes a threshold in place, and the file would still read as
    # a stock session. Every key below a fence must therefore equal the dataclass
    # default; a value chosen for this recording belongs above it.
    stock = {"blackrock": cfg.BlackrockSpec(), "neuropixels": cfg.NeuropixelsSpec()}

    for name, machine in (
        ("session_template_utah_probe.yaml", "windows_rig"),
        ("session_template_1probe.yaml", "windows_rig"),
        ("session_template_utah_only.yaml", "windows_rig"),
        ("session_demo_1probe.yaml", "mac"),
    ):
        text = (CONFIG_DIR / name).read_text()
        session = cfg.load_session_config(CONFIG_DIR / name, machine, CONFIG_DIR)
        fenced = 0
        for block, default in stock.items():
            if f"\n{block}:" not in text:
                continue                             # that system did not record
            keys = _keys_below_the_fence(text, block)
            assert keys, f"{name}: {block}: has no defaults fence"
            fenced += 1
            for key in keys:
                assert getattr(getattr(session, block), key) == getattr(default, key), (
                    f"{name}: {block}.{key} is not a default -- move it above the fence"
                )
        assert fenced, f"{name}: no defaults fence at all"


def test_the_one_probe_template_differs_only_by_the_missing_array():
    # Two near-identical tracked files drift: a threshold gets tuned in the one
    # someone happened to open. Pinning the difference means the drift fails here
    # instead of on the rig, on whichever template was copied that day.
    #
    # This one records the spikes on Neuropixels and keeps Blackrock only for the
    # pulses they are aligned against -- so it must be the utah_probe template
    # minus the array, and nothing else.
    one, none = (
        cfg.load_session_config(CONFIG_DIR / name, "windows_rig", CONFIG_DIR)
        for name in ("session_template_utah_probe.yaml", "session_template_1probe.yaml")
    )

    # Blackrock still recorded, and is still the timebase everything maps onto.
    assert none.blackrock.sync_file == one.blackrock.sync_file
    assert none.has_data("blackrock") and none.aligns_systems

    # There is no array, so: nothing to sort, and no channel map to name.
    assert none.blackrock.spike_file is None
    assert none.blackrock.probe_file is None and none.blackrock.cmp_file is None
    assert none.sorts_blackrock is False
    # ...and the file does not say so: the missing spike_file is what says it, so
    # adding one later starts sorting with nothing else to remember.
    assert none.kilosort_on_blackrock is True

    # Neuropixels is untouched, including its Kilosort block.
    assert none.neuropixels == one.neuropixels
    assert none.kilosort_by_system["neuropixels"] == one.kilosort_by_system["neuropixels"]
    assert "blackrock" not in none.kilosort_by_system   # no array, no settings for one

    stated = {"blackrock", "kilosort_by_system"}
    differing = [
        f
        for f in one.__dataclass_fields__
        if f not in stated and getattr(one, f) != getattr(none, f)
    ]
    assert differing == [], differing
    assert cfg.replace(one.blackrock, spike_file=None, probe_file=None) == none.blackrock


def test_the_utah_only_template_differs_only_by_the_missing_probe():
    # A Utah array and nothing else, so there is no second clock and nothing to
    # align. Same pinning: the utah_probe template minus the Neuropixels half.
    one, utah = (
        cfg.load_session_config(CONFIG_DIR / name, "windows_rig", CONFIG_DIR)
        for name in ("session_template_utah_probe.yaml", "session_template_utah_only.yaml")
    )

    # No neuropixels block at all -- which is how a session says it did not record.
    assert utah.neuropixels == cfg.NeuropixelsSpec()
    assert not utah.has_data("neuropixels")
    assert utah.neuropixels_dir is None

    # So there is nothing to align to, and the cross-system stages skip.
    assert utah.aligns_systems is False
    # ...and the time map has nowhere but the Blackrock folder to live, which is
    # where it would have gone anyway.
    assert utah.paths.aligned == utah.paths.sorted_for("blackrock")

    # The array half is untouched.
    assert utah.blackrock == one.blackrock
    assert utah.sorts_blackrock is True

    stated = {"neuropixels", "neuropixels_dir", "kilosort_on_neuropixels", "kilosort_by_system"}
    differing = [
        f
        for f in one.__dataclass_fields__
        if f not in stated and getattr(one, f) != getattr(utah, f)
    ]
    assert differing == [], differing


def test_two_probes_is_two_session_files_not_a_probes_list():
    # The 2probes template is gone: a probe list cannot exist any more, because a
    # session names one binary. The practice configs are the worked example, and
    # what matters is that two of them cannot collide.
    a, b = (
        cfg.load_session_config(CONFIG_DIR / name, "windows_rig", CONFIG_DIR)
        for name in ("practise_20210819.yaml", "practise_20210825.yaml")
    )

    assert a.neuropixels.bin_file != b.neuropixels.bin_file
    assert a.paths.sorted_for("neuropixels") != b.paths.sorted_for("neuropixels")
    # Each writes inside its own recording's folder, so nothing is shared.
    assert a.paths.sorted_for("neuropixels").parent == a.neuropixels_dir
    assert b.paths.sorted_for("neuropixels").parent == b.neuropixels_dir


def test_the_practice_configs_hand_catgt_the_right_run():
    # Nothing in those files states run_dir/run_name/gate/trigger. All four come
    # from the bin_file name -- including a run name that itself contains "_t1229".
    from spikesorting._io import spikeglx

    session = cfg.load_session_config(
        CONFIG_DIR / "practise_20210825.yaml", "windows_rig", CONFIG_DIR
    )
    layout = spikeglx.run_layout(session.neuropixels.bin_file)

    assert layout.run_name == "Tank_20210825_t1229_d10500_L11_exp"
    assert (layout.gate, layout.trigger, layout.probe) == (0, 0, 0)
    assert layout.directory.name == "practise spike sorting"


def test_output_paths_are_derived_from_each_systems_directory(tmp_path):
    paths = cfg.OutputPaths(tmp_path / "br", tmp_path / "np")
    assert paths.sync_for("blackrock").parent == tmp_path / "br"
    assert paths.aligned.parent == tmp_path / "br"
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


def test_everything_from_one_system_lands_in_one_folder(tmp_path):
    # The flat layout: <system>_dir/<sorter>/ holds the sorting, that system's
    # sync edges and the export bundle. There is no system or probe level above
    # it, because the directory the session names already identifies the recording.
    paths = cfg.OutputPaths(tmp_path / "br", tmp_path / "np")

    assert paths.sorted_for("neuropixels") == tmp_path / "np" / cfg.SORTER_NAME
    assert paths.sorted_for("blackrock") == tmp_path / "br" / cfg.SORTER_NAME
    # Edges share the sorting's folder, and are computable before it exists.
    assert paths.sync_for("neuropixels") == paths.sorted_for("neuropixels")
    assert paths.export_for("neuropixels") == paths.sorted_for("neuropixels") / "export"
    assert paths.figures_for("neuropixels").parts[-3:] == (
        cfg.SORTER_NAME, "export", "figures",
    )
    # No "imec0" anywhere: two probes are two sessions with two directories.
    assert "imec0" not in str(paths.sorted_for("neuropixels"))


def test_the_time_map_goes_under_the_reference_timebase(tmp_path):
    # Cross-system output is owned by neither sorting, so it goes to Blackrock's
    # folder -- Blackrock is the axis everything is mapped onto.
    paths = cfg.OutputPaths(tmp_path / "br", tmp_path / "np")
    assert paths.aligned == paths.sorted_for("blackrock")

    # ...and falls back to the only system that recorded, so a Neuropixels-only
    # session still has somewhere to write.
    npx_only = cfg.OutputPaths(neuropixels_dir=tmp_path / "np")
    assert npx_only.aligned == npx_only.sorted_for("neuropixels")


def test_two_systems_may_not_share_a_directory(tmp_path):
    # Nothing in an output path names the system any more -- the directory *is*
    # the identity -- so both systems pointed at one folder would write one
    # sorting on top of the other. Easy to hit by accident: <root>/<monkey>/
    # <session> is the same string for both whenever the two roots agree.
    path = _write_session(
        tmp_path,
        """
monkey: Monkey Athos
session: "2026-08-13"
roots:
  blackrock: "Z:/"
  neuropixels: "Z:/"
blackrock:
  sync_file: "{blackrock_dir}/NSP-Athos_001.ns5"
neuropixels:
  bin_file: "{neuropixels_dir}/Athos_g0_t0.imec0.ap.bin"
""",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert "same directory" in message
    assert "on top of each other" in message
    assert "imec" in message                    # names the fix


def test_one_system_alone_may_sit_anywhere(tmp_path):
    # The check is about a *collision*, so it must not fire on a session where
    # only one system recorded -- there is nothing to collide with.
    session = cfg.load_session_config(
        _write_session(
            tmp_path,
            """
monkey: Monkey Demo
session: "2026-08-13"
roots:
  neuropixels: "Z:/"
neuropixels:
  bin_file: "{neuropixels_dir}/demo_g0_t0.imec0.ap.bin"
""",
        ),
        "windows_rig",
        CONFIG_DIR,
    )

    assert session.blackrock_dir is None
    assert session.paths.sorted_for("neuropixels").name == cfg.SORTER_NAME


def test_two_sessions_pointed_at_one_directory_would_share_a_folder(tmp_path):
    # Why each recording needs its own <system>_dir: output paths carry no run
    # name and no session, so two sessions sharing a directory share every output.
    # _publish copies with dirs_exist_ok, so the second sort would merge into the
    # first rather than replace it.
    shared = cfg.OutputPaths(neuropixels_dir=tmp_path / "practise")
    separate = cfg.OutputPaths(neuropixels_dir=tmp_path / "practise" / "run_a_g0_imec0")

    assert shared.sorted_for("neuropixels") != separate.sorted_for("neuropixels")
    assert separate.sorted_for("neuropixels").is_relative_to(tmp_path / "practise")


def test_mkdirs_creates_both_systems_folders(tmp_path):
    paths = cfg.OutputPaths(tmp_path / "br", tmp_path / "np")
    paths.mkdirs()

    for system in ("neuropixels", "blackrock"):
        assert paths.sorted_for(system).exists()
        assert paths.export_for(system).exists()
    assert all(p.exists() for p in paths.all())


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
blackrock_dir: "{blackrock}"
neuropixels_dir: "{neuropixels}"
blackrock:
  sync_file: "{blackrock_dir}/NSP-Athos_001.ns5"
  spike_file: "{blackrock_dir}/HUB-Athos_001.ns6"
neuropixels:
  bin_file: "{neuropixels_dir}/Athos_g0_t0.imec0.ap.bin"
""",
    )

    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert str(session.blackrock.sync_file).startswith("Z:/server/Monkey Athos/2026-08-13")
    assert session.blackrock.sync_file.name == "NSP-Athos_001.ns5"
    # The two roots stay independent -- this is why one data_root is not enough,
    # and why the two systems do not collide in the flat output layout.
    assert str(session.neuropixels.bin_file.parent) == "Y:/npx/Monkey Athos/2026-08-13"
    # Cross-system output goes under the reference timebase's directory.
    assert str(session.output_root) == "Z:/server/Monkey Athos/2026-08-13"


def test_the_legacy_output_dir_still_serves_a_single_system(tmp_path):
    # `output_dir:` predates per-system directories and names one tree. That is
    # still enough for a session where only one system recorded -- and it is now
    # only enough for that, since two systems sharing one directory collide.
    session = cfg.load_session_config(
        _write_session(
            tmp_path,
            """
session: Athos_2026_08_13
roots:
  output: "Z:/server/sorted"
output_dir: "{output}/{session}"
neuropixels:
  bin_file: "Y:/npx/Athos_g0_t0.imec0.ap.bin"
""",
        ),
        "windows_rig",
        CONFIG_DIR,
    )

    assert str(session.output_root) == "Z:/server/sorted/Athos_2026_08_13"
    assert session.blackrock_dir is None


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
  bin_file: "{neuropixels_dir}/Athos_2026_08_13_g0_t0.imec0.ap.bin"
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
    assert (
        str(session.neuropixels.bin_file)
        == "Y:/npx/Monkey Athos/2026-08-13/Athos_2026_08_13_g0_t0.imec0.ap.bin"
    )


def test_sorted_results_land_beside_each_systems_own_recording(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.paths.sorted_br) == "Z:/server/Monkey Athos/2026-08-13/kilosort4"
    assert str(session.paths.sorted_np) == "Y:/npx/Monkey Athos/2026-08-13/kilosort4"


def test_sync_edges_follow_the_system_they_came_from(tmp_path):
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)
    paths = session.paths

    # Each system's edges sit in that system's own folder, beside its sorting.
    assert str(paths.sync_for("blackrock")) == "Z:/server/Monkey Athos/2026-08-13/kilosort4"
    assert str(paths.sync_for("neuropixels")) == "Y:/npx/Monkey Athos/2026-08-13/kilosort4"
    # The LF band is a Neuropixels product and goes in its export bundle;
    # Blackrock LFPs are saved separately by Central.
    assert str(paths.export_for("neuropixels")) == (
        "Y:/npx/Monkey Athos/2026-08-13/kilosort4/export"
    )


def test_cross_system_output_goes_under_the_reference_timebase(tmp_path):
    # Blackrock is the reference timebase, so the alignment that maps everything
    # onto it belongs beside the Blackrock recording.
    session = cfg.load_session_config(_write_session(tmp_path, _TWO_SYSTEMS), "windows_rig", CONFIG_DIR)

    assert str(session.paths.aligned) == "Z:/server/Monkey Athos/2026-08-13/kilosort4"


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
  bin_file: "{neuropixels_dir}/demo_g0_t0.imec0.ap.bin"
""",
        ),
        "windows_rig",
        CONFIG_DIR,
    )

    assert session.blackrock_dir is None
    assert str(session.paths.aligned) == "Y:/npx/Monkey Demo/2026-08-13/kilosort4"


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
    assert session.paths.sorted_np == Path.home() / "ephys/demo_data/kilosort4"


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


def test_naming_no_map_at_all_is_not_a_missing_input(tmp_path):
    # Sorting a Utah array without one does fail -- setup_probe raises and
    # check_env reports it -- but a map that was never named is an unmade
    # decision, not an absent file, and this function reports absent files. It is
    # also not needed at all by a session that extracts without sorting.
    session = _flags(tmp_path, "")

    assert not any("probe" in p or "cmp" in p for p in session.missing_inputs())


def test_missing_inputs_reports_absent_files():
    session = cfg.load_session_config(CONFIG_DIR / "session_template_utah_probe.yaml", "mac", CONFIG_DIR)
    problems = session.missing_inputs()
    assert problems
    assert any("does not exist" in p or "required" in p for p in problems)


def test_require_inputs_raises_with_every_problem_at_once():
    session = cfg.load_session_config(CONFIG_DIR / "session_template_utah_probe.yaml", "mac", CONFIG_DIR)
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


def test_neuropixels_is_declared_by_naming_its_binary(tmp_path):
    # One key, and it is the file itself. Everything a run folder used to be
    # needed for is read back off this name -- see the run_layout tests.
    assert not _declared(tmp_path, "neuropixels:\n  sync_bit: 6\n").has_data("neuropixels")
    assert _declared(
        tmp_path, "neuropixels:\n  bin_file: '/a/run_g0_t0.imec0.ap.bin'\n"
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
    # The merge is one level deep, which session_template_utah_probe.yaml now states: a
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


@pytest.mark.parametrize(
    "key, body, expected",
    [
        ("run_dir", "neuropixels:\n  run_dir: '/npx'\n", "bin_file"),
        ("run_name", "neuropixels:\n  run_name: 'r'\n", "bin_file"),
        ("gate", "neuropixels:\n  gate: 0\n", "bin_file"),
        ("trigger", "neuropixels:\n  trigger: 0\n", "bin_file"),
        ("probes", "neuropixels:\n  probes: [0, 1]\n", "two session files"),
        ("by_probe", "neuropixels:\n  by_probe:\n    1: {}\n", "own session file"),
    ],
)
def test_the_removed_run_model_keys_raise_by_name(tmp_path, key, body, expected):
    # These sit in session copies on the rig. Ignoring one would change what a run
    # does without a word, so each raises and says what replaces it.
    path = _write_session(
        tmp_path, f"session: s\nneuropixels_dir: '{tmp_path}'\n" + body
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert key in message
    assert expected in message


def test_catgt_settings_survive_the_removal_of_the_run_model(tmp_path):
    # sync_pulse_ms and the burst_* keys are CatGT arguments, and CatGT still
    # runs -- only the five path arguments became derived. Removing these too
    # would have quietly disabled the -xd/-xa extractions.
    session = _flags(
        tmp_path,
        "neuropixels:\n  bin_file: '/a/run_g0_t0.imec0.ap.bin'\n"
        "  sync_pulse_ms: 250\n  burst_word: 2\n"
        "  burst_threshold_v: [2.0, 0.5]\n  burst_pulse_ms: 10\n",
    )

    assert session.neuropixels.sync_pulse_ms == 250
    assert session.neuropixels.burst_word == 2
    assert session.neuropixels.burst_threshold_v == (2.0, 0.5)
    assert session.neuropixels.burst_pulse_ms == 10


def test_the_lf_band_can_be_named_when_it_is_not_beside_the_ap_file(tmp_path):
    session = _flags(
        tmp_path,
        "neuropixels:\n  bin_file: '/a/run_g0_t0.imec0.ap.bin'\n"
        "  lf_file: '/elsewhere/run_g0_t0.imec0.lf.bin'\n",
    )

    assert session.neuropixels.lf_file == Path("/elsewhere/run_g0_t0.imec0.lf.bin")
    # Unset means "no LF band to export", not a missing input.
    assert _flags(tmp_path, "").neuropixels.lf_file is None


def test_a_named_lf_file_that_is_absent_is_reported(tmp_path):
    session = _flags(
        tmp_path,
        "neuropixels:\n  bin_file: '/a/x.bin'\n  lf_file: '/nope/y.lf.bin'\n",
    )

    assert any("lf_file" in p for p in session.missing_inputs())


def test_kilosort_settings_merge_shared_then_system(tmp_path):
    # Two layers now, not three: the probe layer went with the probe dimension.
    session = _flags(
        tmp_path,
        "kilosort:\n  highpass_cutoff: 300\n"
        "neuropixels:\n  bin_file: '/a/x.bin'\n"
        "  kilosort:\n    bad_channels: [17, 203]\n    nblocks: 1\n",
    )

    settings = session.kilosort_for("neuropixels")
    assert settings["highpass_cutoff"] == 300      # shared
    assert settings["nblocks"] == 1                # per system
    assert settings["bad_channels"] == [17, 203]
    # ...and the other system keeps the shared block alone.
    assert session.kilosort_for("blackrock") == {"highpass_cutoff": 300}


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
    session = cfg.load_session_config(CONFIG_DIR / "session_template_utah_probe.yaml", "windows_rig", CONFIG_DIR)

    assert session.kilosort_for("blackrock")["nblocks"] == 0
    assert session.kilosort_for("neuropixels")["nblocks"] == 1


def _load_session(tmp_path: Path, body: str):
    """A minimal session, loaded. Most waveform tests need only a system block."""
    return cfg.load_session_config(
        _write_session(
            tmp_path,
            "session: s\nblackrock_dir: '/b'\nneuropixels_dir: '/n'\n" + body,
        ),
        "windows_rig",
        CONFIG_DIR,
    )


def test_each_system_states_its_own_waveform_band(tmp_path):
    # The one waveform setting that is not the same everywhere. .ns6 is the
    # broadband 30 kHz group, so its mean has to be filtered to be the signal
    # Kilosort sorted; a Neuropixels AP binary arrives high-passed from the
    # probe, and filtering it again would cascade a second rolloff onto the
    # first. Both are recording-time settings, so both are stated, not derived.
    session = _load_session(
        tmp_path,
        "blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '/b/HUB.ns6'\n"
        "neuropixels:\n  bin_file: '/n/x.bin'\n",
    )
    assert session.waveforms_for("blackrock").highpass_hz == 300.0
    assert session.waveforms_for("neuropixels").highpass_hz is None

    # ...and everything else about the two matches, so the band is the only
    # thing that differs by system rather than by taste.
    br = cfg.replace(session.waveforms_for("blackrock"), highpass_hz=None)
    assert br == session.waveforms_for("neuropixels")


def test_a_waveforms_block_changes_only_the_keys_it_names(tmp_path):
    # Stating one key must not silently reset the other five to the field
    # defaults -- which for Neuropixels would switch its filter back on.
    session = _load_session(
        tmp_path,
        "neuropixels:\n  bin_file: '/n/x.bin'\n  waveforms:\n    max_spikes: 50\n",
    )
    wf = session.waveforms_for("neuropixels")
    assert wf.max_spikes == 50
    assert wf.highpass_hz is None                  # not reset to the 300 default
    assert wf.window_ms == 2.0


def test_null_is_how_a_session_asks_for_every_spike(tmp_path):
    session = _load_session(
        tmp_path,
        "blackrock:\n  spike_file: '/b/HUB.ns6'\n  waveforms:\n    max_spikes: null\n",
    )
    assert session.waveforms_for("blackrock").max_spikes is None


def test_the_renamed_waveform_keys_raise_and_name_their_replacement(tmp_path):
    # Session copies live on the rig. Ignoring a key someone tuned there would
    # change what a run measured without a word, so each is refused by name.
    for old, new in (
        ("waveform_ms: 3.0", "window_ms"),
        ("export_waveforms: true", "export_snippets"),
    ):
        with pytest.raises(ValueError, match=new):
            _load_session(tmp_path, f"{old}\nblackrock:\n  spike_file: '/b/x.ns6'\n")


def test_a_flag_override_reaches_both_systems(tmp_path):
    # --export-waveforms says what this run should do, not which band a system
    # recorded, so it applies to every stream rather than to one.
    session = _load_session(
        tmp_path,
        "blackrock:\n  spike_file: '/b/HUB.ns6'\n"
        "neuropixels:\n  bin_file: '/n/x.bin'\n",
    )
    assert not session.waveforms_for("blackrock").export_snippets

    overridden = cfg.with_overrides(session, waveforms={"export_snippets": True})
    assert overridden.waveforms_for("blackrock").export_snippets
    assert overridden.waveforms_for("neuropixels").export_snippets
    # ...and it changes nothing else, the band included.
    assert overridden.waveforms_for("neuropixels").highpass_hz is None
