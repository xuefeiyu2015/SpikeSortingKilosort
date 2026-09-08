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
    assert session.session == "session_demo_1probe"   # the config's own filename
    assert session.aligns_systems is False   # no blackrock: block
    assert session.has_data("blackrock") is False
    assert session.sorts_blackrock is False
    assert session.sorts_neuropixels is True
    assert session.neuropixels.n_chan_bin == 385
    assert session.neuropixels.sample_rate == 30000.0


def test_demo_paths_are_stated_outright_like_any_other_session():
    # The demo names its directory the same way a real session does -- there is
    # no derivation left to exercise, and the path it produces is the one the
    # file references below as {neuropixels_dir}.
    session = cfg.load_session_config(CONFIG_DIR / "session_demo_1probe.yaml", "mac", CONFIG_DIR)

    assert session.neuropixels_dir == session.paths.dir_for("neuropixels")

    binary = session.neuropixels.bin_file
    assert binary.name == "ZFM-02370_mini.imec0.ap.short.bin"
    assert binary.parent == session.neuropixels_dir
    assert "{" not in str(binary)               # nothing left unexpanded
    assert session.output_root == session.neuropixels_dir


def _template_paths(session):
    return (
        session.blackrock.sync_file,
        session.blackrock.spike_file,
        session.neuropixels.bin_file,
        session.output_root,
    )


def test_template_leaves_no_unresolved_placeholders():
    # _substitute leaves an unknown name alone rather than raising, so without
    # _first_unresolved a typo'd placeholder would become a literal directory
    # named "{...}" and fail much later, if at all.
    session = cfg.load_session_config(CONFIG_DIR / "session_template_utah_probe.yaml", "windows_rig", CONFIG_DIR)
    for path in _template_paths(session):
        assert "{" not in str(path)
        assert str(path).startswith("Z:")
    # The label names outputs, never a directory -- so it must not appear in one.
    assert session.session not in str(session.output_root)


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
        ("session_template_2probes.yaml", "windows_rig"),
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
    assert none.blackrock.probe_file is None
    assert none.sorts_blackrock is False
    # The template says so outright. Unstated, the flag would follow the paths to
    # the same answer -- but writing it down is what makes an explicit `true`
    # with no spike_file a refusal rather than a silent skip.
    assert none.kilosort_on_blackrock is False

    # Neuropixels is untouched, including its Kilosort block.
    assert none.neuropixels == one.neuropixels
    assert none.kilosort_by_system["neuropixels"] == one.kilosort_by_system["neuropixels"]
    assert "blackrock" not in none.kilosort_by_system   # no array, no settings for one

    # `session` is the config filename, so two files necessarily differ there.
    # kilosort_on_blackrock too: this session has no array, and says so, which is
    # exactly the difference being pinned rather than drift.
    stated = {"session", "blackrock", "kilosort_by_system", "kilosort_on_blackrock"}
    differing = [
        f
        for f in one.__dataclass_fields__
        if f not in stated and getattr(one, f) != getattr(none, f)
    ]
    assert differing == [], differing
    assert cfg.replace(one.blackrock, spike_file=None, probe_file=None) == none.blackrock


def test_the_two_probe_template_differs_only_by_the_second_binary():
    # Same pinning as above, for the pair that is easiest to let drift: the
    # 2-probe template is the 1-probe one with bin_file replaced by a bin_files
    # list, so a threshold tuned in whichever file someone happened to open fails
    # here rather than on the rig.
    one, two = (
        cfg.load_session_config(CONFIG_DIR / name, "windows_rig", CONFIG_DIR)
        for name in ("session_template_1probe.yaml", "session_template_2probes.yaml")
    )

    assert one.probes == (0,) and two.probes == (0, 1)
    assert one.neuropixels.bin_files is None   # one probe names a bare bin_file
    assert two.neuropixels.bin_file is None    # ...and two name only the list

    # The Blackrock half is untouched: sync pulses, no array, nothing to sort.
    assert two.blackrock == one.blackrock
    assert two.sorts_blackrock is False and two.has_data("blackrock")
    # ...and, as in the 1-probe file, it says so outright.
    assert two.kilosort_on_blackrock is False

    # Each probe of the pair resolves to the 1-probe template's own session,
    # apart from which binary it names -- that is what "two sessions in one
    # file" has to mean for the pinning to be worth anything. Both binaries are
    # written out in the file; neither is derived from the other.
    for _, probe_config in two.per_probe():
        assert probe_config.neuropixels.bin_file.name.startswith("Athos_2026_08_13_g0_t0.imec")
        assert probe_config.paths.sorted_np.parent == probe_config.neuropixels_dir
    imec0 = two.for_probe(0)
    assert imec0.neuropixels_dir == one.neuropixels_dir      # the same probe folder
    assert imec0.neuropixels.bin_file == one.neuropixels.bin_file
    assert imec0.kilosort_for("neuropixels") == one.kilosort_for("neuropixels")

    # `session` is the config filename, so two files necessarily differ there;
    # neuropixels_dir is the gate folder here and the probe folder there.
    stated = {"session", "neuropixels", "neuropixels_dir"}
    differing = [
        f
        for f in one.__dataclass_fields__
        if f not in stated and getattr(one, f) != getattr(two, f)
    ]
    assert differing == [], differing
    # ...and inside the neuropixels block, only how the binaries are named.
    assert cfg.replace(
        two.neuropixels, bin_files=None, bin_file=one.neuropixels.bin_file
    ) == one.neuropixels


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

    stated = {
        "session",            # the config filename, so necessarily different
        "neuropixels",
        "neuropixels_dir",
        "kilosort_on_neuropixels",
        "kilosort_by_system",
    }
    differing = [
        f
        for f in one.__dataclass_fields__
        if f not in stated and getattr(one, f) != getattr(utah, f)
    ]
    assert differing == [], differing


_RUN = "Y:/npx/Tank_g0"
_BINARIES = (
    f'    - "{_RUN}/Tank_g0_imec0/Tank_g0_t0.imec0.ap.bin"\n'
    f'    - "{_RUN}/Tank_g0_imec1/Tank_g0_t0.imec1.ap.bin"\n'
)


def _two_probe_session(tmp_path: Path, extra: str = "", binaries: str = _BINARIES):
    """One session file naming both binaries of a two-probe run."""
    path = tmp_path / "two.yaml"
    path.write_text(
        f'neuropixels_dir: "{_RUN}"\n'
        "neuropixels:\n  bin_files:\n" + binaries + extra,
        encoding="utf-8",
    )
    return cfg.load_session_config(path, "windows_rig", CONFIG_DIR)


def test_a_bin_files_list_projects_to_one_ordinary_config_per_probe(tmp_path):
    # `bin_files:` is a way to write two sessions in one file, not a dimension
    # the pipeline carries. for_probe hands back a config indistinguishable from
    # a file naming that probe alone, and what matters is that the two cannot
    # collide: every output path comes from neuropixels_dir, so each probe has to
    # land in the folder its own binary sits in.
    session = _two_probe_session(tmp_path)
    assert session.probes == (0, 1)          # read off the two .imec<n> names

    tags = [tag for tag, _ in session.per_probe()]
    assert tags == ["imec0", "imec1"]
    a, b = (config for _, config in session.per_probe())

    # Each projection names the binary the session stated for it -- nothing is
    # derived from the other probe's path.
    assert a.neuropixels.bin_file.name == "Tank_g0_t0.imec0.ap.bin"
    assert b.neuropixels.bin_file.name == "Tank_g0_t0.imec1.ap.bin"
    assert {a.neuropixels.bin_file, b.neuropixels.bin_file} == set(
        session.neuropixels.bin_files
    )

    # Each writes inside its own binary's folder -- including probe 0, whose
    # folder is NOT the directory the session states.
    for config in (a, b):
        assert config.paths.sorted_np.parent == config.neuropixels_dir
        assert config.neuropixels_dir == config.neuropixels.bin_file.parent
        assert config.neuropixels_dir != session.neuropixels_dir
    assert a.paths.sorted_np != b.paths.sorted_np


def test_the_probe_number_comes_from_each_binarys_own_name(tmp_path):
    # No parallel list of probe numbers to drift out of step with the paths --
    # SpikeGLX puts it in the filename. Order in the file does not decide it.
    session = _two_probe_session(
        tmp_path,
        binaries=(
            f'    - "{_RUN}/Tank_g0_imec3/Tank_g0_t0.imec3.ap.bin"\n'
            f'    - "{_RUN}/Tank_g0_imec1/Tank_g0_t0.imec1.ap.bin"\n'
        ),
    )

    assert session.probes == (1, 3)
    assert [tag for tag, _ in session.per_probe()] == ["imec1", "imec3"]
    assert session.for_probe(3).neuropixels.bin_file.name.endswith("imec3.ap.bin")


def test_a_binary_whose_name_carries_no_probe_falls_back_to_its_position(tmp_path):
    # A hand-cut extract has no .imec<n>, and refusing it would make the list
    # useless for exactly the recordings that are hardest to name.
    session = _two_probe_session(
        tmp_path,
        binaries=f'    - "{_RUN}/left.bin"\n    - "{_RUN}/right.bin"\n',
    )

    assert session.probes == (0, 1)
    assert session.for_probe(0).neuropixels.bin_file.name == "left.bin"
    assert session.for_probe(1).neuropixels.bin_file.name == "right.bin"


def test_each_probe_gets_its_own_time_map_under_the_blackrock_folder(tmp_path):
    # Both probes are fitted onto the SAME Blackrock recording, so `aligned` is
    # the one path a probe list can make collide -- one time_map.json, written
    # twice, the second winning. Each probe has its own oscillator and SY word,
    # so those are genuinely different maps.
    path = tmp_path / "two.yaml"
    path.write_text(
        'blackrock_dir: "Y:/br"\n'
        f'neuropixels_dir: "{_RUN}"\n'
        "blackrock:\n  sync_file: '/b/y.ns5'\n"
        "neuropixels:\n  bin_files:\n" + _BINARIES,
        encoding="utf-8",
    )
    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)
    a, b = (config for _, config in session.per_probe())

    assert a.paths.aligned != b.paths.aligned
    assert a.paths.aligned.name == "imec0" and b.paths.aligned.name == "imec1"
    # Still under Blackrock, which is the reference timebase.
    assert a.paths.aligned.parent == session.paths.sorted_br


def test_a_session_naming_one_binary_projects_to_itself(tmp_path):
    # The no-regression pin. Every session written before `bin_files:` existed
    # must resolve to exactly the paths it always did -- no imec<n> level
    # anywhere -- so the probe dimension costs an ordinary session nothing.
    session = _flags(tmp_path, "neuropixels:\n  bin_file: '/a/run_g0_t0.imec0.ap.bin'\n")

    assert session.neuropixels.bin_files is None
    assert session.probes == (0,)                    # read off the binary's name
    assert session.per_probe() == [(None, session.for_probe(0))]

    only = session.for_probe(0)
    assert only.probe_tag is None
    assert only.neuropixels_dir == session.neuropixels_dir
    assert only.neuropixels == session.neuropixels
    assert only.paths == session.paths               # aligned included


def test_a_probe_overrides_kilosort_settings_without_discarding_the_shared_ones(tmp_path):
    # by_probe is merged one level deep, exactly like the per-system layer: a
    # probe's bad_channels replaces the list above it, and everything else it
    # does not mention still applies.
    session = _two_probe_session(
        tmp_path,
        "  kilosort:\n"
        "    nblocks: 1\n"
        "    bad_channels: [7]\n"
        "  by_probe:\n"
        "    1:\n"
        "      kilosort:\n"
        "        bad_channels: [17, 203]\n",
    )
    a, b = (config for _, config in session.per_probe())

    assert a.kilosort_for("neuropixels") == {"nblocks": 1, "bad_channels": [7]}
    assert b.kilosort_for("neuropixels") == {"nblocks": 1, "bad_channels": [17, 203]}
    # Replaced, not extended: keeping [7] would go on sorting a site the session
    # named as dead on that probe.
    assert 7 not in b.kilosort_for("neuropixels")["bad_channels"]


def test_every_probes_binary_is_checked_before_the_run(tmp_path):
    # Both are named, so both are checked -- a session covering two probes with
    # one of them missing would otherwise pass the pre-flight and fail an hour
    # in, on the second sort.
    from conftest import spikeglx_run

    binaries = spikeglx_run(tmp_path / "npx", probes=(0, 1))
    path = _write_session(
        tmp_path,
        f"neuropixels_dir: '{binaries[0].parent.parent}'\n"
        "neuropixels:\n  bin_files:\n"
        f"    - '{binaries[0]}'\n    - '{binaries[1]}'\n",
    )
    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)
    assert not [p for p in session.missing_inputs() if "bin_file" in p]

    binaries[1].unlink()

    problems = [p for p in session.missing_inputs() if "bin_file" in p]
    assert len(problems) == 1
    assert "(imec1)" in problems[0]        # says which probe, not just "a binary"
    assert str(binaries[1]) in problems[0]


def test_naming_both_bin_file_and_bin_files_raises(tmp_path):
    # Two answers to which probe this session sorts, and nothing to say which
    # wins -- so neither does.
    with pytest.raises(ValueError, match="both bin_file and bin_files"):
        _two_probe_session(tmp_path, "  bin_file: '/a/Tank_g0_t0.imec0.ap.bin'\n")


def test_two_binaries_for_one_probe_raise(tmp_path):
    # The probe comes from the name, so a copied line -- or two files from
    # different runs of the same probe -- would collapse to one entry, and the
    # session would quietly sort half of what it names.
    with pytest.raises(ValueError, match="only 1 distinct probe"):
        _two_probe_session(
            tmp_path,
            binaries=(
                f'    - "{_RUN}/a/Tank_g0_t0.imec0.ap.bin"\n'
                f'    - "{_RUN}/b/Other_g0_t0.imec0.ap.bin"\n'
            ),
        )


def test_an_lf_file_cannot_be_named_alongside_several_probes(tmp_path):
    # lf_file is a single key and each probe has its own LF band. Silently
    # applying one probe's to both would export the wrong recording.
    with pytest.raises(ValueError, match="lf_file names one LF band"):
        _two_probe_session(tmp_path, "  lf_file: '/a/Tank_g0_t0.imec0.lf.bin'\n")


def test_by_probe_refuses_a_probe_the_session_does_not_cover(tmp_path):
    # Settings for a probe that never runs are settings nobody reads -- most
    # likely a typo'd probe number, and silently ignoring it would sort the real
    # probe with the wrong bad_channels.
    with pytest.raises(ValueError, match=r"by_probe names probe\(s\) \[2\]"):
        _two_probe_session(
            tmp_path, "  by_probe:\n    2:\n      kilosort:\n        nblocks: 0\n"
        )


def test_by_probe_takes_kilosort_and_nothing_else(tmp_path):
    # Geometry comes from each probe's own .meta and the paths are in bin_files,
    # so anything else here would be read by nobody.
    with pytest.raises(ValueError, match="unknown setting 'neuropixels.by_probe.1.probe_file'"):
        _two_probe_session(
            tmp_path, "  by_probe:\n    1:\n      probe_file: 'configs/probes/np1_default.json'\n"
        )


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
    # sorting on top of the other. Easy to hit by accident whenever one share
    # holds everything and the obvious folder to name is the session folder both
    # systems dropped files into.
    path = _write_session(
        tmp_path,
        """
blackrock_dir: "Z:/Monkey Athos/2026-08-13"
neuropixels_dir: "Z:/Monkey Athos/2026-08-13"
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
neuropixels_dir: "Z:/Monkey Demo/2026-08-13"
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


def test_each_system_names_its_own_directory_and_they_stay_independent(tmp_path):
    # Two systems whose data lives on different shares. Each directory is named
    # once and everything below refers back to it -- which is also why the two
    # do not collide in the flat output layout.
    path = _write_session(
        tmp_path,
        """
blackrock_dir: "Z:/server/Monkey Athos/2026-08-13"
neuropixels_dir: "Y:/npx/Monkey Athos/2026-08-13"
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
    assert str(session.neuropixels.bin_file.parent) == "Y:/npx/Monkey Athos/2026-08-13"
    # Cross-system output goes under the reference timebase's directory.
    assert str(session.output_root) == "Z:/server/Monkey Athos/2026-08-13"


def test_the_session_label_comes_from_the_file_not_from_a_key(tmp_path):
    # It names the .mat products and the cache tag and nothing else, so it is
    # read off the config's own filename -- one fewer thing that can disagree
    # with the file describing it.
    path = tmp_path / "Athos_2026_08_13.yaml"
    path.write_text('neuropixels_dir: "Y:/npx/run_g0_imec0"\n', encoding="utf-8")

    session = cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert session.session == "Athos_2026_08_13"
    assert "Athos_2026_08_13" not in str(session.neuropixels_dir)


def test_a_misspelled_placeholder_raises_instead_of_resolving_to_nothing(tmp_path):
    # The trap this closes: an unknown placeholder used to be left as a literal
    # and a known-but-unset one became "", turning "{root}/Monkey Athos" into
    # "/Monkey Athos" -- a wrong path that only fails much later, if at all.
    path = _write_session(
        tmp_path,
        """
blackrock_dir: "Z:/server"
blackrock:
  sync_file: "{blackrok_dir}/NSP.ns5"
""",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "{blackrok_dir}" in str(excinfo.value)


def test_a_placeholder_from_the_old_composed_layout_raises(tmp_path):
    # {monkey}, {session}, {data_root} and the roots: block are all gone. A
    # session file left over from them must fail loudly rather than build a path
    # around a literal brace.
    path = _write_session(
        tmp_path,
        """
neuropixels_dir: "Z:/npx/{monkey}/2026-08-13"
neuropixels:
  bin_file: "{neuropixels_dir}/x.bin"
""",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert "{monkey}" in message
    assert "{blackrock_dir}" in message and "{neuropixels_dir}" in message


def _flags(tmp_path: Path, body: str):
    """Both directories stated, and both systems declared unless `body` does."""
    lines = [
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
blackrock_dir: "Z:/server/Monkey Athos/2026-08-13"
neuropixels_dir: "Y:/npx/Monkey Athos/2026-08-13"
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
neuropixels_dir: "Y:/npx/Monkey Demo/2026-08-13"
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


def test_a_utah_array_names_its_cmp_in_the_same_key(tmp_path):
    # One key per system whatever the format: a .cmp is named exactly where a
    # .json would be, and setup_probe tells them apart by extension. cmp_file
    # used to be a second key tried first.
    session = _flags(tmp_path, "blackrock:\n  probe_file: '/arrays/athos_A.cmp'\n")

    assert session.blackrock.probe_file == Path("/arrays/athos_A.cmp")


def test_probe_paths_expand_placeholders_like_any_other_path(tmp_path):
    session = _flags(tmp_path, "blackrock:\n  probe_file: '{blackrock_dir}/arrayA.cmp'\n")

    assert str(session.blackrock.probe_file) == str(tmp_path / "br" / "arrayA.cmp")


def test_no_probe_named_is_the_default(tmp_path):
    session = _flags(tmp_path, "")

    assert session.blackrock.probe_file is None
    assert session.neuropixels.probe_file is None


def test_a_named_probe_file_that_is_absent_is_reported(tmp_path):
    # Named but not there is a hard error: the alternative is discovering it after
    # the recording has been copied to the cache.
    session = _flags(
        tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '/nope/utah_A.json'\n"
    )

    assert any("probe_file" in p for p in session.missing_inputs())


def test_a_named_cmp_that_is_absent_is_reported(tmp_path):
    # Same check, and now the same key: the format does not change whether a
    # named map has to be there.
    session = _flags(
        tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n  probe_file: '/nope/array.cmp'\n"
    )

    assert any("probe_file" in p for p in session.missing_inputs())


def test_asking_to_sort_blackrock_with_nothing_to_sort_raises(tmp_path):
    # The flag is a request, and a request that cannot be honoured must fail
    # rather than turn into nothing. Skipping reads like a successful run to
    # someone who meant to sort the array and mistyped the path -- they get no
    # units and one [--] line, hours later.
    path = _write_session(
        tmp_path,
        f"blackrock_dir: '{tmp_path}'\nkilosort_on_blackrock: true\n"
        "blackrock:\n  sync_file: '/b/y.ns5'\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert "kilosort_on_blackrock is true" in message
    assert "spike_file" in message
    # Both ways out are named: the array's .ns6, or saying it is not being sorted.
    assert "kilosort_on_blackrock: false" in message


def test_an_unstated_kilosort_on_blackrock_follows_the_paths(tmp_path):
    # Raising on the *default* would put a line saying "not doing this" in every
    # sync-only session. Unstated, the flag resolves from whether there is
    # anything to sort -- so only a session that writes `true` is held to it.
    sync_only = _flags(tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n")
    with_array = _flags(
        tmp_path, "blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '/b/x.ns6'\n"
    )

    assert sync_only.kilosort_on_blackrock is False
    assert sync_only.sorts_blackrock is False
    assert with_array.kilosort_on_blackrock is True
    assert with_array.sorts_blackrock is True

    # ...and stating false with an array present still turns the sort off, which
    # is the "extract the pulses and the LFP, do not sort" case.
    off = _flags(
        tmp_path,
        "kilosort_on_blackrock: false\n"
        "blackrock:\n  sync_file: '/b/y.ns5'\n  spike_file: '/b/x.ns6'\n",
    )
    assert off.sorts_blackrock is False and off.has_data("blackrock")


def test_the_retired_cmp_file_key_names_probe_file(tmp_path):
    # Session copies live on the rig, so a key that moved must say where it went.
    # This one needs an explicit hint: cmp_file's nearest surviving neighbour is
    # sync_file, and pointing a channel map at the pulse train is worse than no
    # hint at all.
    path = _write_session(
        tmp_path,
        f"blackrock_dir: '{tmp_path}'\n"
        "blackrock:\n  sync_file: '/b/y.ns5'\n  cmp_file: '/b/array.cmp'\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert "blackrock.cmp_file" in message and "blackrock.probe_file" in message
    assert "sync_file" not in message


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

    with pytest.raises(ValueError, match="unknown setting 'neuropixels.probe_name'"):
        _flags(tmp_path, "neuropixels:\n  probe_name: 'NeuroPix1_default.mat'\n")


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
        f"blackrock_dir: '{tmp_path / 'br'}'\n"
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
        f"neuropixels_dir: '{tmp_path}'\n"
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
    "key, body",
    [
        ("run_dir", "neuropixels:\n  run_dir: '/npx'\n"),
        ("run_name", "neuropixels:\n  run_name: 'r'\n"),
        ("gate", "neuropixels:\n  gate: 0\n"),
        ("trigger", "neuropixels:\n  trigger: 0\n"),
    ],
)
def test_the_removed_run_model_keys_raise_by_name(tmp_path, key, body):
    # These sit in session copies on the rig. Ignoring one would change what a run
    # does without a word -- and they are caught by simply not being settings any
    # more, so there is no list of dead keys to keep in step with the code.
    #
    # `probes:` and `by_probe:` are NOT in this list: they are settings again, and
    # are what a two-probe run states. What stayed gone is the four keys that
    # composed a path -- run_layout reads all four back off bin_file's own name.
    path = _write_session(
        tmp_path, f"neuropixels_dir: '{tmp_path}'\n" + body
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert f"neuropixels.{key}" in message
    assert "unknown setting" in message


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
        f"neuropixels_dir: '{tmp_path}'\n"
        "neuropixels:\n  bin_file: '/a/x.bin'\n"
        "preprocess:\n  apply: true\n  bandpass: [300, 6000]\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "unknown setting 'preprocess'" in str(excinfo.value)


def test_a_removed_key_nested_in_a_system_block_is_also_refused(tmp_path):
    # Per-system preprocess: blocks were a real thing. Checking only top-level
    # keys would drop one silently -- the failure this rejection exists to stop.
    path = _write_session(
        tmp_path,
        f"neuropixels_dir: '{tmp_path}'\n"
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
        f"blackrock_dir: '{tmp_path}'\n"
        "blackrock:\n  spike_file: '/b/y.ns6'\n  exclude_channels: ['129']\n",
    )

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    assert "unknown setting 'blackrock.exclude_channels'" in str(excinfo.value)


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
            "blackrock_dir: '/b'\nneuropixels_dir: '/n'\n" + body,
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


def test_the_renamed_waveform_keys_raise_rather_than_being_ignored(tmp_path):
    # Session copies live on the rig. Ignoring a key someone tuned there would
    # change what a run measured without a word. Both moved into the per-system
    # waveforms: block, so at the top level they are simply not settings.
    for old in ("waveform_ms: 3.0", "export_waveforms: true"):
        with pytest.raises(ValueError, match="unknown setting"):
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


# ---------------------------------------------------------------------------
# A key that is not a setting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body, named, suggested",
    [
        ("export_figure: false\n", "export_figure", "export_figures"),
        ("kilosort_on_neuropixel: false\n", "kilosort_on_neuropixel", "kilosort_on_neuropixels"),
        # bin_file and bin_files are both real; a slip in either still lands.
        ("neuropixels:\n  bin_flie: '/a/x.bin'\n", "neuropixels.bin_flie", "neuropixels.bin_file"),
        ("neuropixels:\n  bin_fies: ['/a/x.bin']\n", "neuropixels.bin_fies", "neuropixels.bin_files"),
        (
            "neuropixels:\n  waveforms:\n    window_msec: 2.0\n",
            "neuropixels.waveforms.window_msec",
            "neuropixels.waveforms.window_ms",
        ),
    ],
)
def test_a_mistyped_key_raises_and_names_the_real_one(tmp_path, body, named, suggested):
    # YAML does not care about a key nobody reads, so without this the typo loads
    # clean and the run silently keeps the default -- a wrong result rather than
    # a wrong path, and nothing downstream can tell. Nested blocks are checked
    # too: waveforms: is per system, so a typo there quietly restores a default
    # the session meant to change.
    path = _write_session(tmp_path, f"neuropixels_dir: '{tmp_path}'\n" + body)

    with pytest.raises(ValueError) as excinfo:
        cfg.load_session_config(path, "windows_rig", CONFIG_DIR)

    message = str(excinfo.value)
    assert f"unknown setting '{named}'" in message
    assert f"did you mean: {suggested}?" in message


def test_a_key_inside_a_kilosort_block_is_never_checked(tmp_path):
    # Those keys pass through to Kilosort, and pipeline._kilosort_arguments
    # already splits them against the *installed* DEFAULT_SETTINGS and
    # run_kilosort signature. A second copy of that split here would go stale the
    # first time Kilosort moved a parameter -- and would refuse a real one.
    session = _flags(
        tmp_path,
        "kilosort:\n  a_setting_added_after_this_test: 7\n"
        "neuropixels:\n  bin_file: '/a/x.bin'\n  kilosort:\n    and_another: 3\n",
    )

    assert session.kilosort_for("neuropixels")["a_setting_added_after_this_test"] == 7
    assert session.kilosort_for("neuropixels")["and_another"] == 3


@pytest.mark.parametrize(
    "config", sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))
)
def test_every_tracked_config_uses_only_real_settings(config):
    # The top-level key set is the one part of the check written by hand, so it
    # is pinned against the files that have to keep loading.
    cfg.load_session_config(CONFIG_DIR / config, "windows_rig", CONFIG_DIR)
