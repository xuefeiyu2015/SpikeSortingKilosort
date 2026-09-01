"""Tests for the environment health check.

Pure compute: every subprocess the doctor would run is faked, so these need no
conda, no GPU, no Kilosort and no real recordings -- the same bar as the rest of
the suite. What is being checked is the *reasoning*: that a shared environment is
found the way the rig actually stores it, that a CPU-only torch is called out,
and that warnings never fail a run.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import doctor  # from tools/, added to sys.path by conftest
from spikesorting._config import EnvLocation, MachineProfile, load_machine

KS = doctor.ENV_SPECS[0]
PHY = doctor.ENV_SPECS[1]
ENV_SPECS_PACKAGES = tuple(
    package for spec in doctor.ENV_SPECS for package in spec.packages
)

#: Captured before the autouse fixture below replaces it, for the few tests that
#: exercise the real detector rather than stubbing it out.
REAL_ACTIVE_ENV = doctor.active_env


@pytest.fixture(autouse=True)
def no_active_env(monkeypatch):
    """Pretend nothing is activated unless a test says otherwise.

    Without this the suite would read the *real* ``CONDA_PREFIX``: run the tests
    from inside an env that happens to be named ``kilosort4`` and discovery would
    find it first, quietly invalidating every assertion about the other two
    sources. Tests that care about the active env override this.
    """
    monkeypatch.setattr(doctor, "active_env", lambda: None)


def completed(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["fake"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def make_env(root: Path, name: str, windows: bool = False) -> Path:
    """Create a directory that looks like a conda prefix with an interpreter."""
    prefix = root / name
    if windows:
        (prefix).mkdir(parents=True, exist_ok=True)
        (prefix / "python.exe").write_text("")
    else:
        (prefix / "bin").mkdir(parents=True, exist_ok=True)
        (prefix / "bin" / "python").write_text("")
    return prefix


def probe_payload(present: dict[str, str | None], torch: dict | None = None) -> str:
    packages = {
        spec.module: {
            "present": spec.module in present,
            "version": present.get(spec.module),
        }
        for spec in KS.packages
    }
    return json.dumps(
        {"python": "3.11.9", "executable": "x", "packages": packages, "torch": torch}
    )


def named(name: str) -> EnvLocation:
    """An env configured by name only -- the common case, and the default."""
    return EnvLocation(name=name)


def status_of(checks, name: str) -> str:
    for check in checks:
        if check.name == name:
            return check.status
    raise AssertionError(f"no check named {name!r} in {[c.name for c in checks]}")


# ---------------------------------------------------------------------------
# Environment discovery
# ---------------------------------------------------------------------------


def test_list_conda_envs_names_the_root_prefix_base(monkeypatch):
    """The root env is called 'base', not after its directory."""
    payload = json.dumps(
        {
            "root_prefix": "/opt/anaconda3",
            "envs": ["/opt/anaconda3", "/opt/anaconda3/envs/kilosort4"],
        }
    )
    monkeypatch.setattr(doctor, "conda_executable", lambda: "/opt/anaconda3/bin/conda")
    monkeypatch.setattr(doctor, "_run", lambda argv, timeout_s=60.0: completed(payload))

    envs = doctor.list_conda_envs()

    assert envs["base"] == Path("/opt/anaconda3")
    assert envs["kilosort4"] == Path("/opt/anaconda3/envs/kilosort4")
    assert "anaconda3" not in envs


def test_list_conda_envs_empty_without_conda(monkeypatch):
    monkeypatch.setattr(doctor, "conda_executable", lambda: None)
    assert doctor.list_conda_envs() == {}


def test_shared_env_found_by_path_when_conda_does_not_list_it(tmp_path):
    """The lab-account case: admin created it with `conda create -p`.

    `conda env list` reads the *calling* user's registry, so the env is absent
    from it. Discovery must still find it, or the report blames the wrong thing.
    """
    prefix = make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)

    found, origin = doctor.resolve_env(named("kilosort4"), machine, envs={})

    assert found == prefix
    assert origin == doctor.ORIGIN_SHARED


def test_conda_listing_wins_over_the_shared_directory(tmp_path):
    """Whatever `conda activate <name>` would resolve to is what we report."""
    make_env(tmp_path / "shared", "kilosort4")
    own = make_env(tmp_path / "own", "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path / "shared")

    found, origin = doctor.resolve_env(named("kilosort4"), machine, envs={"kilosort4": own})

    assert found == own
    assert origin == doctor.ORIGIN_CONDA


def test_env_missing_everywhere_resolves_to_none(tmp_path):
    machine = MachineProfile(name="mac")
    found, _ = doctor.resolve_env(named("kilosort4"), machine, envs={})
    assert found is None


# ---------------------------------------------------------------------------
# The active environment
# ---------------------------------------------------------------------------


def make_conda_prefix(root: Path, name: str) -> Path:
    """A conda prefix, identified as one by its conda-meta directory."""
    prefix = make_env(root, name)
    (prefix / "conda-meta").mkdir(parents=True, exist_ok=True)
    return prefix


def test_active_env_is_found_without_conda_or_a_shared_dir(tmp_path):
    """The gap this closes: standing inside the env and being told it is missing.

    Neither conda nor conda_envs_dir is available -- the situation on a batch node
    handed <prefix>/bin/python, or a box whose conda is not on PATH.
    """
    prefix = make_conda_prefix(tmp_path, "kilosort4")
    machine = MachineProfile(name="hpc")

    found, origin = doctor.resolve_env(named("kilosort4"), machine, envs={}, active=prefix)

    assert found == prefix
    assert origin == doctor.ORIGIN_ACTIVE


def test_active_env_is_the_last_resort_not_the_first(tmp_path):
    """It only answers where the other two could not -- conda's answer wins."""
    active = make_conda_prefix(tmp_path / "active", "kilosort4")
    listed = make_env(tmp_path / "own", "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path / "shared")
    make_env(tmp_path / "shared", "kilosort4")

    found, origin = doctor.resolve_env(
        named("kilosort4"), machine, envs={"kilosort4": listed}, active=active
    )

    assert found == listed
    assert origin == doctor.ORIGIN_CONDA


def test_active_env_with_another_name_is_not_adopted(tmp_path):
    """Running from `base` must still report on the env you asked about."""
    base = make_conda_prefix(tmp_path, "base")
    machine = MachineProfile(name="mac")

    found, _ = doctor.resolve_env(named("kilosort4"), machine, envs={}, active=base)

    assert found is None


def test_active_env_ignores_a_prefix_that_is_not_conda(tmp_path, monkeypatch):
    """`sys.prefix` is always set; for a system Python it is not an environment."""
    plain = tmp_path / "usr" / "local"
    plain.mkdir(parents=True)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "prefix", str(plain))

    assert REAL_ACTIVE_ENV() is None


def test_active_env_falls_back_to_sys_prefix_when_nothing_was_activated(
    tmp_path, monkeypatch
):
    """The batch-node case: <prefix>/bin/python invoked by path, no activation."""
    prefix = make_conda_prefix(tmp_path, "kilosort4")
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.setattr(sys, "prefix", str(prefix))

    assert REAL_ACTIVE_ENV() == prefix


def test_active_env_prefers_conda_prefix_over_sys_prefix(tmp_path, monkeypatch):
    """`conda activate` in a shell whose base interpreter is also a conda prefix."""
    activated = make_conda_prefix(tmp_path, "kilosort4")
    base = make_conda_prefix(tmp_path, "base")
    monkeypatch.setenv("CONDA_PREFIX", str(activated))
    monkeypatch.setattr(sys, "prefix", str(base))

    assert REAL_ACTIVE_ENV() == activated


# ---------------------------------------------------------------------------
# Environment names are configuration, not code
# ---------------------------------------------------------------------------


def test_default_names_are_used_when_nothing_is_configured():
    """The whole point of the defaults: existing setups need no config at all."""
    machine = MachineProfile(name="mac")

    assert doctor.env_location(KS, machine) == named("kilosort4")
    assert doctor.env_location(PHY, machine) == named("phy")


def test_machine_profile_renames_the_env():
    machine = MachineProfile(name="rig", conda_envs={"sorting": named("ks5")})

    assert doctor.env_location(KS, machine).name == "ks5"
    assert doctor.env_location(PHY, machine).name == "phy"  # other role untouched


def test_override_outranks_the_machine_profile():
    """`--sorting-env` exists to try a new Kilosort without editing the profile."""
    machine = MachineProfile(name="rig", conda_envs={"sorting": named("ks5")})

    assert doctor.env_location(KS, machine, {"sorting": named("ks4-dev")}).name == "ks4-dev"
    assert doctor.env_location(KS, machine, {}).name == "ks5"


def test_an_override_replaces_the_whole_location_not_half_of_it(tmp_path):
    """A name-only override must not inherit the profile's path.

    Otherwise `--sorting-env ks5` against a profile with a path would look in
    <profile path>/ks5, which is nobody's intent and fails confusingly.
    """
    machine = MachineProfile(
        name="rig", conda_envs={"sorting": EnvLocation("kilosort4", tmp_path)}
    )

    resolved = doctor.env_location(KS, machine, {"sorting": named("ks5")})

    assert resolved == named("ks5")
    assert resolved.path is None


def test_a_renamed_env_is_what_gets_looked_up_and_reported(tmp_path, monkeypatch):
    """End to end: the configured name drives discovery and every check name."""
    make_env(tmp_path, "ks5")
    machine = MachineProfile(
        name="rig", conda_envs_dir=tmp_path, conda_envs={"sorting": named("ks5")}
    )
    installed = {spec.module: "1.0" for spec in KS.packages}
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed(probe_payload(installed)),
    )

    checks = doctor.check_env(KS, machine, envs={})

    assert status_of(checks, "env:ks5") == doctor.OK
    assert status_of(checks, "ks5/neo") == doctor.OK
    assert not any(check.name.startswith("kilosort4") for check in checks)


def test_missing_renamed_env_names_the_role_in_its_fix(tmp_path):
    """The fix has to mention conda_envs, or the reader edits the wrong thing."""
    machine = MachineProfile(name="mac", conda_envs={"sorting": named("ks5")})

    checks = doctor.check_env(KS, machine, envs={})

    assert checks[0].name == "env:ks5"
    assert "conda_envs.sorting" in checks[0].fix


# ---------------------------------------------------------------------------
# A configured path settles it: prefix = path / name
# ---------------------------------------------------------------------------


def test_prefix_is_the_path_joined_with_the_name(tmp_path):
    """The two settings combine; the path holds the env, it is not the env."""
    location = EnvLocation(name="ks5", path=tmp_path / "envs")

    assert location.prefix == tmp_path / "envs" / "ks5"


def test_a_location_without_a_path_has_no_prefix():
    assert named("kilosort4").prefix is None


def test_configured_path_is_used_without_any_search(tmp_path):
    """No conda, no conda_envs_dir, nothing listed: the path alone answers it."""
    prefix = make_env(tmp_path / "envs", "ks5")
    machine = MachineProfile(
        name="rig", conda_envs={"sorting": EnvLocation("ks5", tmp_path / "envs")}
    )

    found, origin = doctor.resolve_env(
        doctor.env_location(KS, machine), machine, envs={}
    )

    assert found == prefix
    assert origin == doctor.ORIGIN_CONFIGURED


def test_configured_path_beats_a_same_named_env_elsewhere(tmp_path):
    """Saying where the env lives is an instruction, so nothing else gets a say."""
    wanted = make_env(tmp_path / "wanted", "kilosort4")
    decoy = make_conda_prefix(tmp_path / "decoy", "kilosort4")
    machine = MachineProfile(
        name="rig",
        conda_envs_dir=tmp_path / "decoy",
        conda_envs={"sorting": EnvLocation("kilosort4", tmp_path / "wanted")},
    )

    found, origin = doctor.resolve_env(
        doctor.env_location(KS, machine),
        machine,
        envs={"kilosort4": decoy},
        active=decoy,
    )

    assert found == wanted
    assert origin == doctor.ORIGIN_CONFIGURED


def test_configured_path_expands_a_tilde(tmp_path, monkeypatch):
    prefix = make_env(tmp_path / "envs", "ks5")
    monkeypatch.setenv("HOME", str(tmp_path))
    machines = tmp_path / "cfg" / "machines"
    machines.mkdir(parents=True)
    (machines / "t.yaml").write_text(
        'conda_envs:\n  sorting:\n    name: ks5\n    path: "~/envs"\n', encoding="utf-8"
    )
    machine = load_machine("t", tmp_path / "cfg")

    found, _ = doctor.resolve_env(doctor.env_location(KS, machine), machine, envs={})

    assert found == prefix


def test_absent_configured_path_fails_instead_of_falling_back(tmp_path):
    """The fallback is for a config that gives no path, not for a broken one.

    Quietly using some other env that happened to match by name is exactly the
    confusion an explicit path was written to prevent.
    """
    elsewhere = make_conda_prefix(tmp_path / "elsewhere", "kilosort4")
    machine = MachineProfile(
        name="rig",
        conda_envs={"sorting": EnvLocation("kilosort4", tmp_path / "gone")},
    )

    checks = doctor.check_env(KS, machine, envs={"kilosort4": elsewhere})

    assert checks[0].status == doctor.MISSING
    assert "no env 'kilosort4' in the configured path" in checks[0].detail
    assert str(tmp_path / "gone") in checks[0].detail
    assert "conda_envs.sorting.path" in checks[0].fix


def test_the_likeliest_path_mistake_is_named_in_the_report(tmp_path):
    """Giving the env's own prefix as `path` doubles the name; say so on sight."""
    prefix = make_env(tmp_path / "envs", "ks5")
    machine = MachineProfile(name="rig", conda_envs={"sorting": EnvLocation("ks5", prefix)})

    checks = doctor.check_env(KS, machine, envs={})

    # The detail prints both halves, so ".../envs/ks5" + "ks5" is visible as such.
    assert str(prefix) in checks[0].detail
    assert "'ks5'" in checks[0].detail
    assert "holds" in checks[0].fix


def test_configured_path_reports_its_origin_and_prefix(tmp_path, monkeypatch):
    prefix = make_env(tmp_path / "envs", "ks5")
    machine = MachineProfile(
        name="rig", conda_envs={"sorting": EnvLocation("ks5", tmp_path / "envs")}
    )
    installed = {spec.module: "1.0" for spec in KS.packages}
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed(probe_payload(installed)),
    )

    checks = doctor.check_env(KS, machine, envs={})

    assert status_of(checks, "env:ks5") == doctor.OK
    assert str(prefix) in checks[0].detail
    assert doctor.ORIGIN_CONFIGURED in checks[0].detail


def test_unknown_role_key_warns_instead_of_being_ignored(tmp_path):
    """config.py drops unknown keys silently; a typo must not read as a default."""
    machine = MachineProfile(name="rig", conda_envs={"sortin": named("ks5")})

    checks = doctor.check_env_names(machine)

    assert status_of(checks, "conda_envs") == doctor.WARN
    assert "sortin" in checks[0].detail
    assert "sorting" in checks[0].detail  # the valid roles are listed


def test_known_roles_produce_no_finding():
    machine = MachineProfile(
        name="rig", conda_envs={"sorting": named("ks5"), "curation": named("phy")}
    )
    assert doctor.check_env_names(machine) == []


def test_prefix_without_interpreter_is_reported_not_probed(tmp_path):
    """A half-deleted env directory is a finding, not a crash."""
    (tmp_path / "kilosort4").mkdir()
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)

    report = doctor.inspect_env(KS, machine, envs={})
    checks = doctor.env_checks(report)

    assert report.python is None
    assert checks[0].status == doctor.MISSING
    assert "no interpreter" in checks[0].detail


# ---------------------------------------------------------------------------
# Package checks
# ---------------------------------------------------------------------------


def test_the_fix_names_the_command_that_actually_installs_each_package(
    tmp_path, monkeypatch
):
    """`pip install -e .` skips the extras, so the fix must name the right one.

    This is the failure it exists for: run the obvious command, then be told
    packages are still missing with no hint of which install brings them in.
    """
    make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)
    installed = {"numpy": "1.0"}  # everything else absent
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed(probe_payload(installed)),
    )

    checks = {check.name: check for check in doctor.check_env(KS, machine, envs={})}

    # The GPU-free stack is a base dependency: a plain editable install gets it.
    for module in ("spikeinterface", "neo", "probeinterface"):
        assert "-e . --" in checks[f"kilosort4/{module}"].fix
    assert '".[sorting]"' in checks["kilosort4/kilosort"].fix
    assert '".[dev]"' in checks["kilosort4/pytest"].fix
    # A package with its own upstream cites it, never the environment's: telling
    # someone to read Kilosort's install page about neo is worse than no link.
    assert doctor.SPIKEINTERFACE_DOCS in checks["kilosort4/spikeinterface"].fix
    assert doctor.NEO_DOCS in checks["kilosort4/neo"].fix
    assert KS.docs not in checks["kilosort4/spikeinterface"].fix
    # No upstream worth citing -> the command alone, no link at all.
    assert "see" not in checks["kilosort4/pytest"].fix
    # torch is deliberately not a declared dependency: the wheel depends on the
    # GPU, so no pip command here would be right. Cite upstream instead.
    assert "pip install" not in checks["kilosort4/torch"].fix
    assert KS.docs in checks["kilosort4/torch"].fix


def test_probeinterface_is_checked_even_though_spikeinterface_requires_it():
    """Relying on the transitive requirement would not actually detect its loss.

    The probe uses ``find_spec``, which does not execute the module, so a
    spikeinterface whose probeinterface was removed underneath it still reports
    present. The failure would then land as a raw ModuleNotFoundError out of
    make_probe.py, which has no ImportError handler.
    """
    modules = {p.module for p in KS.packages}

    assert {"probeinterface", "spikeinterface"} <= modules


def test_find_spec_does_not_detect_a_broken_dependency(tmp_path, monkeypatch):
    """The reason the row above cannot be dropped, pinned as behaviour.

    Uses a package that exists nowhere rather than naming the real ones: asserting
    that probeinterface is *absent* only held on a machine where it happened not
    to be installed, so this test passed on a bare laptop and failed on any
    machine actually set up to sort.
    """
    import importlib
    from importlib import util

    pkg = tmp_path / "pretend_sorter"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import pretend_sorter_backend\n", encoding="utf-8")
    monkeypatch.syspath_prepend(tmp_path)

    # find_spec locates the package without executing it, so the missing import
    # inside is invisible -- the package reports "present".
    assert util.find_spec("pretend_sorter") is not None
    assert util.find_spec("pretend_sorter_backend") is None

    # It only surfaces on a real import, which is where doctor cannot afford to
    # find out: hence a row of its own for every package this repo imports.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pretend_sorter")


def test_base_dependency_hints_a_plain_editable_install():
    numpy = next(p for p in KS.packages if p.module == "numpy")
    assert numpy.extra is None
    assert numpy.install_hint.endswith("-e .")


def test_undeclared_packages_offer_no_install_command():
    undeclared = [p for p in ENV_SPECS_PACKAGES if not p.declared]
    assert {p.module for p in undeclared} == {"torch", "phy"}
    assert all(p.install_hint is None for p in undeclared)


def test_every_extra_tag_matches_pyproject():
    """The hint is only useful while it is true, and pyproject moves without us.

    Reads the real file rather than restating it: a package moved between the
    base set and an extra must fail here, not mislead someone on the rig.
    """
    tomllib = pytest.importorskip("tomllib")
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    def distributions(requirements):
        return {
            re.split(r"[<>=!~\[ ]", req, maxsplit=1)[0].lower() for req in requirements
        }

    base = distributions(project["dependencies"])
    extras = {
        name: distributions(reqs)
        for name, reqs in project["optional-dependencies"].items()
    }

    for spec in ENV_SPECS_PACKAGES:
        if not spec.declared:  # torch, phy -- see PackageSpec.declared
            continue
        dist = spec.dist.lower()
        if spec.extra is None:
            assert dist in base, f"{dist} is tagged base but pyproject disagrees"
        else:
            assert dist in extras[spec.extra], f"{dist} is not in extra {spec.extra}"


def test_missing_neo_blocks_but_missing_phy_only_warns(tmp_path, monkeypatch):
    """Required vs optional is the difference between failing and noting."""
    make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)
    installed = {spec.module: "1.0" for spec in KS.packages if spec.module != "neo"}
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed(probe_payload(installed)),
    )

    ks_checks = doctor.check_env(KS, machine, envs={})
    phy_checks = doctor.env_checks(doctor.EnvReport(PHY, probe={}))

    assert status_of(ks_checks, "kilosort4/neo") == doctor.MISSING
    assert status_of(phy_checks, "env:phy") == doctor.WARN
    assert doctor.exit_code(phy_checks) == 0


def test_optional_package_absence_does_not_block(tmp_path, monkeypatch):
    """pytest is declared not-required, so its absence must not fail the run."""
    make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)
    installed = {spec.module: "1.0" for spec in KS.packages if spec.module != "pytest"}
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed(probe_payload(installed)),
    )

    checks = doctor.check_env(KS, machine, envs={})

    assert status_of(checks, "kilosort4/pytest") == doctor.WARN
    assert doctor.exit_code(checks) == 0


def test_missing_package_names_where_the_import_lives(tmp_path, monkeypatch):
    """The report has to be checkable against the code, not just assertive."""
    make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)
    monkeypatch.setattr(
        doctor, "_run", lambda argv, timeout_s=60.0: completed(probe_payload({}))
    )

    checks = doctor.check_env(KS, machine, envs={})
    neo = next(check for check in checks if check.name == "kilosort4/neo")

    assert "io/blackrock.py:51" in neo.detail
    assert "extract_sync (Blackrock)" in neo.disables
    # Not align/validate: those read the edge *files* via catgt.read_edge_file
    # and never reach neo, so claiming them would overstate the damage.
    assert not any("align" in stage or "validate" in stage for stage in neo.disables)


def test_unparseable_probe_is_a_finding(tmp_path, monkeypatch):
    make_env(tmp_path, "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path)
    monkeypatch.setattr(
        doctor, "_run", lambda argv, timeout_s=60.0: completed("not json")
    )

    checks = doctor.check_env(KS, machine, envs={})

    assert checks[0].status == doctor.MISSING
    assert "could not be probed" in checks[0].detail


# ---------------------------------------------------------------------------
# GPU / torch
# ---------------------------------------------------------------------------


def test_cpu_only_torch_is_caught_when_the_profile_wants_cuda(monkeypatch):
    """The trap this tool exists for: a CPU torch imports fine and fails late."""
    monkeypatch.setattr(
        doctor, "nvidia_smi_info", lambda: {"name": "RTX 5060", "cuda": "12.8"}
    )
    machine = MachineProfile(name="windows_rig", device="cuda")
    torch_info = {"version": "2.3.0", "cuda_available": False, "cuda_version": None}

    checks = doctor.check_gpu(machine, torch_info)

    assert status_of(checks, "torch.cuda") == doctor.MISSING
    cuda_check = next(check for check in checks if check.name == "torch.cuda")
    assert "CPU-only" in cuda_check.detail
    assert "12.8" in (cuda_check.fix or "")


def test_cpu_only_torch_is_merely_a_warning_on_a_cpu_profile(monkeypatch):
    monkeypatch.setattr(doctor, "nvidia_smi_info", lambda: None)
    machine = MachineProfile(name="mac", device="cpu")
    torch_info = {"version": "2.3.0", "cuda_available": False, "cuda_version": None}

    checks = doctor.check_gpu(machine, torch_info)

    assert status_of(checks, "torch.cuda") == doctor.WARN
    assert doctor.exit_code(checks) == 0


def test_absent_gpu_is_fine_on_a_cpu_profile_and_fatal_on_a_cuda_one(monkeypatch):
    monkeypatch.setattr(doctor, "nvidia_smi_info", lambda: None)

    on_mac = doctor.check_gpu(MachineProfile(name="mac", device="cpu"))
    on_rig = doctor.check_gpu(MachineProfile(name="windows_rig", device="cuda"))

    assert status_of(on_mac, "gpu") == doctor.OK
    assert status_of(on_rig, "gpu") == doctor.MISSING


def test_working_cuda_torch_reports_the_device(monkeypatch):
    monkeypatch.setattr(doctor, "nvidia_smi_info", lambda: {"name": "RTX 5060"})
    torch_info = {
        "version": "2.3.0",
        "cuda_available": True,
        "cuda_version": "12.8",
        "device_name": "NVIDIA RTX 5060",
    }

    checks = doctor.check_gpu(MachineProfile(name="windows_rig"), torch_info)

    assert status_of(checks, "torch.cuda") == doctor.OK
    assert doctor.exit_code(checks) == 0


# ---------------------------------------------------------------------------
# CatGT / TPrime
# ---------------------------------------------------------------------------


def test_catgt_found_when_runit_is_present(tmp_path):
    (tmp_path / "runit.bat").write_text("")
    machine = MachineProfile(name="windows_rig", catgt_dir=tmp_path)

    checks = doctor.check_external_tools(machine)

    assert status_of(checks, "CatGT") == doctor.OK


def test_configured_but_empty_catgt_dir_warns_without_failing(tmp_path):
    machine = MachineProfile(name="windows_rig", catgt_dir=tmp_path)

    checks = doctor.check_external_tools(machine)

    assert status_of(checks, "CatGT") == doctor.WARN
    assert doctor.exit_code(checks) == 0


def test_unconfigured_tools_are_ok_because_a_fallback_exists():
    """hpc.yaml sets these to null on purpose; that is not a problem to report."""
    checks = doctor.check_external_tools(MachineProfile(name="hpc"))

    assert status_of(checks, "CatGT") == doctor.OK
    assert status_of(checks, "TPrime") == doctor.OK
    assert "NumPy" in next(c for c in checks if c.name == "CatGT").detail


def test_missing_catgt_directory_warns(tmp_path):
    machine = MachineProfile(name="windows_rig", catgt_dir=tmp_path / "nope")

    assert status_of(doctor.check_external_tools(machine), "CatGT") == doctor.WARN


# ---------------------------------------------------------------------------
# Windows shell
# ---------------------------------------------------------------------------


def test_execution_policy_check_is_skipped_off_windows():
    assert doctor.check_windows_shell(platform="darwin") == []


def test_restricted_execution_policy_warns_but_does_not_fail(monkeypatch):
    """It breaks `conda activate`, not anything this repo runs itself."""
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed("Restricted\nUndefined\n"),
    )

    checks = doctor.check_windows_shell(platform="win32")

    assert status_of(checks, "execution-policy") == doctor.WARN
    assert "Set-ExecutionPolicy" in (checks[0].fix or "")
    assert doctor.exit_code(checks) == 0


def test_remotesigned_execution_policy_is_ok(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_run",
        lambda argv, timeout_s=60.0: completed("RemoteSigned\nRemoteSigned\n"),
    )

    checks = doctor.check_windows_shell(platform="win32")

    assert status_of(checks, "execution-policy") == doctor.OK


# ---------------------------------------------------------------------------
# Machine paths
# ---------------------------------------------------------------------------


def test_cache_dir_on_the_network_share_warns(tmp_path):
    """temp.dat on the share is the documented way to make sorting crawl."""
    machine = MachineProfile(
        name="windows_rig",
        data_root=Path("Z:/server"),
        cache_dir=Path("Z:/server/cache"),
    )

    checks = doctor.check_machine_paths(machine)

    assert status_of(checks, "cache_dir location") == doctor.WARN


def test_unc_cache_dir_warns():
    machine = MachineProfile(name="windows_rig", cache_dir=Path("//nas/share/cache"))

    names = [check.name for check in doctor.check_machine_paths(machine)]

    assert "cache_dir location" in names


def test_local_cache_dir_alongside_a_network_data_root_is_clean(tmp_path):
    machine = MachineProfile(
        name="windows_rig", data_root=Path("Z:/server"), cache_dir=tmp_path
    )

    names = [check.name for check in doctor.check_machine_paths(machine)]

    assert "cache_dir location" not in names


def test_absent_cache_dir_setting_blocks_sorting():
    checks = doctor.check_machine_paths(MachineProfile(name="broken"))

    assert status_of(checks, "cache_dir") == doctor.MISSING
    assert doctor.exit_code(checks) == 1


def test_unset_recording_roots_are_not_a_finding(tmp_path):
    """Every machine profile says "do not re-add data_root here" -- so their
    absence is the correct state, and reporting it trains the reader to skim."""
    machine = MachineProfile(name="windows_rig", cache_dir=tmp_path)

    names = [check.name for check in doctor.check_machine_paths(machine)]

    assert "data_root" not in names
    assert "output_root" not in names


def test_a_root_that_is_set_but_absent_is_still_reported(tmp_path):
    """Silence is for "not configured", not for "configured and broken"."""
    machine = MachineProfile(
        name="rig", data_root=tmp_path / "unmounted", cache_dir=tmp_path
    )

    assert status_of(doctor.check_machine_paths(machine), "data_root") == doctor.WARN


# ---------------------------------------------------------------------------
# conda itself, and the verdict helpers
# ---------------------------------------------------------------------------


def test_absent_conda_is_survivable_when_envs_are_shared_by_path(monkeypatch, tmp_path):
    """Prefix interpreters run without conda's help, so this is not fatal."""
    monkeypatch.setattr(doctor, "conda_executable", lambda: None)

    shared = doctor.check_conda(MachineProfile(name="rig", conda_envs_dir=tmp_path))
    alone = doctor.check_conda(MachineProfile(name="mac"))

    assert status_of(shared, "conda") == doctor.WARN
    assert status_of(alone, "conda") == doctor.MISSING


def test_absent_conda_is_survivable_from_inside_an_active_env(monkeypatch, tmp_path):
    """Same reasoning as the shared dir: we already hold a working interpreter."""
    monkeypatch.setattr(doctor, "conda_executable", lambda: None)
    prefix = make_conda_prefix(tmp_path, "kilosort4")

    checks = doctor.check_conda(MachineProfile(name="mac"), active=prefix)

    assert status_of(checks, "conda") == doctor.WARN
    assert str(prefix) in checks[0].detail


@pytest.mark.parametrize(
    "statuses, expected",
    [
        ([], doctor.OK),
        ([doctor.OK, doctor.OK], doctor.OK),
        ([doctor.OK, doctor.WARN], doctor.WARN),
        ([doctor.WARN, doctor.MISSING], doctor.MISSING),
    ],
)
def test_worst_status(statuses, expected):
    checks = [doctor.Check(name=str(i), status=s) for i, s in enumerate(statuses)]
    assert doctor.worst_status(checks) == expected


def test_exit_code_ignores_warnings():
    warned = [doctor.Check(name="a", status=doctor.WARN)]
    blocked = [doctor.Check(name="b", status=doctor.MISSING)]

    assert doctor.exit_code(warned) == 0
    assert doctor.exit_code(blocked) == 1


def test_importing_doctor_pulls_in_nothing_heavy():
    """The whole point: this must run on a machine with no sorting stack."""
    import sys

    for module in ("torch", "kilosort", "spikeinterface"):
        assert module not in sys.modules


def _blackrock_session(tmp_path, extra: str = ""):
    from spikesorting import _config as cfg

    machines = tmp_path / "machines"
    machines.mkdir(parents=True, exist_ok=True)
    (machines / "m.yaml").write_text(f"cache_dir: '{tmp_path / 'c'}'\n", encoding="utf-8")
    spike = tmp_path / "HUB.ns6"
    spike.write_bytes(b"")
    sync = tmp_path / "NSP.ns5"
    sync.write_bytes(b"")
    (tmp_path / "s.yaml").write_text(
        f"session: s\nblackrock_dir: '{tmp_path}'\n{extra}"
        f"blackrock:\n  sync_file: '{sync}'\n  spike_file: '{spike}'\n",
        encoding="utf-8",
    )
    return cfg.load_session_config(tmp_path / "s.yaml", "m", tmp_path)


def test_a_utah_session_with_no_channel_map_cannot_sort(tmp_path):
    # The silent-wrong-geometry hole. setup_probe raises on it, but that is an
    # hour into a run; check_env is where it should be caught, so it blocks rather
    # than warns and the exit code says so.
    import doctor  # from tools/, added to sys.path by conftest

    checks = doctor.check_session(_blackrock_session(tmp_path))

    blocking = [c for c in checks if c.status == doctor.MISSING and "probe" in c.detail.lower()]
    assert blocking, [(c.status, c.detail) for c in checks]
    assert "cmp_file" in blocking[0].detail
    assert "make_probe.py" in (blocking[0].fix or "")
    assert doctor.exit_code(checks) == 1


def test_a_session_that_does_not_sort_needs_no_channel_map(tmp_path):
    # "Extract the pulses, do not sort" needs no geometry, so demanding a map
    # would block a session that is complete as it stands.
    import doctor

    session = _blackrock_session(tmp_path, extra="kilosort_on_blackrock: false\n")
    checks = doctor.check_session(session)

    assert not [c for c in checks if "probe" in c.detail.lower()]
    assert doctor.exit_code(checks) == 0


# ---------------------------------------------------------------------------
# Recorded band: the measurement the waveform defaults rest on
# ---------------------------------------------------------------------------


def _write_band_binary(path, fs, n_chan, kind: str):
    """A broadband recording, or one already high-passed like an AP band."""
    import numpy as np

    n = int(4 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    spikes = rng.normal(0, 60, size=(n, n_chan))
    if kind == "broadband":
        lfp = 900.0 * np.sin(2 * np.pi * 7.0 * t)[:, None]
        signal = spikes + lfp
    else:
        signal = spikes
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(signal, dtype=np.int16).tobytes())


def test_the_band_a_recording_holds_is_measured_not_assumed(tmp_path):
    # 300 Hz on Blackrock and nothing on Neuropixels are defaults about *settings*
    # -- NP 1.0's AP filter is an imro bit, a Blackrock group's band is set in
    # Central -- so they have to be checkable against the actual file.
    import doctor

    fs, n_chan = 30000.0, 4
    broad, spike_band = tmp_path / "broad.bin", tmp_path / "ap.bin"
    _write_band_binary(broad, fs, n_chan, "broadband")
    _write_band_binary(spike_band, fs, n_chan, "highpassed")

    below_broad = doctor.compute_band_fraction(broad, n_chan, fs, 300.0)
    below_ap = doctor.compute_band_fraction(spike_band, n_chan, fs, 300.0)

    assert below_broad > doctor._BAND_ALREADY_FILTERED > below_ap


def test_a_broadband_stream_with_no_highpass_configured_is_reported(tmp_path):
    # The failure this exists to catch: a run that looks normal until its mean
    # waveforms come out sitting on the LFP.
    import json

    import doctor

    fs, n_chan = 30000.0, 4
    from spikesorting import _config as cfg

    (tmp_path / "machines").mkdir(parents=True, exist_ok=True)
    (tmp_path / "machines" / "m.yaml").write_text(
        f"cache_dir: '{tmp_path / 'c'}'\n", encoding="utf-8"
    )
    (tmp_path / "off.yaml").write_text(
        f"session: s\nblackrock_dir: '{tmp_path}'\n"
        f"blackrock:\n  spike_file: '{tmp_path / 'HUB.ns6'}'\n"
        "  waveforms:\n    highpass_hz: null\n",
        encoding="utf-8",
    )
    session = cfg.load_session_config(tmp_path / "off.yaml", "m", tmp_path)
    sorted_dir = session.paths.sorted_for("blackrock")
    sorted_dir.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "raw" / "HUB.bin"
    _write_band_binary(binary, fs, n_chan, "broadband")
    (sorted_dir / "run_info.json").write_text(
        json.dumps({"binary": str(binary), "settings": {"n_chan_bin": n_chan, "fs": fs}}),
        encoding="utf-8",
    )

    checks = doctor.check_recorded_band(session)

    assert [c.status for c in checks] == [doctor.WARN]
    assert "broadband" in checks[0].detail
    assert "highpass_hz: 300" in (checks[0].fix or "")
    # Advisory, not blocking: the export still produces a usable mean.
    assert doctor.exit_code(checks) == 0


def test_a_spike_band_stream_that_matches_its_config_is_fine(tmp_path):
    import json

    import doctor

    fs, n_chan = 30000.0, 4
    session = _blackrock_session(tmp_path)      # blackrock defaults to 300 Hz
    sorted_dir = session.paths.sorted_for("blackrock")
    sorted_dir.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "raw" / "HUB.bin"
    _write_band_binary(binary, fs, n_chan, "broadband")
    (sorted_dir / "run_info.json").write_text(
        json.dumps({"binary": str(binary), "settings": {"n_chan_bin": n_chan, "fs": fs}}),
        encoding="utf-8",
    )

    checks = doctor.check_recorded_band(session)

    assert [c.status for c in checks] == [doctor.OK]
