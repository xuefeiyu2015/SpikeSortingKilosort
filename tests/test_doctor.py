"""Tests for the environment health check.

Pure compute: every subprocess the doctor would run is faked, so these need no
conda, no GPU, no Kilosort and no real recordings -- the same bar as the rest of
the suite. What is being checked is the *reasoning*: that a shared environment is
found the way the rig actually stores it, that a CPU-only torch is called out,
and that warnings never fail a run.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from spikesorting import doctor
from spikesorting.config import MachineProfile

KS = doctor.ENV_SPECS[0]
PHY = doctor.ENV_SPECS[1]


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

    found, shared = doctor.resolve_env("kilosort4", machine, envs={})

    assert found == prefix
    assert shared is True


def test_conda_listing_wins_over_the_shared_directory(tmp_path):
    """Whatever `conda activate <name>` would resolve to is what we report."""
    make_env(tmp_path / "shared", "kilosort4")
    own = make_env(tmp_path / "own", "kilosort4")
    machine = MachineProfile(name="windows_rig", conda_envs_dir=tmp_path / "shared")

    found, shared = doctor.resolve_env("kilosort4", machine, envs={"kilosort4": own})

    assert found == own
    assert shared is False


def test_env_missing_everywhere_resolves_to_none(tmp_path):
    machine = MachineProfile(name="mac")
    assert doctor.resolve_env("kilosort4", machine, envs={}) == (None, False)


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
    assert "align" in neo.disables


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
