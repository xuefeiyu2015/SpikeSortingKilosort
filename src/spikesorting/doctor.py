"""Environment health check -- is this machine able to run the pipeline?

Answers one question: which pieces of the environment are set up, which are
missing, and what does each gap stop working. It never installs anything and
never creates an environment; installing Kilosort and phy stays a human task
done from the upstream instructions, which change often enough that a copy
frozen into this repo would go stale and mislead.

Two constraints shape the implementation.

**It must not depend on ``conda activate``.** That command is backed by
``conda.ps1``, so under PowerShell's default ``Restricted`` execution policy it
fails outright -- which is exactly the state a fresh Windows box is in, and
exactly when this check is worth running. So environments are inspected from the
outside: resolve the prefix, then run :data:`_PROBE_SRC` in *that* prefix's
interpreter via ``subprocess`` with no shell and no ``.ps1`` involved. That also
keeps the multi-second ``import torch`` inside a throwaway subprocess, leaving
the lazy-import guarantee of this package intact.

**A shared environment may be invisible to ``conda env list``.** That command
reads ``~/.conda/environments.txt`` plus the *calling* user's ``envs_dirs``, so
an env the admin account created with ``conda create -p`` is not necessarily
registered for the lab account. Discovery therefore also stats
``<machine.conda_envs_dir>/<name>`` directly; without that, a perfectly good
shared env is reported as missing.

Computation only -- every function returns :class:`Check` values and prints
nothing. ``scripts/check_env.py`` does the rendering.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import MachineProfile, SessionConfig

__all__ = [
    "Check",
    "PackageSpec",
    "EnvSpec",
    "ENV_SPECS",
    "KILOSORT_DOCS",
    "PHY_DOCS",
    "conda_executable",
    "list_conda_envs",
    "env_python",
    "check_conda",
    "resolve_env",
    "probe_env",
    "EnvReport",
    "inspect_env",
    "env_checks",
    "check_env",
    "check_gpu",
    "check_external_tools",
    "check_windows_shell",
    "check_machine_paths",
    "check_session",
    "run_all",
    "worst_status",
    "exit_code",
]

# Official setup instructions. Reported instead of a hardcoded install command:
# the right torch wheel depends on the GPU, and phy has already moved off conda
# once, so upstream is the only source that stays correct.
KILOSORT_DOCS = "https://github.com/MouseLand/Kilosort#installation"
PHY_DOCS = "https://phy.readthedocs.io/en/latest/installation/"
CATGT_DOCS = "https://billkarsh.github.io/SpikeGLX/More_help/CatGT_ReadMe.html"
TPRIME_DOCS = "https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/"

# Sections, in report order.
SECTION_ENVS = "environments"
SECTION_GPU = "gpu"
SECTION_TOOLS = "command-line tools"
SECTION_SHELL = "shell"
SECTION_PATHS = "paths"
SECTION_SESSION = "session"

OK = "ok"
MISSING = "missing"
WARN = "warn"

#: Status ranked by severity, so a set of checks can be reduced to one verdict.
_SEVERITY = {OK: 0, WARN: 1, MISSING: 2}


@dataclass(frozen=True)
class Check:
    """One finding. ``status`` is ``ok`` / ``warn`` / ``missing``.

    ``warn`` never fails the run: it marks something degraded but covered by a
    fallback, or irrelevant on this machine. ``missing`` means a pipeline stage
    genuinely cannot run.
    """

    name: str
    status: str
    detail: str = ""
    section: str = ""
    #: Pipeline stages that stop working while this is unresolved.
    disables: tuple[str, ...] = ()
    #: What to do about it -- usually an upstream documentation URL.
    fix: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass(frozen=True)
class PackageSpec:
    """A Python package the pipeline reaches for, and what its absence costs."""

    module: str
    #: Where in this repo the import happens, so the report can be checked.
    reached_from: str
    disables: tuple[str, ...] = ()
    #: Distribution name, when it differs from the module name (PyYAML -> yaml).
    distribution: str | None = None
    #: False for packages that are nice to have but block no stage.
    required: bool = True

    @property
    def dist(self) -> str:
        return self.distribution or self.module


@dataclass(frozen=True)
class EnvSpec:
    """A conda environment the workflow expects, and how to set it up."""

    name: str
    purpose: str
    python: str
    docs: str
    packages: tuple[PackageSpec, ...]
    #: Stages lost when the whole environment is absent. Stated rather than
    #: aggregated from the packages, which produces an unreadable union.
    disables: tuple[str, ...] = ()
    #: False for environments whose absence blocks no scripted stage (phy:
    #: curation is manual, so a missing phy env is a warning, not a failure).
    required: bool = True
    #: Whether to pay the cost of importing torch in the probe.
    probe_torch: bool = False


ENV_SPECS: tuple[EnvSpec, ...] = (
    EnvSpec(
        name="kilosort4",
        purpose="sorting, Blackrock/SpikeGLX IO, alignment, export",
        python="3.11",
        docs=KILOSORT_DOCS,
        probe_torch=True,
        disables=(
            "extract_sync",
            "sort_neuropixels",
            "sort_blackrock",
            "align",
            "validate",
            "export",
        ),
        packages=(
            PackageSpec("numpy", "module scope throughout", ("all stages",)),
            PackageSpec(
                "yaml", "config.py:26", ("all stages",), distribution="PyYAML"
            ),
            PackageSpec("scipy", "probes/io.py:129", ("make_probe.py from-mat",)),
            PackageSpec(
                "pandas",
                "export/curated.py:18, export/final.py:26",
                ("export",),
            ),
            PackageSpec(
                "matplotlib",
                "plots/*.py, make_probe.py:104",
                ("figures", "make_probe --plot"),
            ),
            PackageSpec(
                "kilosort",
                "sort.py:95",
                ("sort_neuropixels", "sort_blackrock"),
            ),
            PackageSpec(
                "torch",
                "sort.py:45",
                ("sort_neuropixels", "sort_blackrock"),
            ),
            PackageSpec(
                "spikeinterface",
                "sort.py:173, io/blackrock.py:189, preprocess.py:33",
                ("sort_blackrock", "preprocessing"),
            ),
            PackageSpec(
                "neo",
                "io/blackrock.py:51",
                ("extract_sync (Blackrock)", "align", "validate"),
            ),
            PackageSpec(
                "probeinterface",
                "probes/common.py:25, probes/io.py:167",
                ("probe objects", ".prb export"),
            ),
            PackageSpec("pytest", "tests/", ("test suite",), required=False),
        ),
    ),
    EnvSpec(
        name="phy",
        purpose="manual curation (step 6, not scripted)",
        python="3.12",
        docs=PHY_DOCS,
        required=False,
        disables=("curation",),
        packages=(PackageSpec("phy", "run by hand after sorting", ("curation",)),),
    ),
)


# ---------------------------------------------------------------------------
# Probing another environment's interpreter
# ---------------------------------------------------------------------------

# Runs inside the *target* environment. Detects packages with find_spec so that
# nothing heavy is imported; torch is the one exception, imported deliberately
# because a CPU-only build is indistinguishable from a CUDA one until you ask.
_PROBE_SRC = r"""
import json, sys
from importlib import util as _util

names = json.loads(sys.argv[1])
probe_torch = json.loads(sys.argv[2])

try:
    from importlib.metadata import version as _version
except Exception:
    _version = None

out = {
    "python": "%d.%d.%d" % sys.version_info[:3],
    "executable": sys.executable,
    "packages": {},
    "torch": None,
}

for module, dist in names:
    try:
        present = _util.find_spec(module) is not None
    except Exception:
        present = False
    ver = None
    if present and _version is not None:
        try:
            ver = _version(dist)
        except Exception:
            ver = None
    out["packages"][module] = {"present": present, "version": ver}

if probe_torch and out["packages"].get("torch", {}).get("present"):
    try:
        import torch
        available = bool(torch.cuda.is_available())
        out["torch"] = {
            "version": torch.__version__,
            "cuda_available": available,
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_name": torch.cuda.get_device_name(0) if available else None,
        }
    except Exception as exc:
        out["torch"] = {"error": "%s: %s" % (type(exc).__name__, exc)}

print(json.dumps(out))
"""


def _run(argv: Sequence[str], timeout_s: float = 60.0) -> subprocess.CompletedProcess | None:
    """Run a command, returning None when it cannot be run at all.

    No ``shell=True`` anywhere in this module: that is what keeps the check
    working on a Windows box whose PowerShell execution policy blocks scripts.
    """
    try:
        return subprocess.run(
            list(argv), capture_output=True, text=True, timeout=timeout_s
        )
    except (OSError, subprocess.SubprocessError):
        return None


def conda_executable() -> str | None:
    """Path to conda, from ``CONDA_EXE`` or the PATH. None when absent."""
    exe = os.environ.get("CONDA_EXE")
    if exe and Path(exe).exists():
        return exe
    return shutil.which("conda")


def list_conda_envs(conda: str | None = None) -> dict[str, Path]:
    """Map env name -> prefix as *this account's* conda sees it.

    Uses ``conda info --json`` rather than ``conda env list --json`` because it
    also reports ``root_prefix``: the root environment is named ``base``, not
    after its directory, and keying it by basename would invent an env called
    e.g. ``anaconda3``.

    Empty when conda is missing or the query fails. Deliberately not the only
    discovery source: see the module docstring on shared environments.
    """
    conda = conda or conda_executable()
    if conda is None:
        return {}
    result = _run([conda, "info", "--json"])
    if result is None or result.returncode != 0:
        return {}
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError):
        return {}
    root = payload.get("root_prefix")
    envs: dict[str, Path] = {}
    for entry in payload.get("envs", []):
        prefix = Path(entry)
        name = "base" if root and entry == root else prefix.name
        envs.setdefault(name, prefix)
    return envs


def env_python(prefix: Path) -> Path | None:
    """The interpreter inside ``prefix``, or None if the prefix has no usable one.

    Checks both layouts rather than branching on the host platform, which keeps
    it testable and copes with a prefix that exists but was never populated.
    """
    for candidate in (
        prefix / "python.exe",
        prefix / "bin" / "python",
        prefix / "bin" / "python3",
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_env(
    name: str, machine: MachineProfile, envs: dict[str, Path] | None = None
) -> tuple[Path | None, bool]:
    """Locate an environment. Returns ``(prefix, is_shared)``.

    Conda's own answer wins when it has one, because that is what
    ``conda activate <name>`` would resolve to. The shared directory is the
    fallback, which is the case that matters on the rig: an env created by the
    admin account is not in the lab account's ``conda env list``.
    """
    envs = list_conda_envs() if envs is None else envs
    prefix = envs.get(name)
    if prefix is None and machine.conda_envs_dir is not None:
        candidate = machine.conda_envs_dir / name
        if candidate.is_dir():
            prefix = candidate
    if prefix is None:
        return None, False
    shared = machine.conda_envs_dir is not None and _is_within(
        prefix, machine.conda_envs_dir
    )
    return prefix, shared


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return path.resolve().is_relative_to(parent.resolve())
    except (OSError, ValueError):
        return False


def probe_env(
    python: Path, packages: Iterable[PackageSpec], probe_torch: bool = False
) -> dict[str, Any]:
    """Ask ``python`` which of ``packages`` it has. Returns the parsed probe JSON.

    On failure returns ``{"error": ...}`` rather than raising: a broken
    environment is a finding to report, not an exception to propagate.
    """
    names = [[spec.module, spec.dist] for spec in packages]
    result = _run(
        [
            str(python),
            "-c",
            _PROBE_SRC,
            json.dumps(names),
            json.dumps(bool(probe_torch)),
        ],
        # importing torch on a cold filesystem is slow; this is not a hang.
        timeout_s=180.0,
    )
    if result is None:
        return {"error": f"could not run {python}"}
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        return {"error": detail[-1] if detail else f"exit {result.returncode}"}
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError):
        return {"error": "probe returned unparseable output"}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_conda(machine: MachineProfile) -> list[Check]:
    """Is conda reachable at all?

    Worth its own line: without it every environment would otherwise be reported
    as "not found", which points at the wrong problem. Not fatal on its own when
    the machine declares a shared ``conda_envs_dir``, since environments there
    are found by path and their interpreters run without conda's help.
    """
    conda = conda_executable()
    if conda is not None:
        return [Check(name="conda", status=OK, detail=conda, section=SECTION_ENVS)]
    return [
        Check(
            name="conda",
            status=WARN if machine.has_shared_envs else MISSING,
            detail="not found via CONDA_EXE or PATH",
            section=SECTION_ENVS,
            disables=() if machine.has_shared_envs else ("all stages",),
            fix="install Miniconda/Anaconda, or set CONDA_EXE",
        )
    ]


@dataclass(frozen=True)
class EnvReport:
    """Raw result of looking at one environment, before it becomes findings.

    Exists so the environment is probed exactly once: importing torch costs
    seconds, and both the package checks and the CUDA check need that same probe.
    """

    spec: EnvSpec
    prefix: Path | None = None
    shared: bool = False
    python: Path | None = None
    #: Parsed probe JSON, or ``{"error": ...}``, or ``{}`` when never run.
    probe: dict[str, Any] = field(default_factory=dict)

    @property
    def torch_info(self) -> dict[str, Any] | None:
        return self.probe.get("torch")


def inspect_env(
    spec: EnvSpec, machine: MachineProfile, envs: dict[str, Path] | None = None
) -> EnvReport:
    """Resolve an environment and probe it once."""
    prefix, shared = resolve_env(spec.name, machine, envs)
    if prefix is None:
        return EnvReport(spec, probe={})
    python = env_python(prefix)
    if python is None:
        return EnvReport(spec, prefix, shared, probe={})
    return EnvReport(
        spec, prefix, shared, python, probe_env(python, spec.packages, spec.probe_torch)
    )


def check_env(
    spec: EnvSpec, machine: MachineProfile, envs: dict[str, Path] | None = None
) -> list[Check]:
    """Health of one conda environment: the env itself, then its packages."""
    return env_checks(inspect_env(spec, machine, envs))


def env_checks(report: EnvReport) -> list[Check]:
    """Turn an :class:`EnvReport` into findings. Pure."""
    spec = report.spec
    absent_status = MISSING if spec.required else WARN
    provenance = "shared" if report.shared else "per-user"

    if report.prefix is None:
        return [
            Check(
                name=f"env:{spec.name}",
                status=absent_status,
                detail=(
                    "not found via conda env list or the machine's conda_envs_dir"
                    f" -- needed for {spec.purpose}"
                ),
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=f"create the '{spec.name}' env (python {spec.python}) per {spec.docs}",
            )
        ]

    if report.python is None:
        return [
            Check(
                name=f"env:{spec.name}",
                status=absent_status,
                detail=f"{report.prefix} ({provenance}) exists but holds no interpreter",
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=f"recreate the '{spec.name}' env per {spec.docs}",
            )
        ]

    probe = report.probe or {}
    if "error" in probe:
        return [
            Check(
                name=f"env:{spec.name}",
                status=absent_status,
                detail=f"{report.prefix} ({provenance}) could not be probed: {probe['error']}",
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=f"recreate the '{spec.name}' env per {spec.docs}",
            )
        ]

    prefix = report.prefix
    found = probe.get("packages", {})
    checks = [
        Check(
            name=f"env:{spec.name}",
            status=OK,
            detail=f"{prefix} ({provenance}, python {probe.get('python', '?')})",
            section=SECTION_ENVS,
        )
    ]
    for package in spec.packages:
        entry = found.get(package.module, {})
        if entry.get("present"):
            version = entry.get("version") or "version unknown"
            checks.append(
                Check(
                    name=f"{spec.name}/{package.module}",
                    status=OK,
                    detail=version,
                    section=SECTION_ENVS,
                )
            )
        else:
            checks.append(
                Check(
                    name=f"{spec.name}/{package.module}",
                    status=MISSING if package.required else WARN,
                    detail=f"not installed (used by {package.reached_from})",
                    section=SECTION_ENVS,
                    disables=package.disables,
                    fix=f"install into '{spec.name}' -- see {spec.docs}",
                )
            )
    return checks


def nvidia_smi_info() -> dict[str, str] | None:
    """GPU name, driver version and max supported CUDA, or None without a GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    info: dict[str, str] = {}
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version",
            "--format=csv,noheader",
        ]
    )
    if query is not None and query.returncode == 0 and query.stdout.strip():
        first = query.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if parts:
            info["name"] = parts[0]
        if len(parts) > 1:
            info["driver"] = parts[1]
    banner = _run(["nvidia-smi"])
    if banner is not None and banner.returncode == 0:
        match = re.search(r"CUDA Version:\s*([\d.]+)", banner.stdout)
        if match:
            info["cuda"] = match.group(1)
    return info or None


def check_gpu(
    machine: MachineProfile, torch_info: dict[str, Any] | None = None
) -> list[Check]:
    """Is there a CUDA GPU, and does the installed torch actually reach it?

    The trap this exists for: a CPU-only torch imports perfectly and fails only
    once Kilosort asks for the device, deep inside a long job.
    """
    wants_cuda = machine.device.startswith("cuda")
    gpu = nvidia_smi_info()
    checks: list[Check] = []

    if gpu is None:
        checks.append(
            Check(
                name="gpu",
                status=MISSING if wants_cuda else OK,
                detail=(
                    "no nvidia-smi; sorting is unavailable on this machine"
                    + ("" if wants_cuda else " (profile asks for cpu, so that is fine)")
                ),
                section=SECTION_GPU,
                disables=("sort_neuropixels", "sort_blackrock") if wants_cuda else (),
                fix=(
                    f"machine profile '{machine.name}' sets device: {machine.device}; "
                    "run sorting on a CUDA machine or set device: cpu"
                )
                if wants_cuda
                else None,
            )
        )
    else:
        described = gpu.get("name", "NVIDIA GPU")
        if "driver" in gpu:
            described += f", driver {gpu['driver']}"
        if "cuda" in gpu:
            described += f", CUDA up to {gpu['cuda']}"
        checks.append(
            Check(name="gpu", status=OK, detail=described, section=SECTION_GPU)
        )

    if torch_info is None:
        return checks

    if "error" in torch_info:
        checks.append(
            Check(
                name="torch.cuda",
                status=MISSING if wants_cuda else WARN,
                detail=f"torch present but unusable: {torch_info['error']}",
                section=SECTION_GPU,
                disables=("sort_neuropixels", "sort_blackrock"),
                fix=f"reinstall torch per {KILOSORT_DOCS}",
            )
        )
    elif torch_info.get("cuda_available"):
        checks.append(
            Check(
                name="torch.cuda",
                status=OK,
                detail=(
                    f"torch {torch_info.get('version', '?')} "
                    f"(CUDA {torch_info.get('cuda_version')}) -> "
                    f"{torch_info.get('device_name')}"
                ),
                section=SECTION_GPU,
            )
        )
    else:
        driver_cuda = (gpu or {}).get("cuda")
        hint = f"this driver supports CUDA up to {driver_cuda}; " if driver_cuda else ""
        checks.append(
            Check(
                name="torch.cuda",
                status=MISSING if wants_cuda else WARN,
                detail=(
                    f"torch {torch_info.get('version', '?')} is a CPU-only build "
                    f"(torch.version.cuda={torch_info.get('cuda_version')})"
                ),
                section=SECTION_GPU,
                disables=("sort_neuropixels", "sort_blackrock"),
                fix=f"{hint}install a CUDA torch wheel per {KILOSORT_DOCS}",
            )
        )
    return checks


def check_external_tools(machine: MachineProfile) -> list[Check]:
    """CatGT and TPrime. Never fatal -- both have pure-NumPy fallbacks."""
    from .sync import catgt, tprime

    checks: list[Check] = []
    for label, directory, locator, docs, fallback in (
        (
            "CatGT",
            machine.catgt_dir,
            catgt.catgt_script,
            CATGT_DOCS,
            "sync/edges.py detects the same edges in pure NumPy",
        ),
        (
            "TPrime",
            machine.tprime_dir,
            tprime.tprime_executable,
            TPRIME_DOCS,
            "align falls back to a least-squares fit of the same matched edges",
        ),
    ):
        if directory is None:
            checks.append(
                Check(
                    name=label,
                    status=OK,
                    detail=f"not configured for '{machine.name}'; {fallback}",
                    section=SECTION_TOOLS,
                )
            )
            continue
        if not Path(directory).is_dir():
            checks.append(
                Check(
                    name=label,
                    status=WARN,
                    detail=f"configured directory does not exist: {directory}",
                    section=SECTION_TOOLS,
                    fix=f"install {label} there, or set the dir to null -- {docs}",
                )
            )
            continue
        try:
            checks.append(
                Check(
                    name=label,
                    status=OK,
                    detail=str(locator(directory)),
                    section=SECTION_TOOLS,
                )
            )
        except FileNotFoundError as error:
            checks.append(
                Check(
                    name=label,
                    status=WARN,
                    detail=str(error),
                    section=SECTION_TOOLS,
                    fix=f"reinstall {label} -- {docs}",
                )
            )
    return checks


def check_windows_shell(platform: str | None = None) -> list[Check]:
    """PowerShell execution policy. Empty off Windows.

    Nothing this repo drives needs the policy relaxed -- the pipeline and this
    very check invoke executables directly. What breaks is the human's
    ``conda activate kilosort4``, which is step one of the documented workflow,
    so it is worth naming before it is hit.
    """
    if (platform or sys.platform) != "win32":
        return []
    result = _run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-ExecutionPolicy; Get-ExecutionPolicy -Scope CurrentUser",
        ]
    )
    if result is None or result.returncode != 0:
        return [
            Check(
                name="execution-policy",
                status=WARN,
                detail="could not query PowerShell execution policy",
                section=SECTION_SHELL,
            )
        ]
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    effective = lines[0] if lines else "Unknown"
    scoped = lines[1] if len(lines) > 1 else "Undefined"
    if effective in ("Restricted", "AllSigned"):
        return [
            Check(
                name="execution-policy",
                status=WARN,
                detail=(
                    f"effective policy is {effective} (CurrentUser: {scoped}); "
                    "`conda activate kilosort4` will fail in PowerShell"
                ),
                section=SECTION_SHELL,
                fix=(
                    "in an elevated PowerShell: Set-ExecutionPolicy -Scope "
                    "CurrentUser -ExecutionPolicy RemoteSigned"
                ),
            )
        ]
    return [
        Check(
            name="execution-policy",
            status=OK,
            detail=f"{effective} (CurrentUser: {scoped})",
            section=SECTION_SHELL,
        )
    ]


_DRIVE_LETTER = re.compile(r"^([A-Za-z]):[\\/]")


def _drive_letter(path: Path) -> str | None:
    """Windows drive letter of ``path``, on any host.

    Deliberately not ``Path.drive``, which is always empty under POSIX: checking
    ``--machine windows_rig`` from the laptop is a normal thing to do, and the
    heuristic below has to work when it happens.
    """
    match = _DRIVE_LETTER.match(str(path))
    return match.group(1).lower() if match else None


def _looks_networked(path: Path, data_root: Path | None) -> bool:
    """Heuristic: is ``path`` on a share rather than a local disk?

    A UNC path is conclusive. Sharing a drive letter with ``data_root`` is not,
    but on this rig the data root *is* the network share, so it is the signal
    worth acting on.
    """
    text = str(path)
    if text.startswith("\\\\") or text.startswith("//"):
        return True
    if data_root is None:
        return False
    drive = _drive_letter(data_root)
    return drive is not None and _drive_letter(path) == drive


def check_machine_paths(machine: MachineProfile) -> list[Check]:
    """Roots and scratch declared by the machine profile."""
    checks: list[Check] = []
    for label, path in (
        ("data_root", machine.data_root),
        ("output_root", machine.output_root),
    ):
        if path is None:
            checks.append(
                Check(
                    name=label,
                    status=WARN,
                    detail=f"not set in machine profile '{machine.name}'",
                    section=SECTION_PATHS,
                )
            )
        elif path.exists():
            checks.append(
                Check(name=label, status=OK, detail=str(path), section=SECTION_PATHS)
            )
        else:
            checks.append(
                Check(
                    name=label,
                    status=WARN,
                    detail=f"does not exist: {path} (share not mounted?)",
                    section=SECTION_PATHS,
                    fix=f"mount it, or correct configs/machines/{machine.name}.yaml",
                )
            )

    cache = machine.cache_dir
    if cache is None:
        checks.append(
            Check(
                name="cache_dir",
                status=MISSING,
                detail="not set; Kilosort needs fast local scratch for temp.dat",
                section=SECTION_PATHS,
                disables=("sort_neuropixels", "sort_blackrock"),
                fix=f"set cache_dir in configs/machines/{machine.name}.yaml",
            )
        )
    else:
        if cache.exists():
            checks.append(
                Check(
                    name="cache_dir", status=OK, detail=str(cache), section=SECTION_PATHS
                )
            )
        else:
            checks.append(
                Check(
                    name="cache_dir",
                    status=WARN,
                    detail=f"does not exist yet: {cache} (created on first sort)",
                    section=SECTION_PATHS,
                )
            )
        if _looks_networked(cache, machine.data_root):
            checks.append(
                Check(
                    name="cache_dir location",
                    status=WARN,
                    detail=(
                        f"{cache} looks like it is on the network share; temp.dat "
                        "must be a local SSD"
                    ),
                    section=SECTION_PATHS,
                    fix=f"point cache_dir at a local disk in configs/machines/{machine.name}.yaml",
                )
            )
    return checks


def check_session(config: SessionConfig) -> list[Check]:
    """Session input files, delegated to the existing validator."""
    problems = config.missing_inputs()
    if not problems:
        return [
            Check(
                name=f"session:{config.session}",
                status=OK,
                detail="all declared inputs present",
                section=SECTION_SESSION,
            )
        ]
    return [
        Check(
            name=f"session:{config.session}",
            status=MISSING,
            detail=problem,
            section=SECTION_SESSION,
        )
        for problem in problems
    ]


def run_all(
    machine: MachineProfile, config: SessionConfig | None = None
) -> list[Check]:
    """Every check, in report order."""
    envs = list_conda_envs()
    checks: list[Check] = check_conda(machine)
    torch_info: dict[str, Any] | None = None

    for spec in ENV_SPECS:
        report = inspect_env(spec, machine, envs)
        checks.extend(env_checks(report))
        if spec.probe_torch and report.torch_info is not None:
            torch_info = report.torch_info

    checks.extend(check_gpu(machine, torch_info))
    checks.extend(check_external_tools(machine))
    checks.extend(check_windows_shell())
    checks.extend(check_machine_paths(machine))
    if config is not None:
        checks.extend(check_session(config))
    return checks


def worst_status(checks: Iterable[Check]) -> str:
    """The most severe status present, or ``ok`` when there is nothing to report."""
    return max((check.status for check in checks), key=lambda s: _SEVERITY.get(s, 0), default=OK)


def exit_code(checks: Iterable[Check]) -> int:
    """1 when any check is ``missing``; ``warn`` never fails the run."""
    return 1 if any(check.status == MISSING for check in checks) else 0
