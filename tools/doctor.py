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

**No single source knows about every environment.** Each role ("sorting",
"curation") is configured as an :class:`~spikesorting._config.EnvLocation` -- a
*name* and, optionally, a *path* to the directory holding it -- taken from
``machine.conda_envs[role]``, or an override from the caller, falling back to the
:class:`EnvSpec` default name.

**When a path is given the prefix is ``path / name``, and nothing else is
consulted** -- not even when that prefix turns out to be absent. Saying where the
env lives is an instruction, and a wrong one deserves an error rather than a
silent substitution by whatever else answers to the same name.

Otherwise the name is searched for, across three sources in this order:

1. ``conda info --json`` (:func:`list_conda_envs`), i.e. what ``conda activate
   <name>`` would find.
2. ``<machine.conda_envs_dir>/<name>``, stat'ed directly. ``conda env list`` reads
   ``~/.conda/environments.txt`` plus the *calling* user's ``envs_dirs``, so an env
   the admin account created with ``conda create -p`` is not necessarily registered
   for the lab account; without this fallback a perfectly good shared env is
   reported as missing.
3. *the environment this interpreter is already running in* (:func:`active_env`),
   when its name matches. Last resort, but the only source that survives a machine
   where conda itself is unreachable -- a batch node handed ``<prefix>/bin/python``
   directly, or a box whose conda is not on ``PATH``.

Every source matches on the name: an env is never adopted merely for being active,
so running this from ``base`` while intending to sort in ``kilosort4`` still
reports on ``kilosort4``. Between the two settings and the three sources, a ``ks5``
env can be checked beside a ``kilosort4`` one -- by name where conda knows it, by
path where it does not.

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
from typing import Any, Iterable, Mapping, Sequence

from spikesorting._config import EnvLocation, MachineProfile, SessionConfig

__all__ = [
    "Check",
    "PackageSpec",
    "EnvSpec",
    "ENV_SPECS",
    "KILOSORT_DOCS",
    "PHY_DOCS",
    "ORIGIN_CONFIGURED",
    "ORIGIN_ACTIVE",
    "ORIGIN_CONDA",
    "ORIGIN_SHARED",
    "conda_executable",
    "active_env",
    "list_conda_envs",
    "env_location",
    "env_python",
    "check_conda",
    "check_env_names",
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
SPIKEINTERFACE_DOCS = "https://github.com/SpikeInterface/spikeinterface"
NEO_DOCS = "https://github.com/NeuralEnsemble/python-neo"
PROBEINTERFACE_DOCS = "https://github.com/SpikeInterface/probeinterface"
CATGT_DOCS = "https://billkarsh.github.io/SpikeGLX/More_help/CatGT_ReadMe.html"
TPRIME_DOCS = "https://billkarsh.github.io/SpikeGLX/help/syncEdges/Sync_edges/"

# Sections, in report order.
SECTION_ENVS = "environments"
SECTION_GPU = "gpu"
SECTION_TOOLS = "command-line tools"
SECTION_SHELL = "shell"
SECTION_PATHS = "paths"
SECTION_SESSION = "session"

#: Where an environment was found. Reported verbatim, so these read as prose.
ORIGIN_CONFIGURED = "configured path"  # a prefix named outright in the config
ORIGIN_ACTIVE = "active"  # the env this interpreter is running in
ORIGIN_CONDA = "per-user"  # conda info --json
ORIGIN_SHARED = "shared"  # machine.conda_envs_dir

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
    #: The ``pyproject.toml`` optional-dependency group that provides this, or
    #: None when it is a base dependency. ``pip install -e .`` installs only the
    #: base set, so a package in an extra is missing after the obvious install
    #: command and the report has to say which extra brings it in.
    extra: str | None = None
    #: False for packages this project deliberately does not declare, so no pip
    #: command here would install them: torch (the right wheel depends on the
    #: GPU) and phy (its own env, installed from upstream's instructions). Their
    #: fix is a documentation link, not a command.
    declared: bool = True
    #: This package's *own* upstream, when pointing at the environment's docs
    #: would misdirect -- spikeinterface is not a Kilosort question. None falls
    #: back to :attr:`EnvSpec.docs`.
    docs: str | None = None

    @property
    def dist(self) -> str:
        return self.distribution or self.module

    @property
    def install_hint(self) -> str | None:
        """The pip command that installs this one, or None if we do not ship it."""
        if not self.declared:
            return None
        target = f'".[{self.extra}]"' if self.extra else "."
        return f"pip install --no-build-isolation -e {target}"


@dataclass(frozen=True)
class EnvSpec:
    """A conda environment the workflow expects, and how to set it up."""

    #: Stable identity, and the key a machine profile configures the name under.
    #: Unlike the name it never changes across sorter versions.
    role: str
    #: Name used when neither the machine profile nor the caller supplies one.
    default_name: str
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


# The package lists and docs hang off the *role*, not the version: pointing
# `sorting` at a `ks5` env still checks it for this list and still cites the
# Kilosort docs. That stays right until a future Kilosort changes its
# dependencies, at which point this grows a per-version branch -- inventing one
# now would only guess.
ENV_SPECS: tuple[EnvSpec, ...] = (
    EnvSpec(
        role="sorting",
        default_name="kilosort4",
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
                extra="sorting",
            ),
            PackageSpec(
                "torch",
                "sort.py:45",
                ("sort_neuropixels", "sort_blackrock"),
                declared=False,
            ),
            PackageSpec(
                "spikeinterface",
                "_sort.py:130, _io/blackrock.py:189",
                ("sort_blackrock", "loading either system"),
                docs=SPIKEINTERFACE_DOCS,
            ),
            # Only io/blackrock.py reaches neo. align and validate read the edge
            # *files* through catgt.read_edge_file and never touch it, so losing
            # neo costs Blackrock reading, not the alignment that follows it.
            PackageSpec(
                "neo",
                "io/blackrock.py:51",
                ("extract_sync (Blackrock)", "sort_blackrock"),
                docs=NEO_DOCS,
            ),
            # Checked on its own even though spikeinterface hard-requires it. The
            # probe uses find_spec, which does not execute the module, so a
            # spikeinterface whose probeinterface has been removed underneath it
            # still reports present -- and this repo imports probeinterface
            # directly anyway. Without this row the failure surfaces as a raw
            # ModuleNotFoundError from make_probe.py, which has no ImportError
            # handler, instead of as a line here.
            PackageSpec(
                "probeinterface",
                "probes/common.py:25, probes/io.py:167",
                ("probe objects", ".prb export"),
                docs=PROBEINTERFACE_DOCS,
            ),
            PackageSpec(
                "pytest", "tests/", ("test suite",), required=False, extra="dev"
            ),
        ),
    ),
    EnvSpec(
        role="curation",
        default_name="phy",
        purpose="manual curation (step 6, not scripted)",
        python="3.12",
        docs=PHY_DOCS,
        required=False,
        disables=("curation",),
        packages=(
            PackageSpec(
                "phy", "run by hand after sorting", ("curation",), declared=False
            ),
        ),
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


def active_env() -> Path | None:
    """Prefix of the conda environment this interpreter runs in, or None.

    ``CONDA_PREFIX`` is what ``conda activate`` sets. ``sys.prefix`` is the
    fallback and covers the case that matters on a batch node, where
    ``<prefix>/bin/python`` is invoked by path and nothing was ever activated.

    ``conda-meta`` is the test for "is this a conda prefix at all": ``sys.prefix``
    is *always* set, and for a system Python it points at something like
    ``/usr/local``, which is not an environment anyone meant.
    """
    declared = os.environ.get("CONDA_PREFIX")
    candidates = [Path(declared)] if declared else []
    candidates.append(Path(sys.prefix))
    for candidate in candidates:
        try:
            if (candidate / "conda-meta").is_dir():
                return candidate
        except OSError:
            continue
    return None


def env_location(
    spec: EnvSpec,
    machine: MachineProfile,
    overrides: Mapping[str, EnvLocation] | None = None,
) -> EnvLocation:
    """Where to look for this role's env: caller > machine profile > spec default.

    Name and path are resolved as one unit rather than field by field, so an
    override always replaces a whole location. Overriding the name while silently
    inheriting a path from the profile would send the lookup to
    ``<profile path>/<new name>``, which is nobody's intent.
    """
    override = (overrides or {}).get(spec.role)
    if override is not None:
        return override.resolve(spec.default_name)
    return machine.env_location(spec.role, spec.default_name)


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
    location: EnvLocation,
    machine: MachineProfile,
    envs: dict[str, Path] | None = None,
    active: Path | None = None,
) -> tuple[Path | None, str]:
    """Locate an environment. Returns ``(prefix, origin)``, origin ``ORIGIN_*``.

    A configured ``path`` settles it: the prefix is ``path / name`` and nothing
    else is consulted, **including when it turns out not to be there**. Saying
    where the env lives is an instruction, so a wrong path is an error worth
    reporting; quietly using some other env that matched by name is the outcome
    this avoids.

    Without a path the *name* is searched for, in the order the module docstring
    gives: conda's own listing, the machine's shared directory, then the
    environment this interpreter is running in. Every source matches on the name
    -- an active env called something else is never adopted, because reporting on
    the env you asked about, rather than the one you happen to be in, is what
    makes the report mean anything.

    ``envs`` and ``active`` are injectable so callers can resolve several specs
    against one snapshot, and so tests need not touch the real machine.
    """
    name = location.name or ""

    configured = location.prefix
    if configured is not None:
        return (configured if configured.is_dir() else None), ORIGIN_CONFIGURED

    envs = list_conda_envs() if envs is None else envs

    prefix = envs.get(name)
    if prefix is not None:
        origin = (
            ORIGIN_SHARED
            if machine.conda_envs_dir is not None
            and _is_within(prefix, machine.conda_envs_dir)
            else ORIGIN_CONDA
        )
        return prefix, origin

    if machine.conda_envs_dir is not None:
        candidate = machine.conda_envs_dir / name
        if candidate.is_dir():
            return candidate, ORIGIN_SHARED

    active = active_env() if active is None else active
    if active is not None and active.name == name:
        return active, ORIGIN_ACTIVE

    return None, ORIGIN_CONDA


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


def check_conda(machine: MachineProfile, active: Path | None = None) -> list[Check]:
    """Is conda reachable at all?

    Worth its own line: without it every environment would otherwise be reported
    as "not found", which points at the wrong problem. Not fatal on its own when
    another discovery source can still find environments by path and run their
    interpreters without conda's help -- either a shared ``conda_envs_dir``, or an
    already-active environment, which is the common case for a batch job that was
    handed ``<prefix>/bin/python`` directly.
    """
    conda = conda_executable()
    if conda is not None:
        return [Check(name="conda", status=OK, detail=conda, section=SECTION_ENVS)]

    survivable = machine.has_shared_envs or active is not None
    detail = "not found via CONDA_EXE or PATH"
    if active is not None:
        detail += f"; using the active environment {active}"
    return [
        Check(
            name="conda",
            status=WARN if survivable else MISSING,
            detail=detail,
            section=SECTION_ENVS,
            disables=() if survivable else ("all stages",),
            fix="install Miniconda/Anaconda, or set CONDA_EXE",
        )
    ]


def check_env_names(
    machine: MachineProfile, overrides: Mapping[str, EnvLocation] | None = None
) -> list[Check]:
    """Flag ``conda_envs`` keys that name no known role.

    ``config.py`` ignores keys it does not recognise, so a typo'd ``sortin:``
    would silently leave the default name in place and surface much later as a
    confusing "env not found". Same for an override from a caller.
    """
    roles = {spec.role for spec in ENV_SPECS}
    configured = set(machine.conda_envs) | set(overrides or {})
    unknown = sorted(configured - roles)
    if not unknown:
        return []
    return [
        Check(
            name="conda_envs",
            status=WARN,
            detail=(
                f"unknown role(s) {', '.join(unknown)} -- ignored."
                f" Valid roles: {', '.join(sorted(roles))}"
            ),
            section=SECTION_ENVS,
            fix=f"fix or remove the key in the '{machine.name}' machine profile",
        )
    ]


@dataclass(frozen=True)
class EnvReport:
    """Raw result of looking at one environment, before it becomes findings.

    Exists so the environment is probed exactly once: importing torch costs
    seconds, and both the package checks and the CUDA check need that same probe.
    """

    spec: EnvSpec
    #: What was configured for this role, after the machine profile and any
    #: override. Carried so that :func:`env_checks` stays pure and need not
    #: re-resolve it.
    location: EnvLocation = field(default_factory=EnvLocation)
    prefix: Path | None = None
    origin: str = ORIGIN_CONDA
    python: Path | None = None
    #: Parsed probe JSON, or ``{"error": ...}``, or ``{}`` when never run.
    probe: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The environment's name, as configured. Labels every check."""
        return self.location.name or self.spec.default_name

    @property
    def torch_info(self) -> dict[str, Any] | None:
        return self.probe.get("torch")


def inspect_env(
    spec: EnvSpec,
    machine: MachineProfile,
    envs: dict[str, Path] | None = None,
    active: Path | None = None,
    overrides: Mapping[str, EnvLocation] | None = None,
) -> EnvReport:
    """Resolve an environment and probe it once."""
    location = env_location(spec, machine, overrides)
    prefix, origin = resolve_env(location, machine, envs, active)
    if prefix is None:
        return EnvReport(spec, location, origin=origin, probe={})
    python = env_python(prefix)
    if python is None:
        return EnvReport(spec, location, prefix, origin, probe={})
    return EnvReport(
        spec,
        location,
        prefix,
        origin,
        python,
        probe_env(python, spec.packages, spec.probe_torch),
    )


def check_env(
    spec: EnvSpec,
    machine: MachineProfile,
    envs: dict[str, Path] | None = None,
    active: Path | None = None,
    overrides: Mapping[str, EnvLocation] | None = None,
) -> list[Check]:
    """Health of one conda environment: the env itself, then its packages."""
    return env_checks(inspect_env(spec, machine, envs, active, overrides))


def _package_fix(package: PackageSpec, spec: EnvSpec, env: str) -> str:
    """What to do about one missing package. Pure."""
    if package.install_hint is None:
        # Nothing here installs it; a documentation link is the whole answer.
        return f"install {package.module} into '{env}' -- see {package.docs or spec.docs}"
    fix = f"in '{env}': {package.install_hint}"
    return f"{fix} -- see {package.docs}" if package.docs else fix


def env_checks(report: EnvReport) -> list[Check]:
    """Turn an :class:`EnvReport` into findings. Pure."""
    spec = report.spec
    name = report.name
    absent_status = MISSING if spec.required else WARN
    provenance = report.origin

    if report.prefix is None:
        # A configured path that is not there is a different problem from an env
        # that could not be found, and needs a different instruction: the config
        # is wrong, or the path moved. Saying "create the env" would misdirect.
        if report.origin == ORIGIN_CONFIGURED:
            # Naming both halves makes the join visible, which is what turns the
            # likeliest mistake -- giving the env's own prefix as `path`, so the
            # name gets appended twice -- into something self-evident on sight.
            detail = (
                f"no env '{name}' in the configured path"
                f" {report.location.path} -- needed for {spec.purpose}"
            )
            fix = (
                f"conda_envs.{spec.role}.path is the directory that *holds* the"
                f" env, not the env itself; correct it, or drop it to search for"
                f" an env named '{name}'"
            )
        else:
            detail = (
                "not found via conda env list, in the machine's conda_envs_dir,"
                f" or as the active env -- needed for {spec.purpose}"
            )
            fix = (
                f"create the '{name}' env (python {spec.python}) per {spec.docs},"
                f" or set conda_envs.{spec.role} to the name and path you use"
            )
        return [
            Check(
                name=f"env:{name}",
                status=absent_status,
                detail=detail,
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=fix,
            )
        ]

    if report.python is None:
        return [
            Check(
                name=f"env:{name}",
                status=absent_status,
                detail=f"{report.prefix} ({provenance}) exists but holds no interpreter",
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=f"recreate the '{name}' env per {spec.docs}",
            )
        ]

    probe = report.probe or {}
    if "error" in probe:
        return [
            Check(
                name=f"env:{name}",
                status=absent_status,
                detail=f"{report.prefix} ({provenance}) could not be probed: {probe['error']}",
                section=SECTION_ENVS,
                disables=spec.disables,
                fix=f"recreate the '{name}' env per {spec.docs}",
            )
        ]

    prefix = report.prefix
    found = probe.get("packages", {})
    checks = [
        Check(
            name=f"env:{name}",
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
                    name=f"{name}/{package.module}",
                    status=OK,
                    detail=version,
                    section=SECTION_ENVS,
                )
            )
        else:
            checks.append(
                Check(
                    name=f"{name}/{package.module}",
                    status=MISSING if package.required else WARN,
                    detail=f"not installed (used by {package.reached_from})",
                    section=SECTION_ENVS,
                    disables=package.disables,
                    # Where we ship the package the command *is* the answer, so
                    # the link is added only when the package has an upstream of
                    # its own -- falling back to the env's docs would tell you to
                    # read Kilosort's install page about neo. Undeclared packages
                    # have no command, so there a link is all there is.
                    fix=_package_fix(package, spec, name),
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
    from spikesorting._sync import catgt, tprime

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
    """Roots and scratch declared by the machine profile.

    An unset ``data_root`` / ``output_root`` is **not** a finding. Recording paths
    are a property of the recording, so they live in the session config, written
    out in full; every machine profile says "do not re-add data_root here". These
    are checked when a profile happens to set one, and passed over in silence
    otherwise -- warning about a field the design deliberately leaves empty just
    trains the reader to ignore the report.
    """
    checks: list[Check] = []
    for label, path in (
        ("data_root", machine.data_root),
        ("output_root", machine.output_root),
    ):
        if path is None:
            continue
        if path.exists():
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
    """Session input files, delegated to the existing validator, plus geometry.

    A Utah array with no channel map cannot be sorted: there is no honest default
    to fall back on, so :func:`~spikesorting.pipeline.setup_probe` raises. That is
    an hour into a run, though, and this is the pre-flight -- so the same gap is
    reported here as blocking, and the exit code says so before the job starts.
    """
    problems = config.missing_inputs()
    checks = [
        Check(
            name=f"session:{config.session}",
            status=MISSING,
            detail=problem,
            section=SECTION_SESSION,
        )
        for problem in problems
    ]
    if not problems:
        checks.append(
            Check(
                name=f"session:{config.session}",
                status=OK,
                detail="all declared inputs present",
                section=SECTION_SESSION,
            )
        )

    brk = config.blackrock
    if config.sorts_blackrock and brk.probe_file is None and brk.cmp_file is None:
        checks.append(
            Check(
                name=f"session:{config.session}:probe",
                status=MISSING,
                detail=(
                    "no blackrock.cmp_file or probe_file, so the Utah array "
                    "cannot be sorted: a grid in channel order would attribute "
                    "units to the wrong electrodes, and its channel count would "
                    "silently drop every electrode past it"
                ),
                disables=("sort_blackrock",),
                fix=(
                    "set blackrock.cmp_file to the array's .cmp, or build a map "
                    "once: python tools/make_probe.py utah --cmp array.cmp "
                    "--out configs/probes/utah_<array>.json --plot, then set "
                    "blackrock.probe_file to it"
                ),
                section=SECTION_SESSION,
            )
        )
    return checks


def run_all(
    machine: MachineProfile,
    config: SessionConfig | None = None,
    env_overrides: Mapping[str, EnvLocation] | None = None,
) -> list[Check]:
    """Every check, in report order.

    ``env_overrides`` maps a role to an :class:`~spikesorting._config.EnvLocation`
    and outranks the machine profile, so a one-off run can be pointed at another
    env without editing it.
    """
    envs = list_conda_envs()
    active = active_env()
    checks: list[Check] = check_conda(machine, active)
    checks.extend(check_env_names(machine, env_overrides))
    torch_info: dict[str, Any] | None = None

    for spec in ENV_SPECS:
        report = inspect_env(spec, machine, envs, active, env_overrides)
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
