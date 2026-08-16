"""Session and machine configuration (pipeline step 1).

The configuration is split into two layers so that a single session file runs
unchanged on the Windows rig, on the HPC, and on a laptop:

``SessionConfig``
    What was recorded -- session identity, which files hold the sync pulses and
    the spike data, which channels carry which pulse. Portable.

``MachineProfile``
    Where *this* machine keeps things -- data root, SSD scratch for ``temp.dat``,
    CatGT/TPrime install dirs, torch device. Machine-local.

Session paths may contain the placeholders ``{data_root}``, ``{output_root}``,
``{cache}``, ``{downloads}`` and ``{session}``; they are substituted from the
machine profile when the session is loaded.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "EnvLocation",
    "MachineProfile",
    "BlackrockSpec",
    "NeuropixelsSpec",
    "OutputPaths",
    "SessionConfig",
    "load_machine",
    "load_session_config",
    "default_config_dir",
    "downloads_dir",
]

# Repo root is two levels above src/spikesorting/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]


def default_config_dir() -> Path:
    """Directory holding the YAML configs, overridable with ``SPIKESORTING_CONFIG_DIR``."""
    env = os.environ.get("SPIKESORTING_CONFIG_DIR")
    return Path(env) if env else REPO_ROOT / "configs"


def downloads_dir() -> Path:
    """Kilosort's downloads directory, behind the ``{downloads}`` placeholder.

    Falls back to ``~/.kilosort`` when the ``kilosort`` package is not installed,
    so a config using the placeholder still *loads* on a machine without it.
    """
    try:
        from kilosort.utils import DOWNLOADS_DIR  # type: ignore[import-not-found]

        return Path(DOWNLOADS_DIR)
    except Exception:
        return Path.home() / ".kilosort"


def _as_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def _as_config_path(value: Any) -> Path | None:
    """A path that may be written relative to the repo, like ``configs/probes/x.json``.

    Every other path in a session file is absolute or built from
    ``{blackrock_dir}`` / ``{neuropixels_dir}``. Probe files are the exception --
    they live in the repo, so the examples all write them relative -- and left
    relative to the *working directory* they would resolve only when the pipeline
    happens to be run from the repo root.
    """
    path = _as_path(value)
    if path is None or path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass(frozen=True)
class EnvLocation:
    """Where a machine keeps one conda environment: a name, optionally a place.

    The two are deliberately separate settings. ``name`` is what the environment
    is called; ``path`` is the directory that *holds* it, and the prefix is
    ``path / name`` -- so ``path`` is a per-role counterpart to
    ``conda_envs_dir``, not the environment directory itself. With no ``path``
    the name is searched for instead.
    """

    name: str | None = None
    #: Directory *containing* the environment. Joined with ``name``; never the
    #: prefix on its own.
    path: Path | None = None

    def resolve(self, default_name: str) -> "EnvLocation":
        """This location with ``name`` filled in from the spec default."""
        return EnvLocation(name=self.name or default_name, path=self.path)

    @property
    def prefix(self) -> Path | None:
        """``path / name`` when a path was configured, else None."""
        if self.path is None or not self.name:
            return None
        return self.path / self.name


def _as_env_locations(value: Any) -> dict[str, EnvLocation]:
    """Coerce a YAML ``conda_envs`` block to ``role -> EnvLocation``.

    Accepts either shorthand or the full form, since naming only the name is the
    common case::

        conda_envs:
          sorting: kilosort4
          curation:
            name: phy
            path: /shared/apps/envs

    Anything malformed degrades to empty or to a name-only entry rather than
    raising, so a bad profile still loads and the problem surfaces as a check
    instead of a traceback -- as :func:`_as_path` turns junk into ``None``.
    """
    if not isinstance(value, dict):
        return {}
    located: dict[str, EnvLocation] = {}
    for role, entry in value.items():
        if entry is None:
            continue
        if isinstance(entry, dict):
            name = entry.get("name")
            located[str(role)] = EnvLocation(
                name=str(name) if name else None,
                path=_as_path(entry.get("path")),
            )
        else:
            located[str(role)] = EnvLocation(name=str(entry))
    return located


@dataclass(frozen=True)
class MachineProfile:
    """Machine-local paths and settings."""

    name: str
    data_root: Path | None = None
    output_root: Path | None = None
    #: Fast local scratch for Kilosort's ``temp.dat``. Must be an SSD.
    cache_dir: Path | None = None
    #: Directory containing CatGT's ``runit.bat`` / ``runit.sh``. None on machines
    #: without CatGT (e.g. the HPC) -- the pure-Python fallback is used instead.
    catgt_dir: Path | None = None
    #: Directory containing the TPrime executable. None where TPrime is absent.
    tprime_dir: Path | None = None
    #: Directory holding conda environments shared between accounts on this
    #: machine (e.g. ``C:/ProgramData/anaconda3/envs``). The *last* of three
    #: discovery sources and purely a fallback: the environment this interpreter
    #: runs in is tried first, then ``conda env list``. Leaving it None is normal
    #: -- it is only needed where an env one account created is invisible to
    #: another's ``conda env list``. Used by :mod:`spikesorting.doctor`; nothing
    #: in the pipeline depends on it.
    conda_envs_dir: Path | None = None
    #: The conda environments on this machine, keyed by *role* ("sorting",
    #: "curation"). The role is stable; the name is not, which is the point --
    #: ``sorting: ks5`` points the check at a different Kilosort version without
    #: touching any code, and an optional ``path`` says where to find it instead
    #: of searching. Absent roles fall back to the built-in defaults (kilosort4,
    #: phy), so a machine using the conventional names needs no entry at all.
    #: Read only by :mod:`spikesorting.doctor`, like ``conda_envs_dir``.
    conda_envs: dict[str, EnvLocation] = field(default_factory=dict)
    #: torch device string handed to Kilosort4 ("cuda", "cuda:0", "cpu").
    device: str = "cuda"

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "MachineProfile":
        return cls(
            name=name,
            data_root=_as_path(data.get("data_root")),
            output_root=_as_path(data.get("output_root")),
            cache_dir=_as_path(data.get("cache_dir")),
            catgt_dir=_as_path(data.get("catgt_dir")),
            tprime_dir=_as_path(data.get("tprime_dir")),
            conda_envs_dir=_as_path(data.get("conda_envs_dir")),
            conda_envs=_as_env_locations(data.get("conda_envs")),
            device=str(data.get("device", "cuda")),
        )

    @property
    def has_catgt(self) -> bool:
        return self.catgt_dir is not None

    @property
    def has_tprime(self) -> bool:
        return self.tprime_dir is not None

    @property
    def has_shared_envs(self) -> bool:
        return self.conda_envs_dir is not None

    def env_location(self, role: str, default_name: str) -> EnvLocation:
        """This machine's env for ``role``, with the name defaulted."""
        return self.conda_envs.get(role, EnvLocation()).resolve(default_name)


@dataclass(frozen=True)
class BlackrockSpec:
    """Blackrock inputs.

    Channel defaults follow ``SynchronizePulse_setup.md``: in ``NSP-*.ns5``,
    channel 1 carries the 1 Hz square wave from SpikeGLX and channel 2 carries
    the 14 s coded burst emitted by Blackrock DO1.
    """

    #: NSP-*.ns5 -- holds the two sync channels.
    sync_file: Path | None = None
    #: HUB-*.ns6 -- holds the Utah array spike data.
    spike_file: Path | None = None
    sync_1hz_channel: int = 1
    burst_channel: int = 2
    #: Rising-edge thresholds, in the units returned by the reader (see io.blackrock).
    sync_threshold: float = 2.5
    burst_threshold: float = 2.5
    #: Stream to read from the .nsX file (neo/SpikeInterface stream id).
    stream_id: str | None = None
    #: Channel ids to exclude from sorting (sync / analog inputs sharing the file).
    exclude_channels: tuple[str, ...] = ()
    #: The array's own .cmp wiring map. Preferred, because it describes *this*
    #: array rather than a map that might have been built from another one.
    cmp_file: Path | None = None
    #: Probe JSON from make_probe.py, used when no cmp_file is given.
    probe_file: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BlackrockSpec":
        return cls(
            sync_file=_as_path(data.get("sync_file")),
            spike_file=_as_path(data.get("spike_file")),
            sync_1hz_channel=int(data.get("sync_1hz_channel", 1)),
            burst_channel=int(data.get("burst_channel", 2)),
            sync_threshold=float(data.get("sync_threshold", 2.5)),
            burst_threshold=float(data.get("burst_threshold", 2.5)),
            stream_id=data.get("stream_id"),
            exclude_channels=tuple(str(c) for c in data.get("exclude_channels", ())),
            probe_file=_as_config_path(data.get("probe_file")),
            cmp_file=_as_config_path(data.get("cmp_file")),
        )


@dataclass(frozen=True)
class NeuropixelsSpec:
    """SpikeGLX inputs.

    Two ways to point at the data:

    * a SpikeGLX *run* (``run_dir`` + ``run_name`` + gate/trigger/probe indices),
      which is what CatGT consumes; or
    * a single ``bin_file``, for one-off binaries outside a run folder.
    """

    run_dir: Path | None = None
    run_name: str | None = None
    gate: int = 0
    trigger: int = 0
    probes: tuple[int, ...] = (0,)
    bin_file: Path | None = None
    #: Total channels in the binary, including the SY word. Derived from the
    #: .meta file when absent.
    n_chan_bin: int | None = None
    #: Sampling rate. Only needed for a bare binary with no .meta beside it
    #: (the Kilosort demo file); otherwise read from the meta.
    sample_rate: float | None = None
    #: SY-word bit carrying the SMA1 1 Hz square wave. 6 is the SpikeGLX default.
    sync_bit: int = 6
    #: High duration of the 1 Hz square wave, in ms (1 s period, 50% duty).
    sync_pulse_ms: int = 500
    #: Word index of the XA1 analog input in the obx stream (14 s coded burst).
    burst_word: int = 1
    #: CatGT -xa thresholds in volts: (primary, secondary). Secondary 0 disables it.
    burst_threshold_v: tuple[float, float] = (1.0, 0.0)
    #: Expected burst pulse width in ms; 0 accepts any width.
    burst_pulse_ms: int = 0
    #: Probe JSON from make_probe.py, used when the run has no ``.meta`` to build
    #: the map from. The .meta wins where it exists: it records which sites were
    #: actually active, which a file built from another run cannot know.
    probe_file: Path | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuropixelsSpec":
        thresh = data.get("burst_threshold_v", (1.0, 0.0))
        if isinstance(thresh, (int, float)):
            thresh = (float(thresh), 0.0)
        n_chan = data.get("n_chan_bin")
        rate = data.get("sample_rate")
        return cls(
            run_dir=_as_path(data.get("run_dir")),
            run_name=data.get("run_name"),
            gate=int(data.get("gate", 0)),
            trigger=int(data.get("trigger", 0)),
            probes=tuple(int(p) for p in data.get("probes", (0,))),
            bin_file=_as_path(data.get("bin_file")),
            n_chan_bin=int(n_chan) if n_chan is not None else None,
            sample_rate=float(rate) if rate is not None else None,
            sync_bit=int(data.get("sync_bit", 6)),
            sync_pulse_ms=int(data.get("sync_pulse_ms", 500)),
            burst_word=int(data.get("burst_word", 1)),
            burst_threshold_v=(float(thresh[0]), float(thresh[1])),
            burst_pulse_ms=int(data.get("burst_pulse_ms", 0)),
            probe_file=_as_config_path(data.get("probe_file")),
        )


@dataclass(frozen=True)
class OutputPaths:
    """Where one session's outputs go: beside the recording that produced them.

    Each system has its own data directory, so each system's products stay with
    it -- sorted spikes, its own sync edges, and (Neuropixels only) LFP. Only the
    cross-system results have nowhere natural to live; they go under the
    Blackrock directory, because Blackrock is the reference timebase everything
    is mapped onto.

    ``sorted_np`` and ``sorted_br`` both name a directory holding the sorter's
    own results -- ``spike_times.npy``, ``params.py``, ``cluster_KSLabel.tsv``.
    That is the invariant every consumer depends on: :mod:`.pipeline` reads
    ``spike_times.npy`` straight out of them, and Phy is opened on them. Nesting
    the sorter name one level down keeps a second sorter's run beside the first
    instead of on top of it.
    """

    blackrock_dir: Path | None = None
    neuropixels_dir: Path | None = None
    sorter: str = "kilosort4"

    def dir_for(self, system: str) -> Path | None:
        """That system's directory, or None when it never recorded."""
        return self.blackrock_dir if system == "blackrock" else self.neuropixels_dir

    def _require(self, system: str) -> Path:
        directory = self.dir_for(system)
        if directory is None:
            raise ValueError(
                f"session has no {system}_dir: give one directly, or set a "
                f"roots.{system} together with monkey and session"
            )
        return directory

    @property
    def primary(self) -> Path:
        """Where cross-system output goes -- Blackrock, or the one that recorded."""
        for directory in (self.blackrock_dir, self.neuropixels_dir):
            if directory is not None:
                return directory
        raise ValueError("session has neither a blackrock_dir nor a neuropixels_dir")

    def sync_for(self, system: str) -> Path:
        """Edge files, beside the recording they were extracted from."""
        return self._require(system) / "sync"

    @property
    def lfp(self) -> Path:
        # Neuropixels only: Blackrock LFPs are already saved separately by Central.
        return self._require("neuropixels") / "lfp"

    def sorted_for(self, system: str) -> Path:
        """Where one system's sorting lands. The system-agnostic form."""
        return self._require(system) / system / self.sorter

    @property
    def sorted_np(self) -> Path:
        return self.sorted_for("neuropixels")

    @property
    def sorted_br(self) -> Path:
        return self.sorted_for("blackrock")

    @property
    def aligned(self) -> Path:
        return self.primary / "aligned"

    @property
    def figures(self) -> Path:
        return self.primary / "figures"

    def all(self) -> tuple[Path, ...]:
        """Every directory this session can actually write to."""
        paths: list[Path] = [self.aligned, self.figures]
        if self.neuropixels_dir is not None:
            paths += [self.sync_for("neuropixels"), self.lfp, self.sorted_np]
        if self.blackrock_dir is not None:
            paths += [self.sync_for("blackrock"), self.sorted_br]
        return tuple(dict.fromkeys(paths))

    def mkdirs(self) -> None:
        for path in self.all():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SessionConfig:
    """One recording session, resolved against one machine."""

    session: str
    machine: MachineProfile
    #: Subject folder between a root and the session, e.g. ``"Monkey Athos"``.
    monkey: str = ""
    #: Each system's own directory: raw recording and everything derived from it.
    #: Built as ``<roots.<system>>/<monkey>/<session>`` unless stated outright.
    blackrock_dir: Path | None = None
    neuropixels_dir: Path | None = None
    #: Names the level below ``neuropixels/`` and ``blackrock/`` that results are
    #: written to, so switching sorter versions does not overwrite a previous run.
    sorter: str = "kilosort4"
    blackrock: BlackrockSpec = field(default_factory=BlackrockSpec)
    neuropixels: NeuropixelsSpec = field(default_factory=NeuropixelsSpec)
    #: Skip cross-system alignment (steps 8-9). True when only one system ran.
    skip_sync: bool = False
    # Two independent questions per system, deliberately not one flag:
    #
    #   has_*_data          did this system record at all?
    #   kilosort_on_*       should its spike data be sorted?
    #
    # They come apart in a real case: a session that recorded Neuropixels but is
    # only wanted for its LFP and sync pulses has data (so its inputs must be
    # checked and extraction must run) yet no sorting. Absent data always wins --
    # see :attr:`sorts_neuropixels`.
    has_neuropixels_data: bool = True
    has_blackrock_data: bool = True
    kilosort_on_neuropixels: bool = True
    kilosort_on_blackrock: bool = True
    #: Period of the fine-alignment square wave, in seconds. TPrime -syncperiod.
    sync_period_s: float = 1.0
    #: Nominal interval of the coarse coded burst, in seconds.
    burst_interval_s: float = 14.0
    #: Alignment is considered good when residuals stay below this (seconds).
    alignment_tolerance_s: float = 1e-3
    #: Preprocessing applied before sorting.
    #: Shared default for both systems; see :meth:`preprocess_for`.
    preprocess: dict[str, Any] = field(default_factory=dict)
    #: Per-system ``preprocess:`` blocks, merged over the shared one. Keyed by
    #: "neuropixels" / "blackrock". The two systems genuinely want different
    #: treatment -- an explicit median reference suits a 400 um Utah array, while
    #: a dense probe is usually better left to Kilosort's own internals -- so one
    #: shared block cannot express what a real session needs.
    preprocess_by_system: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Extra settings merged into Kilosort4's settings dict.
    kilosort_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> OutputPaths:
        return OutputPaths(self.blackrock_dir, self.neuropixels_dir, self.sorter)

    @property
    def output_root(self) -> Path:
        """Single directory to name when reporting where a session writes."""
        return self.paths.primary

    @property
    def cache_dir(self) -> Path | None:
        return self.machine.cache_dir

    def preprocess_for(self, system: str) -> dict[str, Any]:
        """Preprocessing settings for one system: shared block, then its override.

        Merged key by key rather than replaced, so ``neuropixels: {preprocess:
        {apply: false}}`` turns it off without also discarding the bandpass and
        reference the shared block set.
        """
        merged = dict(self.preprocess)
        merged.update(self.preprocess_by_system.get(system) or {})
        return merged

    @property
    def sorts_neuropixels(self) -> bool:
        """Whether step 3 runs. ``kilosort_on_*`` is a request, not an override."""
        return self.has_neuropixels_data and self.kilosort_on_neuropixels

    @property
    def sorts_blackrock(self) -> bool:
        """Whether step 4 runs."""
        return self.has_blackrock_data and self.kilosort_on_blackrock

    def missing_inputs(self) -> list[str]:
        """Return human-readable problems with the *input* paths.

        Pure: touches the filesystem read-only and returns findings rather than
        raising, so callers can report every problem at once before a long job.
        """
        problems: list[str] = []

        # Gated on the data existing, not on sorting being wanted: the LFP and
        # sync stages read the same files, so an unsortable session still needs
        # them to be there.
        if self.has_neuropixels_data:
            npx = self.neuropixels
            if npx.bin_file is not None:
                if not npx.bin_file.exists():
                    problems.append(f"neuropixels.bin_file does not exist: {npx.bin_file}")
            elif npx.run_dir is not None:
                if not npx.run_dir.exists():
                    problems.append(f"neuropixels.run_dir does not exist: {npx.run_dir}")
                if not npx.run_name:
                    problems.append("neuropixels.run_name is required when run_dir is set")
            else:
                problems.append(
                    "neuropixels needs either bin_file or run_dir+run_name "
                    "(or set has_neuropixels_data: false)"
                )

        if self.has_blackrock_data:
            brk = self.blackrock
            if brk.sync_file is None:
                problems.append(
                    "blackrock.sync_file is required (or set has_blackrock_data: false)"
                )
            elif not brk.sync_file.exists():
                problems.append(f"blackrock.sync_file does not exist: {brk.sync_file}")
            if brk.spike_file is not None and not brk.spike_file.exists():
                problems.append(f"blackrock.spike_file does not exist: {brk.spike_file}")

        # A channel map that is *named* but absent is a hard error -- the
        # alternative is finding out after the recording has been copied to the
        # cache. Naming none at all is not: that means placeholder geometry, which
        # still sorts and is reported as a warning by doctor instead.
        for system in ("neuropixels", "blackrock"):
            if not getattr(self, f"has_{system}_data"):
                continue
            spec = getattr(self, system)
            for key in ("probe_file", "cmp_file"):
                path = getattr(spec, key, None)
                if path is not None and not path.exists():
                    problems.append(f"{system}.{key} does not exist: {path}")

        if self.machine.cache_dir is None:
            problems.append(
                f"machine '{self.machine.name}' has no cache_dir; Kilosort needs "
                "fast local scratch for temp.dat"
            )

        return problems

    def require_inputs(self) -> None:
        """Raise if any input is missing. Call before starting a long job."""
        problems = self.missing_inputs()
        if problems:
            raise FileNotFoundError(
                f"session '{self.session}' on machine '{self.machine.name}' is not runnable:\n  - "
                + "\n  - ".join(problems)
            )


def _substitute(value: Any, mapping: dict[str, str]) -> Any:
    """Recursively expand ``{placeholder}`` tokens in strings."""
    if isinstance(value, str):
        out = value
        for key, replacement in mapping.items():
            out = out.replace("{" + key + "}", replacement)
        return out
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    return value


#: Matches a placeholder that *looks* deliberate. ``{}`` and ``{0}`` are left
#: alone so a genuine brace in some future setting is not mistaken for one.
_PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")


def _first_unresolved(value: Any) -> str | None:
    """First string still holding a ``{placeholder}`` after substitution."""
    if isinstance(value, str):
        return value if _PLACEHOLDER.search(value) else None
    if isinstance(value, dict):
        values: Any = value.values()
    elif isinstance(value, list):
        values = value
    else:
        return None
    for item in values:
        found = _first_unresolved(item)
        if found is not None:
            return found
    return None


class _StrictLoader(yaml.SafeLoader):
    """A SafeLoader that refuses duplicate keys instead of keeping the last one.

    PyYAML's default silently keeps the last occurrence, which makes pasting a
    template on top of an existing config *look* fine while discarding whole
    blocks -- an entire ``neuropixels:`` section, say, taking ``run_dir`` and
    ``run_name`` with it. That is a wrong sort, not a wrong path, and nothing
    downstream could tell.
    """


def _no_duplicate_keys(loader: yaml.Loader, node: yaml.MappingNode) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ValueError(
                f"duplicate key {key!r} at line {key_node.start_mark.line + 1}. "
                "YAML keeps only the last one, so everything the earlier block set "
                "would be silently discarded."
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=True)


_StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys
)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        try:
            data = yaml.load(handle, Loader=_StrictLoader)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from None
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def _has_data(data: dict[str, Any], system: str) -> bool:
    """Read ``has_<system>_data``, honouring the older ``skip_<system>`` spelling.

    ``skip_*`` meant "this system did not record", which is exactly
    ``has_*_data: false``. Session copies are gitignored and live on the rig, so
    the old key keeps working; the explicit new one wins where both appear.
    """
    explicit = data.get(f"has_{system}_data")
    if explicit is not None:
        return bool(explicit)
    return not bool(data.get(f"skip_{system}", False))


def load_machine(name: str, config_dir: Path | None = None) -> MachineProfile:
    """Load ``configs/machines/<name>.yaml``."""
    config_dir = config_dir or default_config_dir()
    return MachineProfile.from_dict(name, _read_yaml(config_dir / "machines" / f"{name}.yaml"))


def load_session_config(
    path: str | Path,
    machine: str | MachineProfile,
    config_dir: Path | None = None,
) -> SessionConfig:
    """Load a session YAML and resolve its placeholders against a machine profile.

    ``path`` may be a full path or a bare name resolved inside the config dir.
    """
    config_dir = config_dir or default_config_dir()
    session_path = Path(path)
    if not session_path.exists():
        candidate = config_dir / session_path.name
        if candidate.suffix != ".yaml":
            candidate = candidate.with_suffix(".yaml")
        session_path = candidate

    data = _read_yaml(session_path)
    profile = load_machine(machine, config_dir) if isinstance(machine, str) else machine

    session_name = str(data.get("session") or session_path.stem)
    monkey = str(data.get("monkey") or "")

    # Built-ins first. An unset one is *omitted* rather than mapped to "": a
    # machine profile supplies no roots any more, and "{data_root}/Monkey Athos"
    # collapsing to "/Monkey Athos" is a wrong path that fails much later, if at
    # all. Left in place it is caught below instead.
    builtin = {"session": session_name, "downloads": str(downloads_dir())}
    if monkey:
        builtin["monkey"] = monkey
    for key, value in (
        ("cache", profile.cache_dir),
        ("data_root", profile.data_root),
        ("output_root", profile.output_root),
    ):
        if value:
            builtin[key] = str(value)

    # Then the session's own roots, which may themselves use the built-ins --
    # `blackrock: "Z:/server/{monkey}"` is legal, though the usual form stops at
    # the share and lets the monkey/session levels be built below.
    roots = data.get("roots") or {}
    if not isinstance(roots, dict):
        raise ValueError(
            f"{session_path}: 'roots' must be a mapping of name to path, "
            f"got {type(roots).__name__}"
        )
    mapping = dict(builtin)
    for key, value in roots.items():
        mapping[str(key)] = _substitute(str(value), builtin)

    # Finally each system's directory: <root>/<monkey>/<session>, unless the file
    # states one outright. Everything that system produces hangs off it, so it is
    # exposed as a placeholder too -- "{blackrock_dir}/NSP-Athos_001.ns5".
    system_dirs: dict[str, Path | None] = {}
    for system in ("blackrock", "neuropixels"):
        stated = data.get(f"{system}_dir")
        if stated:
            system_dirs[system] = _as_path(_substitute(str(stated), mapping))
        elif system in mapping and monkey:
            system_dirs[system] = _as_path(mapping[system]) / monkey / session_name
        else:
            system_dirs[system] = None
        if system_dirs[system] is not None:
            mapping[f"{system}_dir"] = str(system_dirs[system])

    data = _substitute(data, mapping)

    stale = _first_unresolved(data)
    if stale is not None:
        known = ", ".join(sorted(mapping)) or "(none)"
        raise ValueError(
            f"{session_path}: unresolved placeholder in {stale!r}.\n"
            f"  known names: {known}\n"
            "  declare the missing one under 'roots:' in this file."
        )

    # `output_dir:` is the older single-tree spelling. Outputs now sit beside the
    # recording that produced them, so it is read as the directory of whichever
    # system has no directory of its own -- enough to keep a Neuropixels-only
    # session file working unchanged.
    stated_output = _as_path(data.get("output_dir"))
    if stated_output is not None:
        for system in ("neuropixels", "blackrock"):
            if system_dirs[system] is None and _has_data(data, system):
                system_dirs[system] = stated_output

    if system_dirs["blackrock"] is None and system_dirs["neuropixels"] is None:
        raise ValueError(
            f"{session_path}: no directory for either system. Give a "
            "'monkey:' plus a 'roots:' entry, or state blackrock_dir / "
            "neuropixels_dir outright."
        )

    return SessionConfig(
        session=session_name,
        machine=profile,
        monkey=monkey,
        blackrock_dir=system_dirs["blackrock"],
        neuropixels_dir=system_dirs["neuropixels"],
        sorter=str(data.get("sorter") or "kilosort4"),
        blackrock=BlackrockSpec.from_dict(data.get("blackrock") or {}),
        neuropixels=NeuropixelsSpec.from_dict(data.get("neuropixels") or {}),
        skip_sync=bool(data.get("skip_sync", False)),
        has_neuropixels_data=_has_data(data, "neuropixels"),
        has_blackrock_data=_has_data(data, "blackrock"),
        kilosort_on_neuropixels=bool(data.get("kilosort_on_neuropixels", True)),
        kilosort_on_blackrock=bool(data.get("kilosort_on_blackrock", True)),
        sync_period_s=float(data.get("sync_period_s", 1.0)),
        burst_interval_s=float(data.get("burst_interval_s", 14.0)),
        alignment_tolerance_s=float(data.get("alignment_tolerance_s", 1e-3)),
        preprocess=dict(data.get("preprocess") or {}),
        preprocess_by_system={
            system: dict((data.get(system) or {}).get("preprocess") or {})
            for system in ("neuropixels", "blackrock")
            if (data.get(system) or {}).get("preprocess")
        },
        kilosort_settings=dict(data.get("kilosort_settings") or {}),
    )


def with_overrides(config: SessionConfig, **overrides: Any) -> SessionConfig:
    """Return a copy of ``config`` with fields replaced (CLI flag support)."""
    return replace(config, **overrides)
