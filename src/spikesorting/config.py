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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "MachineProfile",
    "BlackrockSpec",
    "NeuropixelsSpec",
    "OutputPaths",
    "SessionConfig",
    "load_machine",
    "load_session",
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
            device=str(data.get("device", "cuda")),
        )

    @property
    def has_catgt(self) -> bool:
        return self.catgt_dir is not None

    @property
    def has_tprime(self) -> bool:
        return self.tprime_dir is not None


@dataclass(frozen=True)
class BlackrockSpec:
    """Blackrock inputs.

    Channel defaults follow ``SynchronizePulse_setup.md``: in ``NSP-*.ns6``,
    channel 1 carries the 1 Hz square wave from SpikeGLX and channel 2 carries
    the 14 s coded burst emitted by Blackrock DO1.
    """

    #: NSP-*.ns6 -- holds the two sync channels.
    sync_file: Path | None = None
    #: HUB-*.ns6 -- holds the Utah array spike data.
    spike_file: Path | None = None
    sync_1hz_channel: int = 1
    burst_channel: int = 2
    #: Rising-edge thresholds, in the units returned by the reader (see io.blackrock).
    sync_threshold: float = 2.5
    burst_threshold: float = 2.5
    #: Stream to read from the .ns6 file (neo/SpikeInterface stream id).
    stream_id: str | None = None
    #: Channel ids to exclude from sorting (sync / analog inputs sharing the file).
    exclude_channels: tuple[str, ...] = ()

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
    #: Kilosort probe file name (e.g. "NeuroPix1_default.mat"), or None to build
    #: the channel map from the run's .meta.
    probe_name: str | None = None
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
            probe_name=data.get("probe_name"),
            sync_bit=int(data.get("sync_bit", 6)),
            sync_pulse_ms=int(data.get("sync_pulse_ms", 500)),
            burst_word=int(data.get("burst_word", 1)),
            burst_threshold_v=(float(thresh[0]), float(thresh[1])),
            burst_pulse_ms=int(data.get("burst_pulse_ms", 0)),
        )


@dataclass(frozen=True)
class OutputPaths:
    """Derived output layout for one session."""

    root: Path

    @property
    def sync(self) -> Path:
        return self.root / "sync"

    @property
    def lfp(self) -> Path:
        return self.root / "lfp"

    @property
    def sorted_np(self) -> Path:
        return self.root / "sorted_neuropixels"

    @property
    def sorted_br(self) -> Path:
        return self.root / "sorted_blackrock"

    @property
    def aligned(self) -> Path:
        return self.root / "aligned"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    def all(self) -> tuple[Path, ...]:
        return (self.sync, self.lfp, self.sorted_np, self.sorted_br, self.aligned, self.figures)

    def mkdirs(self) -> None:
        for path in self.all():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SessionConfig:
    """One recording session, resolved against one machine."""

    session: str
    machine: MachineProfile
    output_root: Path
    blackrock: BlackrockSpec = field(default_factory=BlackrockSpec)
    neuropixels: NeuropixelsSpec = field(default_factory=NeuropixelsSpec)
    #: Skip cross-system alignment (steps 8-9). True when only one system ran.
    skip_sync: bool = False
    #: Skip everything that needs Blackrock files (step 4 and the Blackrock half
    #: of steps 8-10).
    skip_blackrock: bool = False
    #: Skip Neuropixels sorting (for Blackrock-only sessions).
    skip_neuropixels: bool = False
    #: Period of the fine-alignment square wave, in seconds. TPrime -syncperiod.
    sync_period_s: float = 1.0
    #: Nominal interval of the coarse coded burst, in seconds.
    burst_interval_s: float = 14.0
    #: Alignment is considered good when residuals stay below this (seconds).
    alignment_tolerance_s: float = 1e-3
    #: Preprocessing applied before sorting.
    preprocess: dict[str, Any] = field(default_factory=dict)
    #: Extra settings merged into Kilosort4's settings dict.
    kilosort_settings: dict[str, Any] = field(default_factory=dict)

    @property
    def paths(self) -> OutputPaths:
        return OutputPaths(self.output_root)

    @property
    def cache_dir(self) -> Path | None:
        return self.machine.cache_dir

    def missing_inputs(self) -> list[str]:
        """Return human-readable problems with the *input* paths.

        Pure: touches the filesystem read-only and returns findings rather than
        raising, so callers can report every problem at once before a long job.
        """
        problems: list[str] = []

        if not self.skip_neuropixels:
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
                    "(or set skip_neuropixels: true)"
                )

        if not self.skip_blackrock:
            brk = self.blackrock
            if brk.sync_file is None:
                problems.append("blackrock.sync_file is required (or set skip_blackrock: true)")
            elif not brk.sync_file.exists():
                problems.append(f"blackrock.sync_file does not exist: {brk.sync_file}")
            if brk.spike_file is not None and not brk.spike_file.exists():
                problems.append(f"blackrock.spike_file does not exist: {brk.spike_file}")

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


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_machine(name: str, config_dir: Path | None = None) -> MachineProfile:
    """Load ``configs/machines/<name>.yaml``."""
    config_dir = config_dir or default_config_dir()
    return MachineProfile.from_dict(name, _read_yaml(config_dir / "machines" / f"{name}.yaml"))


def load_session(
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
    mapping = {
        "session": session_name,
        "data_root": str(profile.data_root) if profile.data_root else "",
        "output_root": str(profile.output_root) if profile.output_root else "",
        "cache": str(profile.cache_dir) if profile.cache_dir else "",
        "downloads": str(downloads_dir()),
    }
    data = _substitute(data, mapping)

    output_root = _as_path(data.get("output_dir"))
    if output_root is None:
        if profile.output_root is None:
            raise ValueError(
                f"{session_path} has no output_dir and machine '{profile.name}' "
                "defines no output_root"
            )
        output_root = profile.output_root / session_name

    return SessionConfig(
        session=session_name,
        machine=profile,
        output_root=output_root,
        blackrock=BlackrockSpec.from_dict(data.get("blackrock") or {}),
        neuropixels=NeuropixelsSpec.from_dict(data.get("neuropixels") or {}),
        skip_sync=bool(data.get("skip_sync", False)),
        skip_blackrock=bool(data.get("skip_blackrock", False)),
        skip_neuropixels=bool(data.get("skip_neuropixels", False)),
        sync_period_s=float(data.get("sync_period_s", 1.0)),
        burst_interval_s=float(data.get("burst_interval_s", 14.0)),
        alignment_tolerance_s=float(data.get("alignment_tolerance_s", 1e-3)),
        preprocess=dict(data.get("preprocess") or {}),
        kilosort_settings=dict(data.get("kilosort_settings") or {}),
    )


def with_overrides(config: SessionConfig, **overrides: Any) -> SessionConfig:
    """Return a copy of ``config`` with fields replaced (CLI flag support)."""
    return replace(config, **overrides)
