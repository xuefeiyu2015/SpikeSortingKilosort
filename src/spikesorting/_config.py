"""Session and machine configuration (pipeline step 1).

The configuration is split into two layers so that a single session file runs
unchanged on the Windows rig, on the HPC, and on a laptop:

``SessionConfig``
    What was recorded -- session identity, which files hold the sync pulses and
    the spike data, which channels carry which pulse. Portable.

``MachineProfile``
    Where *this* machine keeps things -- data root, SSD scratch for ``temp.dat``,
    CatGT/TPrime install dirs, torch device. Machine-local.

A session file states each system's directory outright and may refer back to it
as ``{blackrock_dir}`` / ``{neuropixels_dir}``. Those two are the only
placeholders there are: nothing is composed from a subject, a date or a machine
root, so the path in the file is the path on disk.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields, replace
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import yaml

# NumPy-only and stdlib-light, so this stays importable on a machine with no
# Kilosort, torch or SpikeInterface -- which is the whole point of this layer.
# It is what turns `probes: [0, 1]` into each probe's own binary and folder.
from ._io import spikeglx

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
]

#: The sorter this pipeline runs, and the directory results land in. One value so
#: the two cannot disagree -- a session file naming a different one would only
#: mislabel a Kilosort4 sorting.
SORTER_NAME = "kilosort4"

# Repo root is two levels above src/spikesorting/config.py
REPO_ROOT = Path(__file__).resolve().parents[2]


def stream_label(system: str) -> str:
    """What one stream is called in cache tags and status lines.

    One session names one binary per system, so a stream *is* a system and the
    label is just its name. It is no longer a path component: results go to
    ``<system>_dir/<sorter>/``, which the directory itself already identifies.
    Two probes means two session files with two ``neuropixels_dir`` values.
    """
    return system


def default_config_dir() -> Path:
    """Directory holding the YAML configs, overridable with ``SPIKESORTING_CONFIG_DIR``."""
    env = os.environ.get("SPIKESORTING_CONFIG_DIR")
    return Path(env) if env else REPO_ROOT / "configs"


def _as_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value)).expanduser()


def _as_binaries(value: Any) -> tuple[Path, ...] | None:
    """``bin_files:`` as a tuple of paths, or None when the key is absent.

    Order is kept as written, because it is the fallback probe numbering for a
    binary whose name carries none. A bare string is accepted as a one-item list.
    """
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        value = [value]
    return tuple(_as_path(item) for item in value)


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
class WaveformSpec:
    """How the waveform export cuts and filters snippets, for one system.

    Every number the export uses lives here, so a session file is the whole
    record of how a mean waveform was produced. It is a *per-system* block
    because the one setting that matters most -- whether to high-pass -- depends
    on which band the recording holds, and that differs by system and by rig.

    **The high-pass is not optional cleanup; it is what makes the mean comparable
    to anything else.** Kilosort sorts what its own 300 Hz pass produces and Phy
    displays a 150 Hz-filtered trace, so a mean cut from a *broadband* recording
    is neither. Blackrock ``.ns6`` is the broadband 30 kHz group, so it defaults
    to 300 Hz. A Neuropixels AP binary arrives already high-passed on the probe,
    so it defaults to ``None`` -- filtering it again would only cascade a second
    rolloff onto the first.

    Neither default is physics: NP 1.0's AP filter is a programmable imro bit and
    a Blackrock sampling group's band is set in Central. Both are therefore
    settable, which is the point of the block.
    """

    #: Snippet width in milliseconds, centred on the spike sample.
    window_ms: float = 2.0
    #: Spikes per unit to measure, spread uniformly over the *recording* rather
    #: than over the spike list -- see ``_export.waveforms.plan_snippets``. None
    #: measures every spike, which is the only way to get an every-spike mean.
    max_spikes: int | None = 2000
    #: Zero-phase Butterworth high-pass applied before cutting. None disables it.
    highpass_hz: float | None = 300.0
    #: Order of that filter.
    highpass_order: int = 3
    #: Extra signal read either side of a snippet and discarded after filtering,
    #: so the kept samples carry no filter transient. Phy and Kilosort both skip
    #: this and filter the bare window instead.
    filter_pad_ms: float = 10.0
    #: Whether the per-spike snippets are *kept*. The recording is read either
    #: way -- the mean needs it -- so this only decides ~120 MB per million.
    export_snippets: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, default: "WaveformSpec") -> "WaveformSpec":
        """Parse a ``waveforms:`` block, falling back to ``default`` per key.

        ``default`` rather than the field defaults, so each system keeps its own
        starting point and a block that states only one key changes only that.
        """
        data = data or {}
        max_spikes = data.get("max_spikes", default.max_spikes)
        highpass = data.get("highpass_hz", default.highpass_hz)
        return cls(
            window_ms=float(data.get("window_ms", default.window_ms)),
            max_spikes=int(max_spikes) if max_spikes is not None else None,
            highpass_hz=float(highpass) if highpass is not None else None,
            highpass_order=int(data.get("highpass_order", default.highpass_order)),
            filter_pad_ms=float(data.get("filter_pad_ms", default.filter_pad_ms)),
            export_snippets=bool(data.get("export_snippets", default.export_snippets)),
        )


#: The Neuropixels starting point: the AP band is high-passed on the probe.
_NPX_WAVEFORMS = WaveformSpec(highpass_hz=None)


@dataclass(frozen=True)
class BlackrockSpec:
    """Blackrock inputs.

    Channel defaults follow ``SynchronizePulse_setup.md``: in ``NSP-*.ns5``,
    channel 1 carries the 1 Hz square wave from SpikeGLX and channel 2 carries
    the 14 s coded burst emitted by Blackrock DO1.

    Its Neuropixels counterpart is the OneBox ``.obx`` file, found from the AP
    binary's own path -- see ``_io.spikeglx.run_layout``. A binary that is not
    laid out as a SpikeGLX run has no ``.obx`` to match against, and alignment
    falls back to the 1 Hz train alone.
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
    #: Timestamp jumps up to this are absorbed; larger ones split the recording
    #: into segments. PTP files carry occasional corrupted packet timestamps
    #: (measured: 0.8-2 ms, some *negative*), and neo refuses to open the file at
    #: all without a tolerance. Absorbing them loses no time -- they cancel within
    #: a few dozen samples. See ``_io/blackrock.nsp_time_map``.
    gap_tolerance_ms: float = 10.0
    #: Whether to proceed when a gap *above* the tolerance splits the recording.
    #: False refuses and prints neo's gap table: a jump that large is a paused
    #: recording, and whether to sort across it is a decision, not a default.
    allow_segments: bool = False
    #: This array's channel map, in whichever format it is written --
    #: ``.cmp`` (the array's own wiring map), ``.json`` from make_probe.py, or a
    #: ``.mat``. :func:`.._probes.io.load_probe_file` reads it by suffix, so the
    #: format is a property of the file rather than a second setting.
    probe_file: Path | None = None
    #: How the waveform export cuts and filters this system's snippets. The
    #: .ns6 group is broadband, so the default high-passes.
    waveforms: WaveformSpec = field(default_factory=WaveformSpec)

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
            gap_tolerance_ms=float(data.get("gap_tolerance_ms", 10.0)),
            allow_segments=bool(data.get("allow_segments", False)),
            probe_file=_as_config_path(data.get("probe_file")),
            waveforms=WaveformSpec.from_dict(data.get("waveforms"), WaveformSpec()),
        )


@dataclass(frozen=True)
class NeuropixelsSpec:
    """SpikeGLX inputs: the AP binary, and optionally the LF one beside it.

    One session names one binary outright. There is still no SpikeGLX *run*
    model -- no ``run_dir``/``run_name``, no gate/trigger -- because all of it is
    recoverable from that filename by :func:`.._io.spikeglx.run_layout`, and a
    second way to say where the data is is a second thing that can be wrong.

    ``bin_file`` is the AP band: sorted, and the source of the 1 Hz sync train in
    its SY word. ``lf_file`` is the LF band, read only by ``extract_lfp``.

    ``probes`` is what makes one file cover a two-probe run. It lists the probes
    of the run ``bin_file`` belongs to, and the rest of each probe's paths is
    *derived* from that one name -- so the file still states a binary rather than
    composing one out of a run name and a gate. The probe dimension goes no
    further than this class: :meth:`SessionConfig.for_probe` projects a listing
    session back down to an ordinary one-probe config, which is all any verb ever
    sees.
    """

    #: The AP binary. Its ``.meta``, when there is one beside it, supplies the
    #: channel count, the sample rate and the probe geometry.
    bin_file: Path | None = None
    #: The LF binary, for ``export_lfp``. Unset means there is no LF band to
    #: export, and the stage skips itself.
    lf_file: Path | None = None
    #: Total channels in the binary, including the SY word. Derived from the
    #: .meta file when absent.
    n_chan_bin: int | None = None
    #: Sampling rate. Only needed for a bare binary with no .meta beside it
    #: (the Kilosort demo file); otherwise read from the meta.
    sample_rate: float | None = None
    #: SY-word bit carrying the SMA1 1 Hz square wave. 6 is the SpikeGLX default.
    sync_bit: int = 6
    #: High duration of the 1 Hz square wave, in ms (1 s period, 50% duty).
    #: A CatGT ``-xd`` argument.
    sync_pulse_ms: int = 500
    #: Word index of the XA1 analog input in the obx stream (14 s coded burst).
    burst_word: int = 1
    #: CatGT -xa thresholds in volts: (primary, secondary). Secondary 0 disables it.
    burst_threshold_v: tuple[float, float] = (1.0, 0.0)
    #: Expected burst pulse width in ms; 0 accepts any width.
    burst_pulse_ms: int = 0
    #: Probe JSON from make_probe.py, used when the binary has no ``.meta`` to
    #: build the map from. The .meta wins where it exists: it records which sites
    #: were actually active, which a file built from another run cannot know.
    probe_file: Path | None = None
    #: One AP binary per probe, when a session covers several of one run. Each
    #: is named outright, exactly like ``bin_file`` -- nothing is derived from
    #: another probe's path. Which probe each one is comes from its own name
    #: (``.imec<n>.``), falling back to its position in the list.
    #:
    #: It says what the *session* covers and survives
    #: :meth:`SessionConfig.for_probe`, which sets ``bin_file`` to the one probe
    #: in hand. Verbs read ``bin_file``; only the config layer reads this.
    bin_files: tuple[Path, ...] | None = None
    #: Per-probe overrides, keyed by the numbers in ``probes``. Only ``kilosort``
    #: is accepted: geometry already comes from each probe's own ``.meta``, and a
    #: path is what ``bin_file`` plus the run layout already answers. Merged over
    #: the shared and per-system blocks one level deep, so a probe's
    #: ``bad_channels`` replaces the list above it rather than extending it.
    by_probe: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: How the waveform export cuts and filters this system's snippets. The AP
    #: band arrives high-passed from the probe, so the default does not filter.
    waveforms: WaveformSpec = field(default_factory=lambda: _NPX_WAVEFORMS)

    def binaries_by_probe(self) -> dict[int, Path]:
        """``{probe: AP binary}`` for everything this block names.

        The probe number comes from each binary's own name, so a ``bin_files:``
        list needs no parallel list of numbers to drift out of step with it. A
        name carrying none -- a hand-cut extract -- falls back to its position,
        which is why the list order is preserved rather than sorted.

        Duplicates collapse here; :func:`_reject_duplicate_probes` is what turns
        that into an error rather than a silently shorter list.
        """
        named = self.bin_files if self.bin_files else (
            (self.bin_file,) if self.bin_file is not None else ()
        )
        out: dict[int, Path] = {}
        for index, path in enumerate(named):
            probe = spikeglx.probe_number(path)
            out[index if probe is None else probe] = path
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NeuropixelsSpec":
        thresh = data.get("burst_threshold_v", (1.0, 0.0))
        if isinstance(thresh, (int, float)):
            thresh = (float(thresh), 0.0)
        n_chan = data.get("n_chan_bin")
        rate = data.get("sample_rate")
        return cls(
            bin_file=_as_path(data.get("bin_file")),
            lf_file=_as_path(data.get("lf_file")),
            n_chan_bin=int(n_chan) if n_chan is not None else None,
            sample_rate=float(rate) if rate is not None else None,
            sync_bit=int(data.get("sync_bit", 6)),
            sync_pulse_ms=int(data.get("sync_pulse_ms", 500)),
            burst_word=int(data.get("burst_word", 1)),
            burst_threshold_v=(float(thresh[0]), float(thresh[1])),
            burst_pulse_ms=int(data.get("burst_pulse_ms", 0)),
            probe_file=_as_config_path(data.get("probe_file")),
            bin_files=_as_binaries(data.get("bin_files")),
            # YAML gives `1:` as an int and `"1":` as a string; both mean probe 1.
            by_probe={
                int(probe): dict(block or {})
                for probe, block in (data.get("by_probe") or {}).items()
            },
            waveforms=WaveformSpec.from_dict(data.get("waveforms"), _NPX_WAVEFORMS),
        )


@dataclass(frozen=True)
class OutputPaths:
    """Where one session's outputs go: one folder per system, beside its data.

    Everything derived from a system lands in ``<system>_dir/<sorter>/`` -- the
    sorting, that system's sync edges, and the export bundle inside it. Handing
    an analysis a recording means handing it one directory, and there is no
    system or probe level above it: the directory the session names already says
    which recording it is. Cross-system results (the time map) go under the
    Blackrock one, because Blackrock is the reference timebase everything is
    mapped onto.

    ``sorted_np`` and ``sorted_br`` both name a directory holding the sorter's
    own results -- ``spike_times.npy``, ``params.py``, ``cluster_KSLabel.tsv``.
    That is the invariant every consumer depends on: :mod:`.pipeline` reads
    ``spike_times.npy`` straight out of them, and Phy is opened on them. Naming
    the directory after the sorter keeps a second sorter's run beside the first
    instead of on top of it.
    """

    blackrock_dir: Path | None = None
    neuropixels_dir: Path | None = None
    sorter: str = SORTER_NAME
    #: ``imec<n>`` when the session states a ``probes:`` list, else None. It
    #: separates the *cross-system* outputs only -- see :attr:`aligned`. Each
    #: probe's own sorting is already apart, in its own ``neuropixels_dir``.
    probe_tag: str | None = None

    def dir_for(self, system: str) -> Path | None:
        """That system's directory, or None when it never recorded."""
        return self.blackrock_dir if system == "blackrock" else self.neuropixels_dir

    def _require(self, system: str) -> Path:
        directory = self.dir_for(system)
        if directory is None:
            raise ValueError(
                f"session has no {system}_dir: state one in the session file, "
                f"pointing at the folder holding that system's recording"
            )
        return directory

    @property
    def primary(self) -> Path:
        """Where cross-system output goes -- Blackrock, or the one that recorded."""
        for directory in (self.blackrock_dir, self.neuropixels_dir):
            if directory is not None:
                return directory
        raise ValueError("session has neither a blackrock_dir nor a neuropixels_dir")

    def sorted_for(self, system: str) -> Path:
        """Where one system's sorting lands. The system-agnostic form."""
        return self._require(system) / self.sorter

    def sync_for(self, system: str) -> Path:
        """Edge files, in the same folder as the sorting they are aligned with.

        Computable before anything is sorted, which is what lets ``extract_sync``
        run on the rig straight after the session.
        """
        return self.sorted_for(system)

    @property
    def sorted_np(self) -> Path:
        return self.sorted_for("neuropixels")

    @property
    def sorted_br(self) -> Path:
        return self.sorted_for("blackrock")

    @property
    def aligned(self) -> Path:
        """The time map's home: the reference system's folder, or the only one.

        One level deeper when the session states a probe list. Every probe of a
        run is mapped onto the *same* Blackrock recording, so all of them would
        otherwise write ``time_map.json`` -- and the trimmed edge files, and
        ``neuropixels_spike_seconds_blackrock.npy`` -- to one directory, each on
        top of the last. Each probe has its own oscillator and its own SY word, so
        those are genuinely different maps and the last one written would win.

        A single-probe session has ``probe_tag`` None and no such level: there is
        one map, and a directory holding one thing needs no subdivision.
        """
        base = self.primary / self.sorter
        return base if self.probe_tag is None else base / self.probe_tag

    def export_for(self, system: str) -> Path:
        """The copy-paste bundle for one system: everything derived from it.

        Figures, the manifest, the ``.mat`` products and the plain arrays all
        land here, so handing the analysis a sorting means handing it one folder.
        Computable whether or not sorting ran -- a session that only exports the
        LFP gets a bundle holding just that, which keeps one rule instead of two.
        """
        return self.sorted_for(system) / "export"

    def figures_for(self, system: str) -> Path:
        """Per-unit and overview plots, inside that system's bundle."""
        return self.export_for(system) / "figures"

    def all(self) -> tuple[Path, ...]:
        """Every directory this session can actually write to."""
        paths: list[Path] = []
        for system in ("neuropixels", "blackrock"):
            if self.dir_for(system) is not None:
                paths += [self.sorted_for(system), self.export_for(system)]
        return tuple(dict.fromkeys(paths))

    def mkdirs(self) -> None:
        for path in self.all():
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class SessionConfig:
    """One recording session, resolved against one machine."""

    #: What this session is called in output filenames, the cache tag and the
    #: manifests. Derived from the config file's own name -- it is a label, never
    #: a path ingredient, so it cannot disagree with the file that carries it.
    session: str
    machine: MachineProfile
    #: Each system's own directory: raw recording and everything derived from it.
    #: Stated outright in the session file; nothing composes it.
    blackrock_dir: Path | None = None
    neuropixels_dir: Path | None = None
    #: Names the level below ``neuropixels/`` and ``blackrock/`` that results are
    #: written to. Not a session setting: which sorter ran is a property of the
    #: code, so this comes from :data:`SORTER_NAME` and is not read from YAML.
    sorter: str = SORTER_NAME
    blackrock: BlackrockSpec = field(default_factory=BlackrockSpec)
    neuropixels: NeuropixelsSpec = field(default_factory=NeuropixelsSpec)
    # Whether a system recorded is *derived* from whether its paths are declared
    # -- see :meth:`has_data`. Only the second question needs asking, because no
    # path can express it: a session may have recorded Neuropixels and still be
    # wanted for its LFP and sync pulses alone.
    kilosort_on_neuropixels: bool = True
    kilosort_on_blackrock: bool = True
    #: Whether to export the LF band. Off unless asked for: at ``lfp_decimate: 1``
    #: it writes a copy the size of the recording (~7 GB/hour at 385 channels).
    export_lfp: bool = False
    #: Stride for that export. A plain stride with no anti-alias filter, so 2
    #: (-> 1250 Hz) is safe against a band hardware-limited to ~500 Hz and
    #: anything above it aliases.
    lfp_decimate: int = 1
    #: Whether the export stage draws per-unit and overview figures.
    export_figures: bool = True
    #: Which Phy ``cluster_group`` labels the export stage keeps.
    export_groups: tuple[str, ...] = ("good", "mua")
    #: Period of the fine-alignment square wave, in seconds. TPrime -syncperiod.
    sync_period_s: float = 1.0
    #: Nominal interval of the coarse coded burst, in seconds.
    burst_interval_s: float = 14.0
    #: Alignment is considered good when residuals stay below this (seconds).
    alignment_tolerance_s: float = 1e-3
    #: Kilosort settings shared by both systems; see :meth:`kilosort_for`.
    kilosort: dict[str, Any] = field(default_factory=dict)
    #: Per-system ``kilosort:`` blocks, merged over the shared one. The two
    #: systems genuinely want different answers -- a 400 um Utah array and a dense
    #: probe are not the same problem -- but they share most settings.
    kilosort_by_system: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Which probe this config has been projected to by :meth:`for_probe`, or
    #: None for a session covering a single probe. Never read from YAML: the
    #: session says ``probes:``, and this says which one of them is in hand.
    probe: int | None = None

    @property
    def paths(self) -> OutputPaths:
        return OutputPaths(
            self.blackrock_dir,
            self.neuropixels_dir,
            self.sorter,
            probe_tag=self.probe_tag,
        )

    @property
    def probe_tag(self) -> str | None:
        """``imec<n>`` once projected out of a multi-probe session, else None.

        None for a session that never stated ``probes:``, so its output paths are
        exactly what they were before the key existed. The tag is what separates
        two probes' cross-system outputs and their cache work directories.

        Keyed on the session *stating* a list, not on how many probes are left to
        run: ``--probe 1`` narrows the list to one, and its outputs have to land
        where they land when both probes run. Anything counting probes here would
        silently move them.
        """
        if self.probe is None or self.neuropixels.bin_files is None:
            return None
        return f"imec{self.probe}"

    @property
    def probes(self) -> tuple[int, ...]:
        """Every probe this session covers, in ascending order.

        One per binary named -- the ``bin_files:`` list, or the single
        ``bin_file`` -- so a session that never heard of a probe list answers
        this the same way it always would. Empty when Neuropixels did not record.
        """
        return tuple(sorted(self.neuropixels.binaries_by_probe()))

    def for_probe(self, probe: int) -> "SessionConfig":
        """This session as it applies to one probe: an ordinary one-probe config.

        The whole point of the probe dimension, and the whole extent of it.
        ``bin_file`` and ``neuropixels_dir`` become that probe's own -- the
        binary the session named for it, and the folder that binary sits in --
        and its ``by_probe:`` block is merged over the Kilosort settings. What
        comes back is indistinguishable from a session file naming that probe
        alone, which is why no verb, output path or export takes a probe.

        A session naming a single ``bin_file`` is already that config, so it
        comes back with nothing but :attr:`probe` filled in.
        """
        if probe not in self.probes:
            raise ValueError(
                f"session '{self.session}' does not cover probe {probe}; "
                f"it names probes {list(self.probes)}"
            )

        npx = self.neuropixels
        kilosort_by_system = {
            system: dict(block) for system, block in self.kilosort_by_system.items()
        }
        # One level deep, exactly like kilosort_for: a probe's bad_channels
        # replaces the shared list rather than extending it.
        override = (npx.by_probe.get(probe) or {}).get("kilosort") or {}
        if override:
            merged = dict(kilosort_by_system.get("neuropixels") or {})
            merged.update(override)
            kilosort_by_system["neuropixels"] = merged

        if npx.bin_files is None:
            # A session naming one binary already *is* the projection. Leaving
            # neuropixels_dir exactly as stated is the point: it is where that
            # session's outputs have always gone, and bin_file may well sit
            # somewhere else entirely.
            return replace(self, probe=probe, kilosort_by_system=kilosort_by_system)

        bin_file = npx.binaries_by_probe()[probe]
        return replace(
            self,
            probe=probe,
            # Each probe writes inside the folder its own binary sits in.
            # neuropixels_dir is the folder above for a multi-probe session, and
            # two sortings must not both land there -- see _publish, which merges
            # rather than replaces.
            neuropixels_dir=bin_file.parent,
            # lf_file is a single key, so it cannot name both probes' LF bands.
            # _reject_bad_probe_list refuses it alongside bin_files, and
            # _lf_binary finds each probe's own beside its AP file instead.
            # bin_files stays: it records what the *session* covers, which is
            # what probe_tag keys on, while bin_file is the binary in hand. Every
            # verb reads bin_file, so the projection is what it needs to be.
            neuropixels=replace(npx, bin_file=bin_file, lf_file=None),
            kilosort_by_system=kilosort_by_system,
        )

    def per_probe(self) -> list[tuple[str | None, "SessionConfig"]]:
        """``(tag, config)`` for every probe, for the drivers to loop over.

        One entry for an ordinary session, so a driver has a single code path and
        a single-probe run prints exactly what it printed before. ``tag`` is None
        there too, and the config is this one.
        """
        if not self.has_data("neuropixels"):
            return [(None, self)]
        projections = [self.for_probe(p) for p in self.probes]
        return [(config.probe_tag, config) for config in projections]

    @property
    def output_root(self) -> Path:
        """Single directory to name when reporting where a session writes."""
        return self.paths.primary

    @property
    def cache_dir(self) -> Path | None:
        return self.machine.cache_dir

    def kilosort_for(self, system: str) -> dict[str, Any]:
        """Kilosort settings for one system: the shared block, then that system's.

        Merged key by key rather than replaced, so a system can change ``nblocks``
        without discarding the rest of the shared block. The layer is one level
        deep: a per-system ``bad_channels`` *replaces* the list above it rather
        than extending it. Everything here is passed straight to ``run_sorter`` --
        Kilosort's own parameters plus the ones SpikeInterface adds (``do_CAR``,
        ``bad_channels``, ``invert_sign``, ``skip_kilosort_preprocessing``,
        ``save_preprocessed_copy``).

        Empty means Kilosort's defaults, which already highpass at 300 Hz and
        subtract the median across channels on every batch.
        """
        merged = dict(self.kilosort)
        merged.update(self.kilosort_by_system.get(system) or {})
        return merged

    def waveforms_for(self, system: str) -> WaveformSpec:
        """How to cut and filter waveforms for one system.

        A plain lookup rather than a merge: unlike ``kilosort:``, every key here
        already differs by system or is identical everywhere, so a shared layer
        would only add a place for the two to disagree silently.
        """
        if system not in ("blackrock", "neuropixels"):
            raise ValueError(f"unknown system {system!r}")
        return self.blackrock.waveforms if system == "blackrock" else self.neuropixels.waveforms

    def has_data(self, system: str) -> bool:
        """Did this system record? Derived from whether its paths are *declared*.

        Declared, not present on disk. A share that is not mounted must stay
        "this system recorded, and its files are missing" -- which
        :meth:`missing_inputs` reports loudly -- rather than becoming "this
        system never recorded", which would skip it in silence.
        """
        if system == "neuropixels":
            return bool(self.neuropixels.binaries_by_probe())
        brk = self.blackrock
        return brk.sync_file is not None or brk.spike_file is not None

    @property
    def aligns_systems(self) -> bool:
        """Whether cross-system alignment applies: it needs both systems."""
        return self.has_data("neuropixels") and self.has_data("blackrock")

    @property
    def sorts_neuropixels(self) -> bool:
        """Whether step 3 runs. ``kilosort_on_*`` is a request, not an override."""
        return self.has_data("neuropixels") and self.kilosort_on_neuropixels

    @property
    def sorts_blackrock(self) -> bool:
        """Whether step 4 runs.

        ``kilosort_on_blackrock`` decides it, and a session that asks for it must
        name a ``spike_file`` -- :func:`_reject_sorting_nothing` refuses the
        combination at load, so this cannot be true with nothing to sort.
        Recording the sync channels on Blackrock while the spikes come from
        Neuropixels is an ordinary session shape -- a ``sync_file``, no Utah
        array -- and it says so with ``kilosort_on_blackrock: false``.
        ``has_data`` stays true for it, because Blackrock did record and its
        pulses are still extracted.
        """
        return self.blackrock.spike_file is not None and self.kilosort_on_blackrock

    def _probe_configs(self) -> list["SessionConfig"]:
        """Every probe's projection, or just this one if it is already a probe's.

        Guards the recursion: :meth:`missing_inputs` checks each probe's binary
        by asking each projection, and a projection asking again would not stop.
        """
        if self.probe is not None:
            return [self]
        return [config for _, config in self.per_probe()]

    def missing_inputs(self) -> list[str]:
        """Return human-readable problems with the *input* paths.

        Pure: touches the filesystem read-only and returns findings rather than
        raising, so callers can report every problem at once before a long job.
        """
        problems: list[str] = []

        # Gated on the data being *declared*, not on sorting being wanted: the
        # LFP and sync stages read the same files, so an unsortable session still
        # needs them to be there.
        if self.has_data("neuropixels"):
            # Every probe, not just the one bin_file names: the others are
            # derived, so a session covering two probes with only one on disk
            # would otherwise pass here and fail an hour into the run.
            for probe_config in self._probe_configs():
                npx = probe_config.neuropixels
                where = (
                    "neuropixels.bin_file"
                    if probe_config.probe_tag is None
                    else f"neuropixels.bin_file ({probe_config.probe_tag})"
                )
                if not npx.bin_file.exists():
                    problems.append(f"{where} does not exist: {npx.bin_file}")
                # Only when named: an unset lf_file means there is no LF band to
                # export, which is the ordinary case, not a missing input.
                if npx.lf_file is not None and not npx.lf_file.exists():
                    problems.append(f"neuropixels.lf_file does not exist: {npx.lf_file}")

        if self.has_data("blackrock"):
            brk = self.blackrock
            if brk.sync_file is None:
                # Only needed to align against the other system. A Blackrock-only
                # session has nothing to align to, so it needs no pulse train.
                if self.aligns_systems:
                    problems.append(
                        "blackrock.sync_file is required to align against Neuropixels"
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
            if not self.has_data(system):
                continue
            path = getattr(self, system).probe_file
            if path is not None and not path.exists():
                problems.append(f"{system}.probe_file does not exist: {path}")

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


#: Top-level session keys the loader reads itself. The per-system sets below are
#: *derived* from the dataclasses, so they cannot drift; only this one is written
#: by hand, and ``tests/test_config.py`` pins it by loading every tracked config.
_SESSION_KEYS = frozenset({
    "blackrock_dir",
    "neuropixels_dir",
    "blackrock",
    "neuropixels",
    "kilosort",
    "kilosort_on_neuropixels",
    "kilosort_on_blackrock",
    "export_lfp",
    "lfp_decimate",
    "export_figures",
    "export_groups",
    "sync_period_s",
    "burst_interval_s",
    "alignment_tolerance_s",
})


def _spec_keys(spec: type) -> frozenset[str]:
    """The YAML keys one spec dataclass accepts, read off the dataclass itself."""
    return frozenset(f.name for f in fields(spec))


#: ``kilosort`` is not a field of either spec -- the loader lifts it out into
#: ``kilosort_by_system`` -- so it is added here rather than derived.
_BLACKROCK_KEYS = _spec_keys(BlackrockSpec) | {"kilosort"}
_NEUROPIXELS_KEYS = _spec_keys(NeuropixelsSpec) | {"kilosort"}
_WAVEFORM_KEYS = _spec_keys(WaveformSpec)
#: All a per-probe block may say. See :attr:`NeuropixelsSpec.by_probe`.
_BY_PROBE_KEYS = frozenset({"kilosort"})

#: Keys that were renamed rather than merely removed, and whose replacement the
#: fuzzy match gets *wrong*. Retired keys are normally caught by simply not being
#: settings any more, with :func:`get_close_matches` naming the nearest one -- but
#: ``cmp_file`` is closest to ``sync_file``, which is a different file entirely
#: and in the same block, so the hint would tell someone to point their channel
#: map at the pulse train. Keep this to the cases where the guess is actively
#: misleading; a key with no good neighbour needs no entry.
_RENAMED = {"cmp_file": "probe_file"}


def _reject_unknown_keys(data: dict[str, Any], path: Path) -> None:
    """Refuse any key the config model does not define, and suggest the real one.

    YAML does not care about a key nobody reads, so without this a typo loads
    clean and runs with the default: ``export_figure:`` silently keeps figures
    on, ``kilosort_on_neuropixel:`` silently keeps sorting. That is a wrong
    result rather than a wrong path, and nothing downstream can tell. The same
    check retires keys that have been removed -- ``monkey``, ``roots``,
    ``run_dir``, ``exclude_channels`` -- with no list to maintain: they are
    simply not defined any more.

    Nested blocks are checked too, since ``waveforms:`` is per system and a typo
    there would quietly restore a default the session meant to change.

    **The contents of a ``kilosort:`` block are deliberately not checked.** Those
    keys pass through to Kilosort, and ``pipeline._kilosort_arguments`` already
    validates them against the installed ``DEFAULT_SETTINGS`` and
    ``inspect.signature(run_kilosort)``. A second copy of that split here would
    go stale the first time Kilosort moved a parameter.
    """
    scopes: list[tuple[str, dict[str, Any], frozenset[str]]] = [("", data, _SESSION_KEYS)]
    for system, known in (
        ("blackrock", _BLACKROCK_KEYS),
        ("neuropixels", _NEUROPIXELS_KEYS),
    ):
        block = data.get(system)
        if not isinstance(block, dict):
            continue
        scopes.append((f"{system}.", block, known))
        waveforms = block.get("waveforms")
        if isinstance(waveforms, dict):
            scopes.append((f"{system}.waveforms.", waveforms, _WAVEFORM_KEYS))
        # A by_probe: entry takes `kilosort:` and nothing else. Geometry comes
        # from each probe's own .meta and the paths come from the run layout, so
        # a `probe_file:` or `bin_file:` here would be read by nobody.
        for probe, entry in (block.get("by_probe") or {}).items():
            if isinstance(entry, dict):
                scopes.append((f"{system}.by_probe.{probe}.", entry, _BY_PROBE_KEYS))

    for prefix, block, known in scopes:
        for key in sorted(str(k) for k in block):
            if key in known:
                continue
            if key in _RENAMED and _RENAMED[key] in known:
                raise ValueError(
                    f"{path}: '{prefix}{key}' is now '{prefix}{_RENAMED[key]}', "
                    "which takes the channel map in any format it is written in "
                    "-- .cmp, .json or .mat, read by its extension."
                )
            close = get_close_matches(key, sorted(known), n=1, cutoff=0.7)
            hint = f"\n  did you mean: {prefix}{close[0]}?" if close else ""
            raise ValueError(f"{path}: unknown setting '{prefix}{key}'{hint}")


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
    _reject_unknown_keys(data, session_path)
    profile = load_machine(machine, config_dir) if isinstance(machine, str) else machine

    # The session label, and only a label: it names the .mat products, the cache
    # tag and the manifests, and never takes part in building a path. Taken from
    # the file so it cannot disagree with the config that carries it.
    session_name = session_path.stem

    # Each system's directory, stated outright. Everything that system produces
    # hangs off it, so it is exposed as a placeholder for the rest of the file --
    # ``sync_file: "{blackrock_dir}/NSP-Athos_001.ns5"``. These two names are the
    # whole substitution table: nothing is composed from a subject, a date or a
    # machine root, so what the file says is what lands on disk.
    system_dirs: dict[str, Path | None] = {}
    mapping: dict[str, str] = {}
    for system in ("blackrock", "neuropixels"):
        stated = data.get(f"{system}_dir")
        system_dirs[system] = _as_path(stated) if stated else None
        if system_dirs[system] is not None:
            mapping[f"{system}_dir"] = str(system_dirs[system])

    data = _substitute(data, mapping)

    stale = _first_unresolved(data)
    if stale is not None:
        known = ", ".join(sorted(mapping)) or "(none)"
        raise ValueError(
            f"{session_path}: unresolved placeholder in {stale!r}.\n"
            f"  known names: {known}\n"
            "  a session file has only {blackrock_dir} and {neuropixels_dir}; "
            "write every other path out in full."
        )

    if system_dirs["blackrock"] is None and system_dirs["neuropixels"] is None:
        raise ValueError(
            f"{session_path}: no directory for either system. State "
            "blackrock_dir and/or neuropixels_dir, each pointing at the folder "
            "holding that system's recording."
        )

    blackrock_spec = BlackrockSpec.from_dict(data.get("blackrock") or {})
    config = SessionConfig(
        session=session_name,
        machine=profile,
        blackrock_dir=system_dirs["blackrock"],
        neuropixels_dir=system_dirs["neuropixels"],
        blackrock=blackrock_spec,
        neuropixels=NeuropixelsSpec.from_dict(data.get("neuropixels") or {}),
        kilosort_on_neuropixels=bool(data.get("kilosort_on_neuropixels", True)),
        # Unstated means "sort it if there is something to sort", so a sync-only
        # session needs no line at all -- while an explicit `true` is a request,
        # and _reject_sorting_nothing refuses one that cannot be honoured. A
        # Neuropixels block needs no equivalent: its paths *are* its spike data.
        kilosort_on_blackrock=(
            bool(data["kilosort_on_blackrock"])
            if data.get("kilosort_on_blackrock") is not None
            else blackrock_spec.spike_file is not None
        ),
        export_lfp=bool(data.get("export_lfp", False)),
        lfp_decimate=int(data.get("lfp_decimate", 1)),
        export_figures=bool(data.get("export_figures", True)),
        export_groups=tuple(
            str(group) for group in (data.get("export_groups") or ("good", "mua"))
        ),
        sync_period_s=float(data.get("sync_period_s", 1.0)),
        burst_interval_s=float(data.get("burst_interval_s", 14.0)),
        alignment_tolerance_s=float(data.get("alignment_tolerance_s", 1e-3)),
        kilosort=dict(data.get("kilosort") or {}),
        kilosort_by_system={
            system: dict((data.get(system) or {}).get("kilosort") or {})
            for system in ("neuropixels", "blackrock")
            if (data.get(system) or {}).get("kilosort")
        },
    )
    _reject_bad_probe_list(config, session_path)
    _reject_sorting_nothing(config, data, session_path)
    _reject_colliding_system_dirs(config, session_path)
    return config


def _reject_sorting_nothing(
    config: SessionConfig, data: dict[str, Any], path: Path
) -> None:
    """Refuse an explicit ``kilosort_on_blackrock: true`` with no ``spike_file``.

    The flag is a *request*, so a request that cannot be honoured has to fail
    rather than quietly turn into nothing. This combination used to skip with
    "no blackrock.spike_file, so there is no spike data to sort" -- which reads
    fine for a sync-only session, and reads like a successful run for someone who
    meant to sort the array and mistyped the path. The session then ends with no
    Blackrock units and one `[--]` line to say why, hours after the fact.

    **Only when the session actually writes the line.** Unstated, the flag
    follows the paths, so a sync-only session needs no ``kilosort_on_blackrock:
    false`` to stay quiet -- exactly as before. Raising on the default would make
    every such session carry a line saying it is not doing something.

    Neuropixels needs no equivalent: its paths *are* its spike data, so
    ``kilosort_on_neuropixels: true`` with no ``bin_file`` is already "this system
    did not record" -- see :meth:`SessionConfig.has_data`.
    """
    if data.get("kilosort_on_blackrock") is not True:
        return
    if config.blackrock.spike_file is not None:
        return
    if not config.has_data("blackrock"):
        return          # Blackrock did not record at all; the flag is moot.
    raise ValueError(
        f"{path}: kilosort_on_blackrock is true but blackrock.spike_file names "
        "nothing to sort.\n"
        "  If this session recorded a Utah array, name its .ns6:\n"
        "      blackrock:\n"
        "        spike_file: \"{blackrock_dir}/HUB-<subject>_001.ns6\"\n"
        "  If Blackrock is here only to carry the sync pulses, say so:\n"
        "      kilosort_on_blackrock: false\n"
        "  ...which still extracts its pulses and LFP; it only skips the sort."
    )


def _reject_bad_probe_list(config: SessionConfig, path: Path) -> None:
    """Refuse a ``bin_files:`` / ``by_probe:`` block that cannot mean what it says.

    Each of these would otherwise fail much later and much less clearly -- half
    way through a sort, or as a session quietly covering one probe when it named
    two.
    """
    npx = config.neuropixels

    if npx.bin_files is not None and npx.bin_file is not None:
        raise ValueError(
            f"{path}: neuropixels names both bin_file and bin_files. One binary "
            "is bin_file; several are bin_files, bin_file's own included -- "
            "otherwise there are two answers to which probe this session sorts."
        )

    if npx.bin_files is not None:
        if not npx.bin_files:
            raise ValueError(
                f"{path}: neuropixels.bin_files is empty. Name a binary per "
                "probe, or use bin_file for a session with one."
            )
        # binaries_by_probe() keys on the probe number in each name, so two
        # binaries claiming one probe collapse to a single entry -- a session
        # silently sorting half of what it names.
        by_probe = npx.binaries_by_probe()
        if len(by_probe) != len(npx.bin_files):
            raise ValueError(
                f"{path}: neuropixels.bin_files names {len(npx.bin_files)} "
                f"binaries but only {len(by_probe)} distinct probe(s). The probe "
                "comes from each name's .imec<n>, so two binaries for one probe "
                "-- or a copied line -- reads as one.\n"
                "  named: " + ", ".join(f.name for f in npx.bin_files)
            )
        if npx.lf_file is not None:
            raise ValueError(
                f"{path}: neuropixels.lf_file names one LF band, but bin_files "
                "names several probes, and each has its own. Remove it -- the LF "
                "binary beside each AP file is found without being named."
            )

    unknown = sorted(set(npx.by_probe) - set(config.probes))
    if unknown:
        covered = list(config.probes) or "none (no binary is named)"
        raise ValueError(
            f"{path}: neuropixels.by_probe names probe(s) {unknown}, which this "
            f"session does not cover. It covers {covered}."
        )


def _reject_colliding_system_dirs(config: SessionConfig, path: Path) -> None:
    """Two recording systems may not share a directory.

    Everything a system produces lands in ``<system>_dir/<sorter>/`` and nothing
    in that path names the system -- the directory is supposed to *be* the
    identity. So two systems pointed at one directory would write one sorting on
    top of the other: same ``spike_times.npy``, same ``params.py``, same export
    bundle, and ``_publish`` copies with ``dirs_exist_ok`` so the second would
    merge into the first rather than replace it.

    Easy to hit by accident whenever one share holds everything and the obvious
    directory to name is the session folder both systems dropped files into. The
    fix it names is the real layout: a SpikeGLX run nests its own
    ``<run>_g<n>_imec<n>`` folder below that, while the ``.ns5``/``.ns6`` sit at
    the top.
    """
    if not (config.has_data("blackrock") and config.has_data("neuropixels")):
        return
    # Per probe, because a multi-probe session's neuropixels_dir is the gate
    # folder and it is each probe's *own* folder that gets written to. Checking
    # only the stated one would let a two-probe session past the guard.
    if all(
        config.blackrock_dir != probe_config.neuropixels_dir
        for _, probe_config in config.per_probe()
    ):
        return
    raise ValueError(
        f"{path}: blackrock_dir and neuropixels_dir are the same directory "
        f"({config.blackrock_dir}), but each system writes everything it "
        f"produces to <dir>/{config.sorter}/ -- so the two sortings would land on "
        "top of each other.\n"
        "  Point them at the two recordings' own folders. A SpikeGLX run nests "
        "its own, so neuropixels_dir is usually the <run>_g<n>_imec<n> directory "
        "holding the .ap.bin, while blackrock_dir is where the .ns5/.ns6 sit."
    )


def with_overrides(config: SessionConfig, **overrides: Any) -> SessionConfig:
    """Return a copy of ``config`` with fields replaced (CLI flag support).

    ``waveforms=`` is the one nested case: it holds ``WaveformSpec`` keys and is
    applied to *both* systems, since a flag says what this run should do rather
    than which band a system recorded.
    """
    waveforms = overrides.pop("waveforms", None)
    if waveforms:
        config = replace(
            config,
            blackrock=replace(
                config.blackrock,
                waveforms=replace(config.blackrock.waveforms, **waveforms),
            ),
            neuropixels=replace(
                config.neuropixels,
                waveforms=replace(config.neuropixels.waveforms, **waveforms),
            ),
        )
    return replace(config, **overrides) if overrides else config
