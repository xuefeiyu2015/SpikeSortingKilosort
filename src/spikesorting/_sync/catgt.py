"""CatGT wrapper and edge-file IO.

CatGT is SpikeGLX's command-line extractor. It is optional: every extraction it
performs has a pure-NumPy equivalent in :mod:`spikesorting._sync.edges`, and the
two are cross-checked in :mod:`spikesorting._sync.extract`.

Edge files are the interchange format for the whole alignment stage -- one
leading-edge time in seconds per line, relative to stream start. TPrime consumes
exactly this, which is why the Blackrock detector writes the same format.

**Extraction does not copy the binary.** Per the CatGT manual, new .bin/.meta are
written only when files are concatenated, the channel list changes, filters or
tshift are applied, or a time range is exported. Adding any such flag to an
extraction run would duplicate a >100 GB recording, so this wrapper never does.

Reference: https://billkarsh.github.io/SpikeGLX/More_help/CatGT_ReadMe.html
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ExtractorSpec",
    "digital_spec",
    "analog_spec",
    "write_edge_file",
    "read_edge_file",
    "catgt_script",
    "build_extract_args",
    "run_catgt",
    "find_edge_files",
]

#: CatGT stream-type selector (js) -> the command-line flag that enables it.
_JS_FLAG = {0: "-ni", 1: "-ob", 2: "-ap", 3: "-lf"}

#: js values, for readability at call sites.
JS_NI, JS_OB, JS_AP, JS_LF = 0, 1, 2, 3


@dataclass(frozen=True)
class ExtractorSpec:
    """One CatGT extractor.

    ``-xd=js,ip,word,bit,millisec`` for digital, or
    ``-xa=js,ip,word,thresh1,thresh2,millisec`` for analog.

    ``word=-1`` means "last word in the stream", which is how the SY word is
    addressed without knowing the channel count.
    """

    kind: str  # "xd" | "xa"
    js: int
    ip: int
    word: int
    bit: int | None = None
    thresh1_v: float | None = None
    thresh2_v: float | None = None
    millisec: float = 0
    #: What this extractor is for, used to name the copied output file.
    label: str = ""

    def to_flag(self) -> str:
        if self.kind == "xd":
            if self.bit is None:
                raise ValueError("digital extractor needs a bit index")
            return f"-xd={self.js},{self.ip},{self.word},{self.bit},{_num(self.millisec)}"
        if self.kind == "xa":
            if self.thresh1_v is None:
                raise ValueError("analog extractor needs a primary threshold")
            t2 = 0 if self.thresh2_v is None else self.thresh2_v
            return (
                f"-xa={self.js},{self.ip},{self.word},"
                f"{_num(self.thresh1_v)},{_num(t2)},{_num(self.millisec)}"
            )
        raise ValueError(f"unknown extractor kind {self.kind!r}")

    @property
    def stream_flag(self) -> str:
        return _JS_FLAG[self.js]


def _num(value: float | int) -> str:
    """Format a number the way CatGT expects (no trailing .0 on integers)."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def digital_spec(
    js: int, ip: int, word: int, bit: int, millisec: float, label: str = ""
) -> ExtractorSpec:
    """Digital extractor, e.g. the 1 Hz square wave in SY bit 6."""
    return ExtractorSpec("xd", js, ip, word, bit=bit, millisec=millisec, label=label)


def analog_spec(
    js: int,
    ip: int,
    word: int,
    thresh1_v: float,
    thresh2_v: float = 0.0,
    millisec: float = 0,
    label: str = "",
) -> ExtractorSpec:
    """Analog extractor, e.g. the 14 s coded burst on OneBox XA1."""
    return ExtractorSpec(
        "xa", js, ip, word, thresh1_v=thresh1_v, thresh2_v=thresh2_v, millisec=millisec, label=label
    )


def write_edge_file(path: str | Path, times_s: np.ndarray, precision: int = 6) -> Path:
    """Write leading-edge times (seconds, ascending) one per line.

    Byte-compatible with CatGT output, so TPrime cannot tell the difference
    between these and the tool's own files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    if times.size and np.any(np.diff(times) < 0):
        raise ValueError(f"edge times must be ascending before writing {path}")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for value in times:
            handle.write(f"{value:.{precision}f}\n")
    return path


def read_edge_file(path: str | Path) -> np.ndarray:
    """Read an edge file (CatGT's or ours) into a float64 array of seconds."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"edge file not found: {path}")
    text = path.read_text(encoding="utf-8").split()
    if not text:
        return np.empty(0, dtype=np.float64)
    return np.asarray([float(token) for token in text], dtype=np.float64)


def catgt_script(catgt_dir: str | Path) -> Path:
    """Locate ``runit.bat`` (Windows) or ``runit.sh`` (Linux) inside ``catgt_dir``."""
    catgt_dir = Path(catgt_dir)
    for name in ("runit.bat", "runit.sh"):
        candidate = catgt_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no runit.bat or runit.sh in {catgt_dir}")


def build_extract_args(
    run_dir: str | Path,
    run_name: str,
    specs: list[ExtractorSpec],
    gate: int = 0,
    trigger: int | str = 0,
    probes: tuple[int, ...] = (0,),
    dest: str | Path | None = None,
) -> list[str]:
    """Assemble the CatGT argument list for an extraction-only pass.

    Deliberately contains no filter, ``-save`` or ``-startsecs`` flag: any of them
    would make CatGT rewrite the binary.
    """
    if not specs:
        raise ValueError("no extractors requested")

    args = [
        f"-dir={Path(run_dir)}",
        f"-run={run_name}",
        f"-g={gate}",
        f"-t={trigger}",
    ]

    # One stream flag per distinct stream type touched by the extractors.
    for flag in dict.fromkeys(spec.stream_flag for spec in specs):
        args.append(flag)

    if any(spec.js in (JS_AP, JS_LF) for spec in specs):
        args.append("-prb=" + ",".join(str(p) for p in probes))

    args.extend(spec.to_flag() for spec in specs)

    if dest is not None:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        args.append(f"-dest={dest}")

    return args


def run_catgt(
    catgt_dir: str | Path,
    args: list[str],
    timeout_s: float | None = None,
) -> subprocess.CompletedProcess:
    """Run CatGT and return the completed process.

    Raises ``RuntimeError`` on a non-zero exit, with CatGT's own output attached --
    its error messages are specific and worth surfacing verbatim.
    """
    script = catgt_script(catgt_dir)
    command = [str(script), *args]
    result = subprocess.run(
        command,
        cwd=str(Path(catgt_dir)),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "CatGT failed (exit {code})\ncommand: {cmd}\nstdout:\n{out}\nstderr:\n{err}".format(
                code=result.returncode,
                cmd=" ".join(command),
                out=result.stdout,
                err=result.stderr,
            )
        )
    return result


def find_edge_files(dest: str | Path) -> dict[str, Path]:
    """Find the ``.txt`` edge files CatGT wrote under ``dest``.

    Keyed by the extractor part of the filename, e.g. ``imec0.ap.xd_384_6_500``.
    Globbing beats predicting the name: CatGT encodes the *resolved* word index
    (``-1`` becomes 384) and nests output under ``catgt_<run>_g<n>/``.
    """
    dest = Path(dest)
    found: dict[str, Path] = {}
    for path in sorted(dest.rglob("*.txt")):
        name = path.name
        if ".xd_" not in name and ".xa_" not in name:
            continue
        # "run_g0_tcat.imec0.ap.xd_384_6_500.txt" -> "imec0.ap.xd_384_6_500"
        parts = name[: -len(".txt")].split(".", 1)
        found[parts[1] if len(parts) > 1 else parts[0]] = path
    return found
