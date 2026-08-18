"""Write MATLAB v7.3 (``-v7.3``) ``.mat`` files, streaming.

The lab reads this pipeline's products from MATLAB (``JLab/BlackrockLoader.m``)
and from the Python port that matches its conventions
(``JLab_Python/jlab_loader``). Both speak one format: v7.3, one top-level struct
per product. This module writes it.

**Why not ``scipy.io.savemat`` or ``hdf5storage``.** scipy writes MAT-v5, which
caps a variable at 4 GB; a Neuropixels LF band is 7-10 GB. ``hdf5storage`` writes
v7.3 correctly but takes the finished array as an argument, so the whole thing
has to be in memory first. Here the array is *streamed* -- see :class:`Chunked`.

**What makes an HDF5 file a v7.3 ``.mat``**, all three established by dumping a
file ``hdf5storage`` wrote rather than from documentation:

1. a 512-byte userblock whose first 128 bytes carry the header MATLAB checks:
   116 characters of text, space-padded, then ``00000000 00000000 0002494D``;
2. ``MATLAB_class`` on every node -- ``struct`` on a group, the numeric class on
   a dataset, ``char`` (plus ``MATLAB_int_decode = 2``) on a string stored as
   uint16;
3. **reversed dimensions.** MATLAB is column-major and HDF5 is row-major, so a
   MATLAB ``M x N`` array is stored with shape ``(N, M)``. A MATLAB row vector
   ``1 x N`` is stored as ``(N, 1)``.

That last one is the trap, and it cuts both ways: ``jlab_loader/_matio.py``'s
reader transposes on the way back, so an array written here as ``(nSamples,
nChan)`` is what MATLAB and that reader both call ``nChan x nSamples``.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = ["Chunked", "write_mat", "update_mat", "matlab_shape"]

#: numpy kind -> the MATLAB class name that must be stamped on the dataset.
_CLASSES = {
    "int8": "int8", "int16": "int16", "int32": "int32", "int64": "int64",
    "uint8": "uint8", "uint16": "uint16", "uint32": "uint32", "uint64": "uint64",
    "float32": "single", "float64": "double", "bool": "logical",
}


@dataclass
class Chunked:
    """A dataset too big to build in memory: its shape, and how to fill it.

    ``shape`` is the **HDF5** shape, so it is already MATLAB's shape reversed --
    ``(n_samples, n_chan)`` for something MATLAB should see as
    ``nChan x nSamples``. ``fill`` is handed the created dataset and writes into
    it however it likes, a chunk at a time; nothing here holds the array.
    """

    shape: tuple[int, ...]
    dtype: Any
    fill: Callable[[Any], None]
    chunks: tuple[int, ...] | None = None
    compression: str | None = "gzip"


def matlab_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape MATLAB will report for an HDF5 dataset of ``shape``."""
    return tuple(reversed(shape))


def _header_bytes() -> bytes:
    """The 128-byte MATLAB v7.3 signature that opens the file."""
    now = datetime.datetime.now()
    weekday = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")[now.weekday()]
    month = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[now.month - 1]
    text = (
        "MATLAB 7.3 MAT-file, Platform: spikesorting, "
        f"Created on: {weekday} {month} {now:%d %H:%M:%S %Y} HDF5 schema 1.00 ."
    )
    padded = bytearray(text.ljust(116)[:116], encoding="ascii")
    padded.extend(bytes.fromhex("00000000 00000000 0002494D".replace(" ", "")))
    return bytes(padded)


def _matlab_fields(names: list[str]):
    """The ``MATLAB_fields`` attribute: field order, as a vlen array of chars."""
    import h5py

    dtype = h5py.special_dtype(vlen=np.dtype("S1"))
    fields = np.empty(shape=(len(names),), dtype=dtype)
    for i, name in enumerate(names):
        fields[i] = np.array([c.encode("ascii") for c in name], dtype="S1")
    return fields


def _write_value(group, name: str, value: Any) -> None:
    """Write one field of a struct, in whatever form MATLAB expects for it."""
    if isinstance(value, dict):
        sub = group.create_group(name)
        _write_struct(sub, value)
        return

    if isinstance(value, Chunked):
        dataset = group.create_dataset(
            name,
            shape=value.shape,
            dtype=value.dtype,
            chunks=value.chunks,
            compression=value.compression,
        )
        value.fill(dataset)
        dataset.attrs["MATLAB_class"] = np.bytes_(_class_of(np.dtype(value.dtype)))
        return

    if isinstance(value, (str, Path)):
        # MATLAB char: uint16 code points, as a column, flagged for decoding.
        codes = np.array([ord(c) for c in str(value)], dtype=np.uint16).reshape(-1, 1)
        dataset = group.create_dataset(name, data=codes)
        dataset.attrs["MATLAB_class"] = np.bytes_("char")
        dataset.attrs["MATLAB_int_decode"] = np.int64(2)
        return

    array = np.asarray(value)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8)
        matlab_class = "logical"
    else:
        matlab_class = _class_of(array.dtype)

    if array.ndim == 0:
        array = array.reshape(1, 1)              # MATLAB has no 0-D
    elif array.ndim == 1:
        array = array.reshape(-1, 1)             # a MATLAB 1 x N row vector
    else:
        array = array.T                          # MATLAB M x N -> HDF5 (N, M)

    dataset = group.create_dataset(name, data=array)
    dataset.attrs["MATLAB_class"] = np.bytes_(matlab_class)
    if matlab_class == "logical":
        dataset.attrs["MATLAB_int_decode"] = np.int64(1)


def _class_of(dtype: np.dtype) -> str:
    name = np.dtype(dtype).name
    if name not in _CLASSES:
        raise TypeError(f"no MATLAB class for dtype {name}")
    return _CLASSES[name]


def _write_struct(group, fields: dict[str, Any]) -> None:
    group.attrs["MATLAB_class"] = np.bytes_("struct")
    group.attrs["MATLAB_fields"] = _matlab_fields(list(fields))
    for name, value in fields.items():
        _write_value(group, name, value)


def write_mat(path: str | Path, variables: dict[str, Any]) -> Path:
    """Write ``variables`` as a MATLAB v7.3 ``.mat``. Returns the path.

    Each top-level entry becomes one MATLAB variable; a dict becomes a struct, a
    :class:`Chunked` becomes a dataset filled by its callback, and everything
    else is written as an array in MATLAB's orientation. See the module
    docstring for the three things that make the file a ``.mat`` rather than
    plain HDF5.
    """
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The userblock has to be reserved at creation and can only be written once
    # the file is closed -- h5py owns those bytes while the file is open.
    with h5py.File(path, "w", userblock_size=512) as handle:
        for name, value in variables.items():
            if isinstance(value, dict):
                _write_struct(handle.create_group(name), value)
            else:
                _write_value(handle, name, value)

    with open(path, "r+b") as handle:
        handle.write(_header_bytes())
    return path


def update_mat(path: str | Path, variables: dict[str, dict[str, Any]]) -> Path:
    """Replace named fields of an existing ``.mat``, leaving the bulk data alone.

    For metadata a later stage learns -- the fitted time map, say -- when the
    array it describes is gigabytes and was written hours earlier. Only the named
    datasets are rewritten; the file grows by the size of the new values and the
    data is never re-read. The userblock survives, since h5py leaves the first
    512 bytes alone when it appends.

    Field *order* is carried in the group's ``MATLAB_fields`` attribute, so it is
    rebuilt from what the group holds afterwards -- otherwise a field added here
    would be invisible to MATLAB.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "a") as handle:
        for variable, fields in variables.items():
            if variable not in handle:
                raise KeyError(f"{path.name} has no variable {variable!r} to update")
            group = handle[variable]
            for name, value in fields.items():
                if name in group:
                    del group[name]
                _write_value(group, name, value)
            group.attrs["MATLAB_fields"] = _matlab_fields(list(group.keys()))
    return path
