"""The MATLAB v7.3 writer.

Three things separate a v7.3 ``.mat`` from plain HDF5, and getting any of them
wrong produces a file MATLAB refuses or silently reads transposed. All three are
pinned here; the fourth check -- that MATLAB itself opens it -- can only be made
on a machine with MATLAB, and was done by hand when this was written.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._io.matlab import Chunked, matlab_shape, write_mat


def test_the_file_opens_with_the_signature_matlab_looks_for(tmp_path):
    path = write_mat(tmp_path / "x.mat", {"s": {"a": 1.0}})

    head = path.read_bytes()[:128]
    assert head.startswith(b"MATLAB 7.3 MAT-file")
    assert b"HDF5 schema 1.00 ." in head
    # 116 characters of text, then the 12 bytes MATLAB checks: eight nulls, the
    # version 0x0002, and "IM" -- the endianness marker.
    assert len(head) == 128
    assert head[116:] == bytes(8) + b"\x00\x02IM"


def test_dimensions_are_reversed_so_matlab_sees_them_the_right_way_round(tmp_path):
    # MATLAB is column-major and HDF5 is row-major, so an array MATLAB should
    # report as 3 x 2 is stored as (2, 3). Getting this backwards transposes
    # every LFP trace without any error.
    import h5py

    path = write_mat(tmp_path / "x.mat", {"s": {"m": np.arange(6, dtype=np.int16).reshape(3, 2)}})

    with h5py.File(path, "r") as handle:
        assert handle["s/m"].shape == (2, 3)
        assert matlab_shape(handle["s/m"].shape) == (3, 2)


def test_every_node_carries_the_class_matlab_reads_it_as(tmp_path):
    import h5py

    path = write_mat(
        tmp_path / "x.mat",
        {"s": {"i": np.int16(3), "f": 1.5, "name": "hello", "flag": True}},
    )

    with h5py.File(path, "r") as handle:
        assert handle["s"].attrs["MATLAB_class"] == b"struct"
        assert handle["s/i"].attrs["MATLAB_class"] == b"int16"
        assert handle["s/f"].attrs["MATLAB_class"] == b"double"
        assert handle["s/flag"].attrs["MATLAB_class"] == b"logical"
        # char is uint16 code points, flagged so MATLAB decodes rather than shows numbers
        assert handle["s/name"].attrs["MATLAB_class"] == b"char"
        assert handle["s/name"].attrs["MATLAB_int_decode"] == 2
        assert handle["s/name"].shape == (5, 1)
        assert "MATLAB_fields" in handle["s"].attrs


def test_a_scalar_becomes_a_one_by_one(tmp_path):
    import h5py

    path = write_mat(tmp_path / "x.mat", {"s": {"fs": 2500.0}})
    with h5py.File(path, "r") as handle:
        assert handle["s/fs"].shape == (1, 1)
        assert handle["s/fs"][()].ravel()[0] == 2500.0


def test_a_chunked_dataset_is_filled_without_ever_holding_the_array(tmp_path):
    # The reason this module exists rather than hdf5storage: an hour of LF band
    # is ~7 GB, so the writer hands out the dataset and the caller streams into
    # it. Here the callback asserts it is never given the whole thing at once.
    import h5py

    n, chan = 1000, 4
    blocks = []

    def fill(dataset):
        for start in range(0, n, 250):
            block = np.full((250, chan), start, dtype=np.int16)
            dataset[start : start + 250, :] = block
            blocks.append(start)

    path = write_mat(
        tmp_path / "x.mat",
        {"lfp": {"data": Chunked(shape=(n, chan), dtype=np.int16, fill=fill,
                                 chunks=(250, chan))}},
    )

    assert blocks == [0, 250, 500, 750]
    with h5py.File(path, "r") as handle:
        data = handle["lfp/data"]
        assert data.shape == (n, chan)               # MATLAB sees chan x n
        assert data.attrs["MATLAB_class"] == b"int16"
        assert data.chunks == (250, chan)
        assert data.compression == "gzip"
        assert data[0, 0] == 0 and data[999, 0] == 750


def test_a_dtype_matlab_has_no_class_for_is_refused(tmp_path):
    with pytest.raises(TypeError, match="no MATLAB class"):
        write_mat(tmp_path / "x.mat", {"s": {"c": np.array([1 + 2j])}})


def test_nested_structs_round_trip(tmp_path):
    import h5py

    path = write_mat(tmp_path / "x.mat", {"a": {"b": {"c": 7.0}}})
    with h5py.File(path, "r") as handle:
        assert handle["a/b"].attrs["MATLAB_class"] == b"struct"
        assert handle["a/b/c"][()].ravel()[0] == 7.0
