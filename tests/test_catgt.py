"""CatGT flag construction and edge-file interchange.

The flag strings are checked against the examples in the CatGT/SpikeGLX manuals
verbatim: a wrong js/word/bit is not something a run would report, it just
extracts the wrong channel.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting.sync import catgt


def test_digital_flag_matches_the_documented_example():
    # Manual: "-xd=2,0,-1,6,500" produces demo_g0_tcat.imec0.ap.xd_100_6_500.txt
    assert catgt.digital_spec(2, 0, -1, 6, 500).to_flag() == "-xd=2,0,-1,6,500"
    # Manual: "-xd=0,0,1,2,0" -- NI word 1, bit 2, any duration
    assert catgt.digital_spec(0, 0, 1, 2, 0).to_flag() == "-xd=0,0,1,2,0"


def test_analog_flag_matches_the_documented_example():
    # Manual: "-xa=0,0,0,1.1,0,25" -- 25 ms pulses on NI word 0
    assert catgt.analog_spec(0, 0, 0, 1.1, 0, 25).to_flag() == "-xa=0,0,0,1.1,0,25"


def test_stream_flag_follows_the_js_convention():
    assert catgt.digital_spec(0, 0, 0, 0, 0).stream_flag == "-ni"
    assert catgt.digital_spec(1, 0, 0, 0, 0).stream_flag == "-ob"
    assert catgt.digital_spec(2, 0, 0, 0, 0).stream_flag == "-ap"
    assert catgt.digital_spec(3, 0, 0, 0, 0).stream_flag == "-lf"


def test_extractor_requires_its_own_parameters():
    with pytest.raises(ValueError):
        catgt.ExtractorSpec("xd", 2, 0, -1).to_flag()  # no bit
    with pytest.raises(ValueError):
        catgt.ExtractorSpec("xa", 1, 0, 1).to_flag()  # no threshold
    with pytest.raises(ValueError):
        catgt.ExtractorSpec("xz", 1, 0, 1).to_flag()  # unknown kind


def test_build_extract_args_emits_one_stream_flag_per_type():
    specs = [
        catgt.digital_spec(catgt.JS_AP, 0, -1, 6, 500),
        catgt.digital_spec(catgt.JS_OB, 0, -1, 6, 500),
        catgt.analog_spec(catgt.JS_OB, 0, 1, 1.0),
    ]
    args = catgt.build_extract_args("D:/data", "run", specs, gate=0, trigger=0)
    assert args[:4] == ["-dir=D:/data", "-run=run", "-g=0", "-t=0"]
    assert args.count("-ob") == 1
    assert args.count("-ap") == 1
    assert "-prb=0" in args


def test_build_extract_args_contains_no_binary_rewriting_flags():
    # Per the manual a new .bin is written when files are concatenated, the
    # channel list changes, filters/tshift are applied, or a time range is
    # exported. On a 100+ GB recording that would be a very expensive accident.
    specs = [catgt.digital_spec(catgt.JS_AP, 0, -1, 6, 500)]
    args = catgt.build_extract_args("D:/data", "run", specs)
    joined = " ".join(args)
    for forbidden in ("-save", "-startsecs", "-apfilter", "-lffilter", "-tshift", "-gblcar"):
        assert forbidden not in joined


def test_build_extract_args_rejects_an_empty_extractor_list():
    with pytest.raises(ValueError):
        catgt.build_extract_args("D:/data", "run", [])


def test_edge_file_round_trip(tmp_path):
    times = np.array([0.25, 1.25, 2.25, 3.5])
    path = catgt.write_edge_file(tmp_path / "edges.txt", times)
    assert np.allclose(catgt.read_edge_file(path), times)


def test_edge_file_is_one_time_per_line_in_seconds(tmp_path):
    path = catgt.write_edge_file(tmp_path / "edges.txt", np.array([0.5, 1.5]))
    text = path.read_text(encoding="utf-8")
    assert text == "0.500000\n1.500000\n"


def test_edge_file_rejects_unsorted_times(tmp_path):
    with pytest.raises(ValueError, match="ascending"):
        catgt.write_edge_file(tmp_path / "bad.txt", np.array([1.0, 0.5]))


def test_empty_edge_file_round_trips(tmp_path):
    path = catgt.write_edge_file(tmp_path / "empty.txt", np.empty(0))
    assert catgt.read_edge_file(path).size == 0


def test_read_edge_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        catgt.read_edge_file(tmp_path / "nope.txt")


def test_find_edge_files_keys_by_extractor(tmp_path):
    (tmp_path / "catgt_run_g0").mkdir()
    (tmp_path / "catgt_run_g0" / "run_g0_tcat.imec0.ap.xd_384_6_500.txt").write_text("1.0\n")
    (tmp_path / "catgt_run_g0" / "run_g0_tcat.obx0.xa_1_0.txt").write_text("2.0\n")
    (tmp_path / "catgt_run_g0" / "run_g0_tcat.imec0.ap.meta").write_text("ignored")

    found = catgt.find_edge_files(tmp_path)
    assert set(found) == {"imec0.ap.xd_384_6_500", "obx0.xa_1_0"}


def test_catgt_script_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="runit"):
        catgt.catgt_script(tmp_path)
