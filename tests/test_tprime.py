"""TPrime argument construction and the sample/second unit conversion."""

from __future__ import annotations

import numpy as np
import pytest

from spikesorting._sync import tprime


def test_build_args_matches_the_documented_shape():
    args = tprime.build_tprime_args(
        to_stream="/out/blackrock_1hz.txt",
        from_streams={1: "/out/npx_1hz.txt"},
        events=[tprime.TPrimeEvent(1, "/out/spikes.npy", "/out/spikes_br.npy")],
        sync_period_s=1.0,
    )
    assert args[0] == "-syncperiod=1.0"
    assert args[1] == "-tostream=/out/blackrock_1hz.txt"
    assert "-fromstream=1,/out/npx_1hz.txt" in args
    assert "-events=1,/out/spikes.npy,/out/spikes_br.npy" in args


def test_build_args_rejects_events_without_a_matching_stream():
    with pytest.raises(ValueError, match="no -fromstream"):
        tprime.build_tprime_args(
            to_stream="ref.txt",
            from_streams={1: "a.txt"},
            events=[tprime.TPrimeEvent(2, "in.npy", "out.npy")],
        )


def test_build_args_rejects_no_events():
    with pytest.raises(ValueError, match="nothing"):
        tprime.build_tprime_args("ref.txt", {1: "a.txt"}, [])


def test_multiple_from_streams_are_emitted_in_id_order():
    args = tprime.build_tprime_args(
        "ref.txt",
        {2: "b.txt", 1: "a.txt"},
        [tprime.TPrimeEvent(1, "i.npy", "o.npy")],
    )
    from_flags = [a for a in args if a.startswith("-fromstream=")]
    assert from_flags == ["-fromstream=1,a.txt", "-fromstream=2,b.txt"]


def test_sample_to_second_conversion_round_trips():
    samples = np.array([0, 30000, 45000, 90000], dtype=np.int64)
    seconds = tprime.spike_times_to_seconds(samples, 30000.0)
    assert np.allclose(seconds, [0.0, 1.0, 1.5, 3.0])
    assert np.array_equal(tprime.seconds_to_spike_times(seconds, 30000.0), samples)


def test_conversion_rejects_a_bad_sample_rate():
    with pytest.raises(ValueError):
        tprime.spike_times_to_seconds(np.array([1]), 0.0)
    with pytest.raises(ValueError):
        tprime.seconds_to_spike_times(np.array([1.0]), -1.0)


def test_tprime_executable_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="TPrime"):
        tprime.tprime_executable(tmp_path)
