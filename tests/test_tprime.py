"""TPrime argument construction and the sample/second unit conversion."""

from __future__ import annotations

import sys
import time

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


def _fake_tprime(tmp_path, body):
    script = tmp_path / "TPrime"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)


@pytest.mark.skipif(sys.platform == "win32", reason="fake TPrime needs a POSIX shell")
def test_a_stuck_tprime_is_stopped_at_its_deadline(tmp_path):
    _fake_tprime(tmp_path, "echo $$ > pid.txt\nsleep 60\n")
    start = time.monotonic()
    with pytest.raises(tprime.TPrimeTimeout, match="still running"):
        tprime.run_tprime(tmp_path, [], timeout_s=1.0, poll_s=0.1)
    assert time.monotonic() - start < 10
    pid = int((tmp_path / "pid.txt").read_text())
    import os
    with pytest.raises(ProcessLookupError):      # killed, not left holding files
        os.kill(pid, 0)


@pytest.mark.skipif(sys.platform == "win32", reason="fake TPrime needs a POSIX shell")
def test_tprime_never_waits_on_the_keyboard(tmp_path):
    _fake_tprime(tmp_path, 'read answer\necho "got:$answer"\n')
    result = tprime.run_tprime(tmp_path, [], timeout_s=10.0)
    assert result.stdout.strip() == "got:"


@pytest.mark.skipif(sys.platform == "win32", reason="fake TPrime needs a POSIX shell")
def test_tprime_failure_quotes_its_log(tmp_path):
    _fake_tprime(tmp_path, "echo 'bad edge file' > TPrime.log\nexit 3\n")
    with pytest.raises(RuntimeError, match="bad edge file"):
        tprime.run_tprime(tmp_path, ["-syncperiod=1.0"])


def test_the_deadline_grows_with_the_events():
    assert tprime.tprime_timeout_s(0) == 600.0
    assert tprime.tprime_timeout_s(10_000_000) == pytest.approx(1200.0)
