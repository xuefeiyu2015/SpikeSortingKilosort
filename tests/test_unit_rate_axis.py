"""units.pdf's firing-rate panel: seconds since the recording started, not NSP time.

On Blackrock time a recording starts near 1.79e9 s, and the rate panel's axis
used to read '+1.7922e9'. The rates are still binned on the export's own axis;
only the plotted bin centres are shifted, by the same start for every unit.
"""

from __future__ import annotations

import numpy as np
import pytest

START = 1.7922e9              # an NSP/Blackrock-time recording start
DURATION = 600.0


def _units():
    from spikesorting import pipeline
    from spikesorting._export import final, metrics
    from spikesorting._export.curated import PhyResults

    rng = np.random.default_rng(0)
    early = np.sort(rng.uniform(0, DURATION, 3000))            # fires all session
    late = np.sort(rng.uniform(400, DURATION, 800))            # starts 400 s in
    spike_times_s = {0: START + early, 1: START + late}
    clusters = np.r_[np.zeros(early.size, int), np.ones(late.size, int)]
    templates = np.zeros((2, 61, 4), np.float32)
    templates[:, 20, 1] = -1.0
    phy = PhyResults(results_dir=".", spike_samples=np.zeros(clusters.size, np.int64),
                     spike_clusters=clusters, fs=30000.0, templates=templates,
                     amplitudes=rng.uniform(20, 40, clusters.size),
                     channel_map=np.arange(4), spike_templates=clusters.copy(),
                     labels={0: "good", 1: "mua"})
    unit_ids = np.array([0, 1])
    table = final.build_unit_table(phy, unit_ids, spike_times_s)
    drawn = pipeline._unit_page_data(phy, unit_ids, spike_times_s, table, None,
                                     window=(START, DURATION))
    expected = {u: metrics.compute_firing_rate(spike_times_s[u], DURATION,
                                               bin_s=max(1.0, DURATION / 100), start_s=START)
                for u in unit_ids}
    return {u["unit_id"]: u for u in drawn}, expected


def test_the_rate_axis_counts_from_the_recording_start_with_the_same_rates():
    drawn, expected = _units()
    bin_s = DURATION / 100
    for unit_id, unit in drawn.items():
        centres, rate = expected[unit_id]
        # The rates are exactly the original binning's; only the centres moved.
        np.testing.assert_array_equal(unit["rate_hz"], rate)
        np.testing.assert_allclose(unit["rate_centers_s"], centres - START)
        assert unit["rate_centers_s"][0] == pytest.approx(bin_s / 2)   # a bin centre, not 0
    # One reference for every unit: the late unit's bins ending before 400 s are
    # empty, and its bins from 400 s on are not.
    late = drawn[1]
    ends, starts = late["rate_centers_s"] + bin_s / 2, late["rate_centers_s"] - bin_s / 2
    assert not late["rate_hz"][ends <= 400].any()
    assert late["rate_hz"][starts >= 400].all()
    # The amplitude panel shares the axis: each spike at its time since the start.
    assert 400 <= drawn[1]["amp_times_s"][0] < 401 and drawn[0]["amp_times_s"].max() <= DURATION


@pytest.mark.parametrize("draw", ["summary", "page"])
def test_both_unit_figures_label_the_axis_and_start_it_at_zero(draw):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spikesorting._plots import summary as plots

    drawn, _ = _units()
    unit = drawn[1]
    fig = (plots.plot_unit_summary(unit) if draw == "summary"
           else plots.plot_unit_page([unit], rows=1))
    fig.canvas.draw()
    timed = [ax for ax in fig.axes if ax.get_xlabel() == "Time since recording start (s)"]
    assert len(timed) == 2                                      # firing rate and amplitudes
    for ax in timed:
        assert ax.get_xlim()[0] == 0
        assert ax.xaxis.get_offset_text().get_text() == ""      # no '+1.7922e9'
    rate_ax = next(ax for ax in timed if ax.lines)
    assert rate_ax.lines[0].get_xdata()[0] == pytest.approx(DURATION / 100 / 2)
    amp_ax = next(ax for ax in timed if ax.collections)
    assert amp_ax.collections[0].get_offsets()[:, 0].min() >= 400  # the late unit's first spike
    plt.close(fig)
