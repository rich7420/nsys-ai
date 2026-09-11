"""Rule-level contracts for ``root_cause_matcher``.

The matcher reads other skills' summaries, so its correctness depends on picking
the right field out of them. That is what these cover: not whether a rule's
threshold is well chosen, but whether it is applied to the quantity it means.
"""

# ── Which idle figure drives the rules ──────────────────────────────────────


class TestIdleFieldSelection:
    """``total_idle_ms`` is the per-stream sum; ``device_idle_ms`` is the device.

    ``gpu_idle_gaps`` says so in its own formatter -- "on a multi-stream profile
    that overstates the wall-clock lost" -- and prints both for that reason. The
    matcher took the overstating one, which inflates the denominator of the
    synchronisation ratio and makes the rule under-fire on exactly the profiles
    with the most streams to be wrong about.
    """

    @staticmethod
    def _pick(summary):
        from nsys_ai.skills.builtins.root_cause_matcher import _idle_with_matching_pct

        return _idle_with_matching_pct(summary, [])[0]

    def test_the_device_figure_wins_where_they_disagree(self):
        """A four-stream profile: the sum is ~4x the wall-clock loss."""
        assert self._pick({"device_idle_ms": 938.1, "total_idle_ms": 3752.4}) == 938.1

    def test_the_sum_is_used_when_the_device_sweep_could_not_run(self):
        """device_idle_ms is None there, and None is not a measurement."""
        assert self._pick({"device_idle_ms": None, "total_idle_ms": 3752.4}) == 3752.4

    def test_a_no_gaps_summary_yields_zero(self):
        """That shape carries neither field; the caller's guard handles 0."""
        assert self._pick({"gap_count": 0}) == 0.0

    def test_a_device_that_never_idled_is_not_overridden(self):
        """0 is an answer, and falling past it would contradict the measurement.

        Streams can sum to thousands of milliseconds of gaps while the device
        itself was always busy on one of them. Preferring the device figure and
        then ignoring it when it says zero would be the same bug in reverse.
        """
        assert self._pick({"device_idle_ms": 0, "total_idle_ms": 3752.4}) == 0.0

    def test_a_single_stream_profile_is_unchanged(self):
        """Where they agree, nothing moves -- which is why this went unnoticed."""
        assert self._pick({"device_idle_ms": 935.4, "total_idle_ms": 935.4}) == 935.4


class TestIdlePercentageMatchesItsDuration:
    """The duration and the percentage must describe the same measurement.

    ``pct_of_profile`` is ``total_gap_ns / (span x n_streams)`` — normalised per
    stream, so it belongs with the stream sum. Switching the duration to the
    device figure and leaving the percentage put two unrelated quantities in one
    sentence: "0.0ms of idle time (44.1% of profile)" on a profile where one
    stream idled while another kept the device busy.
    """

    @staticmethod
    def _pair(summary):
        from nsys_ai.skills.builtins.root_cause_matcher import _idle_with_matching_pct

        return _idle_with_matching_pct(summary, [])

    def test_the_device_percentage_is_recomputed_from_the_span(self):
        ms, pct = self._pair({
            "device_idle_ms": 250.0, "total_idle_ms": 900.0, "pct_of_profile": 44.1,
            "profile_start_ns": 0, "profile_end_ns": 1_000_000_000,
        })

        assert (ms, pct) == (250.0, 25.0)

    def test_a_device_that_never_idled_reports_zero_percent(self):
        """Not 0.0ms at 44.1%, which is the shape of the bug."""
        ms, pct = self._pair({
            "device_idle_ms": 0.0, "total_idle_ms": 900.0, "pct_of_profile": 44.1,
            "profile_start_ns": 0, "profile_end_ns": 1_000_000_000,
        })

        assert (ms, pct) == (0.0, 0.0)

    def test_the_stream_figures_are_used_together_as_a_pair(self):
        """When the device sweep did not run, both come from the stream side."""
        ms, pct = self._pair({
            "device_idle_ms": None, "total_idle_ms": 900.0, "pct_of_profile": 44.1,
        })

        assert (ms, pct) == (900.0, 44.1)

    def test_a_missing_span_does_not_invent_a_percentage(self):
        """No span, no share — the duration still stands on its own."""
        ms, pct = self._pair({"device_idle_ms": 250.0, "total_idle_ms": 900.0})

        assert ms == 250.0
        assert pct == 0.0
