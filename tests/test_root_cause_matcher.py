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
        from nsys_ai.skills.builtins.root_cause_matcher import _first_measured

        return _first_measured(
            summary.get("device_idle_ms"), summary.get("total_idle_ms")
        )

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
