"""Tests for profile_health_manifest skill and --max-rows truncation.

Uses the minimal_nsys_conn and duckdb_conn fixtures from conftest.py.
"""

import pytest

from nsys_ai.skills.registry import get_skill

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def manifest_skill():
    skill = get_skill("profile_health_manifest")
    assert skill is not None, "profile_health_manifest skill not registered"
    return skill


# ── Manifest skill tests ─────────────────────────────────────────


class TestManifestExecute:
    def test_returns_single_row(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1

    def test_has_required_keys(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        required_keys = [
            "gpu",
            "profile_span_ms",
            "top_kernels",
            "total_kernel_ms",
            "overlap",
            "nccl",
            "communicators",
            "idle",
            "root_cause_count",
            "root_causes",
            "data_quality",
        ]
        for key in required_keys:
            assert key in m, f"Missing key '{key}' in manifest"

    def test_top_kernels_is_list(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert isinstance(m["top_kernels"], list)

    def test_nccl_summary_has_streams(self, minimal_nsys_conn, manifest_skill):
        """Our test data has NCCL kernels, so streams > 0."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert m["nccl"]["streams"] > 0

    def test_communicator_summary_defaults_without_payloads(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert m["communicators"]["communicators"] == 0
        assert m["communicators"]["collective_rows"] == 0

    def test_root_causes_capped(self, minimal_nsys_conn, manifest_skill):
        """root_causes list should never exceed 5 entries."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        m = rows[0]
        assert len(m["root_causes"]) <= 5

    def test_empty_device_returns_manifest(self, minimal_nsys_conn, manifest_skill):
        """Device with no data should still return a valid manifest."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=99)
        assert isinstance(rows, list)
        assert len(rows) == 1
        m = rows[0]
        assert m["nccl"]["streams"] == 0

    def test_data_quality_metrics(self, minimal_nsys_conn, manifest_skill):
        """Ensure data_quality metrics are properly computed."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]

        assert "profiler_overhead_ms" in dq
        assert "overhead_pct" in dq

        assert isinstance(dq["profiler_overhead_ms"], (int, float))
        assert isinstance(dq["overhead_pct"], (int, float))


class TestManifestAutoTrim:
    """Auto-trim picks a representative window on long profiles so the
    manifest doesn't grind through 10-minute / 2 GB exports.

    Boundary: profiles shorter than _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS
    pass through untouched; longer profiles get the middle steady-state
    instance of the top non-aten:: NVTX range, or the middle 20 s as a
    fallback when no NVTX iteration markers are present.
    """

    def _make_profile_stub(self, span_ns: int, nvtx_rows: list[tuple]):
        """Lightweight Profile + connection pair sufficient for auto-trim."""
        from types import SimpleNamespace

        # The auto-trim selector only reaches for `prof.meta.time_range`
        # and `prof._duckdb_query`. Stubbing both decouples the test from
        # the rest of the Profile machinery.
        captured_queries: list[str] = []

        def fake_query(sql, params=None):
            captured_queries.append(sql)
            sl = sql.lower()
            # Probe queries — let the selector think nvtx exists.
            if "from nvtx_high" in sl and "limit 1" in sl and "where" not in sl.split("from nvtx_high")[1].split("limit")[0]:
                return [{"1": 1}]
            # The real CTE query — return the NVTX rows seeded by the test
            # if they match the expected schema (text, start, end).
            if "from ranged" in sl:
                # nvtx_rows is the post-filter set (no aten::*).
                return [{"text": r[0], "start": r[1], "end": r[2]} for r in nvtx_rows]
            return []

        prof = SimpleNamespace(
            meta=SimpleNamespace(time_range=(0, span_ns)),
            _duckdb_query=fake_query,
        )
        return prof, captured_queries

    def test_short_profile_returns_none(self):
        """Profile span below threshold → no auto-trim."""
        from nsys_ai.skills.builtins.profile_health_manifest import (
            _auto_select_trim_window,
        )

        prof, _ = self._make_profile_stub(span_ns=30 * 10**9, nvtx_rows=[])
        assert _auto_select_trim_window(prof, conn=None) is None

    def test_long_profile_picks_middle_nvtx_instance(self):
        """3 stage instances → middle one (skip JIT warmup at idx 0)."""
        from nsys_ai.skills.builtins.profile_health_manifest import (
            _auto_select_trim_window,
        )

        # Three short ranges (each < 20 s) so the selector returns them
        # verbatim rather than slicing to a 20 s sub-window.
        nvtx_rows = [
            # (text, start_ns, end_ns)
            ("stage::DenoisingStage", 100_000_000_000, 110_000_000_000),
            ("stage::DenoisingStage", 200_000_000_000, 210_000_000_000),
            ("stage::DenoisingStage", 300_000_000_000, 310_000_000_000),
        ]
        prof, _ = self._make_profile_stub(span_ns=500 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof, conn=None)
        assert picked is not None
        # Index 1 (the middle of 3, skipping idx 0)
        assert picked == (200_000_000_000, 210_000_000_000)

    def test_long_profile_trims_wide_range_to_middle_20s(self):
        """Range wider than 20 s gets narrowed to its middle 20 s."""
        from nsys_ai.skills.builtins.profile_health_manifest import (
            _AUTO_TRIM_TARGET_WINDOW_NS,
            _auto_select_trim_window,
        )

        # One 200 s range × 3 instances; middle instance spans
        # 400 s → 600 s. Auto-trim should return middle ± 10 s = 490-510 s.
        nvtx_rows = [
            ("stage::DenoisingStage", 100_000_000_000, 300_000_000_000),
            ("stage::DenoisingStage", 400_000_000_000, 600_000_000_000),
            ("stage::DenoisingStage", 700_000_000_000, 900_000_000_000),
        ]
        prof, _ = self._make_profile_stub(span_ns=1000 * 10**9, nvtx_rows=nvtx_rows)
        picked = _auto_select_trim_window(prof, conn=None)
        assert picked is not None
        lo, hi = picked
        assert hi - lo == _AUTO_TRIM_TARGET_WINDOW_NS
        # Centered on the middle of (400e9, 600e9) = 500e9
        assert lo == 500_000_000_000 - _AUTO_TRIM_TARGET_WINDOW_NS // 2
        assert hi == 500_000_000_000 + _AUTO_TRIM_TARGET_WINDOW_NS // 2

    def test_long_profile_no_nvtx_falls_back_to_middle(self):
        """No qualifying NVTX iteration markers → middle 20 s of span."""
        from nsys_ai.skills.builtins.profile_health_manifest import (
            _AUTO_TRIM_TARGET_WINDOW_NS,
            _auto_select_trim_window,
        )

        prof, _ = self._make_profile_stub(span_ns=400 * 10**9, nvtx_rows=[])
        picked = _auto_select_trim_window(prof, conn=None)
        assert picked is not None
        lo, hi = picked
        assert hi - lo == _AUTO_TRIM_TARGET_WINDOW_NS
        # Span midpoint is 200 s; window is [190 s, 210 s]
        assert lo == 200 * 10**9 - _AUTO_TRIM_TARGET_WINDOW_NS // 2
        assert hi == 200 * 10**9 + _AUTO_TRIM_TARGET_WINDOW_NS // 2

    def test_explicit_trim_disables_auto_trim(self, minimal_nsys_conn, manifest_skill):
        """When the caller passes trim_start_ns / trim_end_ns, the manifest
        must honour it and not inject an auto-trim window."""
        rows = manifest_skill.execute(
            minimal_nsys_conn,
            device=0,
            trim_start_ns=0,
            trim_end_ns=10_000_000,
        )
        dq = rows[0]["data_quality"]
        assert "auto_trim" not in dq

    def test_short_profile_no_auto_trim_in_manifest(self, minimal_nsys_conn, manifest_skill):
        """Fixture profile spans ~24 ms — well below threshold; no auto-trim."""
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]
        assert "auto_trim" not in dq

    def test_env_var_disables_auto_trim(self, monkeypatch, minimal_nsys_conn, manifest_skill):
        """NSYS_AI_MANIFEST_AUTO_TRIM=0 opts out even on long profiles."""
        monkeypatch.setenv("NSYS_AI_MANIFEST_AUTO_TRIM", "0")
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        dq = rows[0]["data_quality"]
        assert "auto_trim" not in dq


class TestManifestFormat:
    def test_format_includes_header(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "Profile Health Manifest" in text

    def test_format_includes_gpu(self, minimal_nsys_conn, manifest_skill):
        rows = manifest_skill.execute(minimal_nsys_conn, device=0)
        text = manifest_skill.format_rows(rows)
        assert "GPU:" in text


class TestManifestDuckDB:
    def test_duckdb_path(self, duckdb_conn, manifest_skill):
        rows = manifest_skill.execute(duckdb_conn, device=0)
        assert isinstance(rows, list)
        assert len(rows) == 1
        assert "gpu" in rows[0]


# ── Token budget protection (--max-rows) ─────────────────────────


class TestMaxRowsTruncation:
    def test_truncation_applied(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(10)]
        max_rows = 3
        truncated = _apply_max_rows_truncation(rows, max_rows)
        assert len(truncated) == max_rows + 1
        assert truncated[-1]["_truncated"] is True
        assert truncated[-1]["_total_rows"] == 10
        assert truncated[-1]["_shown_rows"] == 3

    def test_no_truncation_when_under_limit(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        truncated = _apply_max_rows_truncation(rows, 100)
        assert len(truncated) == 3
        assert not any(r.get("_truncated") for r in truncated)

    def test_negative_max_rows_raises(self):
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"id": i} for i in range(3)]
        with pytest.raises(ValueError, match="non-negative integer"):
            _apply_max_rows_truncation(rows, -1)

    def test_error_payload_not_truncated_when_max_rows_zero(self):
        """Error payloads (e.g., [{'error': ...}]) should not be dropped by max-rows."""
        from nsys_ai.cli.handlers import _apply_max_rows_truncation

        rows = [{"error": "Something went wrong"}]
        truncated = _apply_max_rows_truncation(rows, 0)

        # The single error row should be preserved and not replaced by a truncation marker.
        assert isinstance(truncated, list)
        assert len(truncated) == 1
        assert "error" in truncated[0]
        assert truncated[0]["error"] == "Something went wrong"
        # Error payloads should not be marked as truncated.
        assert not truncated[0].get("_truncated")
