"""Profile Health Manifest — one-shot summary for external AI agents.

Returns a compact JSON summary (~500 bytes) that captures the essential
profile characteristics in a single tool call, eliminating the need for
5-8 sequential skill invocations during agent exploration.

Internally orchestrates: overlap_breakdown, nccl_breakdown,
nccl_communicator_analysis, gpu_idle_gaps, root_cause_matcher,
aggregate_nvtx_ranges, and iteration_timing.
"""

import dataclasses
import logging
from datetime import datetime, timezone

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _safe_skill_run(skill_name: str, conn, **kwargs):
    """Run a skill by name, returning [] on any error."""
    import sqlite3

    import duckdb

    from nsys_ai.exceptions import SkillExecutionError

    from ..registry import get_skill

    skill = get_skill(skill_name)
    if skill is None:
        return []
    try:
        return skill.execute(conn, **kwargs)
    except (sqlite3.Error, duckdb.Error, SkillExecutionError) as exc:
        _log.debug("manifest: %s failed: %s", skill_name, exc, exc_info=True)
        return []


# Threshold past which a no-trim manifest is auto-narrowed to a
# representative window. Below this, the full scan is fast enough that
# sub-skills (nvtx_layer_breakdown IEJoin, root_cause_matcher, …) won't
# blow past their soft budgets even on a multi-GPU export.
_AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS = 120 * 10**9   # 120 s
# Target width of the auto-selected window. Picked to be (a) large enough
# to capture at least a few steady-state iterations of any DiT / transformer
# block, (b) small enough that a 15 M-row NVTX IEJoin completes in seconds.
_AUTO_TRIM_TARGET_WINDOW_NS = 20 * 10**9             # 20 s

# Values that turn NSYS_AI_MANIFEST_AUTO_TRIM off. Matches the parsing
# of NSYS_AI_DEFER_NVTX_KERNEL_MAP in parquet_cache.py so users get
# the same on/off semantics across env vars in this repo.
_AUTO_TRIM_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})


def _auto_trim_enabled() -> bool:
    """Read NSYS_AI_MANIFEST_AUTO_TRIM with 0/false/no/off → disabled."""
    import os

    return os.environ.get("NSYS_AI_MANIFEST_AUTO_TRIM", "1").strip().lower() not in _AUTO_TRIM_FALSE_TOKENS


def _auto_select_trim_window(prof) -> tuple[int, int] | None:
    """Pick a representative steady-state window for a long profile.

    Returns ``(start_ns, end_ns)`` to feed into sub-skills as
    ``trim_start_ns`` / ``trim_end_ns``, or ``None`` when no trim is
    needed (profile span below threshold) or no signal is available.

    Strategy:
      1. If profile span ≤ ``_AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS``,
         return None — the full manifest will fit a normal turn budget.
      2. Otherwise, find the top non-`aten::*` NVTX range name with
         ≥ 3 instances (iteration / stage marker) and pick the *middle*
         instance — skips index-0 JIT warmup. If the chosen instance is
         wider than ``_AUTO_TRIM_TARGET_WINDOW_NS``, trim further to the
         middle of that range.
      3. If no qualifying NVTX range exists, fall back to the middle
         ``_AUTO_TRIM_TARGET_WINDOW_NS`` of the profile span.

    Failures are absorbed — auto-trim is best-effort; the caller proceeds
    untrimmed when this returns None or raises.
    """
    import sqlite3

    import duckdb

    try:
        start_ns, end_ns = prof.meta.time_range
    except Exception:
        return None
    # Profiles missing or with a degenerate time_range can't be trimmed.
    if start_ns is None or end_ns is None or end_ns <= start_ns:
        return None
    span_ns = end_ns - start_ns
    if span_ns < _AUTO_TRIM_PROFILE_SPAN_THRESHOLD_NS:
        return None

    # Query the highest-volume non-op-level NVTX range and order its
    # instances by start so we can pick the middle one. The CTE form
    # works on both `nvtx_high` and full `nvtx` (DuckDB cache view) and
    # on raw `NVTX_EVENTS` (SQLite fallback); _duckdb_query translates
    # the [end] dialect when needed.
    #
    # Probe candidates in preference order. The first one that can be
    # opened wins — `nvtx_high` is fastest because aten::* is already
    # filtered, but it only exists on _CACHE_VERSION ≥ 14 caches.
    nvtx_table = "NVTX_EVENTS"  # final fallback; main query catches missing-table
    for candidate in ("nvtx_high", "nvtx", "NVTX_EVENTS"):
        try:
            prof._duckdb_query(f"SELECT 1 FROM {candidate} LIMIT 1")
            nvtx_table = candidate
            break
        except (sqlite3.Error, duckdb.Error):
            continue

    sql = f"""
        WITH ranged AS (
            SELECT text, start, [end]
            FROM {nvtx_table}
            WHERE text IS NOT NULL
              AND [end] > start
              AND text NOT LIKE 'aten::%'
              AND text NOT LIKE 'cudaLaunch%'
              AND text NOT LIKE 'cudaMemcpy%'
        )
        SELECT text, start, [end] FROM ranged
        WHERE text = (
            SELECT text FROM ranged
            GROUP BY text
            HAVING COUNT(*) >= 3
            ORDER BY SUM([end] - start) DESC, text ASC
            LIMIT 1
        )
        ORDER BY start
    """
    try:
        rows = prof._duckdb_query(sql)
    except (sqlite3.Error, duckdb.Error) as exc:
        _log.debug("auto-trim NVTX query failed: %s", exc)
        rows = []

    if rows:
        # Skip the first instance (JIT warmup / one-time setup cost) and
        # pick the middle of what remains so we land in steady state.
        idx = max(1, len(rows) // 2)
        if idx >= len(rows):
            idx = len(rows) - 1
        chosen = rows[idx]
        c_start = int(chosen["start"])
        c_end = int(chosen["end"])
        # Trim very wide stage ranges down to a 20 s slice in their middle
        # so sub-skills still get steady-state behaviour but don't grind.
        if c_end - c_start > _AUTO_TRIM_TARGET_WINDOW_NS:
            mid = (c_start + c_end) // 2
            half = _AUTO_TRIM_TARGET_WINDOW_NS // 2
            return (mid - half, mid + half)
        return (c_start, c_end)

    # No usable NVTX iteration marker — take the middle 20 s of the
    # profile so we at least avoid head-of-trace JIT and tail teardown.
    mid = (start_ns + end_ns) // 2
    half = _AUTO_TRIM_TARGET_WINDOW_NS // 2
    return (mid - half, mid + half)


def _execute(conn, **kwargs):
    """Build a compact profile health manifest."""
    from ...profile import Profile

    device = int(kwargs.get("device", 0))
    overhead_ns = kwargs.get("overhead_ns", 0)

    # Forward trim kwargs if present
    trim_kwargs = {}
    for k in ("trim_start_ns", "trim_end_ns"):
        if kwargs.get(k) is not None:
            trim_kwargs[k] = kwargs[k]
    explicit_trim = bool(trim_kwargs)

    # ── 1. Profile metadata ──────────────────────────────────────
    prof = Profile._from_conn(conn)

    # ── 1a. Auto-trim for very long profiles ────────────────────
    # Without this guard, manifest on a multi-GB / 10-minute profile
    # melts on the NVTX×kernel IEJoin (15M+ rows) and `iteration_timing`
    # full-table scans. Opt out with NSYS_AI_MANIFEST_AUTO_TRIM=0.
    auto_trim_meta: dict | None = None
    if not explicit_trim and _auto_trim_enabled():
        try:
            picked = _auto_select_trim_window(prof)
        except Exception as exc:  # noqa: BLE001 — best-effort, never block manifest
            _log.debug("manifest auto-trim selection failed: %s", exc, exc_info=True)
            picked = None
        if picked is not None:
            t0, t1 = picked
            trim_kwargs = {"trim_start_ns": int(t0), "trim_end_ns": int(t1)}
            # Preserve the original profile span — the standard
            # `profile_span_ms` below will reflect the trim window, so a
            # caller who wants to know "this is a 10-minute profile"
            # would otherwise see 20 s. The pre-trim span lives here.
            full_start, full_end = prof.meta.time_range
            auto_trim_meta = {
                "applied": True,
                "trim_start_ns": int(t0),
                "trim_end_ns": int(t1),
                "window_ms": round((t1 - t0) / 1e6, 1),
                "profile_full_span_ms": round((full_end - full_start) / 1e6, 1),
            }
            _log.info(
                "manifest auto-trimmed to %.2fs–%.2fs (%.1fs window) on long profile",
                t0 / 1e9,
                t1 / 1e9,
                (t1 - t0) / 1e9,
            )
    # Prefer the GPU name for the requested device, if available.
    gpu_name = "unknown"
    gpu_info = getattr(prof.meta, "gpu_info", None)
    if gpu_info is not None:
        device_info = None
        # Support both dict- and list-like gpu_info containers.
        if isinstance(gpu_info, dict):
            device_info = gpu_info.get(device)
        elif isinstance(gpu_info, list) and device < len(gpu_info):
            device_info = gpu_info[device]
        if device_info is not None:
            gpu_name = getattr(device_info, "name", "unknown")

    # Fallback if device info is missing/empty
    if not gpu_name or gpu_name == "unknown":
        from ...profile import get_first_gpu_name

        gpu_name = get_first_gpu_name(conn) or "unknown"
    start_ns, end_ns = prof.meta.time_range
    # Use the trim window (if provided), clamped to the profile range,
    # so the reported span matches the analysis window used by sub-skills.
    effective_start_ns = start_ns
    effective_end_ns = end_ns
    if trim_kwargs.get("trim_start_ns") is not None and trim_kwargs.get("trim_end_ns") is not None:
        effective_start_ns = max(effective_start_ns, trim_kwargs["trim_start_ns"])
        effective_end_ns = min(effective_end_ns, trim_kwargs["trim_end_ns"])
    profile_span_ns = (
        effective_end_ns - effective_start_ns if effective_end_ns > effective_start_ns else 0
    )
    profile_span_ms = round(profile_span_ns / 1e6, 1) if profile_span_ns > 0 else 0

    overhead_ms = round(overhead_ns / 1e6, 1)
    overhead_pct_raw = (overhead_ns / profile_span_ns * 100) if profile_span_ns > 0 else 0
    overhead_pct = round(overhead_pct_raw, 1)
    data_quality = {
        "profiler_overhead_ms": overhead_ms,
        "overhead_pct": overhead_pct,
        "overhead_pct_raw": overhead_pct_raw,
    }
    if auto_trim_meta is not None:
        data_quality["auto_trim"] = auto_trim_meta

    # ── 2. Top kernels (compact: top 5 only) ─────────────────────
    trim_tuple = None
    if trim_kwargs.get("trim_start_ns") is not None and trim_kwargs.get("trim_end_ns") is not None:
        trim_tuple = (trim_kwargs["trim_start_ns"], trim_kwargs["trim_end_ns"])

    try:
        import sqlite3

        import duckdb

        from nsys_ai.exceptions import SkillExecutionError

        # Use aggregate_kernels for correct device filtering
        agg_kernels = prof.aggregate_kernels(device=device, trim=trim_tuple, limit=None)
    except (sqlite3.Error, duckdb.Error, SkillExecutionError) as exc:
        _log.debug("manifest: aggregate_kernels failed: %s", exc, exc_info=True)
        agg_kernels = [{"demangled": f"Error: {exc}", "total_ns": 0, "count": 0}]

    top_kernels = []
    for r in agg_kernels[:5]:
        name = r.get("demangled", "?")
        if len(name) > 60:
            name = name[:57] + "..."
        top_kernels.append(
            {
                "name": name,
                "total_ms": round(r.get("total_ns", 0) / 1e6, 2),
                "count": r.get("count", 0),
            }
        )

    # Compute total kernel time over all kernels
    total_kernel_ms = sum(r.get("total_ns", 0) for r in agg_kernels) / 1e6

    # ── 3. Compute/NCCL overlap ──────────────────────────────────
    overlap_rows = _safe_skill_run("overlap_breakdown", conn, device=device, **trim_kwargs)
    overlap = {}
    if overlap_rows:
        ov = overlap_rows[0]
        if "error" in ov:
            # Preserve error details from overlap_breakdown instead of dropping them.
            # At minimum, surface the primary error message; keep any additional
            # fields that may provide context for callers.
            overlap = dict(ov)
        else:
            overlap = {
                "compute_only_ms": ov.get("compute_only_ms", 0),
                "nccl_only_ms": ov.get("nccl_only_ms", 0),
                "overlap_pct": ov.get("overlap_pct", 0),
                "idle_ms": ov.get("idle_ms", 0),
            }

    # ── 4. NCCL breakdown (compact summary) ──────────────────────
    nccl_rows = _safe_skill_run("nccl_breakdown", conn, device=device, **trim_kwargs)
    nccl_summary = {"streams": 0, "collectives": 0}
    if nccl_rows:
        stream_ids = {r.get("stream_id") for r in nccl_rows}
        nccl_summary["streams"] = len(stream_ids)
        nccl_summary["collectives"] = sum(r.get("count", 1) for r in nccl_rows)
        # Dominant collective by total_ms
        dominant = max(nccl_rows, key=lambda r: r.get("total_ms", 0))
        nccl_summary["dominant_type"] = dominant.get("type", "?")
        nccl_summary["dominant_pct"] = dominant.get("pct", 0)
        nccl_summary["total_nccl_ms"] = round(sum(r.get("total_ms", 0) for r in nccl_rows), 1)

    communicator_rows = _safe_skill_run("nccl_communicator_analysis", conn, device=device, **trim_kwargs)
    communicator_data = [r for r in communicator_rows if not r.get("_diagnostic")]
    communicator_summary = {"communicators": 0, "collective_rows": 0}
    if communicator_data:
        communicator_summary["communicators"] = len(
            {r.get("communicator_hex") for r in communicator_data if r.get("communicator_hex")}
        )
        communicator_summary["collective_rows"] = len(communicator_data)
        dominant_comm = max(communicator_data, key=lambda r: r.get("total_ms", 0))
        communicator_summary["dominant_collective"] = dominant_comm.get("collective_type", "?")
        communicator_summary["dominant_dimension"] = dominant_comm.get(
            "inferred_dimension", "single_rank_or_unknown"
        )
        communicator_summary["top_total_ms"] = round(dominant_comm.get("total_ms", 0), 1)
        communicator_summary["subgroup_count"] = sum(
            1
            for r in communicator_data
            if str(r.get("inferred_dimension", "")).startswith("subgroup_parallelism")
        )
        communicator_summary["low_efficiency_count"] = sum(
            1
            for r in communicator_data
            if r.get("efficiency_pct") is not None and r.get("efficiency_pct", 100) < 20.0
        )

    # ── 5. GPU idle gaps (summary only) ──────────────────────────
    idle_rows = _safe_skill_run("gpu_idle_gaps", conn, device=device, limit=1, **trim_kwargs)
    idle_summary = {"gap_count": 0, "idle_pct": 0}
    summary_row = next((r for r in idle_rows if r.get("_summary")), None)
    if summary_row:
        idle_summary["gap_count"] = summary_row.get("gap_count", 0)
        idle_summary["idle_pct"] = summary_row.get("pct_of_profile", 0)
        idle_summary["total_idle_ms"] = summary_row.get("total_idle_ms", 0)

    # ── 6. Sync Cost Analysis ────────────────────────────────────────
    sync_summary = {}
    try:
        sync_data = _safe_skill_run("sync_cost_analysis", conn, **trim_kwargs)
        if sync_data and "error" not in sync_data[0]:
            sync_summary = sync_data[0]
    except Exception as exc:
        _log.debug("manifest: sync_cost_analysis failed: %s", exc)

    # ── 8. NVTX summary ──────────────────────────────────────────
    nvtx_summary: dict = {"has_nvtx": False}
    try:
        nvtx_ranges = prof.aggregate_nvtx_ranges(limit=5, trim=trim_tuple)
        if nvtx_ranges:
            nvtx_summary["has_nvtx"] = True
            nvtx_summary["top_regions"] = [
                {
                    "name": (r.get("text") or "?")[:50],
                    "total_ms": round(r.get("total_ns", 0) / 1e6, 1),
                    "count": r.get("count", 0),
                }
                for r in nvtx_ranges[:5]
            ]
    except Exception as exc:
        _log.debug("manifest: aggregate_nvtx_ranges failed: %s", exc)

    iter_rows = _safe_skill_run("iteration_timing", conn, device=device, **trim_kwargs)
    if iter_rows:
        nvtx_summary["iteration_count"] = len(iter_rows)
        # Skip iter 0 (warm-up) when computing variance metrics
        steady = iter_rows[1:] if len(iter_rows) > 1 else iter_rows
        if steady:
            durs = sorted(r.get("duration_ms", 0) for r in steady)
            mid = len(durs) // 2
            nvtx_summary["median_iter_ms"] = round(durs[mid], 1)
            nvtx_summary["slowest_iter_ms"] = round(max(durs), 1)

    # ── 7. Root cause findings (count + top severity) ────────────
    # Pass precomputed communicator rows to avoid re-running the expensive
    # nccl_communicator_analysis inside root_cause_matcher.
    rc_rows = _safe_skill_run(
        "root_cause_matcher", conn, device=device,
        communicator_data=communicator_rows, **trim_kwargs,
    )
    root_causes = []
    for r in rc_rows:
        pattern = r.get("pattern", "")
        if pattern == "No Known Anti-Patterns Detected":
            continue
        root_causes.append(
            {
                "pattern": pattern,
                "severity": r.get("severity", "info"),
            }
        )

    sev_rank = {"critical": 0, "high": 1, "warning": 2, "medium": 3, "low": 4, "info": 5}
    root_causes.sort(key=lambda x: sev_rank.get(x["severity"].lower(), 99))

    # ── Assemble manifest ────────────────────────────────────────
    manifest = {
        "analysis_time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gpu": gpu_name,
        "fingerprint": dataclasses.asdict(prof.fingerprint) if prof.fingerprint else None,
        "profile_span_ms": profile_span_ms,
        "top_kernels": top_kernels,
        "total_kernel_ms": round(total_kernel_ms, 1),
        "overlap": overlap,
        "nccl": nccl_summary,
        "communicators": communicator_summary,
        "sync": sync_summary,
        "idle": idle_summary,
        "root_cause_count": len(root_causes),
        "root_causes": root_causes[:5],  # Cap at 5 to keep output compact
        "nvtx": nvtx_summary,
        "data_quality": data_quality,
    }

    # Infer suspected bottleneck
    bottleneck = _infer_bottleneck(manifest)
    if bottleneck:
        manifest["suspected_bottleneck"] = bottleneck

    return [manifest]


def _infer_bottleneck(m: dict) -> str:
    """Heuristic bottleneck inference from manifest data."""
    sync = m.get("sync", {})
    sync_density = sync.get("sync_density_pct", 0)
    # The new threshold
    if sync_density > 20.0:
        return f"High CPU Synchronization Blocking ({sync_density:.1f}% of span)"

    dq = m.get("data_quality", {})
    overhead_pct_val = dq.get("overhead_pct_raw", dq.get("overhead_pct", 0))
    if overhead_pct_val > 1.0:
        return f"Profiler Overhead ({dq.get('overhead_pct', overhead_pct_val)}%) contaminated the profile"

    overlap = m.get("overlap", {})
    idle = m.get("idle", {})
    nccl = m.get("nccl", {})

    # Check NCCL serialization first (most impactful)
    if overlap.get("overlap_pct", 100) < 30 and overlap.get("nccl_only_ms", 0) > 0:
        return "NCCL serialization (overlap < 30%)"

    # Check idle gaps
    if idle.get("idle_pct", 0) > 15:
        return f"GPU idle bubbles ({idle['idle_pct']}% of profile)"

    # Check if one kernel dominates
    top_k = m.get("top_kernels", [])
    total = m.get("total_kernel_ms", 0)
    if top_k and total > 0:
        top_pct = top_k[0]["total_ms"] / total * 100
        if top_pct > 60:
            return f"Kernel hotspot: {top_k[0]['name']} ({top_pct:.0f}%)"

    # Check NCCL dominance
    if nccl.get("total_nccl_ms", 0) > overlap.get("compute_only_ms", float("inf")):
        return "Communication-bound (NCCL > compute)"

    # Check iteration variance (spike pattern)
    nvtx = m.get("nvtx", {})
    median_ms = nvtx.get("median_iter_ms", 0)
    slowest_ms = nvtx.get("slowest_iter_ms", 0)
    if median_ms > 0 and slowest_ms > median_ms * 1.5:
        ratio = slowest_ms / median_ms
        return f"Iteration variance spike (slowest {slowest_ms:.0f}ms = {ratio:.1f}× median)"

    return ""


def _format(rows):
    if not rows:
        return "(No manifest data)"
    m = rows[0]
    lines = ["══ Profile Health Manifest ══"]

    fp = m.get("fingerprint")
    if fp:
        dist_str = "Distributed: yes" if fp.get("distributed") else "Distributed: no"
        mn_str = "Multi-node: yes" if fp.get("multi_node") else "Multi-node: no"
        lines.append(f"  Framework:    {fp.get('framework', 'Unknown')} ({dist_str}, {mn_str})")

    lines.append(f"  GPU:          {m.get('gpu', '?')}")
    lines.append(f"  Profile span: {m.get('profile_span_ms', 0):.1f}ms")

    dq = m.get("data_quality", {})
    overhead_pct_raw = dq.get("overhead_pct_raw", dq.get("overhead_pct", 0))
    if overhead_pct_raw >= 0.1:
        lines.append(
            f"  ⚠️ Profiler Overhead: {dq.get('profiler_overhead_ms', 0):.1f}ms ({dq.get('overhead_pct', overhead_pct_raw)}% of span)"
        )

    # Top kernels
    lines.append("")
    lines.append("  Top Kernels:")
    for k in m.get("top_kernels", []):
        lines.append(f"    {k['name'][:50]:<52s}  {k['total_ms']:>8.1f}ms  ×{k['count']}")

    # Overlap
    ov = m.get("overlap", {})
    if ov:
        lines.append("")
        lines.append("  Compute/NCCL Overlap:")
        err = ov.get("error")
        if err:
            # Preserve and surface overlap computation errors instead of showing 0.0ms metrics.
            if isinstance(err, dict):
                msg = err.get("message") or str(err)
            else:
                msg = str(err)
            lines.append(f"    ERROR: {msg}")
        else:
            lines.append(f"    Compute only: {ov.get('compute_only_ms', 0):.1f}ms")
            lines.append(f"    NCCL only:    {ov.get('nccl_only_ms', 0):.1f}ms")
            lines.append(f"    Overlap:      {ov.get('overlap_pct', 0)}%")

    # NCCL
    nccl = m.get("nccl", {})
    if nccl.get("streams", 0) > 0:
        lines.append("")
        lines.append("  NCCL Summary:")
        lines.append(f"    Streams: {nccl['streams']}, Collectives: {nccl['collectives']}")
        lines.append(
            f"    Dominant: {nccl.get('dominant_type', '?')} ({nccl.get('dominant_pct', 0)}%)"
        )
        lines.append(f"    Total: {nccl.get('total_nccl_ms', 0):.1f}ms")

    comm = m.get("communicators", {})
    if comm.get("communicators", 0) > 0:
        lines.append("")
        lines.append("  NCCL Communicators:")
        lines.append(
            f"    Communicators: {comm.get('communicators', 0)}, grouped rows: {comm.get('collective_rows', 0)}"
        )
        lines.append(
            f"    Dominant: {comm.get('dominant_collective', '?')} / {comm.get('dominant_dimension', '?')}"
        )
        if comm.get("low_efficiency_count", 0):
            lines.append(f"    Low efficiency groups: {comm.get('low_efficiency_count', 0)}")

    # Idle
    idle = m.get("idle", {})
    sync = m.get("sync", {})
    if idle.get("gap_count", 0) > 0:
        lines.append("")
        lines.append(f"  GPU Idle: {idle['gap_count']} gaps, {idle.get('idle_pct', 0)}% of profile")
    if sync.get("total_sync_wall_ms"):
        lines.append("")
        lines.append(
            f"  CPU Sync Block: {sync.get('total_sync_wall_ms', 0):.1f}ms ({sync.get('sync_density_pct', 0)}% of profile)"
        )

    # Root causes
    rcs = m.get("root_causes", [])
    if rcs:
        lines.append("")
        lines.append(f"  Root Causes ({m.get('root_cause_count', 0)} findings):")
        for rc in rcs:
            sev_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(rc["severity"], "⚪")
            lines.append(f"    {sev_icon} {rc['pattern']}")

    # NVTX summary
    nvtx = m.get("nvtx", {})
    if nvtx.get("has_nvtx"):
        lines.append("")
        iter_count = nvtx.get("iteration_count")
        if iter_count:
            median_ms = nvtx.get("median_iter_ms", 0)
            slowest_ms = nvtx.get("slowest_iter_ms", 0)
            lines.append(f"  NVTX Iterations: {iter_count} detected, median {median_ms:.1f}ms, slowest {slowest_ms:.1f}ms")
        top = nvtx.get("top_regions", [])
        if top:
            lines.append("  Top NVTX Regions:")
            for r in top:
                lines.append(f"    {r['name'][:48]:<50s}  {r['total_ms']:>8.1f}ms  ×{r['count']}")

    # Bottleneck
    bn = m.get("suspected_bottleneck", "")
    if bn:
        lines.append("")
        lines.append(f"  ⚡ Suspected bottleneck: {bn}")

    return "\n".join(lines)


SKILL = Skill(
    name="profile_health_manifest",
    title="Profile Health Manifest",
    description=(
        "One-shot profile health summary for AI agents. Returns a compact JSON manifest "
        "covering GPU info, top kernels, compute/NCCL overlap, NCCL summary, "
        "communicator-aware NCCL hints, idle gaps, root cause findings, and NVTX summary "
        "(top regions + iteration count/median/slowest) — all in a single call. "
        "If Profiler Overhead is >1%, advise the user to use torch.cuda.profiler.start/stop() "
        "and --capture-range=cudaProfilerApi instead of full-script profiling. "
        "Use this as the FIRST skill to call on any new profile. "
        "The nvtx.iteration_count, nvtx.median_iter_ms, and nvtx.slowest_iter_ms fields "
        "let you skip the first iteration_timing call for Mode 5 and Mode 9."
    ),
    category="utility",
    execute_fn=_execute,
    format_fn=_format,
    params=[
        SkillParam("device", "GPU device ID", "int", False, 0),
    ],
    tags=["manifest", "summary", "health", "overview", "agent", "triage"],
)
