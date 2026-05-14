"""Compute/Communication overlap breakdown.

Uses overlap_analysis() from overlap.py to quantify how much GPU compute
overlaps with NCCL communication.  This is a Python-level skill (execute_fn)
rather than a SQL skill because it needs interval merging and intersection
logic that can't be expressed in a single SQL query.
"""

import logging

from nsys_ai.connection import DB_ERRORS

from ..base import Skill, SkillParam

_log = logging.getLogger(__name__)


def _execute(conn, **kwargs):
    from ...overlap import overlap_analysis
    from ...profile import Profile

    prof = Profile._from_conn(conn)
    device = int(kwargs.get("device", 0))
    # Support --trim passthrough from agent analyze
    trim = None
    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    if trim_start is not None and trim_end is not None:
        trim = (int(trim_start), int(trim_end))
    result = overlap_analysis(prof, device, trim=trim)
    if "error" in result:
        return [result]

    # Same-stream diagnosis for evidence builder overlap check
    same_stream = []
    try:
        kernel_tbl = prof.schema.kernel_table
        if kernel_tbl:
            trim_clause = ""
            params = [device]
            if trim is not None:
                trim_clause = " AND k.[end] >= ? AND k.start <= ? "
                params.extend([trim[0], trim[1]])
            same_stream = prof._duckdb_query(
                f"""
                SELECT k.streamId,
                    SUM(CASE WHEN s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'
                        THEN 1 ELSE 0 END) AS nccl_count,
                    SUM(CASE WHEN NOT (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
                        THEN 1 ELSE 0 END) AS compute_count
                FROM {kernel_tbl} k
                JOIN StringIds s ON k.shortName = s.id
                WHERE k.deviceId = ? {trim_clause}
                GROUP BY k.streamId
                HAVING nccl_count > 0 AND compute_count > 0
                """,
                params,
            )
            if same_stream:
                result["same_stream_diagnosis"] = [str(r["streamId"]) for r in same_stream]
                # Quantify the diagnosis: a near-clean multi-stream layout
                # with a few stray cross-stream kernels reads identically
                # to 100 % collision without these percentages. Trigger
                # above is unchanged; the proportions are purely additive.
                totals = prof._duckdb_query(
                    f"""
                    SELECT
                        SUM(CASE WHEN s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%'
                            THEN 1 ELSE 0 END) AS nccl_total,
                        SUM(CASE WHEN NOT (s.value LIKE '%nccl%' OR s.value LIKE '%NCCL%')
                            THEN 1 ELSE 0 END) AS compute_total
                    FROM {kernel_tbl} k
                    JOIN StringIds s ON k.shortName = s.id
                    WHERE k.deviceId = ? {trim_clause}
                    """,
                    params,
                )
                if totals:
                    nccl_total = totals[0]["nccl_total"] or 0
                    compute_total = totals[0]["compute_total"] or 0
                    nccl_same = sum((r["nccl_count"] or 0) for r in same_stream)
                    compute_same = sum((r["compute_count"] or 0) for r in same_stream)
                    if compute_total > 0:
                        result["same_stream_compute_pct"] = round(
                            100 * compute_same / compute_total, 1
                        )
                    if nccl_total > 0:
                        result["same_stream_nccl_pct"] = round(
                            100 * nccl_same / nccl_total, 1
                        )
    except DB_ERRORS:
        _log.debug("Failed to enrich stream compute/nccl overlap", exc_info=True)

    # Surface other GPUs that exist in the profile. Otherwise a multi-rank
    # single-node profile is invisible to the agent (which only sees
    # device=0 by default).
    try:
        kernel_tbl = prof.schema.kernel_table
        if kernel_tbl:
            devs = prof._duckdb_query(
                f"SELECT DISTINCT deviceId FROM {kernel_tbl} ORDER BY deviceId"
            )
            if devs:
                result["present_devices"] = [
                    int(r["deviceId"]) for r in devs if r["deviceId"] is not None
                ]
    except DB_ERRORS:
        _log.debug("Failed to enumerate present devices", exc_info=True)

    try:
        from ...skills.registry import get_skill

        sync_skill = get_skill("sync_cost_analysis")
        if sync_skill:
            sync_data = sync_skill.execute(conn, **kwargs)
            if sync_data and "error" not in sync_data[0]:
                result["sync_ms"] = sync_data[0].get("total_sync_wall_ms", 0)
    except Exception:
        _log.debug("Failed to enrich sync cost", exc_info=True)

    result["device_id"] = device
    return [result]


def _format(rows):
    if not rows:
        return "(No overlap data available)"
    r = rows[0]
    if "error" in r:
        return f"(Overlap analysis: {r['error']})"
    lines = [
        "── Compute/Communication Overlap ──",
        f"  Total span:    {r['total_ms']:.1f}ms",
        f"  Compute only:  {r['compute_only_ms']:.1f}ms",
        f"  NCCL only:     {r['nccl_only_ms']:.1f}ms",
        f"  Overlap:       {r['overlap_ms']:.1f}ms ({r['overlap_pct']}% of NCCL overlapped)",
        f"  Idle:          {r['idle_ms']:.1f}ms",
    ]
    if r.get("sync_ms"):
        lines.append(f"  ⚡ CPU Sync:     {r['sync_ms']:.1f}ms (global host-side blocking)")
    lines.append(f"  Kernels:       {r['compute_kernels']} compute + {r['nccl_kernels']} NCCL")

    # Same-stream serialization warning — surfaced in textual output so it's
    # visible without parsing JSON. Proportions added when available.
    streams = r.get("same_stream_diagnosis")
    if streams:
        compute_pct = r.get("same_stream_compute_pct")
        nccl_pct = r.get("same_stream_nccl_pct")
        suffix = ""
        if compute_pct is not None and nccl_pct is not None:
            suffix = f" — {compute_pct}% of compute + {nccl_pct}% of NCCL"
        lines.append(
            f"  ⚠️ Same-stream:  stream(s) [{', '.join(streams)}] carry both"
            f"{suffix} → overlap blocked"
        )

    # Multi-device hint — textual users miss the device list otherwise.
    devs = r.get("present_devices") or []
    if len(devs) > 1:
        analyzed = r.get("device_id", 0)
        others = [str(d) for d in devs if d != analyzed]
        lines.append(
            f"  Devices:       {analyzed} (analyzed); profile also has "
            f"[{', '.join(others)}] — re-run with -p device=N for each"
        )
    return "\n".join(lines)


def _to_findings(rows: list[dict]) -> list:
    from nsys_ai.annotation import Finding

    findings = []
    if not rows or "error" in rows[0]:
        return findings

    r = rows[0]
    nccl_ms = r.get("nccl_only_ms", 0) + r.get("overlap_ms", 0)
    compute_ms = r.get("compute_only_ms", 0)
    overlap_pct = r.get("overlap_pct", 0)
    total_ms = r.get("total_ms", 1)
    start_ns = r.get("span_start_ns", 0)
    end_ns = r.get("span_end_ns", 0)
    device = r.get("device_id", 0)

    # Low overlap: NCCL not well hidden behind compute
    if nccl_ms > 0 and overlap_pct < 30:
        note = (
            f"Only {overlap_pct}% of NCCL time overlaps with compute. "
            f"NCCL-only: {r.get('nccl_only_ms', 0):.1f}ms out of "
            f"{total_ms:.1f}ms total span."
        )
        streams = r.get("same_stream_diagnosis")
        if streams:
            compute_pct = r.get("same_stream_compute_pct")
            nccl_pct = r.get("same_stream_nccl_pct")
            note += (
                f" Streams [{', '.join(streams)}] run both NCCL and "
                f"compute — overlap is impossible on same stream."
            )
            if compute_pct is not None and nccl_pct is not None:
                note += (
                    f" ({compute_pct}% of compute and {nccl_pct}% of NCCL "
                    f"share these streams.)"
                )

        findings.append(
            Finding(
                type="region",
                label=f"Low Compute/NCCL Overlap ({overlap_pct}%)",
                start_ns=start_ns,
                end_ns=end_ns,
                gpu_id=device,
                severity="warning",
                note=note,
            )
        )

    # Communication dominated: NCCL > compute
    if nccl_ms > 0 and compute_ms > 0:
        ratio = compute_ms / nccl_ms
        if ratio < 0.5:
            findings.append(
                Finding(
                    type="region",
                    label=f"Communication Dominated (ratio={ratio:.2f})",
                    start_ns=start_ns,
                    end_ns=end_ns,
                    gpu_id=device,
                    severity="critical",
                    note=(
                        f"Compute/Communication ratio is {ratio:.2f} "
                        f"(healthy > 2.0). Compute: {compute_ms:.1f}ms, "
                        f"NCCL: {nccl_ms:.1f}ms."
                    ),
                )
            )

    return findings


SKILL = Skill(
    name="overlap_breakdown",
    title="Compute/Communication Overlap Breakdown",
    description=(
        "Quantifies how much GPU compute overlaps with NCCL communication. "
        "Shows compute-only, NCCL-only, overlap, and idle time. "
        "overlap_pct > 60% means NCCL is well-hidden behind compute."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    to_findings_fn=_to_findings,
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    tags=["overlap", "nccl", "compute", "communication", "distributed"],
)
