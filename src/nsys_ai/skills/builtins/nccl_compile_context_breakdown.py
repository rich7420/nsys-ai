"""Classify NCCL kernels by their leaf NVTX label (call mode).

Answers a single question: of the NCCL kernels in this profile, what
fraction were called eagerly (user code reaching c10d / NCCL directly),
captured inside an inductor-compiled graph, or running under some
unrelated NVTX scope?

The fix recommendation depends on the dominant bucket:

  - eager-dominant         → caller-side fix: wrap collectives with
                              ``async_op=True`` or move them to a
                              dedicated stream
  - inductor-dominant      → ``torch._inductor.config`` knobs:
                              functional collectives,
                              ``reorder_for_compute_comm_overlap``
  - temporal_only-dominant → reach for ``nccl_payload_breakdown`` and
                              ``overlap_breakdown`` to find the real
                              driver; the leaf NVTX is uninformative

IMPORTANT (the lesson from §4 B audit gap #14): classification is by
**leaf NVTX label**, not ancestor-path match. NVTX path containment
is *temporal* — any scope that happened to still be open when the
kernel launched. A filter like ``nvtx_path LIKE '%Torch-Compiled%'``
would falsely tag eager calls whose enclosing compile-region NVTX
hadn't closed yet at launch time. A real fastvideo audit shipped the
wrong fix recommendation on exactly that mistake (96% of NCCL kernels
matched the path filter, but only 5.4% were actually inductor-captured
by their leaf label).
"""

from ..base import Skill

# Inductor-captured leaf marker — observed in PyTorch ≥ 2.1 compiled graphs.
# When PyTorch lowers a collective through Dynamo + Inductor, the launching
# NVTX scope at kernel time is the compiled fx-graph call frame.
_INDUCTOR_LEAF_MARKERS = ("## Call CompiledFxGraph",)

# Eager-call leaf prefixes. ``c10d::`` is the public C++ binding namespace
# (e.g. ``c10d::all_reduce_``), ``nccl`` covers raw NCCL wrappers and the
# functional collectives runtime calls (e.g. ``nccl:all_reduce``).
_EAGER_LEAF_PREFIXES = ("c10d::", "nccl")


def _classify_leaf(leaf: str) -> str:
    """Map a leaf NVTX label to one of the three call-mode buckets."""
    if not leaf:
        return "temporal_only"
    if any(leaf.startswith(p) for p in _EAGER_LEAF_PREFIXES):
        return "eager"
    if any(m in leaf for m in _INDUCTOR_LEAF_MARKERS):
        return "inductor_captured"
    return "temporal_only"


def _is_nccl_kernel(name: str) -> bool:
    """Match NCCL kernels by lowercase substring — covers `ncclKernel_*`,
    `ncclDevKernel_*`, and `nccl_AllReduce_*` variants across NCCL versions."""
    return bool(name) and "nccl" in name.lower()


def _execute(conn, **kwargs):
    from ...nvtx_attribution import attribute_kernels_to_nvtx

    trim_start = kwargs.get("trim_start_ns")
    trim_end = kwargs.get("trim_end_ns")
    trim = (
        (int(trim_start), int(trim_end))
        if trim_start is not None and trim_end is not None
        else None
    )
    sqlite_path = kwargs.get("_sqlite_path")

    rows = attribute_kernels_to_nvtx(conn, sqlite_path=sqlite_path, trim=trim, limit=None)

    buckets: dict[str, dict[str, int]] = {
        "eager": {"count": 0, "ns": 0},
        "inductor_captured": {"count": 0, "ns": 0},
        "temporal_only": {"count": 0, "ns": 0},
    }
    for r in rows:
        if not _is_nccl_kernel(r.get("kernel_name", "")):
            continue
        bucket = _classify_leaf(r.get("nvtx_text", "") or "")
        buckets[bucket]["count"] += 1
        buckets[bucket]["ns"] += int(r.get("k_dur_ns") or 0)

    total_count = sum(b["count"] for b in buckets.values())
    total_ns = sum(b["ns"] for b in buckets.values())

    if total_count == 0:
        return [{
            "error": (
                "No NCCL kernels found in this profile (filter: kernel name "
                "contains 'nccl'). Single-GPU profiles or captures without "
                "NCCL tracing will land here."
            ),
        }]

    return [
        {
            "_summary": True,
            "total_nccl_kernels": total_count,
            "total_nccl_ms": round(total_ns / 1e6, 3),
        },
        *[
            {
                "bucket": name,
                "count": b["count"],
                "ms": round(b["ns"] / 1e6, 3),
                "pct": round(b["count"] / total_count * 100, 1),
                "ms_pct": round(b["ns"] / total_ns * 100, 1) if total_ns > 0 else 0.0,
            }
            for name, b in buckets.items()
        ],
    ]


def _format(rows):
    if not rows or "error" in rows[0]:
        return f"(NCCL compile context: {rows[0].get('error', 'no data') if rows else 'no data'})"
    s = rows[0]
    lines = [
        "── NCCL Call-Mode Breakdown (by leaf NVTX) ──",
        f"  Total NCCL kernels: {s['total_nccl_kernels']:,}  ({s['total_nccl_ms']:.3f} ms)",
        "",
        f"  {'bucket':<20}  {'count':>8}  {'ms':>12}  {'count_pct':>10}  {'ms_pct':>10}",
        "  " + "─" * 66,
    ]
    for r in rows[1:]:
        lines.append(
            f"  {r['bucket']:<20}  {r['count']:>8}  {r['ms']:>12.3f}  "
            f"{r['pct']:>9.1f}%  {r['ms_pct']:>9.1f}%"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="nccl_compile_context_breakdown",
    title="NCCL Call-Mode Breakdown (eager vs inductor-captured vs temporal-only)",
    description=(
        "Classifies NCCL kernels by the leaf NVTX label open at launch time: "
        "eager (c10d::* / nccl* leaf), inductor_captured (## Call CompiledFxGraph "
        "leaf), or temporal_only (anything else). Decides whether a collective "
        "perf fix lives in user code (stream wrap, async_op) or in inductor "
        "config (functional collectives, reorder_for_compute_comm_overlap). "
        "Classifies by LEAF label, not ancestor-path containment — see module "
        "docstring for why."
    ),
    category="communication",
    execute_fn=_execute,
    format_fn=_format,
    tags=[
        "nccl", "communication", "distributed", "torch-compile", "inductor",
        "call-mode", "eager", "nvtx",
    ],
)
