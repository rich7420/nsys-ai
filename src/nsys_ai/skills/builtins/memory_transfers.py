"""Memory transfer summary — H2D, D2H, D2D, P2P breakdown."""

from ..base import Skill, SkillParam, is_abstention

_COPY_KINDS = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}


def _format(rows):
    if not rows:
        return "(No memory transfers found)"
    lines = [
        "── Memory Transfers Summary ──",
        f"{'Direction':<10s}  {'Count':>7s}  {'Total(MB)':>10s}  {'Total(ms)':>10s}",
        "─" * 44,
    ]
    for r in rows:
        direction = _COPY_KINDS.get(r["copyKind"], f"kind={r['copyKind']}")
        lines.append(
            f"{direction:<10s}  {r['count']:>7d}  {r['total_mb']:>10.2f}  {r['total_ms']:>10.2f}"
        )
    return "\n".join(lines)


SKILL = Skill(
    name="memory_transfers",
    title="Memory Transfer Summary",
    description=(
        "Breaks down memory copy operations by direction (Host→Device, Device→Host, "
        "Device→Device, Peer-to-Peer). Excessive H2D transfers in the critical path "
        "often indicate data not being pre-staged on GPU."
    ),
    category="memory",
    sql="""\
SELECT k.copyKind,
       COUNT(*) AS count,
       ROUND(SUM(k.bytes) / 1e6, 2) AS total_mb,
       ROUND(SUM(k.[end] - k.start) / 1e6, 2) AS total_ms
FROM {memcpy_table} k
WHERE 1=1 {trim_clause}
GROUP BY k.copyKind
ORDER BY total_ms DESC, copyKind ASC""",
    format_fn=_format,
    tags=["memory", "transfer", "H2D", "D2H", "copy", "bandwidth"],
)


#: A window shorter than this has no distribution to classify: across two
#: seconds any "leading portion" is most of the window. Measured in elapsed
#: seconds, not in buckets that happen to carry a transfer.
_MIN_SECONDS_FOR_SHAPE = 3

#: The leading share of the window that "front-loaded" means. A fraction rather
#: than a fixed duration, so the same transfer pattern gets the same verdict in a
#: 4-second window and a 40-second one.
_INIT_LEAD_FRACTION = 0.25


def _observed_seconds(conn, rows: list, kwargs: dict) -> float | None:
    """Seconds from the first H2D transfer to the end of what was observed.

    The capture's own span for the selected device, narrowed to the trim window
    when one was given -- narrowed, never widened: a requested window is not
    evidence that anything was watched for that long. None when no bound can be
    established, and the caller falls back to the buckets -- which is the wrong
    answer, but a bounded one.
    """
    starts = [r.get("window_start") for r in rows if r.get("window_start") is not None]
    if not starts:
        return None
    first_transfer = min(int(s) for s in starts)

    # The last transfer is a floor on how long we watched: we saw it, so
    # observation reached at least that far. Taking the kernel table's MAX(end)
    # alone said otherwise wherever copies outlive the last kernel -- on a
    # fixture with one kernel ending at 1.0s and transfers running to 4.2s it
    # reported a one-second window and refused to classify a plainly steady
    # spread.
    ends = [r.get("window_end") for r in rows if r.get("window_end") is not None]
    end_ns = max((int(e) for e in ends), default=None)

    # The kernel bound has to be scoped the way the buckets are. This query had
    # no device predicate while the transfers it is measuring are filtered to
    # one GPU, so a kernel on another card ending late stretched this card's
    # window: steady transfers across GPU 0's six seconds classified as
    # init_heavy because GPU 1 ran until second sixty.
    try:
        from nsys_ai.connection import wrap_connection

        adapter = wrap_connection(conn)
        kernel_tbl = adapter.resolve_activity_tables().get("kernel")
        if kernel_tbl:
            device = int(kwargs.get("device", 0) or 0)
            row = adapter.execute(
                f'SELECT MAX(k."end") FROM {kernel_tbl} k WHERE k.deviceId = {device}'  # nosec B608
            ).fetchone()
            kernel_end = row[0] if row else None
            if kernel_end is not None:
                end_ns = max(int(kernel_end), end_ns) if end_ns is not None else int(kernel_end)
    except Exception:  # noqa: BLE001 - an optional bound, never a failure
        pass

    # A requested window is not observed time. --trim accepts an endpoint past
    # the end of the capture and nothing clamps it, so asking for sixty seconds
    # of a six-second profile put the init lead (a quarter of the window) at
    # fifteen seconds -- swallowing every transfer and reporting init_heavy for
    # data that is plainly spread out. Observation stops where the capture does.
    trim_end = kwargs.get("trim_end_ns")
    if trim_end is not None:
        end_ns = min(int(trim_end), end_ns) if end_ns is not None else int(trim_end)
    if end_ns is None:
        return None

    span_ns = int(end_ns) - first_transfer
    return max(1.0, span_ns / 1e9) if span_ns > 0 else None


def _classify_h2d_pattern(rows: list, kwargs: dict | None = None) -> dict:
    """Classify H2D transfer distribution into init_heavy / spread_out / spike."""
    if not rows:
        return {"_pattern": True, "type": "none", "detail": "No H2D transfers"}

    total_mb = sum(r["total_mb"] for r in rows)
    if total_mb <= 0:
        return {"_pattern": True, "type": "none", "detail": "No bytes transferred"}

    # A window has to be long enough to have a shape before one can be read off
    # it. With two buckets, "the first two seconds" is the whole window: the
    # ratio below is 1.0 by construction and init_heavy wins whatever the data
    # says, so steady per-batch loading in a sub-2s trim was reported as normal
    # weight loading -- the one verdict that tells a reader to stop looking.
    # Elapsed seconds, not row count. The query returns only the seconds that
    # carried a transfer, so len(rows) is how many buckets have data and says
    # nothing about how long the window is. Slicing by position then took "the
    # first quarter of six rows" on buckets [0, 50, 51, 52, 53, 54] and called a
    # 900 MB spike at second 50 front-loading -- reported, absurdly, as "the
    # first 51 seconds of a 6-second window" -- which also swallowed the spike
    # finding that root_cause_matcher consumes.
    first_second = min(r["second"] for r in rows)
    last_second = max(r["second"] for r in rows)
    # How long we watched, not how long transfers ran. Deriving the window from
    # the buckets makes the commonest init_heavy profile unclassifiable: a
    # 60-second capture whose weights load in the first two seconds returns
    # buckets 0 and 1 and nothing else, which reads as a two-second window. And
    # the advice to widen --trim cannot help, because the rest of the capture
    # holds no transfers to find.
    observed_seconds = (kwargs or {}).get("_observed_seconds")
    window_seconds = (
        int(observed_seconds)
        if observed_seconds
        else last_second - first_second + 1
    )

    if window_seconds < _MIN_SECONDS_FOR_SHAPE:
        return {
            "_pattern": True,
            "type": "undetermined",
            "detail": (
                f"Window spans {window_seconds} second-bucket(s); at least "
                f"{_MIN_SECONDS_FOR_SHAPE} are needed to tell a front-loaded "
                f"distribution from a steady one. Widen --trim to classify."
            ),
        }

    # Init-heavy: the leading portion of the window accounts for >80% of bytes.
    # Measured as a fraction of elapsed time rather than a fixed two seconds, so
    # the verdict describes the distribution instead of the window length.
    lead_seconds = max(1, round(window_seconds * _INIT_LEAD_FRACTION))
    lead_cutoff = first_second + lead_seconds
    lead_mb = sum(r["total_mb"] for r in rows if r["second"] < lead_cutoff)
    if lead_mb / total_mb > 0.8:
        return {
            "_pattern": True,
            "type": "init_heavy",
            "detail": (
                f"H2D concentrated in the first {lead_seconds} second(s) of a "
                f"{window_seconds}-second window "
                f"({lead_mb:.1f}/{total_mb:.1f} MB = "
                f"{100 * lead_mb / total_mb:.0f}%). "
                f"Likely model weight loading — normal behavior."
            ),
        }

    # Spike detection: any second exceeds 3× median
    sorted_mb = sorted(r["total_mb"] for r in rows)
    median_mb = sorted_mb[len(sorted_mb) // 2]
    if median_mb > 0:
        spikes = [r for r in rows if r["total_mb"] > 3 * median_mb]
        if spikes:
            spike_secs = ", ".join(str(r["second"]) for r in spikes[:5])
            gpu_id = kwargs.get("device", 0) if kwargs else 0
            return {
                "_pattern": True,
                "type": "spike",
                "detail": (
                    f"{len(spikes)} spike(s) detected at second(s) [{spike_secs}] "
                    f"(>{3 * median_mb:.1f} MB vs median {median_mb:.1f} MB). "
                    f"Check timeline at those timestamps for unexpected data movement."
                ),
                "spike_seconds": [r["second"] for r in spikes],
                "spikes": [
                    {
                        "second": r["second"],
                        "total_mb": r["total_mb"],
                        "window_start": r.get("window_start"),
                        "window_end": r.get("window_end"),
                    }
                    for r in spikes
                ],
                "gpu_id": gpu_id,
            }

    # Spread-out: transfers happen across many seconds
    if len(rows) >= 3:
        return {
            "_pattern": True,
            "type": "spread_out",
            "detail": (
                f"H2D transfers spread across {len(rows)} seconds "
                f"({total_mb:.1f} MB total). Every step has H2D — "
                f"consider pin_memory=True with prefetching — pin_memory raises transfer bandwidth on its own, but hiding the transfer behind compute additionally needs a copy stream — DataLoader prefetching stages batches on the CPU and does not by itself overlap H2D with kernels."
            ),
        }

    return {"_pattern": True, "type": "unknown", "detail": "Too few data points to classify"}


def _format_dist(rows):
    if not rows:
        return "(No H2D memory transfers found)"

    # Separate data rows from pattern metadata
    data_rows = [r for r in rows if not r.get("_pattern")]
    pattern = next((r for r in rows if r.get("_pattern")), None)

    if not data_rows:
        return "(No H2D memory transfers found)"

    lines = [
        "── H2D Time Distribution ──",
        f"{'Second':<8s}  {'Count':>7s}  {'Total(MB)':>10s}  {'Avg GB/s':>10s}",
        "─" * 41,
    ]
    for r in data_rows:
        lines.append(
            f"{r['second']:<8d}  {r['ops']:>7d}  {r['total_mb']:>10.2f}  {r['avg_gbps']:>10.2f}"
        )

    if pattern:
        ptype = pattern.get("type", "unknown")
        icon = {"init_heavy": "✅", "spread_out": "⚠️", "spike": "🔴"}.get(ptype, "ℹ️")
        lines.append(f"\n  {icon} Pattern: {ptype}")
        lines.append(f"     {pattern.get('detail', '')}")

    return "\n".join(lines)


H2D_DIST_SKILL = Skill(
    name="h2d_distribution",
    title="H2D Transfer Time Distribution",
    description=(
        "Groups Host-to-Device (H2D) memory transfers by second. "
        "Useful for distinguishing between initial model loading (concentrated at start) "
        "and continuous data feeding in the training/inference loop."
    ),
    category="memory",
    params=[SkillParam("device", "GPU device ID", "int", False, 0)],
    sql="""\
WITH baseline AS (
    SELECT MIN(k.start) AS min_start
    FROM {memcpy_table} k
    WHERE k.copyKind = 1 AND k.deviceId = {device} {trim_clause}
)
SELECT
    -- Elapsed whole seconds since the first H2D copy. Both operands are kept
    -- integral on purpose: subtracting the remainder floors the value exactly,
    -- so the bucket does not depend on how the engine rounds. Casting a
    -- fractional value instead would split the profile differently per engine
    -- (0.6 truncates to 0 under SQLite, rounds to 1 under DuckDB), putting the
    -- same copies in different buckets. (k.start - b.min_start) is never
    -- negative, so flooring and truncating agree here.
    CAST(
        (k.start - b.min_start - ((k.start - b.min_start) % 1000000000))
        / 1000000000 AS INT
    ) AS second,
    COUNT(*) AS ops,
    SUM(k.bytes) / 1e6 AS total_mb,
    -- Bytes per nanosecond is already GB/s. Cast the numerator to DOUBLE
    -- first: both sums are integers, and SQLite truncates integer division
    -- to 0 where DuckDB promotes to a real quotient.
    COALESCE(
        CAST(SUM(k.bytes) AS DOUBLE) / NULLIF(SUM(k.[end] - k.start), 0), 0
    ) AS avg_gbps,
    MIN(k.start) AS window_start,
    MAX(k.[end]) AS window_end
FROM {memcpy_table} k CROSS JOIN baseline b
WHERE k.copyKind = 1 AND k.deviceId = {device} {trim_clause}
GROUP BY 1
ORDER BY 1""",
    format_fn=_format_dist,
    tags=["memory", "transfer", "H2D", "distribution", "time", "leak"],
)


def _to_findings_dist(rows: list[dict]) -> list:
    from nsys_ai.annotation import Finding

    findings = []

    pattern = next((r for r in rows if r.get("_pattern")), None)
    if not pattern:
        return findings

    if pattern.get("type") == "spike":
        # Cap at top 5 spikes by size to avoid unbounded findings on long profiles.
        spikes = sorted(
            pattern.get("spikes", []), key=lambda s: s.get("total_mb", 0), reverse=True
        )[:5]
        for spike in spikes:
            start = spike.get("window_start")
            end = spike.get("window_end")
            if start is not None and end is not None and end > start:
                findings.append(
                    Finding(
                        type="region",
                        label=f"H2D Spike ({spike['total_mb']:.1f}MB)",
                        start_ns=start,
                        end_ns=end,
                        gpu_id=pattern.get("gpu_id", 0),
                        severity="warning",
                        note=f"Spike at second {spike['second']}: {spike['total_mb']:.1f}MB transferred",
                    )
                )
    return findings


H2D_DIST_SKILL.to_findings_fn = _to_findings_dist


# Replace the direct SQL with a safe execute_fn for the module
def _execute_h2d_dist(conn, **kwargs):
    """Execute the H2D distribution query, then classify the pattern it shows.

    This function is assigned as ``H2D_DIST_SKILL.execute_fn`` and wraps the
    underlying SQL execution. A profile with no memcpy table now abstains
    inside ``Skill.execute`` rather than reaching the database at all, and that
    abstention is returned untouched: classifying it would read ``total_mb``
    off the marker row and raise ``KeyError``. The ``SkillExecutionError``
    branch below stays for the errors the guard does not cover — a table that
    resolves but cannot be read, for instance — and returns ``[]`` there as it
    always has.

    After executing the SQL, appends a pattern classification dict as the
    last element of the result list.
    """

    # Create a temporary Skill that uses the same SQL but no custom execute_fn
    temp_skill = Skill(
        name=H2D_DIST_SKILL.name,
        title=H2D_DIST_SKILL.title,
        description=H2D_DIST_SKILL.description,
        category=H2D_DIST_SKILL.category,
        params=H2D_DIST_SKILL.params,
        sql=H2D_DIST_SKILL.sql,
        format_fn=H2D_DIST_SKILL.format_fn,
        tags=getattr(H2D_DIST_SKILL, "tags", None),
    )

    from nsys_ai.exceptions import SkillExecutionError

    try:
        rows = temp_skill.execute(conn, **kwargs)
    except SkillExecutionError as exc:
        err_msg = str(exc).lower()
        if "no such table" in err_msg or "does not exist" in err_msg:
            return []
        raise

    if is_abstention(rows):
        return rows

    # Append pattern classification as metadata
    if rows:
        rows.append(
            _classify_h2d_pattern(rows, {**kwargs, "_observed_seconds": _observed_seconds(conn, rows, kwargs)})
        )
    return rows


H2D_DIST_SKILL.execute_fn = _execute_h2d_dist


SKILLS = [SKILL, H2D_DIST_SKILL]
