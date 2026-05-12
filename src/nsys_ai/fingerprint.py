"""
fingerprint.py — Detects the ML framework and network topology from Nsight SQLite traces.

Extracts a ProfileFingerprint efficiently using O(1) string pool queries.
Also exposes ``get_profile_id`` — a content-derived stable identifier
for a profile, used as the canonical ``profile_id`` in evidence
artefacts (envelope and ``TraceSelection``).
"""

import hashlib
import json
import os
import typing
from dataclasses import dataclass, field

from .connection import DB_ERRORS, wrap_connection


@dataclass
class ProfileFingerprint:
    framework: str
    distributed: bool
    multi_node: bool
    nic_summary: str = ""
    precision_notes: list[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        lines = [
            f"Framework: {self.framework}",
            f"Distributed training: {'yes' if self.distributed else 'no'}",
            f"Multi-node (RDMA): {'yes' if self.multi_node else 'no'}",
        ]
        if self.nic_summary:
            import re

            safe_nic = re.sub(r"[^\w\s\-.,()]", "", str(self.nic_summary)).strip()[:200]
            lines.append(f"Network: {safe_nic}")
        if self.precision_notes:
            lines.append("Notes: " + "; ".join(self.precision_notes))
        return "\n".join(lines)


# Ranked by priority / specificity.
# An environment matching multiple (e.g. Megatron and PyTorch) resolves to the first match.
FRAMEWORK_PRIORITY = [
    ("vLLM", ["paged_attention", "vllm", "SamplerOutput", "ModelRunner"]),
    ("SGLang", ["sglang", "RadixAttention", "TokenAttention"]),
    ("Megatron-LM", ["Megatron", "p2p_comm", "FlushGroups", "MegatronModule"]),
    ("DeepSpeed", ["DeepSpeed", "ZeRO", "offload", "DeepSpeedEngine"]),
    ("PyTorch", ["forward", "backward", "optimizer_step", "flash_attn"]),
]

_LOWERCASE_FRAMEWORK_PRIORITY = [
    (fw, [kw.lower() for kw in keywords]) for fw, keywords in FRAMEWORK_PRIORITY
]

# Known high-performance interconnect vendors
KNOWN_NIC_VENDORS = {
    5555: "Mellanox / NVIDIA",
    5348: "Broadcom",
    6082: "Cray",
    32902: "Intel",
}


def get_fingerprint(conn: typing.Any) -> ProfileFingerprint:
    adapter = wrap_connection(conn)
    tables = adapter.get_table_names()

    # Step A: O(1) String Search via C-engine SQLite LIMIT Sweeps
    # We sweep by priority. The moment we find vLLM, it breaks,
    # scanning exactly 1 matching row avoiding parsing bounds.
    framework = "Generic CUDA"

    def _check_framework(table: str, column: str) -> str | None:
        try:
            for fw, lower_keywords in _LOWERCASE_FRAMEWORK_PRIORITY:
                like_conds = " OR ".join(f"{column} LIKE '%{kw}%'" for kw in lower_keywords)
                cur = adapter.execute(f"SELECT 1 FROM {table} WHERE {like_conds} LIMIT 1")
                if cur.fetchone():
                    return fw
        except DB_ERRORS:
            pass
        return None

    if "StringIds" in tables:
        found = _check_framework("StringIds", "value")
        if found:
            framework = found

    if framework == "Generic CUDA" and "NVTX_EVENTS" in tables:
        # Fallback to direct event traces if no canonical stringIds are hit
        found = _check_framework("NVTX_EVENTS", "text")
        if found:
            framework = found

    # Step B: Topology Search
    multi_node = False
    nic_summary = ""
    if "TARGET_INFO_NIC_INFO" in tables:
        try:
            vendor_keys = ",".join(map(str, KNOWN_NIC_VENDORS.keys()))
            cur = adapter.execute(
                f"SELECT vendorId, name FROM TARGET_INFO_NIC_INFO "
                f"WHERE CAST(vendorId AS INTEGER) IN ({vendor_keys}) "
                f"OR name LIKE 'mlx5_%' OR name LIKE 'cxi%' LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                multi_node = True
                v_id = -1
                try:
                    v_id = int(row[0])
                except (ValueError, TypeError):
                    pass
                vendor_name = KNOWN_NIC_VENDORS.get(v_id, "NIC")
                nic_summary = (
                    f"{vendor_name} hardware detected (vendorId: {row[0]}, name: {row[1]})"
                )
        except DB_ERRORS:
            pass

    distributed = False
    if "NVTX_PAYLOAD_SCHEMAS" in tables:
        try:
            cur = adapter.execute(
                "SELECT 1 FROM NVTX_PAYLOAD_SCHEMAS WHERE name LIKE '%NCCL%' LIMIT 1"
            )
            if cur.fetchone():
                distributed = True
        except DB_ERRORS:
            pass

    return ProfileFingerprint(
        framework=framework,
        distributed=distributed,
        multi_node=multi_node,
        nic_summary=nic_summary,
        precision_notes=[],
    )


# ---------------------------------------------------------------------------
# profile_id — stable content-derived identifier
# ---------------------------------------------------------------------------

PROFILE_ID_VERSION = "nsys1"
"""Algorithm tag for :func:`get_profile_id`. Bump (``nsys2``, …) if the
contributing columns or the canonical serialisation change."""


def get_profile_id(
    conn: typing.Any, *, fallback_path: str | os.PathLike[str] | None = None
) -> str:
    """Return a stable content-derived id for a Nsight Systems profile.

    The hash spans only fields stamped at *profile-capture* time, so it
    survives ``.nsys-rep`` → ``.sqlite`` re-export, ``VACUUM``,
    ``journal_mode`` changes, and filesystem moves:

      - ``TARGET_INFO_SESSION_START_TIME.utcEpochNs``
      - ``ANALYSIS_DETAILS`` rows (``duration / startTime / stopTime``)
      - ``TARGET_INFO_GPU`` rows (``id``, ``name``)
      - ``TARGET_INFO_GPU.uuid`` values when the column exists
        (newer Nsight); older exports keep ``uuid`` on
        ``TARGET_INFO_CUDA_DEVICE`` and that contribution degrades to
        empty rather than failing
      - distinct ``ANALYSIS_FILE.globalPid`` values
      - ``CUPTI_ACTIVITY_KIND_KERNEL`` row count

    Each contribution is JSON-encoded as a ``(label, value)`` pair before
    hashing, so values containing ``|`` / ``;`` / ``\\n`` can never collide
    with neighbouring values via separator ambiguity.

    Format::

        nsys1:sha256:<64-hex>   # preferred — content-derived
        nsys1:path:<64-hex>     # fallback — when no Nsight metadata is reachable

    The fallback fires for backends that expose only the parquet cache
    (``backend='parquetdir'``) where META_DATA / TARGET_INFO tables were
    never materialised. Callers should pass ``self.prof.conn`` (the
    SQLite source where available); the function never reads
    ``self.prof.db`` semantics — it just runs SQL.

    ``fallback_path`` is hashed verbatim — pass an absolute path if you
    want it to compare equal across working-directory changes.

    .. note::
       When *both* ``conn`` is None / empty and ``fallback_path`` is
       None, the function returns ``nsys1:sha256:<sha256 of empty>``.
       That is a recognisable null-id sentinel: every such caller
       collapses to the same value. Always pass ``fallback_path`` if
       cross-call distinguishability matters.

    .. note::
       ``NULLS LAST`` requires SQLite ≥ 3.30 (2019-10-04) and is native
       in DuckDB. Older SQLite raises ``OperationalError`` on the
       ``ORDER BY ... NULLS LAST`` clause; the offending contribution
       is caught and degraded to empty rather than crashing.
    """
    # Coerce ``fallback_path`` once: callers often pass ``pathlib.Path``
    # (e.g. via ``Profile(Path(...))`` → ``Profile.path``). Both fallback
    # branches below would otherwise crash with AttributeError on
    # ``.encode("utf-8")``.
    fallback_str = os.fspath(fallback_path) if fallback_path is not None else None

    def _path_id(p: str) -> str:
        digest = hashlib.sha256(p.encode("utf-8")).hexdigest()
        return f"{PROFILE_ID_VERSION}:path:{digest}"

    def _null_id() -> str:
        """The shared null-id sentinel — same value whether ``conn`` is
        None or merely empty, so consumers can detect "no usable
        identity" with a single equality check."""
        return f"{PROFILE_ID_VERSION}:sha256:{hashlib.sha256(b'').hexdigest()}"

    # None-safe shortcut: wrap_connection(None) returns a SQLiteAdapter
    # over None, whose .execute() raises AttributeError (not a DB error),
    # so the loop's try/except wouldn't catch it. Skip the queries.
    if conn is None:
        return _path_id(fallback_str) if fallback_str else _null_id()

    adapter = wrap_connection(conn)

    def _scalar(sql: str) -> typing.Any:
        """Return the single cell from a one-row/one-column query, or None."""
        try:
            row = adapter.execute(sql).fetchone()
            return row[0] if row and row[0] is not None else None
        except DB_ERRORS:
            return None

    def _rows(sql: str) -> list[list[typing.Any]]:
        """Return all rows as a list of lists (JSON-serialisable)."""
        try:
            return [list(r) for r in adapter.execute(sql).fetchall() if r]
        except DB_ERRORS:
            return []

    # NULLS LAST: SQLite defaults to NULLS FIRST, DuckDB to NULLS LAST —
    # pin it so two engine adapters on the same data hash identically.
    parts: list[tuple[str, typing.Any]] = [
        ("session_start_utc_ns", _scalar("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME")),
        # ANALYSIS_DETAILS is conventionally single-row, but serialise all
        # rows in deterministic order so the contribution stays stable
        # if a future profile carries more than one row.
        (
            "analysis_details",
            _rows(
                "SELECT duration, startTime, stopTime FROM ANALYSIS_DETAILS "
                "ORDER BY startTime, stopTime"
            ),
        ),
        # GPU id + name works on every Nsight schema we've seen.
        (
            "gpus",
            _rows("SELECT id, name FROM TARGET_INFO_GPU ORDER BY id NULLS LAST"),
        ),
        # uuid lives on TARGET_INFO_GPU in newer Nsight exports and on
        # TARGET_INFO_CUDA_DEVICE in older ones (and in our minimal test
        # fixture). Query each in its own contribution so a schema
        # mismatch degrades only that part, not the whole id.
        (
            "gpu_uuids",
            _rows("SELECT uuid FROM TARGET_INFO_GPU ORDER BY id NULLS LAST"),
        ),
        (
            "cuda_device_uuids",
            _rows(
                "SELECT uuid FROM TARGET_INFO_CUDA_DEVICE "
                "ORDER BY gpuId NULLS LAST, cudaId NULLS LAST"
            ),
        ),
        (
            "pids",
            _rows(
                "SELECT DISTINCT globalPid FROM ANALYSIS_FILE "
                "ORDER BY globalPid NULLS LAST"
            ),
        ),
        ("kernel_count", _scalar("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")),
    ]

    # If every contribution is missing/empty the conn carries no Nsight
    # metadata (e.g. parquetdir backend). Fall back to a clearly-labelled
    # path-derived id rather than collapse every such profile to the
    # same constant hash.
    def _empty(v: typing.Any) -> bool:
        return v is None or v == "" or v == [] or v == 0

    if all(_empty(v) for _, v in parts):
        # Same shape as the ``conn is None`` branch above: path-fallback
        # if available, else the null-id sentinel. This keeps the
        # "no usable identity" check a single equality.
        return _path_id(fallback_str) if fallback_str else _null_id()

    # JSON canonical: structured, unambiguous, no separator collisions.
    # ``sort_keys=False`` because ``parts`` is already ordered; flipping
    # the order would change the hash, which is intended (algorithm
    # change → bump PROFILE_ID_VERSION).
    canonical = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{PROFILE_ID_VERSION}:sha256:{digest}"
