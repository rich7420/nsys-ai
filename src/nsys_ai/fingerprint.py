"""
fingerprint.py — Detects the ML framework and network topology from Nsight SQLite traces.

Extracts a ProfileFingerprint efficiently using O(1) string pool queries.
Also exposes ``get_profile_id`` — a content-derived stable identifier
for a profile, used as the canonical ``profile_id`` in evidence
artefacts (envelope and ``TraceSelection``).
"""

import hashlib
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


def get_profile_id(conn: typing.Any, *, fallback_path: str | None = None) -> str:
    """Return a stable content-derived id for a Nsight Systems profile.

    The hash spans only fields stamped at *profile-capture* time, so it
    survives ``.nsys-rep`` → ``.sqlite`` re-export, ``VACUUM``,
    ``journal_mode`` changes, and filesystem moves:

      - ``TARGET_INFO_SESSION_START_TIME.utcEpochNs``
      - ``ANALYSIS_DETAILS`` ``duration / startTime / stopTime``
      - ``TARGET_INFO_GPU`` rows (``id``, ``name``, hardware ``uuid``)
      - distinct ``ANALYSIS_FILE.globalPid`` values
      - ``CUPTI_ACTIVITY_KIND_KERNEL`` row count

    Format::

        nsys1:sha256:<64-hex>   # preferred — content-derived
        nsys1:path:<64-hex>     # fallback — when no Nsight metadata is reachable

    The fallback fires for backends that expose only the parquet cache
    (``backend='parquetdir'``) where META_DATA / TARGET_INFO tables were
    never materialised. Callers should pass ``self.prof.conn`` (the
    SQLite source where available); the function never reads
    ``self.prof.db`` semantics — it just runs SQL.
    """
    adapter = wrap_connection(conn)
    parts: list[tuple[str, str]] = []

    def _one(sql: str) -> str:
        try:
            row = adapter.execute(sql).fetchone()
            return str(row[0]) if row and row[0] is not None else ""
        except DB_ERRORS:
            return ""

    def _all(sql: str) -> str:
        try:
            rows = adapter.execute(sql).fetchall()
            return ";".join(str(r[0]) for r in rows if r and r[0] is not None)
        except DB_ERRORS:
            return ""

    parts.append(
        ("session_start_utc_ns", _one("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"))
    )
    parts.append(
        (
            "trace_range",
            _one(
                "SELECT duration || '|' || startTime || '|' || stopTime FROM ANALYSIS_DETAILS"
            ),
        )
    )
    parts.append(
        (
            "gpus",
            _all(
                "SELECT id || '|' || name || '|' || COALESCE(uuid, '') "
                "FROM TARGET_INFO_GPU ORDER BY id"
            ),
        )
    )
    parts.append(
        ("pids", _all("SELECT DISTINCT globalPid FROM ANALYSIS_FILE ORDER BY globalPid"))
    )
    parts.append(("kernel_count", _one("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL")))

    canonical = "\n".join(f"{k}={v}" for k, v in parts)

    # If every contribution is empty the conn carries no Nsight metadata
    # (e.g. parquetdir backend). Fall back to a clearly-labelled
    # path-derived id rather than collapse every such profile to the same
    # constant hash.
    if not any(v for _, v in parts) and fallback_path:
        digest = hashlib.sha256(fallback_path.encode("utf-8")).hexdigest()
        return f"{PROFILE_ID_VERSION}:path:{digest}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{PROFILE_ID_VERSION}:sha256:{digest}"
