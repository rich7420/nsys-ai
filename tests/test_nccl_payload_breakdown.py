"""Tests for nccl_payload_breakdown skill.

Builds a minimal sqlite that mirrors Nsight's NVTX_PAYLOAD_SCHEMAS +
NVTX_PAYLOAD_SCHEMA_ENTRIES + NVTX_EVENTS.binaryData layout, then asserts
the decoder + aggregator produce the expected per-category breakdown.

Binary layout encoded by `_pack_event` mirrors the format documented in the
skill's module docstring (32-byte header + payload-bytes from schema).
"""

import sqlite3
import struct

from nsys_ai.skills.builtins.nccl_payload_breakdown import (
    SKILL,
    _classify,
    _decode_row,
    _load_schemas,
    _percentile,
)


def _pack_event(schema_id: int, payload_bytes: bytes) -> bytes:
    """Build a 32-byte header + payload, matching Nsight's binaryData format."""
    payload_size = len(payload_bytes)
    header = struct.pack(
        "<IIIIQQ", 1, 0, schema_id, 0, payload_size, 32 + payload_size
    )
    return header + payload_bytes


def _build_db_with_nccl_payloads():
    """Create an in-memory sqlite with two schemas + a handful of decoded rows."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMAS "
        "(domainId INTEGER, schemaId INTEGER, name TEXT, type INTEGER, "
        " flags INTEGER, numEntries INTEGER, payloadSize INTEGER, alignTo INTEGER)"
    )
    c.execute(
        "CREATE TABLE NVTX_PAYLOAD_SCHEMA_ENTRIES "
        "(domainId INTEGER, schemaId INTEGER, idx INTEGER, flags INTEGER, "
        " type INTEGER, name TEXT, description TEXT, arrayOrUnionDetail INTEGER, offset INTEGER)"
    )
    c.execute(
        "CREATE TABLE NVTX_EVENTS "
        "(start INTEGER, eventType INTEGER, binaryData BLOB)"
    )

    # Schema A: simple collective (16 bytes payload — comm_id + msg_size)
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 100, NULL, 1, NULL, 2, 16, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 100, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 100, 1, NULL, 22, 'Message size [bytes]', NULL, NULL, 8)"
    )

    # Schema B: p2p (24 bytes — comm_id + msg_size + peer_rank)
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (1, 200, NULL, 1, NULL, 3, 24, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 0, NULL, 18, 'NCCL communicator ID', NULL, NULL, NULL)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 1, NULL, 22, 'Message size [bytes]', NULL, NULL, 8)"
    )
    c.execute(
        "INSERT INTO NVTX_PAYLOAD_SCHEMA_ENTRIES VALUES "
        "(1, 200, 2, NULL, 5, 'Peer rank', NULL, NULL, 16)"
    )

    # Insert events:
    #   - 3 collective events on schema 100 with msg_sizes [1024, 2048, 4096]
    #   - 2 p2p events on schema 200 with msg_sizes [8192, 16384], peers [1, 2]
    comm = 0xCAFE_BABE_DEAD_BEEF
    for ms in (1024, 2048, 4096):
        payload = struct.pack("<QQ", comm, ms)
        c.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(100, payload))
        )
    for ms, peer in ((8192, 1), (16384, 2)):
        payload = struct.pack("<QQI", comm, ms, peer) + b"\x00\x00\x00\x00"
        c.execute(
            "INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(200, payload))
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Decoder unit tests
# ---------------------------------------------------------------------------


def test_load_schemas_returns_field_layout():
    conn = _build_db_with_nccl_payloads()
    schemas = _load_schemas(conn)
    assert set(schemas.keys()) == {100, 200}
    assert schemas[100]["payload_size"] == 16
    assert schemas[200]["payload_size"] == 24
    # field offsets: first is NULL → coerced to 0
    s100_fields = {f["name"]: f["offset"] for f in schemas[100]["fields"]}
    assert s100_fields == {"NCCL communicator ID": 0, "Message size [bytes]": 8}


def test_decode_row_extracts_fields():
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 22, "name": "Message size [bytes]", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xDEAD_BEEF, 65536)
    bd = _pack_event(100, payload)
    decoded = _decode_row(bd, schemas)
    assert decoded == {
        "schema_id": 100,
        "fields": {"NCCL communicator ID": 0xDEAD_BEEF, "Message size [bytes]": 65536},
    }


def test_decode_row_returns_none_on_truncated_blob():
    assert _decode_row(b"", {}) is None
    assert _decode_row(b"\x00" * 10, {}) is None  # shorter than 32-byte header


def test_decode_row_returns_none_on_unknown_schema():
    bd = _pack_event(99999, struct.pack("<QQ", 1, 1))
    assert _decode_row(bd, {}) is None


def test_classify_buckets():
    assert _classify(["NCCL communicator ID"]) == "group_marker"
    assert _classify(["NCCL communicator ID", "Message size [bytes]"]) == "collective"
    assert _classify(["NCCL communicator ID", "Message size [bytes]", "Peer rank"]) == "p2p"
    assert _classify(["NCCL communicator ID", "No. of ranks", "Rank", "CUDA device"]) == "init"


def test_percentile_handles_edge_cases():
    assert _percentile([], 0.5) == 0
    assert _percentile([42], 0.5) == 42
    # p50 of [10, 20, 30] = 20 (linear interp at index 1.0)
    assert _percentile([10, 20, 30], 0.5) == 20
    # p99 of [10, 20, 30] ≈ 29.6 → int → 29
    assert _percentile([10, 20, 30], 0.99) == 29


# ---------------------------------------------------------------------------
# Skill execute + aggregate
# ---------------------------------------------------------------------------


def test_skill_aggregates_per_schema_correctly():
    conn = _build_db_with_nccl_payloads()
    rows = SKILL.execute_fn(conn)

    # First row is the summary.
    s = rows[0]
    assert s["_summary"] is True
    assert s["total_payload_events"] == 5  # 3 collective + 2 p2p
    assert s["distinct_schemas"] == 2

    # Per-schema rows.
    data = {r["schema_id"]: r for r in rows[1:]}
    assert 100 in data and 200 in data

    coll = data[100]
    assert coll["category"] == "collective"
    assert coll["count"] == 3
    assert coll["msg_size_bytes_total"] == 1024 + 2048 + 4096
    assert coll["msg_size_bytes_min"] == 1024
    assert coll["msg_size_bytes_max"] == 4096
    assert coll["distinct_communicators"] == 1

    p2p = data[200]
    assert p2p["category"] == "p2p"
    assert p2p["count"] == 2
    assert p2p["msg_size_bytes_total"] == 8192 + 16384
    assert "Peer rank" not in p2p["field_set"] or "Peer rank" in p2p["field_set"]
    # The peer field is captured in field_set:
    assert "Peer rank" in p2p["field_set"]


def test_skill_returns_error_when_no_payload_schemas():
    """If the profile has no NVTX_PAYLOAD_SCHEMAS table, return a clear error."""
    conn = sqlite3.connect(":memory:")
    rows = SKILL.execute_fn(conn)
    assert "error" in rows[0]
    assert "NVTX_PAYLOAD" in rows[0]["error"]


def test_skill_handles_empty_events_table():
    """Schemas defined but no binaryData events → summary with zero counts."""
    conn = _build_db_with_nccl_payloads()
    conn.execute("DELETE FROM NVTX_EVENTS")
    conn.commit()
    rows = SKILL.execute_fn(conn)
    assert rows[0]["_summary"] is True
    assert rows[0]["total_payload_events"] == 0
    assert rows[0]["total_bytes_all"] == 0


def test_format_produces_human_readable_output():
    conn = _build_db_with_nccl_payloads()
    text = SKILL.format_fn(SKILL.execute_fn(conn))
    assert "NCCL Payload Breakdown" in text
    assert "collective" in text
    assert "p2p" in text
    assert "MiB" in text  # message size formatted


# ---------------------------------------------------------------------------
# Regression / coverage gaps surfaced during real-profile end-to-end debugging
# ---------------------------------------------------------------------------


def test_decode_row_accepts_hex_string_blob():
    """DuckDB returns BLOB columns as hex-encoded strings (e.g. '0100...').
    Sqlite3 returns raw bytes. The decoder must handle both — this test
    locks in the hex-string path so a future regression that re-breaks
    DuckDB compatibility fails fast at unit-test time."""
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 22, "name": "Message size [bytes]", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xCAFE_BEEF, 32768)
    raw = _pack_event(100, payload)
    hex_str = raw.hex()  # DuckDB-style representation
    decoded = _decode_row(hex_str, schemas)
    assert decoded is not None
    assert decoded["schema_id"] == 100
    assert decoded["fields"]["Message size [bytes]"] == 32768


def test_decode_row_rejects_non_hex_string():
    """A str blob that isn't valid hex (e.g. garbage) must return None,
    not raise."""
    assert _decode_row("not-actually-hex-zzz", {}) is None


def test_percentile_boundary_p0_and_p1():
    """Percentile at the endpoints should return min/max exactly."""
    vals = [10, 20, 30, 40, 50]
    assert _percentile(vals, 0.0) == 10
    assert _percentile(vals, 1.0) == 50


def test_decode_row_skips_unknown_field_type():
    """If a schema declares a field with a type not in TYPE_SIZES (e.g. a
    future Nsight emits type 99 = string), the decoder must skip that
    field — never crash. The other fields with known types still decode."""
    schemas = {100: {"payload_size": 16, "fields": [
        {"idx": 0, "type": 18, "name": "NCCL communicator ID", "offset": 0},
        {"idx": 1, "type": 99, "name": "Future field (unknown)", "offset": 8},
    ]}}
    payload = struct.pack("<QQ", 0xCAFE_BEEF, 12345)
    decoded = _decode_row(_pack_event(100, payload), schemas)
    assert decoded is not None
    # Known field decoded:
    assert decoded["fields"]["NCCL communicator ID"] == 0xCAFE_BEEF
    # Unknown-type field skipped (not crashed, not present):
    assert "Future field (unknown)" not in decoded["fields"]


def test_skill_counts_distinct_communicators():
    """When two events share a communicator ID, distinct_communicators must
    not double-count. When they differ, it must reflect both."""
    conn = _build_db_with_nccl_payloads()
    # The fixture uses the same comm_id (0xCAFEBABEDEADBEEF) for all events.
    rows = SKILL.execute_fn(conn)
    data = {r["schema_id"]: r for r in rows[1:]}
    assert data[100]["distinct_communicators"] == 1
    assert data[200]["distinct_communicators"] == 1

    # Add an event on schema 100 with a different communicator and re-check.
    c = conn.cursor()
    payload = struct.pack("<QQ", 0xFEED_FACE_BEEFCAFE, 8192)
    c.execute("INSERT INTO NVTX_EVENTS VALUES (?, 59, ?)", (0, _pack_event(100, payload)))
    conn.commit()
    rows = SKILL.execute_fn(conn)
    data = {r["schema_id"]: r for r in rows[1:]}
    assert data[100]["distinct_communicators"] == 2
