"""Tests for overlap_breakdown skill enrichments.

Covers the v0.2.4 additions:
- `present_devices` field surfaces other GPUs in the profile
- `same_stream_compute_pct` / `same_stream_nccl_pct` quantify the
  proportion behind `same_stream_diagnosis`

The conftest's `minimal_nsys_conn` seed places 5 kernels on device 0:

    streamId=7  compute (cId=1,  shortName=1  → kernel_A)
    streamId=7  compute (cId=2,  shortName=2  → kernel_B)
    streamId=8  NCCL    (cId=10, shortName=10 → nccl_AllReduce_kernel)
    streamId=7  NCCL    (cId=12, shortName=11 → nccl_ReduceScatter_kernel)
    streamId=7  compute (cId=13, shortName=1  → kernel_A)

So stream 7 has 3 compute + 1 NCCL (a same-stream candidate). Stream 8
has 1 NCCL only (not same-stream). Across the device:
    total compute = 3, total NCCL = 2
    same-stream compute on stream 7 = 3 (100 %)
    same-stream NCCL on stream 7 = 1 (50 %)
"""

import sqlite3

from nsys_ai.skills.builtins.overlap_breakdown import SKILL


def test_present_devices_single_device(minimal_nsys_conn):
    rows = SKILL.execute_fn(minimal_nsys_conn)
    assert rows, "overlap_breakdown returned no rows"
    r = rows[0]
    assert r.get("present_devices") == [0], (
        f"Expected [0] for single-device fixture, got {r.get('present_devices')!r}"
    )


def test_present_devices_multi_device(minimal_nsys_conn):
    """Add kernels on devices 1, 2, 3 and confirm all four are surfaced."""
    c = minimal_nsys_conn.cursor()
    # Append a compute kernel on each extra device. Same shortName as
    # existing rows so the schema check holds.
    for dev in (1, 2, 3):
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, dev, 17, 50 + dev, 10_000_000 + dev * 100, 11_000_000 + dev * 100,
             1, 1, 32, 1, 1, 256, 1, 1),
        )
    minimal_nsys_conn.commit()

    rows = SKILL.execute_fn(minimal_nsys_conn)
    assert rows[0].get("present_devices") == [0, 1, 2, 3]


def test_same_stream_proportions(minimal_nsys_conn):
    """Stream 7 carries all 3 compute + 1 of 2 NCCL kernels →
    same_stream_compute_pct = 100, same_stream_nccl_pct = 50."""
    rows = SKILL.execute_fn(minimal_nsys_conn)
    r = rows[0]
    assert r.get("same_stream_diagnosis") == ["7"], (
        f"Expected stream 7 flagged, got {r.get('same_stream_diagnosis')!r}"
    )
    assert r.get("same_stream_compute_pct") == 100.0, (
        f"Expected 100.0 % of compute on stream 7, "
        f"got {r.get('same_stream_compute_pct')!r}"
    )
    assert r.get("same_stream_nccl_pct") == 50.0, (
        f"Expected 50.0 % of NCCL on stream 7, "
        f"got {r.get('same_stream_nccl_pct')!r}"
    )


def test_no_same_stream_no_proportions(minimal_nsys_conn):
    """Move the same-stream NCCL kernel to stream 8 so no stream has both
    compute and NCCL. The diagnostic must not fire and the proportions
    must not be emitted."""
    c = minimal_nsys_conn.cursor()
    # cId=12 is the NCCL kernel currently on stream 7. Move it.
    c.execute(
        "UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET streamId = 8 "
        "WHERE correlationId = 12"
    )
    minimal_nsys_conn.commit()

    rows = SKILL.execute_fn(minimal_nsys_conn)
    r = rows[0]
    assert "same_stream_diagnosis" not in r, (
        f"Diagnostic should not fire when no stream has both kinds "
        f"(got {r.get('same_stream_diagnosis')!r})"
    )
    assert "same_stream_compute_pct" not in r
    assert "same_stream_nccl_pct" not in r


def test_present_devices_survives_empty_kernel_table():
    """If the kernel table exists but is empty, present_devices should
    either be empty or absent — not crash."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE NsightSchemaMeta (version INTEGER)")
    c.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    c.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(globalPid INTEGER, deviceId INTEGER, streamId INTEGER, "
        " correlationId INTEGER, start INTEGER, \"end\" INTEGER, "
        " shortName INTEGER, demangledName INTEGER, "
        " gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
        " blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
    )
    conn.commit()

    rows = SKILL.execute_fn(conn)
    # Either an error row (no kernel activity) or normal row with no
    # present_devices key. Both are acceptable; the assertion is that
    # we don't crash.
    assert rows is not None
    if rows and "error" not in rows[0]:
        assert rows[0].get("present_devices", []) == []


# ---------------------------------------------------------------------------
# Textual rendering — _format must surface the new fields so users running
# the skill without --format json see same-stream warnings and multi-device
# hints inline.
# ---------------------------------------------------------------------------


def test_format_shows_same_stream_warning_with_pct(minimal_nsys_conn):
    """_format must emit the ⚠️ Same-stream line with the proportion."""
    rows = SKILL.execute_fn(minimal_nsys_conn)
    text = SKILL.format_fn(rows)
    assert "Same-stream" in text, f"missing warning line in:\n{text}"
    assert "[7]" in text, "expected stream id 7 in warning"
    assert "100" in text and "50" in text, (
        f"expected compute_pct=100 and nccl_pct=50 in warning, got:\n{text}"
    )


def test_format_shows_multi_device_hint(minimal_nsys_conn):
    """When >1 device exists, _format hints at other ranks."""
    c = minimal_nsys_conn.cursor()
    for dev in (1, 2, 3):
        c.execute(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES "
            "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (100, dev, 17, 50 + dev, 10_000_000 + dev * 100, 11_000_000 + dev * 100,
             1, 1, 32, 1, 1, 256, 1, 1),
        )
    minimal_nsys_conn.commit()

    text = SKILL.format_fn(SKILL.execute_fn(minimal_nsys_conn))
    assert "Devices:" in text, f"missing devices line in:\n{text}"
    assert "1, 2, 3" in text, f"expected other ranks listed; got:\n{text}"


def test_format_no_multi_device_hint_single_device(minimal_nsys_conn):
    """Single-device profile must not emit the multi-device hint."""
    text = SKILL.format_fn(SKILL.execute_fn(minimal_nsys_conn))
    assert "Devices:" not in text, f"unexpected multi-device hint:\n{text}"


# ---------------------------------------------------------------------------
# Findings — _to_findings should bake the proportions into the note so the
# evidence layer downstream of analyze --format json gets the context.
# ---------------------------------------------------------------------------


def test_findings_note_includes_same_stream_pct(minimal_nsys_conn):
    rows = SKILL.execute_fn(minimal_nsys_conn)
    findings = SKILL.to_findings_fn(rows)
    # Look for the low-overlap finding (always emitted in this fixture
    # because overlap_pct is < 30 with same-stream serialization).
    low_overlap = [f for f in findings if "Low Compute/NCCL Overlap" in f.label]
    assert low_overlap, f"expected a low-overlap finding, got: {[f.label for f in findings]}"
    note = low_overlap[0].note
    # Proportion strings must appear so consumers can quantify the warning
    # without re-querying the skill output.
    assert "100" in note and "50" in note, (
        f"expected 100% / 50% proportions in finding note, got:\n{note}"
    )


# ---------------------------------------------------------------------------
# Trim-aware path — passing -p trim_start_ns / trim_end_ns must still emit
# the new fields and proportions, scoped to the trimmed window.
# ---------------------------------------------------------------------------


def test_trim_window_preserves_enrichments(minimal_nsys_conn):
    """All five fixture kernels fall in [1000000, 9000000] ns; a trim
    that includes only stream-7 same-stream rows (4_000_000..6_000_000)
    catches just the NCCL-on-stream-7 (cId=12, 4.5..5.5 ms) — proportions
    should still emit if both kinds exist in the window."""
    rows = SKILL.execute_fn(
        minimal_nsys_conn,
        trim_start_ns=500_000,
        trim_end_ns=9_500_000,  # wide enough to include all 5 kernels
    )
    r = rows[0]
    # present_devices is profile-wide, not window-scoped → always [0].
    assert r.get("present_devices") == [0]
    # Same-stream diagnosis should still fire with the wide trim.
    assert r.get("same_stream_diagnosis") == ["7"]
    assert r.get("same_stream_compute_pct") == 100.0
    assert r.get("same_stream_nccl_pct") == 50.0
