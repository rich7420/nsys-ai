"""Tests for framework fingerprinting logic."""

import re
import sqlite3

from nsys_ai.fingerprint import PROFILE_ID_VERSION, get_fingerprint, get_profile_id


def build_mock_db(
    string_ids: list[str], target_info: list[tuple] = None, payload_schemas: list[str] = None
):
    conn = sqlite3.connect(":memory:")

    # We must patch PRAGMA table_info natively or just use sqlite3 correctly so adapter understands
    c = conn.cursor()
    c.execute("CREATE TABLE StringIds (value TEXT)")

    # Needs to be a valid schema for adapter
    c.execute("CREATE TABLE NsightSchemaMeta (version INTEGER)")

    for s in string_ids:
        c.execute("INSERT INTO StringIds VALUES (?)", (s,))

    if target_info:
        c.execute("CREATE TABLE TARGET_INFO_NIC_INFO (vendorId TEXT, name TEXT)")
        for v in target_info:
            c.execute("INSERT INTO TARGET_INFO_NIC_INFO VALUES (?, ?)", v)

    if payload_schemas:
        c.execute("CREATE TABLE NVTX_PAYLOAD_SCHEMAS (name TEXT)")
        for p in payload_schemas:
            c.execute("INSERT INTO NVTX_PAYLOAD_SCHEMAS VALUES (?)", (p,))

    conn.commit()
    return conn


def test_fingerprint_megatron():
    conn = build_mock_db(["MegatronModule", "flash_attn"])
    fp = get_fingerprint(conn)
    assert fp.framework == "Megatron-LM"
    assert not fp.distributed
    assert not fp.multi_node


def test_fingerprint_vllm():
    conn = build_mock_db(["SamplerOutput", "vllm"])
    fp = get_fingerprint(conn)
    assert fp.framework == "vLLM"


def test_fingerprint_pytorch_generic():
    conn = build_mock_db(["forward", "backward"])
    fp = get_fingerprint(conn)
    assert fp.framework == "PyTorch"


def test_fingerprint_ambiguous():
    # A profile with both PyTorch generic strings and Megatron clusters.
    # To ensure Megatron-LM wins over PyTorch, we add 3 Megatron hits, and 2 PyTorch hits.
    conn = build_mock_db(["Megatron_1", "Megatron_2", "Megatron_3", "forward", "backward"])
    fp = get_fingerprint(conn)
    assert fp.framework == "Megatron-LM"


def test_topology():
    conn = build_mock_db(
        [], target_info=[("5555", "mlx5_0")], payload_schemas=["NCCL communicator"]
    )
    fp = get_fingerprint(conn)
    assert fp.distributed is True
    assert fp.multi_node is True


def test_legacy_fallback():
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE NVTX_EVENTS (text TEXT)")
    c.execute("INSERT INTO NVTX_EVENTS VALUES ('DeepSpeedEngine')")
    conn.commit()

    fp = get_fingerprint(conn)
    assert fp.framework == "DeepSpeed"


# ---------------------------------------------------------------------------
# get_profile_id — content-derived stable identifier
# ---------------------------------------------------------------------------


_SHA256_RE = re.compile(r"^nsys1:sha256:[0-9a-f]{64}$")
_PATH_RE = re.compile(r"^nsys1:path:[0-9a-f]{64}$")


def _nsight_meta_db(
    *,
    session_start_ns: int = 1_700_000_000_000_000_000,
    duration_ns: int = 100_000_000,
    gpus: list[tuple[int, str, str]] | None = None,
    pids: list[int] | None = None,
    kernel_count: int = 0,
) -> sqlite3.Connection:
    """Build an in-memory sqlite that looks enough like a Nsight export
    for ``get_profile_id`` to read its META / TARGET_INFO tables.
    """
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)")
    c.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)", (session_start_ns,))
    c.execute(
        "CREATE TABLE ANALYSIS_DETAILS "
        "(globalVid INTEGER, duration INTEGER, startTime INTEGER, stopTime INTEGER)"
    )
    c.execute(
        "INSERT INTO ANALYSIS_DETAILS VALUES (1, ?, 0, ?)",
        (duration_ns, duration_ns),
    )
    c.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, name TEXT, uuid TEXT)")
    for gid, name, uuid in gpus or []:
        c.execute("INSERT INTO TARGET_INFO_GPU VALUES (?, ?, ?)", (gid, name, uuid))
    c.execute("CREATE TABLE ANALYSIS_FILE (id INTEGER, filename TEXT, globalPid INTEGER)")
    for i, pid in enumerate(pids or []):
        c.execute("INSERT INTO ANALYSIS_FILE VALUES (?, '/tmp/x', ?)", (i, pid))
    c.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (correlationId INTEGER)")
    for i in range(kernel_count):
        c.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?)", (i,))
    conn.commit()
    return conn


def test_get_profile_id_format():
    """Default-shaped result is ``nsys1:sha256:<64-hex>``."""
    conn = _nsight_meta_db()
    pid = get_profile_id(conn)
    assert _SHA256_RE.match(pid), pid
    assert pid.startswith(f"{PROFILE_ID_VERSION}:sha256:")


def test_get_profile_id_deterministic():
    """Two builds of the same META content yield the same id."""
    a = _nsight_meta_db(
        gpus=[(0, "NVIDIA H100", "abc-uuid-0")], pids=[1234, 5678], kernel_count=10
    )
    b = _nsight_meta_db(
        gpus=[(0, "NVIDIA H100", "abc-uuid-0")], pids=[1234, 5678], kernel_count=10
    )
    assert get_profile_id(a) == get_profile_id(b)


def test_get_profile_id_differs_on_content_change():
    base = _nsight_meta_db(kernel_count=10)
    # Different kernel count alone changes the id.
    other = _nsight_meta_db(kernel_count=11)
    assert get_profile_id(base) != get_profile_id(other)
    # Different GPU set alone changes the id.
    diff_gpu = _nsight_meta_db(gpus=[(0, "NVIDIA H100", "xyz")], kernel_count=10)
    assert get_profile_id(base) != get_profile_id(diff_gpu)


def test_get_profile_id_path_independent():
    """``profile_id`` is content-derived; fallback_path does not influence
    the content hash when META tables are present."""
    conn = _nsight_meta_db(kernel_count=3)
    assert get_profile_id(conn, fallback_path="/a/x.sqlite") == get_profile_id(
        conn, fallback_path="/b/y.sqlite"
    )


def test_get_profile_id_missing_tables_falls_back_to_path():
    """Empty conn (no Nsight tables) + fallback_path → nsys1:path:..."""
    conn = sqlite3.connect(":memory:")
    pid = get_profile_id(conn, fallback_path="/some/profile.sqlite")
    assert _PATH_RE.match(pid), pid


def test_get_profile_id_missing_tables_without_fallback_is_null_sentinel():
    """Empty conn + no fallback → the *same* null-id sentinel as
    ``get_profile_id(None)``. Consumers should be able to detect
    "no usable identity" with a single equality check regardless of
    whether the caller passed ``None`` or a real-but-empty conn."""
    pid = get_profile_id(sqlite3.connect(":memory:"))
    assert pid == get_profile_id(None), (
        "conn=empty and conn=None must produce the same null-id sentinel"
    )
    # And they must both be the well-known empty-string SHA-256.
    empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert pid == f"nsys1:sha256:{empty_sha}"


def test_get_profile_id_partial_metadata_still_content_path():
    """If *any* META contribution is non-empty we stay on the sha256 path,
    even when fallback_path is supplied."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    c.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)")
    c.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (999)")
    conn.commit()
    pid = get_profile_id(conn, fallback_path="/x.sqlite")
    assert _SHA256_RE.match(pid), pid


def test_get_profile_id_none_conn_with_fallback():
    """``conn=None`` short-circuits to the path-derived fallback."""
    pid = get_profile_id(None, fallback_path="/some/profile.sqlite")
    assert _PATH_RE.match(pid), pid


def test_get_profile_id_accepts_pathlib_fallback():
    """Regression for Copilot review: ``Profile.path`` is often a
    ``pathlib.Path`` in tests. Both fallback branches must coerce it
    via ``os.fspath`` before ``.encode("utf-8")`` — passing a Path
    used to crash with ``AttributeError`` in the very scenario the
    fallback is meant to handle."""
    from pathlib import Path

    # conn=None branch
    pid_none = get_profile_id(None, fallback_path=Path("/some/profile.sqlite"))
    assert _PATH_RE.match(pid_none), pid_none
    # Equality with str-form to confirm the coercion is value-stable.
    assert pid_none == get_profile_id(None, fallback_path="/some/profile.sqlite")

    # Empty-conn → fallback branch (real conn, no Nsight tables)
    empty_conn = sqlite3.connect(":memory:")
    pid_empty = get_profile_id(empty_conn, fallback_path=Path("/some/profile.sqlite"))
    assert _PATH_RE.match(pid_empty), pid_empty
    assert pid_empty == get_profile_id(empty_conn, fallback_path="/some/profile.sqlite")


def test_get_profile_id_none_conn_without_fallback_is_null_sentinel():
    """``conn=None`` and no fallback yields the recognisable empty-sha256
    sentinel — every such caller collapses to the same value, which is
    intentional (it's a null-id marker, not a usable identifier)."""
    pid = get_profile_id(None)
    assert _SHA256_RE.match(pid), pid
    # The well-known SHA-256 of the empty string. If this constant ever
    # changes, hashlib has changed, not our algorithm.
    empty_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert pid == f"nsys1:sha256:{empty_sha}"


def test_get_profile_id_no_separator_collision():
    """Regression for Copilot review: previous canonical built strings by
    concatenating values with ``|`` / ``;`` separators, so a GPU named
    ``"A|B"`` could collide with two distinct fields ``("A", "B")`` etc.
    JSON canonical eliminates this entirely."""
    # Two profiles differing only in *where* the ``|`` lives.
    a = _nsight_meta_db(gpus=[(0, "A|B", "uuid")])
    b = _nsight_meta_db(gpus=[(0, "A", "B|uuid")])
    assert get_profile_id(a) != get_profile_id(b)


def test_get_profile_id_handles_missing_gpu_uuid_column():
    """The repo's own minimal fixture stores ``uuid`` on
    ``TARGET_INFO_CUDA_DEVICE``, not ``TARGET_INFO_GPU``. The ``gpu_uuids``
    contribution must fail gracefully on such schemas without aborting
    the whole id computation."""
    conn = sqlite3.connect(":memory:")
    c = conn.cursor()
    # Older-style schema: TARGET_INFO_GPU has no uuid column.
    c.execute(
        "CREATE TABLE TARGET_INFO_GPU "
        "(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT)"
    )
    c.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100', '0000:91:00.0')")
    # uuid lives here in this schema:
    c.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE "
        "(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT)"
    )
    c.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, 'older-style-uuid')")
    conn.commit()

    pid = get_profile_id(conn, fallback_path="/x.sqlite")
    assert _SHA256_RE.match(pid), pid
    # Sanity: a profile with the *same* GPU id+name but a *different*
    # cuda_device uuid must hash differently — the contribution must
    # actually be reaching the canonical bytes.
    other = sqlite3.connect(":memory:")
    oc = other.cursor()
    oc.execute(
        "CREATE TABLE TARGET_INFO_GPU "
        "(id INTEGER PRIMARY KEY, name TEXT, busLocation TEXT)"
    )
    oc.execute("INSERT INTO TARGET_INFO_GPU VALUES (0, 'NVIDIA H100', '0000:91:00.0')")
    oc.execute(
        "CREATE TABLE TARGET_INFO_CUDA_DEVICE "
        "(gpuId INTEGER, cudaId INTEGER, pid INTEGER, uuid TEXT)"
    )
    oc.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES (0, 0, 100, 'different-uuid')")
    other.commit()
    assert get_profile_id(other, fallback_path="/x.sqlite") != pid
