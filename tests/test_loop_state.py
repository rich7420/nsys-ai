import json
import os
import shutil
import time
from pathlib import Path

import pytest

from nsys_ai import profile as _profile
from nsys_ai.loop_state import (
    DiffLoopState,
    detect_h100_replay_preset,
    normalize_profile_path,
    profile_display_name,
    same_profile_path,
)


def test_loop_state_phase_and_proposal():
    state = DiffLoopState(before_path="/tmp/before.sqlite")
    assert state.phase == "diagnose"

    state.set_phase("propose")
    assert state.phase == "propose"

    state.set_proposal("Try FA3", expected_impact="Lower step time")
    assert state.proposal == "Try FA3"
    assert state.expected_impact == "Lower step time"
    assert state.phase == "propose"

    state.set_decision("accept", reason="speedup confirmed")
    assert state.phase == "accept"
    assert state.decision == "accept"
    assert state.decision_reason == "speedup confirmed"


def test_detect_h100_replay_preset_picks_newest_complete_snapshot(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    base = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
    )
    old_snap = base / "old_rev"
    new_snap = base / "new_rev"
    for snap in (old_snap, new_snap):
        (snap / "profiles").mkdir(parents=True)
    (old_snap / "profiles" / "perf_h100_sp1.sqlite").write_text("")
    new_before = new_snap / "profiles" / "perf_h100_sp1.sqlite"
    new_after = new_snap / "profiles" / "perf_h100_sp1_fa3.sqlite"
    new_before.write_text("")
    new_after.write_text("")

    os.utime(old_snap, (time.time() - 100, time.time() - 100))
    os.utime(new_snap, (time.time(), time.time()))

    out = detect_h100_replay_preset()
    assert out is not None
    assert "new_rev" in out["before_path"]
    assert "new_rev" in out["after_path"]


def test_detect_h100_replay_preset(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    snap = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
        / "abc123"
        / "profiles"
    )
    snap.mkdir(parents=True)
    before = snap / "perf_h100_sp1.sqlite"
    after = snap / "perf_h100_sp1_fa3.sqlite"
    before.write_text("")
    after.write_text("")

    out = detect_h100_replay_preset()
    assert out is not None
    assert out["before_path"].endswith("perf_h100_sp1.sqlite")
    assert out["after_path"].endswith("perf_h100_sp1_fa3.sqlite")


def test_set_decision_writes_diff_json_matching_cli_shape(minimal_nsys_db_path, tmp_path):
    before = Path(minimal_nsys_db_path)
    after = tmp_path / "after.sqlite"
    shutil.copy(before, after)

    state = DiffLoopState(before_path=str(before), after_path=str(after))
    state.run_diff(gpu=0)

    out_path = tmp_path / "diff.json"
    warnings = state.set_decision(
        "accept",
        reason="candidate is faster",
        write_path=str(out_path),
        decider="tester@example.com",
        decided_at="2026-06-18T00:00:00Z",
    )
    assert isinstance(warnings, list)
    assert out_path.exists()
    assert state.decision == "accept"
    assert state.decision_path == str(out_path)
    assert state.phase == "accept"

    record = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["decision"]["status"] == "accepted"
    assert record["decision"]["reason"] == "candidate is faster"
    assert record["decision"]["decider"] == "tester@example.com"
    assert record["decision"]["decided_at"] == "2026-06-18T00:00:00Z"

    # Byte-shape identical to what the CLI writer produces for the same summary.
    from nsys_ai.diff_decision import build_diff_decision_record

    expected_payload, _ = build_diff_decision_record(
        state.diff_summary_obj,
        decision="accepted",
        reason="candidate is faster",
        decider="tester@example.com",
        decided_at="2026-06-18T00:00:00Z",
    )
    expected_text = json.dumps(expected_payload, indent=2, sort_keys=True) + "\n"
    assert out_path.read_text(encoding="utf-8") == expected_text


def test_set_decision_persists_across_reload(minimal_nsys_db_path, tmp_path):
    before = Path(minimal_nsys_db_path)
    after = tmp_path / "after.sqlite"
    shutil.copy(before, after)

    state = DiffLoopState(before_path=str(before), after_path=str(after))
    state.run_diff(gpu=0)
    out_path = tmp_path / "diff.json"
    state.set_decision("reject", reason="too slow", write_path=str(out_path))

    restored = DiffLoopState.from_dict(state.to_dict())
    assert restored.decision == "reject"
    assert restored.decision_reason == "too slow"
    assert restored.decision_path == str(out_path)
    # The on-disk audit record survives regardless of in-memory reload.
    assert out_path.exists()


def test_set_decision_empty_reason_leaves_state_unchanged(minimal_nsys_db_path, tmp_path):
    before = Path(minimal_nsys_db_path)
    after = tmp_path / "after.sqlite"
    shutil.copy(before, after)

    state = DiffLoopState(before_path=str(before), after_path=str(after))
    state.run_diff(gpu=0)
    out_path = tmp_path / "diff.json"

    with pytest.raises(ValueError):
        state.set_decision("accept", reason="   ", write_path=str(out_path))

    assert state.decision is None
    assert state.decision_path == ""
    assert state.phase == "diff"
    assert not out_path.exists()


def test_set_decision_without_diff_records_but_writes_nothing(tmp_path):
    state = DiffLoopState(before_path="/tmp/before.sqlite")
    out_path = tmp_path / "diff.json"
    warnings = state.set_decision("accept", reason="manual", write_path=str(out_path))
    assert warnings == []
    assert state.decision == "accept"
    assert state.decision_reason == "manual"
    assert state.decision_path == ""
    assert not out_path.exists()


def test_loop_state_run_diff(minimal_nsys_db_path, tmp_path):
    before = Path(minimal_nsys_db_path)
    after = tmp_path / "after.sqlite"
    shutil.copy(before, after)

    state = DiffLoopState(before_path=str(before), after_path=str(after))
    payload = state.run_diff(gpu=0)
    assert isinstance(payload, dict)
    assert payload.get("verdict") is not None
    assert state.diff_summary is not None
    assert state.phase == "diff"


def test_loop_state_run_diff_reuses_baseline_prof(minimal_nsys_db_path, tmp_path):
    before = Path(minimal_nsys_db_path)
    after = tmp_path / "after.sqlite"
    shutil.copy(before, after)

    with _profile.open(str(before)) as prof:
        state = DiffLoopState(before_path="/nonexistent/before.sqlite", after_path=str(after))
        payload = state.run_diff(gpu=0, baseline_prof=prof)
    assert payload.get("verdict") is not None
    assert state.before_path == str(before.resolve())


def test_normalize_profile_path_rejects_missing(tmp_path):
    missing = tmp_path / "missing.sqlite"
    try:
        normalize_profile_path(str(missing), label="before")
    except FileNotFoundError as exc:
        assert "before not found" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_same_profile_path_symlink(tmp_path):
    target = tmp_path / "profile.sqlite"
    target.write_text("x")
    link = tmp_path / "link.sqlite"
    link.symlink_to(target)
    assert same_profile_path(str(target), str(link))


def test_profile_display_name_maps_h100_blob(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    snap = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "snapshots"
        / "abc123"
        / "profiles"
    )
    snap.mkdir(parents=True)
    blob = (
        home
        / ".cache"
        / "huggingface"
        / "hub"
        / "datasets--rich7421--fastvideo-wan-h100-sp1-nsys"
        / "blobs"
        / ("a" * 40)
    )
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(b"SQLite format 3\x00")
    before = snap / "perf_h100_sp1.sqlite"
    after = snap / "perf_h100_sp1_fa3.sqlite"
    before.symlink_to(blob)
    after.write_bytes(b"SQLite format 3\x00")
    preset = detect_h100_replay_preset()
    assert preset is not None
    assert profile_display_name(str(blob), preset) == "perf_h100_sp1.sqlite"
    assert profile_display_name(str(after), preset) == "perf_h100_sp1_fa3.sqlite"


def test_normalize_profile_path_keeps_symlink_name(tmp_path):
    blob = tmp_path / "blobs" / "803cf28fff228c523caf78689e65d39b8a33f6555cc677bdf00000000"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"SQLite format 3\x00")
    profiles = tmp_path / "snapshots" / "rev1" / "profiles"
    profiles.mkdir(parents=True)
    link = profiles / "perf_h100_sp1.sqlite"
    link.symlink_to(blob)
    out = normalize_profile_path(str(link), label="before")
    assert out.endswith("perf_h100_sp1.sqlite")
    assert "803cf28" not in out


def test_record_reprofile_artifact_missing_path(tmp_path):
    state = DiffLoopState(before_path="/tmp/before.sqlite")
    missing = tmp_path / "missing.sqlite"
    try:
        state.record_reprofile_artifact(str(missing))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")


def test_loop_state_run_diagnose(minimal_nsys_db_path):
    state = DiffLoopState(before_path=str(minimal_nsys_db_path))
    with _profile.open(minimal_nsys_db_path) as prof:
        findings = state.run_diagnose(prof, device=0)
    assert isinstance(findings, list)
    assert state.diagnose_findings_count >= 0
    assert state.diagnose_ran is True


def test_diagnose_ran_roundtrip_and_default():
    state = DiffLoopState(diagnose_ran=True, diagnose_findings_count=0)
    restored = DiffLoopState.from_dict(state.to_dict())
    assert restored.diagnose_ran is True
    assert restored.diagnose_findings_count == 0

    legacy = DiffLoopState.from_dict({"diagnose_findings_count": 3})
    assert legacy.diagnose_ran is False
