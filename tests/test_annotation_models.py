"""Tests for v0.1 evidence schema models in annotation.py.

Covers EvidenceRow, TraceSelection, DiffLineage, Diagnostic.
Existing Finding / EvidenceReport tests live in test_evidence_build.py.
"""

import json

from nsys_ai.annotation import (
    Diagnostic,
    DiffLineage,
    EvidenceRow,
    Finding,
    TraceSelection,
)


class TestEvidenceRow:
    def test_minimal_construction(self):
        row = EvidenceRow(id="r0", source_skill="gpu_idle_gaps")
        assert row.id == "r0"
        assert row.source_skill == "gpu_idle_gaps"
        assert row.values == {}
        assert row.units == {}
        assert row.selection_id is None
        assert row.provenance == {}

    def test_full_construction(self):
        row = EvidenceRow(
            id="r1",
            source_skill="overlap_breakdown",
            values={"overlap_ratio": 0.12, "exposed_ms": 48},
            units={"overlap_ratio": "ratio", "exposed_ms": "ms"},
            selection_id="sel_001",
            provenance={"trim_start_ns": 0, "trim_end_ns": 1_000_000},
        )
        assert row.values["overlap_ratio"] == 0.12
        assert row.selection_id == "sel_001"
        assert row.provenance["trim_end_ns"] == 1_000_000

    def test_round_trip(self):
        original = EvidenceRow(
            id="r2",
            source_skill="top_kernels",
            values={"count": 42},
            units={"count": "count"},
        )
        restored = EvidenceRow.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_json(self):
        row = EvidenceRow(id="r3", source_skill="nccl_breakdown", values={"a": 1})
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored == row

    def test_from_dict_ignores_unknown_keys(self):
        row = EvidenceRow.from_dict({"id": "r4", "source_skill": "x", "future_field": "ignored"})
        assert row.id == "r4"
        assert row.source_skill == "x"

    def test_to_dict_drops_none(self):
        row = EvidenceRow(id="r5", source_skill="x")
        d = row.to_dict()
        assert "selection_id" not in d


class TestTraceSelection:
    def test_minimal_construction(self):
        sel = TraceSelection(id="s0", profile_id="abc123", source="user")
        assert sel.id == "s0"
        assert sel.profile_id == "abc123"
        assert sel.source == "user"
        assert sel.start_ns is None
        assert sel.gpu_ids is None

    def test_full_construction(self):
        sel = TraceSelection(
            id="s1",
            profile_id="abc123",
            source="skill:gpu_idle_gaps",
            start_ns=1_000_000,
            end_ns=2_000_000,
            gpu_ids=[0, 1],
            rank_ids=[5],
            stream_ids=[7, 13],
            nvtx_path=["iteration_142", "backward"],
            event_ids=["evt_99"],
            label="NCCL exposed",
        )
        assert sel.gpu_ids == [0, 1]
        assert sel.nvtx_path[-1] == "backward"
        assert sel.label == "NCCL exposed"

    def test_round_trip_minimal(self):
        sel = TraceSelection(id="s2", profile_id="def", source="gui")
        restored = TraceSelection.from_dict(sel.to_dict())
        assert restored == sel

    def test_round_trip_full(self):
        sel = TraceSelection(
            id="s3",
            profile_id="def",
            source="diff",
            start_ns=10,
            end_ns=20,
            gpu_ids=[0],
            nvtx_path=["a", "b"],
            label="region",
        )
        restored = TraceSelection.from_dict(sel.to_dict())
        assert restored == sel

    def test_round_trip_json(self):
        sel = TraceSelection(id="s4", profile_id="x", source="user", start_ns=1, gpu_ids=[0, 1])
        restored = TraceSelection.from_dict(json.loads(json.dumps(sel.to_dict())))
        assert restored == sel

    def test_to_dict_drops_none_optionals(self):
        sel = TraceSelection(id="s5", profile_id="x", source="user")
        d = sel.to_dict()
        # Optional fields with None values are not serialized
        assert "start_ns" not in d
        assert "gpu_ids" not in d
        assert "label" not in d
        # Required fields are
        assert d["id"] == "s5"
        assert d["source"] == "user"

    def test_empty_list_distinct_from_none(self):
        """Empty list ≠ None: empty list means 'selection contains zero gpus'."""
        sel = TraceSelection(id="s6", profile_id="x", source="user", gpu_ids=[])
        d = sel.to_dict()
        assert d["gpu_ids"] == []

    def test_from_dict_ignores_unknown_keys(self):
        sel = TraceSelection.from_dict(
            {
                "id": "s7",
                "profile_id": "x",
                "source": "user",
                "unknown": 123,
            }
        )
        assert sel.id == "s7"


class TestDiffLineage:
    def test_construction(self):
        lin = DiffLineage(
            diff_id="diff_20260511",
            role="regression",
            rank=2,
            baseline_profile_id="base_v1_0",
        )
        assert lin.role == "regression"
        assert lin.rank == 2
        assert lin.baseline_profile_id == "base_v1_0"

    def test_round_trip(self):
        lin = DiffLineage(diff_id="d", role="improvement", rank=0, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored == lin

    def test_round_trip_json(self):
        lin = DiffLineage(diff_id="d", role="stable", rank=5, baseline_profile_id="b")
        restored = DiffLineage.from_dict(json.loads(json.dumps(lin.to_dict())))
        assert restored == lin

    def test_from_dict_ignores_unknown_keys(self):
        lin = DiffLineage.from_dict(
            {
                "diff_id": "d",
                "role": "stable",
                "rank": 0,
                "baseline_profile_id": "b",
                "extra": "ignored",
            }
        )
        assert lin.diff_id == "d"


class TestDiagnostic:
    def test_minimal_construction(self):
        diag = Diagnostic(
            id="diag_0",
            summary="Test summary",
            recommendation="Do nothing",
            verification_command="nsys-ai diff a.sqlite b.sqlite",
            confidence=0.9,
        )
        assert diag.primary_findings == []
        assert diag.root_cause_hypotheses == []
        assert diag.confidence == 0.9

    def test_with_findings_and_hypotheses(self):
        f = Finding(type="marker", label="L", start_ns=0)
        diag = Diagnostic(
            id="diag_1",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.7,
            primary_findings=[f],
            root_cause_hypotheses=["h1", "h2"],
        )
        assert len(diag.primary_findings) == 1
        assert diag.primary_findings[0].label == "L"
        assert diag.root_cause_hypotheses == ["h1", "h2"]

    def test_to_dict_serializes_nested_findings(self):
        f1 = Finding(type="region", label="A", start_ns=10, end_ns=20)
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag_2",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.5,
            primary_findings=[f1, f2],
        )
        d = diag.to_dict()
        # Nested findings should be dicts, not Finding objects
        assert isinstance(d["primary_findings"][0], dict)
        assert d["primary_findings"][0]["label"] == "A"
        assert d["primary_findings"][1]["start_ns"] == 30

    def test_round_trip_with_nested_findings(self):
        f1 = Finding(type="region", label="A", start_ns=10, end_ns=20)
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag_3",
            summary="S",
            recommendation="R",
            verification_command="V",
            confidence=0.5,
            primary_findings=[f1, f2],
            root_cause_hypotheses=["h1"],
        )
        restored = Diagnostic.from_dict(diag.to_dict())
        assert len(restored.primary_findings) == 2
        assert restored.primary_findings[0].label == "A"
        assert restored.primary_findings[1].start_ns == 30
        assert restored.root_cause_hypotheses == ["h1"]

    def test_round_trip_json(self):
        f = Finding(type="region", label="L", start_ns=1)
        diag = Diagnostic(
            id="diag_4",
            summary="S",
            recommendation="R",
            verification_command="nsys-ai diff x y",
            confidence=0.6,
            primary_findings=[f],
            root_cause_hypotheses=["h"],
        )
        restored = Diagnostic.from_dict(json.loads(json.dumps(diag.to_dict())))
        assert restored.id == diag.id
        assert restored.confidence == diag.confidence
        assert restored.verification_command == diag.verification_command
        assert restored.primary_findings[0].label == "L"

    def test_from_dict_ignores_unknown_keys(self):
        diag = Diagnostic.from_dict(
            {
                "id": "d",
                "summary": "s",
                "recommendation": "r",
                "verification_command": "v",
                "confidence": 0.5,
                "future_field": "ignored",
            }
        )
        assert diag.id == "d"
        assert diag.primary_findings == []


class TestDiagnosticEdgeCases:
    """Edge cases for Diagnostic: None lists, mutation safety, unicode, zero values."""

    def _minimal(self, **overrides):
        defaults = dict(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
        )
        defaults.update(overrides)
        return Diagnostic.from_dict(defaults)

    def test_from_dict_with_null_primary_findings(self):
        """JSON null → empty list, not a crash."""
        diag = self._minimal(primary_findings=None)
        assert diag.primary_findings == []

    def test_from_dict_with_null_root_cause_hypotheses(self):
        diag = self._minimal(root_cause_hypotheses=None)
        assert diag.root_cause_hypotheses == []

    def test_from_dict_with_missing_optional_lists(self):
        """Missing keys use dataclass defaults."""
        diag = self._minimal()
        assert diag.primary_findings == []
        assert diag.root_cause_hypotheses == []

    def test_to_dict_does_not_share_list_state(self):
        """Mutating the returned dict's lists does not affect the Diagnostic."""
        f = Finding(type="region", label="L", start_ns=0)
        diag = Diagnostic(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
            primary_findings=[f],
            root_cause_hypotheses=["h"],
        )
        d = diag.to_dict()
        d["root_cause_hypotheses"].append("mutation")
        d["primary_findings"].append({"injected": True})
        assert diag.root_cause_hypotheses == ["h"]
        assert len(diag.primary_findings) == 1

    def test_to_dict_repeated_calls_produce_independent_results(self):
        """Two consecutive to_dict() calls return independent dicts."""
        diag = Diagnostic(
            id="d",
            summary="s",
            recommendation="r",
            verification_command="v",
            confidence=0.5,
            root_cause_hypotheses=["a"],
        )
        d1 = diag.to_dict()
        d2 = diag.to_dict()
        d1["root_cause_hypotheses"].append("mutation")
        assert d2["root_cause_hypotheses"] == ["a"]

    def test_full_json_round_trip_with_unicode(self):
        f1 = Finding(
            type="region",
            label="A",
            start_ns=10,
            end_ns=20,
            severity="warning",
            note="中文 note — em-dash",
        )
        f2 = Finding(type="marker", label="B", start_ns=30)
        diag = Diagnostic(
            id="diag",
            summary="unicode: 中文 ✓ — em-dash",
            recommendation="action",
            verification_command="nsys-ai diff a b",
            confidence=0.0,  # edge: zero confidence
            primary_findings=[f1, f2],
            root_cause_hypotheses=["h1", "h2"],
        )
        s = json.dumps(diag.to_dict(), indent=2, ensure_ascii=False)
        restored = Diagnostic.from_dict(json.loads(s))
        assert restored == diag

    def test_confidence_boundary_values(self):
        """confidence accepts 0.0 and 1.0; no validation enforced (yet)."""
        for c in (0.0, 0.5, 1.0):
            diag = self._minimal(confidence=c)
            assert diag.confidence == c

    def test_verification_command_can_be_multiline(self):
        """Verification commands may be multi-line shell snippets."""
        cmd = (
            "nsys-ai diff before.sqlite after.sqlite \\\n"
            "  --gate exposed_comm:+10ms \\\n"
            "  --exit-on regression"
        )
        diag = self._minimal(verification_command=cmd)
        restored = Diagnostic.from_dict(json.loads(json.dumps(diag.to_dict())))
        assert restored.verification_command == cmd


class TestEvidenceRowEdgeCases:
    def test_nested_dict_in_values(self):
        row = EvidenceRow(
            id="r",
            source_skill="x",
            values={"nested": {"a": 1, "b": [2, 3]}},
        )
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored.values["nested"]["b"] == [2, 3]

    def test_unicode_in_values(self):
        row = EvidenceRow(
            id="r",
            source_skill="x",
            values={"label": "中文 — em-dash"},
        )
        s = json.dumps(row.to_dict(), ensure_ascii=False)
        restored = EvidenceRow.from_dict(json.loads(s))
        assert restored.values["label"] == "中文 — em-dash"

    def test_empty_dicts_preserved(self):
        """Empty dicts (default values) round-trip; only None is dropped."""
        row = EvidenceRow(id="r", source_skill="x")
        d = row.to_dict()
        assert d["values"] == {}
        assert d["units"] == {}
        assert d["provenance"] == {}
        assert "selection_id" not in d  # None → dropped

    def test_to_dict_mutation_safety(self):
        row = EvidenceRow(id="r", source_skill="x", values={"a": 1})
        d = row.to_dict()
        d["values"]["a"] = 999
        assert row.values["a"] == 1  # original unchanged

    def test_from_dict_normalizes_null_dict_fields(self):
        """JSON null for values / units / provenance is coerced to {}.

        Preserves the dict invariant on these fields so callers never see
        ``None`` where a dict is expected.
        """
        row = EvidenceRow.from_dict(
            {
                "id": "r",
                "source_skill": "x",
                "values": None,
                "units": None,
                "provenance": None,
            }
        )
        assert row.values == {}
        assert row.units == {}
        assert row.provenance == {}

    def test_from_dict_null_dict_fields_round_trip(self):
        """A payload with null dict-fields normalizes and round-trips cleanly."""
        row = EvidenceRow.from_dict({"id": "r", "source_skill": "x", "values": None})
        restored = EvidenceRow.from_dict(json.loads(json.dumps(row.to_dict())))
        assert restored == row
        assert restored.values == {}

    def test_from_dict_mixed_null_and_present_dict_fields(self):
        """Only null dict fields are normalized; present ones are preserved."""
        row = EvidenceRow.from_dict(
            {
                "id": "r",
                "source_skill": "x",
                "values": {"a": 1},
                "units": None,
                "provenance": {"trim_start_ns": 0},
            }
        )
        assert row.values == {"a": 1}
        assert row.units == {}
        assert row.provenance == {"trim_start_ns": 0}


class TestTraceSelectionEdgeCases:
    def test_all_lists_empty_distinct_from_none(self):
        sel = TraceSelection(
            id="s",
            profile_id="x",
            source="user",
            gpu_ids=[],
            rank_ids=[],
            stream_ids=[],
            nvtx_path=[],
            event_ids=[],
        )
        d = sel.to_dict()
        assert d["gpu_ids"] == []
        assert d["rank_ids"] == []
        assert d["stream_ids"] == []
        assert d["nvtx_path"] == []
        assert d["event_ids"] == []
        restored = TraceSelection.from_dict(d)
        assert restored == sel

    def test_zero_timestamps_preserved(self):
        """start_ns=0 is meaningful (not None) and must round-trip."""
        sel = TraceSelection(
            id="s",
            profile_id="x",
            source="user",
            start_ns=0,
            end_ns=0,
        )
        d = sel.to_dict()
        assert d["start_ns"] == 0
        assert d["end_ns"] == 0
        restored = TraceSelection.from_dict(d)
        assert restored.start_ns == 0
        assert restored.end_ns == 0

    def test_to_dict_mutation_safety(self):
        sel = TraceSelection(id="s", profile_id="x", source="user", gpu_ids=[0, 1])
        d = sel.to_dict()
        d["gpu_ids"].append(99)
        assert sel.gpu_ids == [0, 1]

    def test_unicode_label(self):
        sel = TraceSelection(id="s", profile_id="x", source="user", label="iteration_142 / 反向")
        s = json.dumps(sel.to_dict(), ensure_ascii=False)
        restored = TraceSelection.from_dict(json.loads(s))
        assert restored.label == "iteration_142 / 反向"


class TestDiffLineageEdgeCases:
    def test_rank_zero(self):
        lin = DiffLineage(diff_id="d", role="regression", rank=0, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored.rank == 0

    def test_large_rank(self):
        lin = DiffLineage(diff_id="d", role="improvement", rank=999_999, baseline_profile_id="b")
        restored = DiffLineage.from_dict(lin.to_dict())
        assert restored.rank == 999_999

    def test_all_three_roles_round_trip(self):
        for role in ("regression", "improvement", "stable"):
            lin = DiffLineage(diff_id="d", role=role, rank=0, baseline_profile_id="b")
            restored = DiffLineage.from_dict(json.loads(json.dumps(lin.to_dict())))
            assert restored.role == role


class TestModuleExports:
    """All v0.1 schema models are importable from nsys_ai.annotation."""

    def test_imports(self):
        from nsys_ai.annotation import (
            Diagnostic,
            DiffLineage,
            EvidenceReport,
            EvidenceRow,
            Finding,
            FindingCategory,
            TraceSelection,
            load_findings,
            save_findings,
        )

        # Touch each binding so a future accidental removal fails this test.
        assert Diagnostic is not None
        assert DiffLineage is not None
        assert EvidenceReport is not None
        assert EvidenceRow is not None
        assert Finding is not None
        assert FindingCategory is not None
        assert TraceSelection is not None
        assert callable(load_findings)
        assert callable(save_findings)
