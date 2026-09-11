"""`root_cause_matcher` must not treat "could not run" as evidence.

`_safe_execute` returns whatever the sub-skill returned, and every consumer
guards with `if <rows>:`. An abstention row is a non-empty list, so it passes
that guard and is read as data.

Nothing goes wrong today, and the reason is worth stating because it is not a
design: the only sub-skill that can abstain is `nvtx_layer_breakdown`, abstention
is exactly one row, and the layer analysers happen to sit behind
`len(layer_data) >= 2`. Add a second abstaining skill to a `len >= 1` path, or
give `abstain()` a detail row, and the accident stops holding.

`evidence_builder` already filters for this reason — its docstring names the same
hazard, that the safe skills are "safe only through unrelated guards ... which a
refactor could remove without anyone noticing". This puts `root_cause_matcher` on
the same footing.
"""

import sqlite3
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NO_NVTX = REPO / "tests" / "fixtures" / "healthy_1pct.sqlite"


@pytest.fixture
def conn():
    """Read-only, so the run cannot modify the checked-in fixture.

    `Skill.execute` calls `ensure_performance_indexes`, which writes `_nsysai_*`
    indexes into whatever profile it is handed. Against a committed fixture that
    leaves the working tree dirty and changes the file for every later run.
    """
    c = sqlite3.connect(f"file:{NO_NVTX}?mode=ro", uri=True)
    try:
        yield c
    finally:
        c.close()


def test_the_fixture_really_has_no_nvtx(conn):
    """Guard the premise: without this the assertions below go vacuous."""
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert not any(t.startswith("NVTX_EVENTS") for t in tables)


def test_the_subskill_really_abstains(conn):
    """The other half of the premise: the skill must abstain, not return []."""
    from nsys_ai.skills.base import is_abstention
    from nsys_ai.skills.registry import get_skill

    rows = get_skill("nvtx_layer_breakdown").execute(conn)
    assert is_abstention(rows), f"expected an abstention row, got {rows[:1]}"


def test_safe_execute_does_not_pass_abstention_through(conn):
    """The assertion that fails without the guard."""
    from nsys_ai.skills.builtins.root_cause_matcher import _safe_execute

    rows = _safe_execute("nvtx_layer_breakdown", conn)

    assert rows == [], (
        f"_safe_execute returned an abstention row as if it were data; every "
        f"consumer guards with `if rows:` and a non-empty list passes that: {rows}"
    )


def test_safe_execute_still_returns_real_rows(conn):
    """The other half of the contract: a skill that ran must pass its data through.

    The change restructured the return, so guard the path it moved: without this
    a `return rows` lost in the edit would only surface as some unrelated
    matcher test going quiet.
    """
    from nsys_ai.skills.builtins.root_cause_matcher import _safe_execute

    rows = _safe_execute("top_kernels", conn, limit=5)

    assert rows, "top_kernels ran against a profile with kernels and returned nothing"
    assert "kernel_name" in rows[0], f"rows came back reshaped: {rows[0]}"


def test_matcher_reports_no_layer_findings_without_annotation(conn):
    """End to end: an unannotated profile must not yield layer-attributed causes."""
    from nsys_ai.skills.registry import get_skill

    findings = get_skill("root_cause_matcher").execute(conn)

    for f in findings:
        assert "_abstained" not in f, f"an abstention row was emitted as a finding: {f}"
        # Layer findings are the ones nvtx_layer_breakdown feeds; none can be
        # justified on a profile that carries no annotation at all.
        assert "layer" not in str(f.get("pattern", "")).lower(), (
            f"layer-attributed cause on an unannotated profile: {f}"
        )


# ── Pipeline Imbalance only claims a pipeline where one is plausible ────────


def test_phase_annotations_are_not_reported_as_a_pipeline_to_rebalance():
    """forward / backward / optimizer / data_load differ by design.

    The check compared whatever nvtx_layer_breakdown returned and called the
    spread "Pipeline Imbalance", recommending stage repartitioning — on a
    single-GPU run that may have no pipeline parallelism at all. backward taking
    120x data_load is what a healthy iteration looks like.
    """
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    findings = _check_pipeline_imbalance([
        {"nvtx_region": "forward", "compute_ms": 120.0},
        {"nvtx_region": "backward", "compute_ms": 240.0},
        {"nvtx_region": "optimizer", "compute_ms": 15.0},
        {"nvtx_region": "data_load", "compute_ms": 2.0},
    ])

    assert len(findings) == 1
    assert findings[0]["pattern"] == "Uneven NVTX Regions"
    assert findings[0]["severity"] == "info"
    assert "not necessarily pipeline stages" in findings[0]["recommendation"]
    # The measurement is still reported; it is true and worth seeing.
    assert "120.0×" in findings[0]["evidence"]


def test_stage_like_regions_keep_the_rebalancing_advice():
    """Where the regions do look like peers, the original finding stands."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    findings = _check_pipeline_imbalance([
        {"nvtx_region": f"stage_{i}", "compute_ms": ms}
        for i, ms in enumerate([240.0, 80.0, 75.0, 20.0])
    ])

    assert findings[0]["pattern"] == "Pipeline Imbalance"
    assert findings[0]["severity"] == "warning"
    assert "Rebalance pipeline stage partitioning" in findings[0]["recommendation"]


def test_a_single_phase_label_is_enough_to_withhold_the_claim():
    """A mixed set is not a clean list of peers either."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    findings = _check_pipeline_imbalance([
        {"nvtx_region": "stage_0", "compute_ms": 240.0},
        {"nvtx_region": "stage_1", "compute_ms": 80.0},
        {"nvtx_region": "backward", "compute_ms": 20.0},
    ])

    assert findings[0]["pattern"] == "Uneven NVTX Regions"


def test_nesting_does_not_change_the_verdict():
    """nvtx_layer_breakdown returns full paths, so ancestors are in the label.

    Sibling stages under a 'train_step' parent all carry "step" in their path
    and were read as phases — the warning suppressed because an enclosing
    annotation existed, which is the opposite of what that annotation says.
    """
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    nested_stages = [
        {"nvtx_path": f"train_step > stage_{i}", "compute_ms": ms}
        for i, ms in enumerate([240.0, 80.0, 75.0, 20.0])
    ]
    nested_phases = [
        {"nvtx_path": f"iteration > {name}", "compute_ms": ms}
        for name, ms in [("forward", 120.0), ("backward", 240.0), ("data_load", 2.0)]
    ]

    assert _check_pipeline_imbalance(nested_stages)[0]["pattern"] == "Pipeline Imbalance"
    assert _check_pipeline_imbalance(nested_phases)[0]["pattern"] == "Uneven NVTX Regions"


# ── The claim needs positive evidence, and the spread needs a magnitude ──────


def test_per_operation_annotations_are_not_a_pipeline():
    """132 'aten::linear, op_id = N' regions are operations, not stages.

    The label check was a denylist, so anything that was not PyTorch phase
    naming defaulted to "pipeline stage". This is the repository's own fixture:
    every region wraps exactly one kernel, which is what an operation looks like
    and what a stage never does.
    """
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    regions = [
        {"nvtx_path": f"aten::linear, op_id = {318000 + i}", "kernel_count": 1,
         "compute_ms": 40.0 if i == 0 else 4.0}
        for i in range(132)
    ]

    findings = _check_pipeline_imbalance(regions)

    assert findings[0]["pattern"] == "Uneven NVTX Regions"
    assert findings[0]["severity"] == "info"


def test_a_sub_millisecond_spread_is_not_reported_at_all():
    """0.3ms against 0.1ms clears a 3x ratio and is still noise.

    The only floor was compute_ms > 10us, so the fixture reported a 4.4x
    "Pipeline Imbalance" between two operations whose entire difference was
    0.2ms.
    """
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    assert _check_pipeline_imbalance([
        {"nvtx_region": "stage_0", "compute_ms": 0.31},
        {"nvtx_region": "stage_1", "compute_ms": 0.07},
    ]) == []


def test_a_real_pipeline_survives_both_new_guards():
    """The complement: few stages, each spanning many kernels, a real spread."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    findings = _check_pipeline_imbalance([
        {"nvtx_region": f"stage_{i}", "kernel_count": 400, "compute_ms": ms}
        for i, ms in enumerate([240.0, 80.0, 75.0, 20.0])
    ])

    assert findings[0]["pattern"] == "Pipeline Imbalance"
    assert findings[0]["severity"] == "warning"


def test_a_missing_kernel_count_is_not_read_as_evidence():
    """Absence is not a single-kernel region; callers pass rows without it."""
    from nsys_ai.skills.builtins.root_cause_matcher import _check_pipeline_imbalance

    findings = _check_pipeline_imbalance([
        {"nvtx_region": f"stage_{i}", "compute_ms": ms}
        for i, ms in enumerate([240.0, 80.0, 75.0, 20.0])
    ])

    assert findings[0]["pattern"] == "Pipeline Imbalance"
