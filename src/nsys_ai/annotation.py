"""
annotation.py — Evidence annotation schema.

Agents produce findings (bottleneck highlights, time-range markers, etc.)
that overlay onto the timeline viewer for human verification.

This module also defines the v0.1 evidence schema models that downstream
surfaces (CLI, GUI, agent, diff) share:

    EvidenceRow      — one row of evidence backing a Finding
    TraceSelection   — a region in a profile (time, GPU, rank, stream, NVTX)
    DiffLineage      — links a Finding to the diff that surfaced it
    Diagnostic       — an agent's summarized diagnosis with verification command

Existing models (``Finding``, ``EvidenceReport``) are unchanged in this slice
and will be extended additively in a follow-up.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Finding:
    """A single agent-authored finding to overlay on the timeline."""

    type: str  # "highlight" | "region" | "marker"
    label: str
    start_ns: int
    end_ns: int | None = None  # None for marker type
    stream: str | None = None  # target stream ID (for highlight)
    gpu_id: int | None = None
    color: str = "rgba(255,68,68,0.3)"
    severity: str = "info"  # "critical" | "warning" | "info"
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Finding":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class EvidenceReport:
    """A collection of findings for a profile, produced by an AI agent."""

    title: str
    profile_path: str = ""
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "profile_path": self.profile_path,
            "findings": [f.to_dict() for f in self.findings],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceReport":
        findings = [Finding.from_dict(f) for f in d.get("findings", [])]
        return cls(
            title=d.get("title", "Untitled"),
            profile_path=d.get("profile_path", ""),
            findings=findings,
        )


def load_findings(path: str) -> EvidenceReport:
    """Load an evidence report from a JSON file."""
    with open(path) as f:
        return EvidenceReport.from_dict(json.load(f))


def save_findings(report: EvidenceReport, path: str) -> None:
    """Save an evidence report to a JSON file."""
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# v0.1 evidence schema models
# ──────────────────────────────────────────────────────────────────────

# FindingCategory: step-time category for findings.
# The first four values map to the step-time decomposition
# ``Step Time = Compute + Communication + Launch/Overhead + Idle``.
# The remaining values are orthogonal tags, not step-time buckets.
FindingCategory = Literal[
    "compute",
    "communication",
    "launch_overhead",
    "idle",
    "memory",
    "sync",
    "nvtx",
    "profile_quality",
    "kernel_internal",
    "framework",
]


@dataclass
class EvidenceRow:
    """One row of evidence backing a Finding.

    A skill emits zero or more ``EvidenceRow`` instances; an evidence-citing
    ``Finding`` references them either by id (via ``selection_id``-style
    pointers) or by embedding.
    """

    id: str
    source_skill: str
    values: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    selection_id: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceRow":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Normalize JSON null → {} for dict-typed fields so the dict invariant
        # holds even when callers serialize an explicit ``null`` value.
        for key in ("values", "units", "provenance"):
            if filtered.get(key) is None and key in filtered:
                filtered[key] = {}
        return cls(**filtered)


@dataclass
class TraceSelection:
    """A region in a profile.

    ``profile_id`` is the canonical fingerprint of the source profile
    (see ``nsys_ai.fingerprint.get_fingerprint``); two surfaces looking at
    the same ``.sqlite`` will agree on this id without depending on the
    filesystem path.

    All location fields are optional. A selection may be time-only,
    GPU-only, NVTX-only, or any combination.

    ``source`` records who produced the selection, using the convention
    ``"skill:<name>"`` | ``"gui"`` | ``"user"`` | ``"diff"``.
    """

    id: str
    profile_id: str
    source: str
    start_ns: int | None = None
    end_ns: int | None = None
    gpu_ids: list[int] | None = None
    rank_ids: list[int] | None = None
    stream_ids: list[int] | None = None
    nvtx_path: list[str] | None = None
    event_ids: list[str] | None = None
    label: str | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "TraceSelection":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class DiffLineage:
    """Links a Finding to the diff that surfaced it.

    Lets a Finding inside an ``after.sqlite`` profile carry "I am regression
    #2 of the YYYY-MM-DD diff against baseline:v1.0". Agent and GUI use
    this for provenance and narration.
    """

    diff_id: str
    role: Literal["regression", "improvement", "stable"]
    rank: int  # 0-indexed position in top_regressions / top_improvements
    baseline_profile_id: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DiffLineage":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class Diagnostic:
    """An agent-authored diagnosis with a runnable verification command.

    ``verification_command`` is the runnable ``nsys-ai`` command the user
    should run to confirm whether the proposed fix works. Narration is
    not verification; if no runnable command can be constructed the agent
    should say so explicitly rather than emit prose here.
    """

    id: str
    summary: str
    recommendation: str
    verification_command: str
    confidence: float
    primary_findings: list[Finding] = field(default_factory=list)
    root_cause_hypotheses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "verification_command": self.verification_command,
            "confidence": self.confidence,
            "primary_findings": [f.to_dict() for f in self.primary_findings],
            "root_cause_hypotheses": list(self.root_cause_hypotheses),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Diagnostic":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        # Normalize JSON null / missing → empty list for the nested-Finding path.
        if "primary_findings" in filtered:
            raw = filtered["primary_findings"] or []
            filtered["primary_findings"] = [Finding.from_dict(f) for f in raw]
        if "root_cause_hypotheses" in filtered and filtered["root_cause_hypotheses"] is None:
            filtered["root_cause_hypotheses"] = []
        return cls(**filtered)
