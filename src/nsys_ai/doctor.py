"""doctor.py — environment and profile-health diagnostics for nsys-ai.

``run_doctor()`` returns a structured :class:`DoctorReport` that every surface
consumes the same way:

* CLI       — ``format_doctor_text(report)`` for humans, ``report.to_dict()`` for ``--format json``.
* Web GUI   — a ``/api/doctor`` route serves ``report.to_dict()``.
* TUI       — a health badge / panel reads the same report.
* Claude    — the ``nsys-ai`` skill runs ``nsys-ai doctor --format json`` as a preflight.

All detection logic lives here, never in the CLI handler, so ``web.py`` and the
TUI can ``import nsys_ai.doctor`` and call ``run_doctor(...)`` in-process without
shelling out.

Scope: this checks the *analysis* environment (can we read and analyze this
profile, are optional features configured) and the *quality* of a profile (is
there enough data for a useful diagnosis). It does NOT check whether the host
can capture a profile — for that, run ``nsys status -e`` (NVIDIA's own
profiling-environment check).
"""

from __future__ import annotations

import importlib.util
import platform
import shutil
from dataclasses import dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "0.1"

# Profiler-overhead thresholds (percent of profile span).
_OVERHEAD_WARN_PCT = 10.0
_OVERHEAD_FAIL_PCT = 20.0

CheckStatus = Literal["ok", "warn", "fail", "not_configured", "skipped"]

# Fixed-width status tokens for the text renderer (ASCII — portable across
# terminals, CI logs, and Windows; the project keeps emoji out of output).
_STATUS_TOKEN: dict[CheckStatus, str] = {
    "ok": "OK  ",
    "warn": "WARN",
    "fail": "FAIL",
    "not_configured": "----",
    "skipped": "SKIP",
}

# Statuses whose hint is shown by default (actionable). ``skipped`` hints are
# only surfaced with ``--verbose``.
_HINT_BY_DEFAULT: set[CheckStatus] = {"warn", "fail", "not_configured"}


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """One diagnostic line: a named check, its status, and how to fix it.

    *sub* marks a check that refines the one above it (e.g. the CUDA-match
    detail under CUTracer); the text renderer indents it, while JSON keeps the
    name clean so consumers can match on it directly.
    """

    name: str
    status: CheckStatus
    detail: str = ""
    hint: str | None = None
    sub: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "hint": self.hint,
            "sub": self.sub,
        }


@dataclass
class DoctorSection:
    """A group of related checks (e.g. "System", "Profile health")."""

    name: str
    checks: list[CheckResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "checks": [c.to_dict() for c in self.checks]}


@dataclass
class DoctorReport:
    """The full diagnostic, consumed identically by every surface."""

    schema_version: str
    producer: str
    producer_version: str
    profile_path: str | None
    profile_id: str | None
    sections: list[DoctorSection]

    def all_checks(self) -> list[CheckResult]:
        return [c for s in self.sections for c in s.checks]

    def summary(self) -> dict[str, int]:
        counts = {status: 0 for status in _STATUS_TOKEN}
        for c in self.all_checks():
            counts[c.status] = counts.get(c.status, 0) + 1
        return counts

    def has_failures(self) -> bool:
        return any(c.status == "fail" for c in self.all_checks())

    def has_warnings(self) -> bool:
        return any(c.status == "warn" for c in self.all_checks())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "producer": self.producer,
            "producer_version": self.producer_version,
            "profile_path": self.profile_path,
            "profile_id": self.profile_id,
            "sections": [s.to_dict() for s in self.sections],
            "summary": self.summary(),
        }


# ---------------------------------------------------------------------------
# Small detection helpers
# ---------------------------------------------------------------------------


def _have_pkg(name: str) -> bool:
    """Return True if *name* is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _producer_version() -> str:
    try:
        from importlib.metadata import version

        return version("nsys-ai")
    except Exception:  # noqa: BLE001 — version lookup is best-effort
        return "unknown"


def _skill_conn(prof: Any) -> Any:
    """The connection a skill should run against (DuckDB cache if present)."""
    return prof.db if getattr(prof, "db", None) is not None else prof.conn


# ---------------------------------------------------------------------------
# Environment sections (no profile required)
# ---------------------------------------------------------------------------


def _check_system() -> DoctorSection:
    checks: list[CheckResult] = []

    # nsys-ai requires Python >= 3.10 at install time, so a running interpreter
    # is supported by construction; report the version for bug reports.
    checks.append(CheckResult("Python", "ok", platform.python_version()))
    checks.append(CheckResult("nsys-ai", "ok", _producer_version()))
    checks.append(CheckResult("Platform", "ok", platform.platform()))
    return DoctorSection("System", checks)


def _check_profile_support() -> DoctorSection:
    checks: list[CheckResult] = []

    # SQLite analysis is always available (stdlib).
    checks.append(CheckResult("SQLite analysis", "ok", "stdlib sqlite3"))

    # .nsys-rep conversion needs the nsys CLI on PATH.
    nsys = shutil.which("nsys")
    if nsys:
        checks.append(CheckResult(".nsys-rep conversion", "ok", nsys))
    else:
        checks.append(
            CheckResult(
                ".nsys-rep conversion",
                "warn",
                "nsys CLI not found on PATH",
                hint=(
                    "Install NVIDIA Nsight Systems to analyze .nsys-rep files directly. "
                    ".sqlite profiles work without it."
                ),
            )
        )

    # Parquet cache acceleration needs duckdb + pyarrow.
    missing = [p for p in ("duckdb", "pyarrow") if not _have_pkg(p)]
    if not missing:
        checks.append(CheckResult("Parquet cache", "ok", "duckdb + pyarrow"))
    else:
        checks.append(
            CheckResult(
                "Parquet cache",
                "warn",
                f"missing: {', '.join(missing)}",
                hint="pip install nsys-ai  (duckdb + pyarrow accelerate large profiles)",
            )
        )

    return DoctorSection("Profile support", checks)


def _check_ai_provider() -> CheckResult:
    if not _have_pkg("litellm"):
        return CheckResult(
            "AI provider (litellm)",
            "not_configured",
            "litellm not installed",
            hint="pip install 'nsys-ai[agent]' to enable ask/chat/agent features.",
        )
    try:
        from nsys_ai.chat_config import get_available_models

        models = get_available_models()
    except Exception as exc:  # noqa: BLE001 — never let a config probe crash doctor
        return CheckResult(
            "AI provider (litellm)", "warn", f"model probe failed: {exc}"
        )

    if models:
        labels = ", ".join(m["label"] for m in models[:3])
        more = f" (+{len(models) - 3} more)" if len(models) > 3 else ""
        return CheckResult(
            "AI provider (litellm)", "ok", f"{len(models)} model(s): {labels}{more}"
        )
    return CheckResult(
        "AI provider (litellm)",
        "not_configured",
        "litellm OK; no API key set",
        hint=(
            "Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY "
            "(and optionally NSYS_AI_MODEL)."
        ),
    )


def _check_cutracer() -> list[CheckResult]:
    """CUTracer split into 'installed' and 'CUDA-compatible' (ds_report style)."""
    checks: list[CheckResult] = []

    # 1. Installed: package + prebuilt .so.
    try:
        from nsys_ai.cutracer.installer import _find_cutracer_so_path

        so_path = _find_cutracer_so_path()
    except Exception:  # noqa: BLE001
        so_path = None
    pkg = _have_pkg("cutracer")

    if pkg and so_path:
        checks.append(CheckResult("CUTracer", "ok", so_path))
    else:
        missing = []
        if not pkg:
            missing.append("cutracer package")
        if not so_path:
            missing.append("cutracer.so")
        checks.append(
            CheckResult(
                "CUTracer",
                "not_configured",
                f"missing: {', '.join(missing)}",
                hint="nsys-ai cutracer install  (requires CUDA toolkit + g++)",
            )
        )

    # 2. Compatible: nvdisasm CUDA major must match the framework's CUDA build.
    #    A mismatch silently drops the hot kernel's SASS — see docs/cutracer-modal.md.
    #    Only relevant (and only worth importing torch for) when CUTracer is
    #    present; skip it entirely otherwise.
    if pkg or so_path:
        checks.append(_check_cuda_toolchain_match())
    return checks


def _framework_cuda() -> str | None:
    """Best-effort CUDA version the local ML framework was built against."""
    try:
        import torch  # type: ignore[import]

        if getattr(torch, "version", None) and torch.version.cuda:
            return str(torch.version.cuda)
    except Exception:  # noqa: BLE001 — torch is optional on an analysis host
        pass
    try:
        from nsys_ai.cutracer.installer import detect_cuda_version

        ver = detect_cuda_version()
        if ver:
            return f"{ver[0]}.{ver[1]}"
    except Exception:  # noqa: BLE001
        pass
    return None


def _check_cuda_toolchain_match() -> CheckResult:
    name = "nvdisasm <-> framework CUDA"
    if not shutil.which("nvdisasm"):
        return CheckResult(
            name,
            "skipped",
            "nvdisasm not found",
            hint="nvdisasm (CUDA toolkit) is needed only for CUTracer SASS resolution.",
            sub=True,
        )
    try:
        from nsys_ai.cutracer.installer import _run_version_cmd

        nvd = _run_version_cmd(["nvdisasm", "--version"], r"release (\d+\.\d+)")
    except Exception:  # noqa: BLE001
        nvd = None
    fw = _framework_cuda()

    if not nvd or nvd == "?":
        return CheckResult(
            name, "skipped", "could not read nvdisasm CUDA version", sub=True
        )
    if not fw:
        return CheckResult(
            name,
            "skipped",
            f"nvdisasm {nvd}; framework CUDA unknown",
            hint="Install torch on this host (or run on the training host) to compare.",
            sub=True,
        )

    nvd_major = nvd.split(".")[0]
    fw_major = fw.split(".")[0]
    if nvd_major == fw_major:
        return CheckResult(name, "ok", f"{nvd} <-> {fw}", sub=True)
    return CheckResult(
        name,
        "warn",
        f"{nvd} <-> {fw} (CUDA major mismatch)",
        hint=(
            "nvdisasm older than the framework CUDA silently drops a kernel's SASS. "
            "Match the CUDA toolkit to your framework — see docs/cutracer-modal.md."
        ),
        sub=True,
    )


def _check_optional_features() -> DoctorSection:
    checks: list[CheckResult] = []

    if _have_pkg("textual"):
        checks.append(CheckResult("GUI / chat TUI (textual)", "ok", "textual"))
    else:
        checks.append(
            CheckResult(
                "GUI / chat TUI (textual)",
                "not_configured",
                "textual not installed",
                hint="pip install 'nsys-ai[chat]' for the chat TUI.",
            )
        )

    checks.append(_check_ai_provider())
    checks.extend(_check_cutracer())
    return DoctorSection("Optional features", checks)


# ---------------------------------------------------------------------------
# Profile health section (requires an opened profile)
# ---------------------------------------------------------------------------


def _safe_profile_id(prof: Any, fallback_path: str | None) -> str | None:
    try:
        from nsys_ai.fingerprint import get_profile_id

        return get_profile_id(getattr(prof, "conn", None), fallback_path=fallback_path)
    except Exception:  # noqa: BLE001 — identity is informational, never fatal
        return None


def _nccl_kernel_count(prof: Any) -> int | None:
    """Count NCCL kernels by name — a cheap presence probe (no NVTX IEJoin).

    Returns the count, or None if the kernel table / StringIds is unavailable.
    """
    kt = getattr(getattr(prof, "schema", None), "kernel_table", None)
    if not kt:
        return None
    try:
        cur = prof.adapter.execute(
            f"SELECT COUNT(*) FROM {kt} k "  # noqa: S608 — kt is a validated table name
            "JOIN StringIds s ON k.shortName = s.id "
            "WHERE lower(s.value) LIKE '%nccl%'",
            [],
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001 — presence probe is best-effort
        return None


def _nccl_call_modes(prof: Any, device: int) -> str | None:
    """eager / inductor-captured / temporal-only split via the breakdown skill.

    Runs ``nccl_compile_context_breakdown``, which joins ``nvtx_kernel_map`` and
    can take minutes on an NVTX-heavy profile — so doctor only calls this under
    ``--deep``, never on the default fast path.
    """
    try:
        from nsys_ai.skills.registry import get_skill

        skill = get_skill("nccl_compile_context_breakdown")
        if skill is None:
            return None
        rows = skill.execute(_skill_conn(prof), device=device)
    except Exception:  # noqa: BLE001
        return None

    buckets = {r["bucket"]: r for r in rows if "bucket" in r}
    if not buckets:
        return None
    return (
        f"eager {buckets.get('eager', {}).get('pct', 0):.0f}% / "
        f"inductor {buckets.get('inductor_captured', {}).get('pct', 0):.0f}% / "
        f"temporal {buckets.get('temporal_only', {}).get('pct', 0):.0f}%"
    )


def _profiler_overhead_pct(prof: Any, span_ns: int) -> float | None:
    """Profiler overhead as a percent of the full profile span, or None."""
    if span_ns <= 0:
        return None
    try:
        from nsys_ai.skills.base import compute_profiler_overhead_ns

        overhead_ns = compute_profiler_overhead_ns(_skill_conn(prof))
    except Exception:  # noqa: BLE001
        return None
    return overhead_ns / span_ns * 100.0


def _gpu_model_check(meta: Any, devices: list[int]) -> CheckResult:
    names = {
        (meta.gpu_info[d].name or "").strip()
        for d in devices
        if d in meta.gpu_info
    }
    named = sorted(n for n in names if n and n.lower() != "unknown")
    if named:
        return CheckResult("GPU model", "ok", ", ".join(named))
    return CheckResult(
        "GPU model",
        "warn",
        "unknown",
        hint="GPU model missing from CUPTI TARGET_INFO; MFU / efficiency cannot be computed.",
    )


def _overhead_check(pct: float | None) -> CheckResult:
    if pct is None:
        return CheckResult(
            "Profiler overhead", "skipped", "no profiler-overhead table in profile"
        )
    if pct > 100.0:
        # Physically impossible — a degenerate/very short profile, not a real
        # overhead problem (cf. silencing impossible profile-overhead findings).
        return CheckResult(
            "Profiler overhead",
            "skipped",
            f"{pct:.0f}% (unreliable; profile too short to measure)",
        )
    if pct > _OVERHEAD_FAIL_PCT:
        return CheckResult(
            "Profiler overhead",
            "fail",
            f"{pct:.1f}%",
            hint=(
                f"Overhead > {_OVERHEAD_FAIL_PCT:.0f}% badly distorts timings; "
                "re-profile with fewer trace domains or a capture range."
            ),
        )
    if pct > _OVERHEAD_WARN_PCT:
        return CheckResult(
            "Profiler overhead",
            "warn",
            f"{pct:.1f}%",
            hint=f"Overhead > {_OVERHEAD_WARN_PCT:.0f}% may skew kernel timings.",
        )
    return CheckResult("Profiler overhead", "ok", f"{pct:.1f}%")


def _check_profile_health(prof: Any, device: int | None, *, deep: bool = False) -> DoctorSection:
    checks: list[CheckResult] = []
    meta = prof.meta
    devices = list(meta.devices or [])
    dev = device if device is not None else (devices[0] if devices else 0)
    span_ns = (meta.time_range[1] - meta.time_range[0]) if meta.time_range else 0

    checks.append(CheckResult("Duration", "ok", f"{span_ns / 1e9:.1f}s"))
    # GPUs — COUNT(DISTINCT deviceId), not the fingerprint heuristic.
    checks.append(CheckResult("GPUs", "ok", str(len(devices))))
    checks.append(_gpu_model_check(meta, devices))

    nccl_count = _nccl_kernel_count(prof)
    has_nccl = bool(nccl_count and nccl_count > 0)

    # Multi-GPU: device count is authoritative; NCCL on a single device hints
    # at a per-rank profile from a larger distributed run.
    if len(devices) > 1:
        checks.append(CheckResult("Multi-GPU run", "ok", f"yes ({len(devices)} GPUs)"))
    elif has_nccl:
        checks.append(
            CheckResult("Multi-GPU run", "ok", "single device; NCCL present (per-rank profile?)")
        )
    else:
        checks.append(CheckResult("Multi-GPU run", "ok", "no (single GPU)"))

    # NVTX presence (coverage percentage is a future enhancement).
    if meta.nvtx_count and meta.nvtx_count > 0:
        checks.append(CheckResult("NVTX events", "ok", f"{meta.nvtx_count} events"))
    else:
        checks.append(
            CheckResult(
                "NVTX events",
                "warn",
                "none",
                hint=(
                    "No NVTX annotations — layer/region attribution and code "
                    "tracing will be weak. Annotate with torch.cuda.nvtx or nvtx.annotate."
                ),
            )
        )

    if nccl_count is None:
        checks.append(CheckResult("NCCL events", "skipped", "could not probe NCCL kernels"))
    elif has_nccl:
        checks.append(CheckResult("NCCL events", "ok", f"{nccl_count} kernels"))
        # The eager/inductor/temporal split joins the NVTX map and can take
        # minutes on an NVTX-heavy profile, so keep it off the fast path.
        if deep:
            modes = _nccl_call_modes(prof, dev)
            if modes:
                checks.append(CheckResult("NCCL by call mode", "ok", modes, sub=True))
            else:
                checks.append(
                    CheckResult("NCCL by call mode", "skipped", "classification unavailable", sub=True)
                )
        else:
            checks.append(
                CheckResult(
                    "NCCL by call mode",
                    "skipped",
                    "deep check (use --deep)",
                    hint="--deep classifies NCCL as eager vs inductor-captured; slow on NVTX-heavy profiles.",
                    sub=True,
                )
            )
    else:
        checks.append(CheckResult("NCCL events", "ok", "none"))

    checks.append(_overhead_check(_profiler_overhead_pct(prof, span_ns)))

    # RunSpec presence (forward-compatible; recording lands with `profile --runspec`).
    checks.append(
        CheckResult(
            "RunSpec attached",
            "not_configured",
            "no",
            hint="RunSpec recording (nsys-ai profile --runspec) is not yet available.",
        )
    )

    return DoctorSection("Profile health", checks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_doctor(
    profile_path: str | None = None,
    *,
    profile: Any = None,
    device: int | None = None,
    include_env: bool = True,
    include_health: bool = True,
    deep: bool = False,
) -> DoctorReport:
    """Build a :class:`DoctorReport`.

    *profile_path* — open this profile for the health section.
    *profile* — an already-open ``Profile`` (used by in-process surfaces like
      the web GUI / TUI); takes precedence over *profile_path*.
    *include_env* / *include_health* — toggle the environment and profile-health
      sections so a surface can request only what it needs.
    *deep* — also run slow checks (e.g. the NCCL eager/inductor call-mode split,
      which joins the NVTX map). Off by default to keep doctor a fast triage.
    """
    sections: list[DoctorSection] = []
    if include_env:
        sections.append(_check_system())
        sections.append(_check_profile_support())
        sections.append(_check_optional_features())

    profile_id: str | None = None
    if include_health and (profile is not None or profile_path):
        prof = profile
        own = False
        try:
            if prof is None:
                from nsys_ai import profile as _profile_mod

                prof = _profile_mod.open(profile_path)
                own = True
        except Exception as exc:  # noqa: BLE001 — a bad path/file is a real finding
            sections.append(
                DoctorSection(
                    "Profile health",
                    [
                        CheckResult(
                            "Open profile",
                            "fail",
                            str(exc),
                            hint="Check the path and that the file is a valid Nsight profile.",
                        )
                    ],
                )
            )
            prof = None
        if prof is not None:
            try:
                profile_id = _safe_profile_id(prof, profile_path)
                # _check_profile_health is internally best-effort; each enrichment
                # degrades to a skipped check rather than raising.
                sections.append(_check_profile_health(prof, device, deep=deep))
            finally:
                if own:
                    try:
                        prof.close()
                    except Exception:  # noqa: BLE001
                        pass

    return DoctorReport(
        schema_version=SCHEMA_VERSION,
        producer="nsys-ai",
        producer_version=_producer_version(),
        profile_path=profile_path,
        profile_id=profile_id,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Text renderer (CLI)
# ---------------------------------------------------------------------------


def format_doctor_text(report: DoctorReport, *, verbose: bool = False) -> str:
    """Render a report as aligned, human-readable text."""
    lines: list[str] = []
    if report.profile_path:
        lines.append(f"Profile: {report.profile_path}")
        if report.profile_id:
            lines.append(f"  id: {report.profile_id}")
        lines.append("")

    # Column width for check names across the whole report (sub-checks indent
    # by two spaces, so account for that when sizing the column).
    _sub_indent = 2
    width = max(
        (len(c.name) + (_sub_indent if c.sub else 0) for c in report.all_checks()),
        default=0,
    )

    for section in report.sections:
        lines.append(section.name)
        for c in section.checks:
            token = _STATUS_TOKEN.get(c.status, c.status)
            detail = f"  {c.detail}" if c.detail else ""
            label = ("  " if c.sub else "") + c.name
            lines.append(f"  {label.ljust(width)}  [{token}]{detail}")
            show_hint = c.hint and (c.status in _HINT_BY_DEFAULT or verbose)
            if show_hint:
                lines.append(f"  {' ' * width}     -> {c.hint}")
        lines.append("")

    s = report.summary()
    lines.append(
        "Summary: "
        f"{s.get('ok', 0)} ok, {s.get('warn', 0)} warning, {s.get('fail', 0)} failed, "
        f"{s.get('not_configured', 0)} not configured, {s.get('skipped', 0)} skipped"
    )
    if report.profile_path is None:
        lines.append("Pass a profile (nsys-ai doctor profile.sqlite) for a health summary.")
    lines.append("Capture-side checks (can this host profile?): run  nsys status -e")
    return "\n".join(lines)
