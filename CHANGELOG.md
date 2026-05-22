# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `nccl_compile_context_breakdown` skill — classifies NCCL kernels by their
  **leaf** NVTX label into eager / inductor_captured / temporal_only buckets.
  Decides whether a collective perf fix lives in user code (stream wrap,
  `async_op=True`) or in `torch._inductor.config` (functional collectives,
  `reorder_for_compute_comm_overlap`).
- `nsys-ai diff --format json` emits a v0.1 envelope (`schema_version`,
  `producer`, `producer_version`, `diff_id`) plus top-level `verdict`,
  `comparability_confidence`, step-time `category_attribution` (HTA
  convention: `overlap_ms` counts as compute, `nccl_only_ms` is exposed
  comm), and content-derived `profile_id` on each side (PR #130).
- `nccl_payload_breakdown` skill — decodes NVTX typed-payload `binaryData`
  to extract real NCCL message sizes, peer ranks, and communicator IDs
  (PR #128).

### Changed

- `nvtx_kernel_map` docstring now documents temporal-containment semantics:
  `nvtx_text` is the **leaf** NVTX label (innermost open scope at kernel
  launch), and `nvtx_path` containment is temporal, not lexical.
  Classifying by ancestor-path substring (e.g. `LIKE '%Torch-Compiled%'`)
  can pick up kernels whose enclosing scope merely happened to still be
  open — not kernels that the enclosing tool actually traced.

### Fixed

- `fingerprint.distributed` falls back to `COUNT(DISTINCT deviceId) > 1`
  so multi-GPU single-node profiles report `distributed=True`;
  `overlap_breakdown` surfaces `present_devices` and warns when the caller
  did not pass `-p device=N` on a multi-device profile (PR #127).
- Parquet cache builds are `flock`-serialized to prevent concurrent crashes
  when two `nsys-ai` invocations target the same cache directory (PR #122).

### Performance

- `overlap_analysis` SQL sweep-line — 3.2× speedup on the analysis call
  (PR #125).
- NVTX layer breakdown 43s → 1.4s via `nvtx_high` parquet + `ORDER BY`
  (PR #123).
- `profile_health_manifest` auto-trims long profiles to a representative
  window (PR #124).

### Docs

- Evidence schema reference brought up to v0.1: new optional fields,
  envelope keys, supporting types, enriched example (PR #119).
- `docs/agent_skills/.../perf_budget.md` §9 rewritten for NVTX-heavy
  profiles (PR #126).

## [0.2.3] — 2026-05-12 and earlier

See `git log v0.2.3` for changes prior to the introduction of this
changelog.

[Unreleased]: https://github.com/GindaChen/nsys-ai/compare/v0.2.3...HEAD
[0.2.3]: https://github.com/GindaChen/nsys-ai/releases/tag/v0.2.3
