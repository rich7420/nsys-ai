<!--
Thanks for sending a PR! Keep this template tight — empty sections are fine.
Delete any section that does not apply, but please at least keep Summary and Test plan.
-->

## Summary

<!-- 1–3 bullets: what this PR changes and why. -->

## Changes

<!-- Group by area / file. Bullets only — no narrative. -->

## Backward compatibility

<!-- Yes / No. If "no", state the breaking change and the migration path explicitly. -->

## Test plan

- [ ] Tests added or updated for the new behavior
- [ ] `python -m nsys_ai --help` — CLI loads without error
- [ ] `pytest tests/ -v --tb=short` passes locally
- [ ] `ruff check src/ tests/` clean
- [ ] Manually exercised the changed CLI / GUI path (if applicable)

## Out of scope

<!-- What this PR deliberately does NOT do. List follow-up PRs / issues. -->

## Linked issues

<!-- Closes #N, Refs #N. Remove this section if none. -->
