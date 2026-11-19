---
name: sdd-audit
description: Audits current git diff against the active OpenSpec specification, project licenses (no GPL/AGPL), async safety, and test coverage.
---

# SDD Audit Checklist

When invoked with `/sdd-audit`, execute the following steps sequentially:

1. **Locate Active Spec:** Read the relevant proposal/spec/tasks files inside `openspec/changes/` (or ask which change folder to audit if multiple exist).
2. **Inspect Diff:** Run `git status` and `git diff` to see all modified/added files.
3. **Requirement Traceability:** Produce a table mapping every requirement ID from the spec to the exact file/function implementing it and the test covering it. Flag any missing or partially implemented requirements.
4. **License & Security Check (K-13 / Closed Contour):**
   - Inspect `pyproject.toml`, `uv.lock`, and `docker-compose*.yml`.
   - Flag any GPL/AGPL library or container, hardcoded secrets, or unapproved external network calls outside `egress-gateway`.
5. **Async & Performance Check:**
   - Verify no blocking I/O or PyTorch forward passes run directly on the FastAPI event loop.
   - Verify timeouts and retries exist for all inter-service HTTP calls.
6. **Run Quality Gates:** Execute `uv run ruff check .` and `uv run pytest`. Report pass/fail counts verbatim. Do NOT mark as passed if tests are mocked to hide real failures.