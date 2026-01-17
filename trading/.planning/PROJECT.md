# Trading Documentation Alignment

## What This Is

This is a documentation alignment effort for the DASHI trading stack. The goal is to bring the recorded theory notes in this repo into line with the current PDF sources without inventing new semantics.

## Core Value

Keep the written framework faithful to the PDFs and avoid over-committing definitions that are not explicitly supported.

## Requirements

### Validated

(None yet - align and confirm)

### Active

- [ ] Update M9 and supervision text to match PDF wording (no extra derivations)
- [ ] Clarify all-red spine as a failure signature without hard-coded triple equality
- [ ] Note the special code layer (VOID/PARADOX) and severity ordering
- [ ] Keep doc updates scoped to theory text; do not change trading code

### Out of Scope

- Any changes to trading execution behavior or strategy code - this is a doc-only effort
- Rewriting the entire theory corpus - only the sections in question

## Context

The current notes include a detailed M9 explanation with explicit triple checks and semantic labels that are not explicitly claimed in the PDFs. The user provided corrections that should be reflected in the write-up.

## Constraints

- **Scope**: Docs only - no code changes
- **Fidelity**: Stay within PDF language and avoid new derivations
- **Traceability**: Record the update in TODOs and compactified context

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Update doc sections only | Keep behavior unchanged while fixing theory text | ✓ Good |

---
*Last updated: 2026-01-17 after doc alignment update*
