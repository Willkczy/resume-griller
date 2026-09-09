# Refactor handoff

Last updated: 2026-09-10.

## Start here

Read [AGENTS.md](../AGENTS.md), [AI_ONBOARDING.md](../AI_ONBOARDING.md),
[the canonical roadmap](../DEVELOPMENT_ROADMAP.md) and
[the assessment](PROJECT_REASSESSMENT.md). Inspect `git status`, current branch and
recent commits before editing. Do not assume an old chat or remote machine has the
same worktree. The roadmap's M0–M9 numbering supersedes the previous M0–M8 plan.

## Current state

The owner approved documenting the learning-oriented plan and pushing the reviewed
working tree for use on another machine. This is not authorization to implement all
milestones, select optional tracks, deploy publicly, or rewrite history.

Planning/documentation is complete. No functional refactor fixes have been implemented
by this planning change. Earlier pending edits align documentation, code comments and
example configuration with the implemented runtime; they must not be interpreted as
completion of M0 or repair of the startup/scoring defects.

| Milestone | Engineering status | Owner learning status | Evidence / blocker |
|---|---|---|---|
| M0 | Open; documentation aligned | Not recorded | Provenance/data inventory and voice contract unresolved |
| M1 | Not started | Not recorded | Depends on M0 |
| M2 | Not started | Not recorded | Depends on M1 |
| M3 | Not started | Not recorded | No human-reviewed benchmark yet |
| M4 | Not started | Not recorded | No evaluation ledger/report yet |
| M5 | Not started | Not recorded | Retry/restart guarantees not established |
| M6 | Not started | Not recorded | Shared UI controller/support cleanup not implemented |
| M7 | Not selected | Not recorded | Optional; do not start automatically |
| M8 | Not selected | Not recorded | Optional; depends on M7 |
| M9 | Not started | Not recorded | Real-user/public-beta gate remains closed |

## Exact next slice: M0 inventory and provenance

1. Inspect the tracked PDF/TXT fixtures, `LLM_Inference.ipynb`, nested upload paths,
   `.gitignore`, and relevant history without printing contact details or secrets.
2. Create `docs/DATA_INVENTORY.md` documenting categories, storage locations, copies,
   processors, retention/deletion gaps and provenance status. This file does not exist
   yet; create it as implementation evidence, not as a speculative completed checklist.
3. Ask the owner for fixture origin/synthetic status only where repository evidence
   cannot establish it. Mark unresolved artifacts unknown. Prepare separately scoped
   remediation if real PII is confirmed; no force push/history rewrite or runtime deletion.
4. Have the owner draw upload/answer flows and explain one synthetic checkpoint.
   Agent can provide a walkthrough and review, but must not claim that exercise happened.
5. Next bounded M0 slice: choose and test a text-only default or properly optional voice
   dependency contract. Inspect config, `.env.example`, TTS/STT initialization and UI
   capability reporting. Do not implement streaming voice.

Until provenance is established, use newly authored synthetic fixtures for tests and do
not send tracked resume content to external model services. Continue independent read-only
inventory while waiting on provenance; do not guess an answer to unblock a gate.

## Known defects to reproduce after M0

- Startup: session `_invoke_graph` overwrites start with initial null action; null error
  content then fails response validation. Use real graph + mocked model services.
- Follow-ups: evaluator/current-question APIs use main-question index, not actual turn.
- Standard evaluation parser accepts empty objects and unbounded scores.
- Routing policies disagree; evaluation history is cleared on advance.
- WS lacks REST-equivalent lifecycle checks; reconnect/double submission are not safe.
- Result counts and custom-mode derived scores cannot support quality claims.

See the assessment and milestone acceptance checks for details. Do not fix these as
unrelated cleanup during the documentation/provenance slice.

## Verification evidence

Assessment on 2026-09-09:

- Existing backend suite: 21 passed (providers mocked).
- TypeScript `--noEmit --incremental false`: passed.
- Frontend ESLint: passed.
- Targeted in-memory startup reproduction: failed with response validation before LLM
  invocation, as described above. Empty JSON gave 0.425; out-of-range dimensions gave 100.
- Coverage, production build, live providers, voice and browser usability were not
  revalidated in that assessment. Do not copy the earlier coverage/build baseline as fresh.

Planning handoff on 2026-09-10:

- Local Markdown link targets/code fences, milestone references and `git diff --check`:
  passed. Backend AST comparison confirmed pending Python edits are comments/docstrings
  only; root dependency and optional-dependency lists are unchanged.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest tests/
  -p no:cacheprovider -q`: 21 passed (command entered as one line).
- `.venv/bin/ruff check backend rag/retriever.py tests --config pyproject.toml`: passed.
- `.venv/bin/black --check backend rag/retriever.py tests`: passed.
- From frontend: `./node_modules/.bin/tsc --noEmit --incremental false` and
  `npm run lint`: passed.
- Focused recognizable-credential/private-key scan of added outgoing diff lines found
  no matches. This does not establish fixture provenance or constitute a full audit.
- Production build, coverage, live providers and browser/voice testing were not rerun.

These checks validate the documentation handoff and existing baseline, not milestone
completion. Machine-local `.claude/settings.local.json` changes are excluded from this
handoff commit. The branch also includes earlier stabilization commits; inspect Git
history for their implementation scope. No new sensitive runtime artifacts are included.

## New-machine workflow

Use the branch named in the push confirmation or inspect `git branch -r`; do not assume
`main` contains this plan. For an existing clone, fetch, switch to the pushed branch and
fast-forward pull after preserving any local edits. For a new clone, clone the repository
and switch to that remote branch. Inspect status and commit identity before work.

Follow [CONTRIBUTING](CONTRIBUTING.md) for `uv sync --extra dev` and frontend `npm ci`.
Create local `.env` / frontend `.env.local` from examples only if absent; never overwrite
existing credentials. Runtime uploads, parsed resumes, checkpoints, credentials and
installed dependencies are not portable through Git. Synthetic mock tests need no
provider key; live new-upload parsing needs the configured Groq credential.

Suggested prompt for the next agent:

> Read AGENTS.md, AI_ONBOARDING.md, DEVELOPMENT_ROADMAP.md and
> docs/REFACTOR_HANDOFF.md. Inspect the current worktree. Start the next eligible M0
> inventory/provenance slice, explaining what I should investigate myself and what you
> will handle. Use synthetic data, preserve existing changes/runtime data, and do not
> implement later milestones or declare my learning exercise complete without evidence.

## Record after each implementation slice

Append a dated entry with:

- milestone/slice and code commit (do not create self-referential commit hashes);
- expected invariant and files changed;
- commands actually run, result and sanitized evidence links;
- benchmark versions/delta, or why model quality was not measured;
- owner exercise: pending, demonstrated, or explicitly delegated (with remaining learning);
- engineering gate status, unresolved risks and exact next action.

Keep current status table synchronized. Updating a checkbox is not acceptance evidence.
