# Refactor handoff

Last updated: 2026-10-13.

## Start here

Read [AGENTS.md](../AGENTS.md), [AI_ONBOARDING.md](../AI_ONBOARDING.md),
[the canonical roadmap](../DEVELOPMENT_ROADMAP.md) and
[the assessment](PROJECT_REASSESSMENT.md). Inspect `git status`, current branch and
recent commits before editing. Do not assume an old chat or remote machine has the
same worktree. The roadmap's M0–M9 numbering supersedes the previous M0–M8 plan.

## Current state

The owner explicitly delegated implementation, testing and documentation for the
current milestone. A separate learning agent guides optional exercises; pending
learning is not an engineering blocker. Do not automatically start the next milestone.
The owner authorized isolated rewriting, then the five remote branch updates and
local adoption on 2026-09-20. Those actions are complete. The owner subsequently
authorized local implementation commits; see the commit record below. On 2026-10-10
the owner authorized pushing the latest work and updating GitHub to reflect project
status. On 2026-10-13 the owner authorized fixing the PR review findings, merging
PR #6 and continuing M2 stage 4. Deployment and the deferred Support request remain
outside that authorization.

M0's local synthetic-data baseline, clean setup, and local/remote-branch history
sanitation are verified. Five GitHub PR refs still retain historical artifacts.
On 2026-09-20 the owner explicitly deferred that host-cleanup work and authorized
M1. This is a scoped dependency exception for local synthetic-data implementation,
not a claim of full historical erasure or permission for broader distribution.
Do not send the prepared Support request while it is deferred. Public beta and
real-user processing remain blocked by M9; resolve retained PR data before release.

| Milestone | Engineering status | Owner learning status | Evidence / blocker |
|---|---|---|---|
| M0 | Local baseline accepted for M1; host cleanup deferred by owner | Pending / not assessed | Five GitHub PR refs still retain data; release/distribution issue remains open |
| M1 | Engineering complete | Pending / not assessed | 73 tests passed at M1; real graph/engine integration, lifecycle parity and active-turn evidence below |
| M2 | Stages 1–3 and PR review corrections complete; stage 4 authorized | Pending / not assessed | Resume storage regressions pass; metadata/timings and CI remain open |
| M3 | Not started; Jev comparison planned after M2 | Not recorded | No human-reviewed benchmark yet |
| M4 | Not started | Not recorded | No evaluation ledger/report yet |
| M5 | Not started | Not recorded | Retry/restart guarantees not established |
| M6 | Not started | Not recorded | Shared UI controller/support cleanup not implemented |
| M7 | Not selected | Not recorded | Optional; do not start automatically |
| M8 | Not selected | Not recorded | Optional; depends on M7 |
| M9 | Not started | Not recorded | Real-user/public-beta gate remains closed |

## Current scope: merge reviewed corrections, then M2 stage 4

The owner authorized M2 with a local commit and a stop between each stage, then
explicitly delegated stage 3. Stages 1–3 enforce runtime/public model contracts,
remove fabricated fallback scores, halt progression on unavailable evaluation and
restore outcomes/submitted drafts through the UI. Provider metadata/timings and CI
remain open. The 2026-10-13 review findings are fixed with synthetic regressions;
see the correction record below. The owner authorized merging PR #6 and then
implementing stage 4, with the existing review stop afterward. Learning remains
separately pending.

On 2026-09-22 the owner requested adding the Jev comparison to the plan and stated
they would return to finish M2 first. That September planning update performed no
M2 or Jev implementation. The planned M3 comparison is recorded in the
[roadmap](../DEVELOPMENT_ROADMAP.md#planned-m3-comparison-jev-answer-gap-detection).
Keep M2 as the immediate priority, then evaluate Jev without changing interview
behavior before considering adoption.

Deferred M0 item: GitHub PR refs 1–5 and the draft Support request remain documented
in HISTORY_REMEDIATION and GITHUB_HISTORY_CLEANUP_REQUEST. The owner chose to defer
that work; do not treat it as an implementation dependency for M1 or as resolved.

## Remaining engineering limitations

- Model outputs have runtime/public validation and dedicated non-scored outcome UI.
  Provider metadata/timings and deterministic CI remain M2. Custom/Hybrid compact
  evaluation is explicitly unsupported, not scalar-derived.
- The latest accepted feedback survives advancement until the next mutating command;
  there is no durable per-turn evaluation ledger and reports remain count-based (M4).
- Sequential validation is not atomic concurrency control or idempotency (M5).
- Checkpointed conversation, failed outcomes and submitted drafts restore on reconnect.
  Partial-node interruption, unsent-draft persistence and process-restart recovery are
  not guaranteed. Missing legacy follow-up prompts reject commands with 409.
- Model-quality calibration and human-reviewed benchmark evidence do not yet exist (M3).
- M0 GitHub PR-reference cleanup is deferred; public beta/real-user processing stays blocked.

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

> Read the roadmap and this handoff; inspect the current worktree. Implementation is
> delegated, and learning is tracked separately. M1 and M2 stages 1–3 are implemented.
> The 2026-10-13 review corrections are verified. The owner authorized merging PR #6
> and continuing M2 stage 4. Inspect the latest delivery record before acting; stop
> for review after stage 4. M0 host cleanup remains deferred. Preserve runtime data
> and existing edits.

## Record after each implementation slice

Append a dated entry with:

- milestone/slice and code commit (do not create self-referential commit hashes);
- expected invariant and files changed;
- commands actually run, result and sanitized evidence links;
- benchmark versions/delta, or why model quality was not measured;
- owner exercise: pending, demonstrated, or explicitly delegated (with remaining learning);
- engineering gate status, unresolved risks and exact next action.

Keep current status table synchronized. Updating a checkbox is not acceptance evidence.

## 2026-09-12 — M0 slice 1: inventory and test-data isolation

Implementation delegated explicitly by the owner, including tests and documentation;
a separate learning agent owns exercises. Pending learning does not block technically
eligible work. No authorization to commit, push, delete runtime data or rewrite history.

- Added `docs/DATA_INVENTORY.md`: artifacts, trust boundaries, persisted copies,
  deletion gaps and explicit unknown provenance. Owner provenance question is pending.
- Changed `.gitignore` to cover nested upload/parsed/Chroma paths. Before the change,
  `git check-ignore --no-index` missed nested uploads and parsed Markdown.
- Replaced `tests/test_parser.py`'s tracked-PDF/contact-printing smoke test with a
  temporary, newly authored synthetic PDF and real extraction assertions.
- Ran `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest
  tests/test_parser.py -p no:cacheprovider -q`: 1 passed; PyMuPDF deprecation warnings.
- Inspected path history, compared PDF hashes (identical), and counted notebook cells
  and outputs (4 / 0). No historical PII clearance or synthetic provenance inferred.
- No AI behavior change; no benchmark version exists or model-quality delta measured.
- Learning-agent handoff: trace upload -> normalization -> Markdown -> checkpoints;
  compare workflow state with retained copies; inspect `resume.py`, `rag/retriever.py`,
  `graph/checkpointer.py`. Discuss why ignore rules cannot erase tracked/history blobs.
  Owner learning remains pending/not assessed.
- Next eligible independent slice: text-only voice default and disabled-path checks.
  M0 stays open pending provenance and reproducible setup evidence; M1 is not started.

## 2026-09-12 — M0 slice 2: authorized removals and text-only contract

Implementation complete for this slice; M0 remains open.

- Owner requested removing the four TXT fixtures, both PDF copies and notebook;
  separately approved removal of `data/exported_prompts.json`. Those eight paths
  are removed from the working tree. No provenance assertion, history rewrite,
  commit, push or unrelated runtime deletion occurred. The inventory records the
  historical-copy limitation and a bounded future review/remediation scope.
- `backend/app/config.py` defaults voice off. The existing REST/WS guards already
  avoid voice initialization when disabled; no orchestration or wire-schema change
  was needed. `tests/test_voice_baseline.py` adds actual HTTP route checks and a
  real in-memory graph/WS rejection-and-ping check proving no checkpoint mutation.
- `ResumeUploader.tsx` and the interview page default to chat. The page falls back
  to chat if voice capabilities are unavailable; `InterviewRoom.tsx` hides recording
  until STT availability is reported. Existing `VoiceStatus` TypeScript/API fields
  are reused. No mandatory camera/microphone permission is introduced for text.
- README, onboarding, contributing guide, inventory and
  `docs/decisions/0001-text-only-baseline.md` describe the implemented contract.
- Regression before config change: `tests/test_voice_baseline.py` had 1 failure
  (default unexpectedly true), 6 passes. Final focused run after adding WS coverage:
  8 passed. Final full command `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
  .venv/bin/python -m pytest tests/ -p no:cacheprovider -q`: 29 passed;
  five PyMuPDF deprecation warnings.
- `.venv/bin/ruff check backend rag/retriever.py tests --config pyproject.toml` and
  `.venv/bin/black --check backend rag/retriever.py tests`: passed.
- From `frontend`, `./node_modules/.bin/tsc --noEmit --incremental false` and
  `npm run lint`: passed after final frontend edits.
- `npm run build`: first failed because the sandbox blocked Turbopack's local
  worker port; authorized rerun outside sandbox passed. Build warns that
  baseline-browser-mapping data is old. No dependency upgrade was made for that warning.
- Root/nested synthetic upload, parsed-resume and Chroma probe paths are ignored
  (`git check-ignore --no-index`). `git diff --check` passed during review.
- No AI behavior was changed. Dataset/benchmark version and model-quality delta
  are not applicable/not measured; deterministic tests are not quality benchmarks.

### Clean-setup evidence and limitations

A temporary source copy contains pyproject/lock/README, backend, rag, tests and
tracked frontend sources, excluding removed data and local environment files.
Its path is saved in `/tmp/resume-griller-m0-setup-path`; no existing environment
was replaced. This verifies a clean source-copy install, not a committed checkout.

- `uv sync --frozen --extra dev --offline --project <temporary-copy>` initially
  hit sandbox cache access; authorized rerun found the locked Torch wheel absent.
- `uv sync --frozen --extra dev --project <temporary-copy>` downloaded dependencies
  but failed on a pypdfium2 wheel download timeout. Longer-timeout retry recorded below.
- `npm ci --prefix <temporary-copy>/frontend --cache /tmp/resume-griller-npm-cache
  --no-audit --no-fund` failed. Its local log shows repeated registry.npmjs.org
  ENOTFOUND errors followed by npm's `Exit handler never called` error. Clean
  frontend installation remains unverified; existing-install type/lint/build passes
  do not substitute for it. Retry in an environment with working registry DNS.

### Remaining prerequisites and learning-agent handoff

- Historical artifact disposition remains unresolved. Removal does not establish
  synthetic origin or erase historical copies. Follow the inventory's prepared scope;
  history rewriting/remote changes require separate authorization.
- Retry unsuccessful clean setup, then run checks in those new environments.
- M1 startup/follow-up issues were reconfirmed by current code inspection (initial
  state overwrites action; evaluator indexes the main question). They were not fixed
  or freshly reproduced as behavioral tests in this M0 slice. Other historical
  model-quality/lifecycle findings remain reproduction targets, not new test claims.
- Live providers, browser/device behavior, optional voice provider success/failure,
  checkpoint restart recovery and model-quality benchmarks were not verified.
  Public beta/real-user processing remain blocked until M9.
- Learning-agent topics: fail-closed capability UI versus provider health; lazy client
  initialization; a real graph at a WS test seam; synthetic PDF extraction; why a
  clean install and an existing-install build provide different evidence. Relevant
  paths: config, voice/WS routes, `tests/test_voice_baseline.py`, interview page,
  `tests/test_parser.py`, and the data inventory. Owner learning: pending/not assessed.
- Next action: resolve historical distribution disposition and clean-setup failures.
  Do not start M1 until M0 engineering gates are resolved and the owner authorizes
  the next milestone. No learning exercise is a prerequisite for implementation.

Clean-setup follow-up:

- `UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=4 uv sync --frozen --extra dev
  --project <temporary-copy>`: passed, installing the locked dependencies into a
  new temporary virtual environment. An automatic permission-review timeout on
  this retry was retried successfully; no permission denial remains.
- Temporary-copy `.venv/bin/python` with an actual application `TestClient` lifespan:
  startup/shutdown passed; `/health` and `/api/v1/voice/status` returned 200 and
  reported voice disabled. No dotenv file or voice credentials were required.
- `npm ci --offline --prefix <temporary-copy>/frontend --no-audit --no-fund`:
  failed with ENOTCACHED for Zustand. Together with registry DNS failures above,
  this leaves clean frontend setup unverified; no lockfile changes were made.
- In the temporary copy, `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
  .venv/bin/python -m pytest tests/ -p no:cacheprovider -q`: 29 passed, five
  PyMuPDF deprecation warnings. Temporary-environment Ruff and Black checks using
  the same commands above also passed. Python clean setup is now verified;
  the remaining setup issue is frontend installation.
- Final local Markdown target check found no missing links in the changed guides;
  `git diff --check` passed. Existing `.claude/settings.local.json` changes preserved.


## 2026-09-15 — M0 slice 3: clean frontend verification and historical review

Both requested tasks completed. No application code changed in this slice.
M0 remains open for historical data disposition/remediation, not setup or learning.

- Inspected current branch, dirty worktree and recent commits; preserved all existing
  changes, including `.claude/settings.local.json`. No commit/push/rewrite performed.
- Created a fresh temporary frontend source copy from existing tracked source paths,
  using current worktree contents. Excluded deleted artifacts, data, local dotenv,
  node_modules and build output. Path recorded in `/tmp/rg-m0-frontend-path`.
- `npm ci --prefix <temporary-frontend> --cache /tmp/rg-m0-npm-cache --no-audit
  --no-fund`: passed (430 packages). No existing installation or lockfile changed.
- From that fresh directory: `./node_modules/.bin/tsc --noEmit --incremental false`,
  `npm run lint`, and `npm run build`: all passed. Build ran with permission for
  Turbopack's local worker port, without `.env.local`. Existing stale
  baseline-browser-mapping warning persists; no dependency upgrade was introduced.
- Historical review: `git rev-list --all`, `git ls-tree -r -z`, `git cat-file blob`,
  and `git for-each-ref --format=%(refname)` traversed 71 locally reachable commits.
  Selected eight removed paths resolved to seven unique blobs; identical copies
  matched only the known PDF pair. PDF extraction stayed in memory via PyMuPDF.
- Sanitized pattern counts found contact-like data in four TXT fixtures, PDF, and
  notebook source. No selected blob matched the limited credential patterns used;
  this is not a comprehensive secret/PII clearance. Remote-tracking refs are local
  snapshots; no fetch or remote inspection occurred. Origin remains unknown.
- Added `docs/HISTORICAL_DATA_REVIEW.md` with blob identities, reachability, method,
  limitations and concrete remediation scope; updated inventory and this handoff.
- No backend tests rerun for these documentation/verification changes. Prior Python
  clean-install evidence remains dated 2026-09-12. No AI behavior or benchmark change;
  model-quality delta not applicable/not measured. Browser/live providers not tested.
- Learning-agent handoff: Git tree removal versus blob reachability, source data in
  notebooks without outputs, regex evidence versus provenance, and clean locked
  dependency installation. Relevant docs: historical review and inventory. Owner
  learning remains pending/not assessed and does not block technical work.
- Exact next action: obtain authorization for the historical remediation scope (or
  verified provenance that changes disposition), then execute/verify that scope.
  History rewriting/force pushes remain unauthorized. Report M0 completion only
  after its engineering gate is satisfied; do not automatically start M1.
- Final checks: `git diff --check` passed; local Markdown link checks for the three
  changed documents passed. Historical review outputs contain only artifact paths,
  blob identities, counts and ref names, not contact values or source payloads.


## 2026-09-20 — M0 slice 4: isolated historical remediation

- Owner chose artifact removal, explicitly authorized an isolated local rewrite and
  verification, and stated they are the only repository user. Learning remains
  pending/not assessed; provenance remains unknown, with removal as disposition.
- Completed the isolated rewrite and exhaustive selected-path/blob/tree/ref checks.
  See `HISTORY_REMEDIATION.md` for candidate location, exact method, initial rejected
  attempt, corrected remote-snapshot preservation, commands/results and promotion map.
- Verification: all 71 original commits mapped; all 14 refs accounted for; zero
  selected blobs reachable or physically present; zero unrelated tree changes;
  `git fsck --full --no-reflogs` passed; source worktree hash comparison passed.
- Read-only `git ls-remote --heads --tags origin` succeeded: five branch tips are
  represented in the map, no tags advertised. No fetch, push or remote write occurred.
- Changed docs: remediation report, historical review/inventory cross-links and this
  handoff. No application edits; prior dated runtime/setup checks were not rerun.
  Benchmark/model-quality delta not applicable; no AI behavior changed.
- Learning-agent handoff: reachability versus physical object removal, Git ref
  namespaces, remote-tracking versus local branches, commit-map/tree invariants,
  and explicit force-push leases. Relevant code/evidence: filter-repo commit map,
  remediation report and its old/new branch table. No owner understanding inferred.
- Next action: approve/promote sanitized remote branch replacements and deliberately
  adopt sanitized history while preserving pending M0 changes. Current clone and
  remote still contain historical artifacts; M0 remains open for that scope. Do not
  start M1 automatically. No source rewrite/commit/push has been performed.


## 2026-09-20 — M0 slice 5: authorized remote promotion and local adoption

- Completed the authorized atomic push with explicit old-tip leases for the five
  branches; read-back matched the prepared IDs. No ordinary implementation commit.
- Discovered and preserved all three stashes, including reflog-only history; extended
  isolated tree comparisons to 75 commits with zero unrelated changes. Installed
  sanitized Git metadata while leaving worktree files unchanged (hash-verified).
- Active local refs/reflogs: eight removed paths absent, seven blobs physically
  absent; `git fsck --full` passed. Unstaged implementation/user changes remain,
  including `.claude/settings.local.json`; the eight historical deletions no longer
  appear as worktree deletions because sanitized HEAD never contains those paths.
- Preserved original Git metadata in the private sibling recovery directory documented
  in HISTORY_REMEDIATION. No runtime data deleted. That backup is not distributable.
- Fresh remote `git clone --mirror` revealed PR refs 1–5 still retaining all seven
  blobs. Remote absence assertion failed as intended. Branch histories are clean;
  full remote history is not. Subsequent scan: 117 remote commits, five affected PR
  refs, seven retained blobs; fsck passed. Mirror marked private/DO NOT DISTRIBUTE.
- Read-only GitHub API checks: affected PRs all closed/merged; repository reports zero
  forks. Checked official GitHub documentation: PR refs are read-only; eligible
  cleanup requires Support and non-sensitive data is not eligible for that process.
- Added a factual support-request draft with unknown provenance, PR IDs and first
  changed commit. No message sent, no PR changed/deleted, no repository deleted.
- Updated handoff/remediation/inventory/README to distinguish completed branch/local
  work from unresolved PR retention. No app edits or model-quality change; benchmark
  delta not applicable. Earlier Python/frontend acceptance evidence remains dated;
  no unnecessary application test rerun after hash-preserving metadata adoption.
- Learning-agent handoff: atomic ref updates and explicit leases; refs versus reflogs;
  retaining stashes during metadata replacement; branch-only versus complete remote
  verification; host-managed refs. Owner learning pending/not assessed, not a blocker.
- Exact next action: owner reviews/authorizes the prepared GitHub request. If Support
  requires provenance/sensitivity confirmation, obtain the missing fact rather than
  inventing it. After cleanup, repeat full remote mirror verification. M0 remains
  open for this host-side issue; do not begin M1 or claim global erasure.


## 2026-09-20 — Owner decision: defer host cleanup, authorize M1

The owner chose to skip GitHub host cleanup for now and proceed to M1 if it does not
impede the interview work. Retained PR refs do not affect local graph execution,
so this is accepted as an explicit M0 dependency exception for synthetic-data work.
No real/synthetic provenance is inferred and no Support message is sent. Historical
distribution and M9 release gates remain open; owner learning remains pending.

## 2026-09-20 — M1 slice 1: real session startup

- Reproduced two failures with synthetic input: actual HTTP creation traversed the
  real graph but failed Pydantic response validation on null content; an invalid
  graph action with initial `error=None` returned null error content.
- Changed both invocation adapters to apply initial defaults before the explicit
  command, and to distinguish missing answers from falsey values. Normalized nullable
  graph response fields; `handle_error` now supplies nonempty fallback text.
- Added `tests/test_interview_flow.py` using the real graph, real GrillingEngine and
  temporary Markdown retriever with mocked external LLM calls. No runtime data used.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest
  tests/test_interview_flow.py -p no:cacheprovider -q`: initially 2 failed.
- Same command with `tests/test_api_contract.py tests/test_graph_flow.py` added after
  repair: 12 passed. No benchmark exists; model-quality delta not measured.
- Learning-agent concepts: dictionary merge precedence, null versus absent defaults,
  and why replacing a graph with a fake can hide integration defects. Learning pending.
- Next slice: active turn identity, follow-up scoring/reconnect and one shared policy.
  Later shared invocation work will replace the duplicated adapter input construction.

## 2026-09-20 — M1 slice 2: actual turns and shared follow-up policy

- Added three deterministic regressions that failed before changes: sufficient first
  answer was forced into a follow-up; repeated follow-up answers used the main prompt;
  REST/reconnect returned the main prompt instead of the active follow-up.
- Added checkpointed active-question records and stable session/question/follow-up IDs.
  Candidate messages and current evaluations identify the question actually answered.
  Legacy checkpoints derive their active prompt read-only from conversation; missing
  follow-up text is never guessed from the main question. Terminal states have none.
- Graph nodes, REST detail and WS connected/resume projections use that active turn.
  Updated public detail/conversation fields and frontend types; chat reconnect marks
  follow-ups correctly. No second workflow store or runtime migration was introduced.
- Added `core/follow_up_policy.py` (`follow-up-v1`), shared by engine and graph.
  Preserved the engine's existing thresholds/gap rules; removed evaluator mutation
  that separately forced follow-ups. A sufficient high-scoring first answer can now
  advance; insufficient later answers no longer require a gap label to trigger one.
- Files: graph state/nodes/edges, grilling engine and new policy, session/WS adapters,
  frontend types/chat, integration tests and policy regression matrix.
- Focused startup/turn/API/graph run: 15 passed. Ruff passed after fixing a Python 3.11
  f-string syntax issue caught by lint. Added 10 policy cases for cap, threshold and
  gap boundaries; final combined result recorded with subsequent verification.
- Deterministic delta: the three new defect cases changed from fail to pass. These
  are software/controlled-output evaluation cases, not a model-quality benchmark.
  Human-reviewed benchmark and measured quality delta remain unavailable until M3.
- Learning-agent concepts: prompt identity versus main-question index, projections,
  lazy legacy compatibility, pure shared policies. Owner learning pending/not assessed.
- Next slice: shared command validation/invocation; lifecycle parity and invalid-input
  recovery. Provider-output validity/score calibration remain M2/M3, not claimed fixed.


## 2026-09-20 — M1 slice 3: shared commands and lifecycle acceptance

- Completed `graph/commands.py` and `transitions.py`: both REST/WS validate the latest
  graph checkpoint before provider creation or invocation. No separate state store.
  Graph routing uses the same preconditions as a defensive check; asking/start resumes
  without regenerating. Skip/end do not initialize model providers.
- Before correction, the completed-session/end regression changed completed to cancelled
  (HTTP 200), and whitespace answers were accepted (HTTP 200). Both now reject with 400;
  shared rejected commands leave checkpoint state unchanged. Busy phases reject 409.
- Changed session/WS adapters, graph edges/builder/nodes, graph usage documentation,
  API/graph/integration tests, frontend types and chat/video terminal reconnect handling.
  Fixed the frontend detail type's inherited list-only field. Updated README/onboarding,
  the smoke script's obsolete forced-first-follow-up wording, and decision 0002 with
  lifecycle table, compatibility, failure boundaries and policy tradeoffs.
- Added real graph/engine cases for multi-question WS progression, skip of an active
  follow-up, repeated read-only start, terminal/busy commands, malformed input, pending
  cancellation, provider-setup rejection, provider-independent skip/end, legacy follow-up
  recovery/refusal and the compact custom evaluation path. Temporary synthetic inputs;
  no real provider, STT/TTS, Chroma or runtime data calls in these cases.
- Verification commands actually run:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest tests/ -p no:cacheprovider -q`:
    **73 passed**, five existing PyMuPDF/SWIG deprecation warnings.
  - `.venv/bin/ruff check backend rag/retriever.py tests --config pyproject.toml`:
    passed; `.venv/bin/black --check backend rag/retriever.py tests`: passed (43 files).
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/test_graph.py`:
    passed start/answer/follow-up/skip/end smoke run. This older smoke mocks the engine;
    the new integration suite exercises the real engine.
  - In `frontend`: `./node_modules/.bin/tsc --noEmit --incremental false` and
    `npm run lint`: passed. `npm run build`: passed with sandbox escalation for the
    local Turbopack worker; only stale baseline-browser-mapping warnings/npm notice.
  - `git diff --check`: passed. Additional smoke-script Ruff check found an unused
    MagicMock import after removing its obsolete stub; removed it. Black requested
    formatting for the script; ran `.venv/bin/black scripts/test_graph.py`, then
    its Ruff/Black checks, `git diff --check` and the smoke command again: all passed.
- Policy version: `follow-up-v1`. Ten fixed policy decisions and targeted integration
  regressions establish deterministic routing/targeting changes. No model/prompt/rubric
  benchmark was run; dataset/model quality delta is **not measured**. Prompt templates
  and existing truncation budgets remain unchanged; actual follow-up prompt selection
  is corrected. No threshold calibration claim.
- M1 acceptance: session creation runs the real graph; main/repeated follow-up answers
  link to the presented question; start/answer/advance/skip/end/terminal/invalid cases
  are covered; both transports use shared validation; graph and engine use one policy;
  reads/reconnect project the active follow-up. Owner explanation/learning remains
  pending/not assessed under explicit implementation delegation.
- Limitations: live provider/browser behavior and SQLite process-restart/concurrency
  are unverified here; invalid model output (M2), quality benchmark (M3), durable ledger
  (M4), retries/recovery (M5) remain open. Existing fallback behavior is not improved by
  these tests. Terminal reconnect UI passed types/lint/build, not browser automation.
- Learning-agent handoff: study `commands.execute_command` and `transitions.command_error`
  as validation versus orchestration; trace `state.QuestionTurn` through evaluate and
  follow-up nodes; compare append-only conversation reducers to replaceable active state.
  Explore why a route fake hid startup merging, why deterministic policy checks are not
  a quality benchmark, and why stable turn IDs alone do not make retries safe.
- Exact next action: await explicit M2 authorization, then reproduce invalid model
  output handling with synthetic tests and add validated contracts/CI per the roadmap.
  Do not begin M2 automatically. No implementation commit, push or deployment performed.


## 2026-09-20 — Owner-authorized traceable local commits

The owner requested commits that make the implementation details traceable for learning.
Used focused conventional commits, preserving the original implementation chronology
in earlier dated entries rather than claiming each development slice was committed then.

- `af60e12`: M0 synthetic parser fixture, nested runtime ignore rules, voice defaults,
  capability-gated frontend, baseline tests and decision 0001.
- `c7c1d2a`: coupled M1 graph/engine/transport changes, synchronized types/UI,
  real-component regression tests, smoke updates and decision 0002.
- A following documentation commit records runtime guidance, historical-data evidence,
  the owner dependency exception, this handoff and the commit reading guide.

See [M0–M1 commit reading guide](M0_M1_COMMIT_GUIDE.md) for the execution path and
read-only Git commands. Implementation and test changes stay together; no fake
intermediate red/green history was manufactured. The voice and M1 hunks in
InterviewRoom were staged separately without modifying the working file.

Verification of commit boundaries: exported each implementation commit with
`git archive HEAD` into a fresh temporary directory, then ran the repository's
`.venv/bin/python -m pytest tests/ -p no:cacheprovider -q` with PYTHONPATH set to
that exported source and bytecode disabled. The committed M0 snapshot passed 29
checks; the committed M1 snapshot passed 73 checks. Both reported only the five
existing PyMuPDF/SWIG deprecation warnings. Dependencies
were reused from the existing venv; this was not another clean dependency install.
Full frontend build/type/lint and Python quality evidence above still applies to the
unchanged implementation. Final staged whitespace and file inventory checks are part
of this commit pass. No runtime payloads or local settings were staged.

Owner learning remains pending/not assessed. M2 remains eligible but unauthorized.
The earlier 'uncommitted'/'no commit authorized' statements are dated historical
records superseded by this explicit owner request; no new push or deployment occurred.

## 2026-09-22 — Plan Jev comparison after M2

- Owner requested recording the proposed experiment and will return to M2 first.
  Updated the roadmap and current handoff; no implementation or milestone completion.
- M3 will compare Jev answer-gap judgments with the existing evaluator and human
  labels on synthetic cases, initially without changing interview behavior.
  Adoption depends on predefined quality, uncertainty, latency and cost evidence.
- Owner exercise: label gaps and follow-up need before viewing model outputs;
  learning remains pending/not assessed. Agent scope: future adapter, runner and
  analysis, after M2. No benchmark run or model-quality delta exists for this change.
- Check: `git diff --check` passed. Documentation-only change; tests not rerun.
  Existing local settings changes preserved. No commit, push or provider call made.
- Exact next step: finish M2 contracts and deterministic CI before starting the
  planned M3 comparison.


## 2026-09-22 — M2 stage 1: contracts and defect reproductions

- Owner authorized M2 implementation with a local commit and a stop between each
  stage. Agent scope: contracts, synthetic examples, tests and documentation.
  Owner exercise: distinguish JSON validity from domain validity and trace one
  parser; learning remains pending/not assessed.
- Added internal `model-output-v1` contracts for valid evaluation, insufficient
  evidence, invalid output, provider failure, unsupported mode, question batches
  and follow-ups. Required fields, strict types, bounded finite numbers, gap
  coherence and duplicate JSON keys are checked without retaining invalid payloads.
- Moved the existing GapType enum to a shared model module; the engine imports it
  and preserves its existing import surface. No runtime scoring or public API
  behavior changed. Seven explicit dimensions remain provisional, not calibrated.
- Synthetic fixtures: `model-output-cases-v1` in `tests/test_model_outputs.py`.
  Focused run: 55 passed, 3 strict xfailed. Running with `--runxfail -k test_engine_`
  reproduced all three defects: empty JSON -> 0.425; dimensions of 100 -> 100;
  mocked timeout -> heuristic score 0.4. These are still live engine defects.
- Verification: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest
  tests/ -p no:cacheprovider -q`: 128 passed, 3 xfailed, five existing SWIG warnings.
  `.venv/bin/ruff check backend rag/retriever.py tests --config pyproject.toml` and
  `.venv/bin/black --check backend rag/retriever.py tests`: passed.
  No frontend changes; frontend checks not rerun. No live provider calls, runtime
  data changes or model-quality benchmark. Deterministic parser cases pass;
  production defect delta remains unchanged until wiring in stage 2.
- Reading guide and decisions: [0003](decisions/0003-model-output-contracts.md),
  `backend/app/models/model_outputs.py`, `tests/test_model_outputs.py`.
- Existing Jev planning edits and local settings are preserved outside this stage's
  commit. No push or deployment. Stage 1 ends here per the requested review stop.
- Exact next step after owner resumes: stage 2 engine/question contract enforcement,
  remove strict xfails as defects are repaired, eliminate scalar fabrication and
  heuristic failure scoring. Keep the intermediate commit runnable with explicit
  failure handling; full graph/API/UI integration is stage 3. M2 is not complete.


## 2026-09-22 — M2 stage 2: enforce runtime model contracts

- Owner requested continuation to stage 2, with a corresponding local commit and
  review stop. Agent scope: engine/question enforcement, minimal graph compatibility,
  synthetic regression coverage and documentation. Owner exercise: trace validated
  model output through projection and routing; learning remains pending/not assessed.
- Standard evaluation now validates complete JSON before constructing a scored
  result. Removed missing-field defaults at the model boundary, exception/length
  heuristic scores, compact scalar-to-seven-dimension expansion, and the unvalidated
  second consistency call. Consistency is judged inside the validated response;
  absent resume evidence is explicitly distinguished from contradiction in the prompt.
- Custom/Hybrid session compact evaluation and its legacy helper methods report
  unsupported with no score. API-mode adapters must satisfy the standard contract.
  All interview, prepared Hybrid and standalone endpoint question outputs require
  the exact requested number of valid strings; no padding, substring or punctuation
  repair. Follow-up generation likewise validates output and has no silent fallback.
- Minimal graph integration: non-scored outcomes retain the active question and
  halt before follow-up policy, even at a zero cap/final question. Existing REST/WS
  error envelopes expose safe, distinct codes without a success evaluation payload.
  Failed startup stays pending. Failed follow-up keeps the accepted answer/evaluation
  and records a failure; another answer is rejected with 409, while skip/end work.
  This is a bounded guard, not concurrency, idempotency or restart recovery.
- A non-scored answer is not appended as an accepted conversation answer; its text
  remains in current graph state. A new explicit submission may succeed. Dedicated
  UI, failure restoration on reconnect and full public outcome contracts remain
  stage 3. Added only the optional valid tag to the existing frontend evaluation type.
- Removed timestamp prompt entropy while changing question format. Changed prompt
  identities: evaluation-v2, follow-up-generation-v2, question-generation-v2,
  standalone-questions-v2 and hybrid-questions-v2; rubric identity is
  question-aware-seven-dimension-v1. Schema remains model-output-v1; policy is
  follow-up-v2 for the new outcome guard, with existing numeric thresholds unchanged.
  Full metadata propagation and broader provider logging/timings remain stage 4.
- Deterministic delta: all three stage 1 engine defects now pass as normal tests
  (strict xfail markers removed). Expanded synthetic fixture revision:
  model-output-cases-v2. Short answers now reach the mocked evaluator; invalid output,
  provider failure, insufficient evidence and unsupported mode cannot become scores.
  New graph tests cover failed startup/follow-up, REST/WS parity, no progression at
  cap, valid low-score follow-up, exact question counts and stable prompts.
  No human-reviewed benchmark or live model-quality delta is claimed.
- Verification actually run:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest tests/
    -p no:cacheprovider -q`: **174 passed**, five existing SWIG warnings, no xfails.
  - `.venv/bin/ruff check backend rag/retriever.py tests scripts/test_graph.py
    --config pyproject.toml`: passed; `.venv/bin/black --check backend
    rag/retriever.py tests scripts/test_graph.py`: passed.
  - Mocked graph smoke: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.
    .venv/bin/python scripts/test_graph.py`: passed start/answer/follow-up/skip/end.
  - Frontend `./node_modules/.bin/tsc --noEmit --incremental false` and
    `npm run lint`: passed. Only an additive type declaration changed; production
    build/browser tests were not rerun. No live LLM/STT/TTS/Chroma calls or runtime
    data migrations/deletions. `git diff --check`: passed.
- Initial integration run found ten policy test fixtures missing the new valid tag;
  updated fixtures and reran the suite, without changing score thresholds.
- Read [decision 0003](decisions/0003-model-output-contracts.md), then
  `core/grilling_engine.py`, `graph/nodes.py`, `graph/edges.py` and
  `tests/test_interview_flow.py`. Paths are under `backend/app/` unless prefixed tests.
- Existing Jev planning edits and local settings remain outside this stage's commit.
  No push/deployment. Exact next step: stop for owner review, then stage 3 public
  outcome schemas/types and UI handling, including unavailable/insufficient outcomes,
  draft treatment and reconnect visibility. M2 remains incomplete; do not start M3.


## 2026-10-10 — M2 stage 3: public outcomes and checkpoint presentation

The owner explicitly delegated stage 3 implementation. Agent scope: public schemas,
checkpoint projection, REST/WS adapters, outcome UI, deterministic fixtures and
verification. Owner exercise: predict score availability and allowed actions for a
valid zero, insufficient evidence, malformed output and provider timeout, then trace
one outcome from graph to UI. Owner learning remains pending/not assessed. This
slice's implementation and evidence are recorded together in its local commit;
no self-referential commit identity is embedded here.

- Added public contract `interview-outcome-v1` in `backend/app/models/interview.py`
  and one read-only projection in `graph/public.py`. REST commands/reads and WS
  commands/connections expose the same typed outcomes, generation failures,
  submitted draft, active-question answer status, supported actions and conversation.
  Unknown/corrupt legacy evaluations project safely without a score; older valid tags
  and stage 2 startup failure codes normalize in memory without data migration.
- Graph records safe generation failure and the latest accepted evaluation output.
  Feedback retains question/answer identity after advancement and survives reads/resume;
  the next mutating command clears it. Completed sessions retain final valid feedback;
  cancellation clears it. No durable ledger/report or concurrency guarantee was added.
- Insufficient/invalid/provider evaluation outcomes retain the active prompt and
  submitted draft. A new explicit answer, skip or end remains allowed. Unsupported
  evaluation and accepted follow-up failure reject another answer with 409; read/resume,
  skip and end remain available. Failed startup permits explicit start/end. Supported
  commands are derived from the same validation used for execution.
- Chat and video consume checkpoint snapshots, render distinct outcomes, associate
  feedback with its evaluated prompt, and display a valid zero score. They reconcile
  accepted conversation without optimistic duplicates and retain local edits on manual
  reconnect. Reconnect never resends commands. Failed startup has explicit Try again;
  terminal feedback uses View results rather than automatic navigation. Voice transcripts
  and local recordings remain available for explicit submission; devices were not tested.
- Removed raw answer/audio WS payload console logs in the touched transport. Added a
  dependency-free frontend `npm test` command for the typed presentation/reconciliation
  helper. No package/lockfile dependency changes or live provider calls.
- Deterministic fixture revision: `stage3-outcomes-v1`, 37 new backend cases. The full
  suite increased from 174 to 211 passing cases. Ten frontend fixtures cover all outcome
  presentations, zero score, edited/cleared drafts, accepted history, generic failures,
  startup and explicit reconnect behavior. Scoring prompts/rubric/model are unchanged;
  no human-reviewed benchmark or model-quality delta was measured.
- Checks actually run:
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -m pytest tests/
    -p no:cacheprovider -q`: **211 passed**, five existing SWIG deprecation warnings.
  - `.venv/bin/ruff check backend rag/retriever.py tests scripts/test_graph.py
    --config pyproject.toml`: passed; `.venv/bin/black --check backend
    rag/retriever.py tests scripts/test_graph.py`: passed.
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python scripts/test_graph.py`:
    mocked start/answer/follow-up/skip/end passed.
  - Frontend `npm test`: **10 passed**; `./node_modules/.bin/tsc --noEmit
    --incremental false` and `npm run lint`: passed after the browser-driven UI fixes.
  - Final `NEXT_TELEMETRY_DISABLED=1 npm run build`: passed after the final UI fixes.
    Initial build exposed a missing installed native SWC binary; restored the exact
    installed compiler archive only in ignored node_modules. The compiler worker's
    local port required sandbox permission. Existing baseline-browser-mapping advisory
    remains; no dependency upgrade was introduced.
  - Browser check used a temporary localhost mock server, real graph/engine and memory
    checkpoints, with newly authored synthetic state only. Verified timeout + refresh,
    explicit successful submission with one accepted answer, accepted follow-up failure
    with disabled submission and skip/end, pending startup with manual retry, valid zero,
    insufficient evidence, and edited draft retained after forced disconnect/manual
    reconnect. Temporary servers/tab were stopped. The first mock selected on instruction
    text rather than answer text; corrected the fixture and repeated the affected check.
  - `git diff --check`: passed after final documentation updates.
- README, onboarding, contributing guides and decision 0003 describe the implemented
  public contract and bounded feedback lifetime. Existing Jev planning and local settings
  edits are preserved outside this stage's commit. No push, deployment, support message,
  runtime-data migration/deletion or real-data processing occurred.
- Stage 3 engineering complete; M2 remains incomplete. Exact next step after owner
  resumes: stage 4 provider/model/usage/finish-reason/version metadata and safe timings/
  logging. Stage 5 then establishes deterministic CI including frontend outcome fixtures.
  Do not start M3/Jev, M4 ledger or M5 recovery automatically. M0 host cleanup stays
  deferred and M9 real-user/public-beta gate remains closed.

## 2026-10-10 — Publish implementation and refresh GitHub status

- Owner authorized pushing the latest work and checking/updating GitHub project
  status. Agent scope: remote comparison, normal branch push, documentation and
  repository metadata. The stage 3 owner exercise remains pending/not assessed;
  publication does not establish human understanding or complete M2.
- Fetched remote refs and confirmed no newer commits on the active refactor branch.
  Published the existing M0/M1 and M2 stages 1–3 implementation commits, together
  with current-status corrections and the previously approved Jev comparison plan.
  No history rewriting was needed. Machine-local settings remain excluded.
- GitHub had no open PRs/issues, no workflows/releases and an empty About section.
  Created [draft PR #6](https://github.com/Willkczy/resume-griller/pull/6) with current
  scope, verification and unfinished milestones. Added a prototype/status description
  and linked the current handoff from About. The default-branch README remains the
  earlier version until this PR is reviewed and merged; historical merged PR records
  were preserved.
- Corrected current documentation to distinguish bounded latest feedback from the
  missing durable ledger, describe received question-audio playback, and identify
  M2 stage 4 metadata/timings as next. Kept the roadmap's initial baseline/table and
  dated no-push records historical rather than rewriting them.
- Checks: remote branch HEAD matched local HEAD after push; GitHub reported the PR
  open, draft and mergeable. About changes were read back successfully. Changed
  Markdown link targets/code fences and `git diff --check` passed. Source was unchanged
  in this publication slice, so the stage 3 evidence above remains the latest code
  verification: 211 backend tests, 10 frontend fixtures, formatting/lint/type/build
  and synthetic browser checks. These are local checks, not GitHub Actions results.
- Exact next implementation step remains M2 stage 4 after the owner continues, then
  stage 5 deterministic CI. M3/Jev is planned only. Deployment, deferred Support
  cleanup and runtime-data migration/deletion were not performed; M9 remains closed.

## 2026-10-13 — Review draft PR #6 and reassess the next step

- Owner requested review and plan status. Agent scope: inspect the PR diff and
  roadmap, rerun deterministic checks, reproduce suspected defects and record evidence.
  Reviewed PR head `b31c26329178ba7ab38c7b24ae9c66d8e49fcda1`, which matched the local
  checkout and remained unchanged on the final remote read. GitHub reported an open,
  mergeable draft with no CI checks/statuses or review comments. Mergeability is not
  correctness evidence. Existing machine-local settings changes were preserved.
- Three actionable findings, all reproduced using newly authored temporary synthetic
  files or mocked Chroma metadata; no live provider or runtime-data access:
  - **P1: resume path containment.** `POST /api/v1/sessions` with resume ID
    `../review-note` or an absolute filename stem returned 200 and passed a synthetic
    Markdown file outside the configured parsed directory to the mocked LLM.
    `rag/retriever.py:get_full_resume_text` joins an unrestricted ID directly to the
    directory. Validate IDs and resolved path containment before reading or invoking
    a provider; cover traversal, absolute paths and normal/legacy IDs. This is an
    immediate correction under the roadmap's early-security rule, not work to defer
    merely because M9 also includes path containment.
  - **P2: legacy summary regression.** With flat Chroma metadata containing
    `skills_list="Python, FastAPI"`, one experience chunk and one education chunk,
    `GET /api/v1/resume/{id}` returned empty skills and zero counts. The route now
    reads those fields from a summary, but `_summary_from_chunks` does not provide
    them. Preserve the previous metadata extraction in the legacy projection.
  - **P2: Markdown education boundary.** Two synthetic `###` degree entries under
    `## EDUCATION` returned an education count of one, or zero with a blank line
    before the first entry. The `\n##` section terminator also matches `###` headings.
    Match the next level-two heading and test multiple entries and blank lines.
- Fresh checks: backend pytest **211 passed** with five existing SWIG warnings;
  Ruff and Black passed for backend, retriever, tests and graph smoke script;
  mocked graph start/answer/follow-up/skip/end passed. Frontend **10 fixtures passed**,
  TypeScript with incremental output disabled and ESLint passed. Production build
  passed after granting the local compiler worker's required port access; the first
  sandboxed attempt failed on that port restriction. Existing browser-mapping advisory
  remains. PR diff and final handoff whitespace checks passed.
- No new browser/device, clean-install, coverage or live model-quality run. The
  reproductions reveal gaps in the existing green suite, not a measured model-quality
  delta. No prompt/rubric/schema/model change was made. Only this handoff was edited;
  no fix, commit, push, GitHub review submission, merge or deployment occurred.
- Owner learning remains pending/not assessed and separate from engineering eligibility.
  Retain the stage 3 exercise: predict score availability/actions for valid zero,
  insufficient evidence, malformed output and timeout, then trace graph to UI.
  The correction exercise is to trace resume ID -> file path -> provider prompt and
  compare expected summary metadata across the two storage formats.
- Recommendation: keep PR #6 draft, fix the three findings with focused regressions,
  then continue M2 stage 4 metadata/timings/safe logging and stage 5 deterministic CI.
  M2 is incomplete; do not start M3/Jev yet. M0 host cleanup remains deferred, M4–M6
  remain unstarted, M7/M8 remain optional/unselected, and M9 remains closed.

## 2026-10-13 — Correct PR #6 resume storage findings

- Owner explicitly requested the fixes, PR #6 merge and subsequent M2 stage 4.
  Agent scope: all three review corrections, regression tests and delivery.
  Owner exercise: trace resume ID -> validated path -> provider prompt, and compare
  summary metadata across Markdown and legacy storage. Learning remains pending.
- Added a shared resume-ID/path boundary. Rejects traversal, absolute/Windows path
  syntax, control characters and symlink escapes before reading or parsing. Safe
  legacy stems with spaces, dots and Unicode remain supported. Session creation
  rejects invalid access with 422 before a model call. Read/write/delete paths use
  the same containment check; deletion validates every target before removing any.
  No migration or deletion of existing runtime data was performed.
- Legacy summaries again project skills and experience/education counts. Markdown
  education sections now end at level-two headings, preserving multiple degrees
  and blank lines. Synthetic fixture revision: `resume-storage-review-v1`.
- Backend suite: **227 passed**, including 16 new storage/route regressions; five
  existing SWIG warnings. Ruff and Black checks passed. `git diff --check` passed.
  The immediately preceding review's frontend build/types/lint/10 fixtures and mock
  graph smoke remain applicable: this slice changes only backend resume access and
  summaries. No live providers, device tests, runtime payloads or model-quality run.
  Prompt/rubric/schema behavior is unchanged; the three reproduced defects now have
  passing regressions, not an AI-quality benchmark delta.
- Exact next action: publish this correction commit and merge PR #6 as authorized,
  preserving the staged learning history, then implement M2 stage 4 and stop for
  review. Stage 5 CI and M3/Jev remain unstarted; M0 host cleanup remains deferred.
