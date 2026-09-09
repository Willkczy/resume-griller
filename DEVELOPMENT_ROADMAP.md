# Resume Griller learning-oriented refactor roadmap

Plan adopted: 2026-09-10. Assessment performed: 2026-09-09.
Status: planning complete; implementation gates remain open.

This is the canonical implementation plan. It supersedes the previous 12-week
M0–M8 plan. Milestone numbers below have new meanings: do not use old chats or old
milestone references as implementation authority. There is no fixed completion date;
progress is determined by evidence and learning, not elapsed weeks.

## Purpose and authority

The owner wants to understand and implement the engineering, strengthen an Applied
AI / AI product engineering portfolio, and learn skills relevant to U.S. early-career
roles. Optimize for explainable behavior and measured quality, not technology count.

- [README.md](README.md): implemented runtime and setup, including known defects.
- [AGENTS.md](AGENTS.md): agent working rules; CLAUDE.md points there.
- [AI_ONBOARDING.md](AI_ONBOARDING.md): short architecture context.
- [docs/PROJECT_REASSESSMENT.md](docs/PROJECT_REASSESSMENT.md): assessment evidence
  and recruiting rationale; historical findings, not proof they remain unfixed.
- [docs/REFACTOR_HANDOFF.md](docs/REFACTOR_HANDOFF.md): current progress, next slice,
  unresolved questions, and the handoff record. Update it after implementation work.
- [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md): setup and verification commands.

No application behavior is fixed merely by adopting this plan. Start with M0.

## Verified baseline and immediate risks

The 2026-09-09 review ran 21 backend tests successfully, TypeScript checking and
frontend lint successfully. It did not rerun coverage, production build, live model
calls, voice, or a browser usability study. Earlier documentation reported a
2026-08-01 coverage/build baseline; those numbers are not current release evidence.

Targeted in-memory checks with the real graph and mocked services established:

1. Session creation merges initial `action=None` over `action="start"`, reaches an
   error response with null content, and fails response validation before an LLM call.
2. The standard evaluation parser accepts `{}` as an ordinary score of 0.425 and
   accepts out-of-range dimension scores, producing an overall score of 100.

Inspection established additional problems, to reproduce with tests when addressed:

- Follow-up evaluation and reconnect identify the indexed main question rather than
  the actual current interviewer turn.
- Graph routing and GrillingEngine disagree about follow-up policy.
- Advancing clears current evaluation; complete evaluations are not a business ledger.
- Custom-mode dimensional scores are derived from one scalar, not independently judged.
- Full resume text is checkpointed, but evaluator prompts truncate context.
- REST/WS lifecycle validation differs; concurrent/repeated commands lack a shared guard.
- Reports infer answer quality from follow-up counts and completion from questions shown.
- Chat/video duplicate lifecycle code; Zustand store is unused by those controllers.
- Voice defaults/dependencies disagree; video read-aloud is a timer placeholder.
- Tracked resume artifacts and inference notebook have unresolved provenance. Two
  tracked PDFs have identical bytes. No complete historical PII audit was performed.

Tests currently isolate away several integration seams and most evaluator behavior.
Passing the existing suite is necessary but insufficient to close these findings.

## Target design and constraints

Keep FastAPI, Next.js/React, a deterministic LangGraph workflow, and direct LLM SDK
adapters. Keep a modular monolith and preserve the working API surface incrementally.

### Workflow and contracts

- LangGraph remains the sole owner of interview progression and checkpointed state.
- REST/WS become adapters around one shared command validation/invocation boundary.
  Do not create a second orchestration service or independent current-session store.
- Assign explicit question/answer/turn IDs. A follow-up has a parent main-question ID.
- Use runtime-validated Pydantic contracts and synchronized TypeScript types for active
  endpoints. Legacy unused schemas are not evidence of runtime validation.
- Separate valid evaluation, insufficient evidence, invalid output, and provider failure.
  Reject invalid numeric values; do not silently clamp them into valid judgments.
- Use question-appropriate rubrics. Fewer calibrated dimensions are preferable to seven
  unvalidated ones. Concise answers do not automatically deserve low scores; factual
  technical answers do not always require STAR structure or business metrics.

### Data and evidence

Proposed relationships (contracts to refine in the owning milestone):

```text
ResumeVersion -> InterviewSession -> QuestionTurn -> AnswerTurn -> EvaluationEvent
                                      |                             |
                                 parent question              FollowUpDecision
Report = projection of persisted turns and evaluations
```

- Preserve source extraction and evidence references; LLM normalization is not ground
  truth. Compare raw extraction versus normalization before making normalization required.
- Use explicit context/token budgets and handle omitted material deliberately. Full
  context for one short resume/JD is appropriate; do not introduce vector retrieval.
- SQLite remains sufficient for local synthetic-data work. Introduce migration-managed
  relational records for turns/evaluations in M4. Use PostgreSQL for the controlled
  deployment when concurrent access and managed persistence justify it.
- Ledger records are business facts, not a duplicate mutable workflow state. Reports
  read them; graph state references stable identities. Define the failure semantics
  between ledger writes and graph checkpoints. They are not automatically atomic.
- Append-only evaluations can be superseded by correction events, but must still be
  removable under the deletion/retention policy. Backups need a deletion policy too.

### Model services, frontend and operations

- Support one default provider well. Keep existing adapters until deliberately retired;
  reuse another for an experiment if useful. No additional providers just for breadth.
- Return provider/model identity, usage where available, finish reason, timing, outcome,
  request/trace identity, and prompt version. Unknown usage/cost is null, not zero.
- Establish bounded timeouts/retries and safe logging early. Never silently switch
  evaluator/model and imply identical calibrated scores.
- Keep REST/WS compatibility while consolidating behavior. Streaming is optional.
- Share one frontend interview controller; retain local UI state where adequate.
  Default to text, restore server history, preserve drafts, support keyboard access,
  and allow transcript review before scoring. Hide unsupported modes and remove claims
  of validated proprietary models. Camera preview is not model video analysis.
- Start with structured logs and timings. Add tracing when it answers an operational
  question. Do not introduce an observability vendor before defining needed signals.
- One containerized backend and frontend hosting suffice. Object storage is justified
  when instance replacement makes local files unsafe. A queue, Redis, circuit breaker,
  or distributed lock needs a measured requirement, not an automatic checklist entry.

## Dependencies and completion tracking

| Milestone | Classification | Prerequisite | Initial status |
|---|---|---|---|
| M0 — System ownership and safe baseline | Quality + learning | None | Open |
| M1 — Correct text interview | Quality + learning | M0 | Not started |
| M2 — Model output contracts and CI | Quality + learning | M1 | Not started |
| M3 — Human-reviewed benchmark | Learning + quality | M2 | Not started |
| M4 — Durable feedback and report | Quality + learning | M3 | Not started |
| M5 — Retry, interruption and recovery | Quality + learning | M4 | Not started |
| M6 — Product/dependency simplification | Quality + learning | M5 | Not started |
| M7 — Job-description targeting | Optional product + learning | M6; owner selects track | Not selected |
| M8 — Adaptive-selection experiment | Optional experiment | M7; owner selects track | Not selected |
| M9 — Controlled deployment | Required for real-user operation | M6; M7/M8 not required | Not started |

```text
M0 -> M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M9
                                      \-> M7 -> M8 (optional)
```

M4 is the first local synthetic-data portfolio demonstration. M5/M6 strengthen it.
M9 is not required to demonstrate the local prototype, but its security and release
checks must pass before public beta or collection/processing of real-user data.
Security design starts at M0; never postpone an exposed vulnerability just because
its broader deployment milestone is later. Implement an urgent prerequisite as an
explicit, tested slice and record it in the handoff.

The table is the initial plan. Current status and evidence live only in the handoff.
Completion requires both engineering evidence and an owner explanation; record those
separately so technical completion does not imply the owner has learned the material.

## Milestones

### M0 — System ownership and safe baseline

**Change / why:** Establish what actually runs and which data can safely be used.
Resolve provenance for every tracked PDF/TXT fixture and notebook; inventory raw files,
parsed files, checkpoints, audio, browser/server logs and external processors. Verify
ignore coverage, including nested upload directories. Decide and implement a minimal
voice default/dependency contract; text-only should work without voice credentials.

**Learn:** Dependency graphs, trust boundaries, checkpoint inspection and reproducibility.

**Owner work:** Draw upload and answer flows, inspect a synthetic checkpoint, explain
external calls and persisted copies; supply provenance that cannot be inferred from code.

**Agent work:** Inventory paths without echoing PII/secrets; check the diagram, reproduce
setup, and implement small agreed baseline changes. Do not infer synthetic provenance
from plausible names. Do not rewrite Git history or destroy runtime data implicitly;
if real data is confirmed, prepare a separately scoped remediation and coordinate it.

**Acceptance:**

- Every tracked resume-shaped artifact has a documented synthetic/provenance disposition;
  unresolved artifacts are explicitly blocked, not silently declared safe.
- No known real PII or secrets remain in the intended distributable tree/history after
  any separately authorized remediation; no new sensitive content is committed.
- Data inventory and voice support decision are recorded in `docs/decisions/` or a
  linked data-inventory document created during this milestone.
- Clean setup and relevant checks are recorded with actual results, including failures.
- Owner can trace a request and describe where sensitive copies persist.

### M1 — Correct one complete text interview

**Change / why:** Fix start-action merging/null error handling; model actual turns,
consolidate follow-up policy, and centralize start/answer/skip/end validation. Return the
actual current question on REST reads and WS reconnects. Preserve existing wire behavior
unless a documented contract change is necessary.

**Learn:** State machines, invariants, reducer merge behavior and integration seams.

**Owner work:** Write a transition table and failing startup/follow-up tests; implement
the core state/policy correction and explain why it works.

**Agent work:** Review edge cases, adapt response mappings and expand tests around the
owner's implementation. Main files: `backend/app/graph/`, session/WS routes and
`tests/test_graph_flow.py`, `tests/test_api_contract.py`.

**Acceptance:**

- Actual session creation passes through the real graph with mocked model services and
  returns the first question; the test does not replace the graph with a fake.
- Main and repeated follow-up answers identify exactly the question presented.
- Start, answer, follow-up, advance, skip, end, completed/cancelled and invalid actions
  have explicit transitions; REST/WS accept/reject them consistently.
- One pure follow-up policy is used in production. Reconnect restores the active prompt.
- Owner can explain state merging and why previous green tests missed the bug.

### M2 — Model output contracts and a deterministic CI gate

**Change / why:** Validate evaluations and question outputs at runtime, remove scalar-to-
7D fabrication and silent failure-to-score behavior, version contracts/prompts/rubrics,
and remove timestamp prompt entropy. Add provider result metadata and safe timings.
Add GitHub Actions for deterministic backend and frontend checks early.

**Learn:** Valid JSON versus valid domain output, error taxonomy, reproducibility and CI.

**Owner work:** Write valid/invalid examples, implement one contract/parser and define the
UI outcome for unavailable evaluation. Inspect which SDK metadata is actually available.

**Agent work:** Synchronize public schemas/types, mechanical adapter changes, malformed-
output tests, CI wiring and documentation. Main files: GrillingEngine, LLM adapters,
active API schemas, frontend types and `.github/workflows/` (to be created).

**Acceptance:**

- Missing required fields, wrong types, unknown enum values, non-finite/out-of-range
  scores, malformed/truncated output and provider failure never become ordinary scores.
- Evidence insufficiency and valid low scores are distinct from infrastructure failure.
- Disabled/experimental modes either meet the contract or report an explicit limitation.
- Version identity and provider outcomes are captured without full payload logging.
- CI runs backend tests, Python checks, frontend type/lint/build and deterministic
  contract fixtures without live LLM/STT/TTS/Chroma or provider credentials.
- Owner can explain the error categories and diagnose a rejected response.

### M3 — Human-reviewed evaluation benchmark

**Change / why:** Begin with 10–15 carefully labeled synthetic cases; expand to a focused
30–50-case set after the labeling protocol works. Measure the evaluator, not just the
application. Proposed home: `evals/` with a runner, versioned safe fixtures and data card;
choose exact layout before implementation and avoid another duplicate ML package.

**Learn:** Annotation agreement, data leakage, precision/recall, controlled experiments
and uncertainty. A model judge is a secondary signal, not independent ground truth.

**Owner work:** Label evidence and follow-up need before reading model output. Have a
second human review a subset; resolve disagreements explicitly. Define the initial role,
language and question-type scope rather than claiming every possible slice is covered.

**Agent work:** Implement runner/aggregation, suggest missing failure cases, generate
candidate synthetic examples for human review and summarize experiment results.

**Acceptance:**

- Data card records provenance, rubric, scenario grouping, labels, reviewer process,
  supported slices, limitations and immutable dataset version.
- Separate development and held-out cases by resume/scenario; do not tune on test output.
- Compare current baseline, simpler rubric and one controlled change on the same inputs.
- Report schema validity, evidence support, follow-up precision/recall/usefulness, severe
  errors, latency and cost with case counts/denominators; repeat selected cases for variance.
- Include concise-correct answers, contradictions versus absent facts, prompt injection,
  and relevant language/bias cases. No broad fairness claim from a tiny sample.
- Predetermine project acceptance thresholds after pilot labeling, before held-out runs.
  Do not treat former numerical roadmap targets as industry standards or calibrated gates.
- Store sanitized result artifacts and failure analysis. Paid runs are explicit commands,
  separate from unit tests; scheduled runs require separate scheduling authorization.
- Owner can defend labels, metrics, experiment conclusions and limitations.

### M4 — Durable feedback and coaching report

**Change / why:** Introduce relational turns/evaluation events with migrations, uniqueness
and evidence identity. Replace count-based quality claims with reports reconstructed from
persisted events. Record corrections/helpfulness without overwriting original evaluations.

**Learn:** Relational design, constraints, query design, event history and provenance.

**Owner work:** Draw the schema, implement a report query, and decide correction/version
semantics and the graph/ledger source-of-truth boundary.

**Agent work:** Scaffold migrations/persistence, propagate contracts, implement report UI
and regression tests. Record a storage decision before introducing new infrastructure.

**Acceptance:**

- Every evaluation identifies session, actual question/answer, evidence, outcome,
  prompt/rubric/schema/model versions, trace ID and nullable usage/cost.
- Advancing never erases the business record; a report rebuilds from turns/events alone.
- Skipped, presented and answered questions are separate; follow-up counts are not scores.
- Reports show evidence, uncertainty and a next practice action. Rewrites never invent
  candidate achievements. Self-reported model confidence is not a calibrated probability.
- Identity constraints and basic repeat-write safety are tested. Document the remaining
  checkpoint/ledger crash windows for M5; do not claim cross-store atomicity.
- Owner can reconstruct one report and explain its provenance.

### M5 — Retry, interruption and recovery

**Change / why:** Add command/turn idempotency, stale-turn checks, per-session serialization
or version control, bounded provider timeouts/retries, and correct client recovery.
Initialize/close shared resources through application lifespan. Move blocking extraction
work off the event loop where needed; add a queue only if measured processing needs it.

**Learn:** Race conditions, partial failure, replay, cancellation and backpressure.

**Owner work:** Reproduce double submission and process interruption; predict state before
running the experiment. Distinguish command retry, model retry and graph replay.

**Agent work:** Build failure-injection harnesses, temporary-database recovery tests,
shared command adapters and frontend reconnect/draft delivery handling.

**Acceptance:**

- Repeated logical commands do not duplicate answers/evaluations or advance twice.
- Stale answers, REST/WS races and two clients for one session have explicit outcomes.
- Temporary SQLite tests survive actual saver close/reopen; checkpoint/ledger writes
  interrupted on either side recover without missing reports or duplicate evaluation.
- Timeout/429/5xx failures are bounded; layered SDK/application retries do not multiply
  unnoticed. Unknown completion does not trigger blind command resubmission.
- UI restores history/active follow-up, preserves drafts and distinguishes sent/accepted/
  failed states. Errors retain recoverable state and do not leak internal payloads.
- Owner can diagnose a recorded failure and explain the recovery guarantee's limits.

### M6 — Product and dependency simplification

**Change / why:** Extract one shared frontend interview controller, default to text, add
parsed-information review/correction and transcript review before scoring. Replace video
read-aloud simulation with supported behavior or hide it. Isolate legacy imports and
unsupported modes; remove unused store/scaffolding only after reference checks.

**Learn:** Cohesion, compatibility, dependency lifecycle, accessibility and user trust.

**Owner work:** Decide supported paths, compare raw extraction/normalization quality,
and demonstrate one complete browser refresh/retry flow with synthetic data.

**Agent work:** Mechanical extraction, dependency audits, UI wiring, selected component
and browser tests. Do not initialize a legacy model merely to inspect an absent ID.

**Acceptance:**

- Text works without camera, mic, custom endpoint or voice SDK; capabilities reflect
  backend availability. Infrastructure details move to developer settings.
- Parsed data and transcripts can be reviewed; corrections have provenance.
- Shared controller owns connection/lifecycle behavior across retained presentations.
- Keyboard access, focus/status feedback, permission denial and retry flows are checked.
- Full payload console logging and unsupported scoring/model claims are removed.
- Legacy resume IDs still work or have an explicitly scoped/tested migration plan;
  no runtime database or original uploads are silently removed.
- No provider/ML/state dependency remains solely to appear in a portfolio stack list.
- Owner can justify each retained component and demonstrate its failure state.

### M7 — Optional job-description targeting

**Change / why:** After the owner selects this track, accept one JD, extract a typed
JobProfile with cited requirements, and create a user-reviewed competency plan. Preserve
resume-only mode. Use deterministic matching and context, not a vector database.

**Learn:** Structured extraction, evidence grounding and ranking.

**Owner work:** Define requirements versus preferences, label a small extraction set,
and implement initial question-selection rules.

**Agent work:** Form/API plumbing, evidence display and benchmark expansion.

**Acceptance:** Every chosen competency traces to the JD; unsupported extraction and
injection cases are tested; the user can correct focus; coverage/duplication metrics and
resume-only regression results are recorded. Owner can explain a question's selection.

### M8 — Optional adaptive-selection experiment

**Change / why:** Only after M7 and explicit track selection, compare fixed order with
simple deterministic coverage/gap rules under the same turn budget. Do not deploy an
adaptive policy solely because it is more elaborate.

**Learn:** Baselines, policy evaluation, confounding and uncertainty.

**Owner work:** Define success before coding and manually explain choices for sample
states. Define initial/unknown values and coverage guarantees; the old multiplicative
priority formula is not an approved algorithm and can suppress topics through zeros.

**Agent work:** Replay tooling, reason-code records and experiment summaries.

**Acceptance:** Equal inputs give the same choice/reason; results show whether relevance,
coverage and repetition improved without more turns. Publish negative results too.
Retain fixed order if adaptation fails the predefined comparison. Owner can explain why.

### M9 — Controlled deployment and real-user gate

**Change / why:** After M6, independently of M7/M8, prepare authenticated operation with
managed identity, resource ownership, migrations, persistence, retention and recoverability.
Use PostgreSQL for production metadata/checkpoints when justified by the recorded runtime
design; avoid an unnecessary intermediate production migration. Add object storage if
files must survive instance replacement. Keep one backend application.

**Learn:** Authorization, data lifecycle, deployment, backups, operations and release risk.

**Owner work:** Write the threat model, implement one ownership check, deploy manually
once, restore a backup, and diagnose a simulated provider outage.

**Agent work:** Container/config templates, authorization matrices, deployment automation,
redacted instrumentation and migration/release tests.

**Acceptance — all required before public beta or real-user data processing:**

- User A cannot access/change/delete User B's resume, session, report, audio or WS stream.
- Opaque IDs/path containment, type/signature/size validation, upload limits, WS command
  limits, resource isolation and third-party processing consent are implemented/tested.
- Retention/export/deletion covers originals, parsed copies, checkpoint history, ledger,
  reports, audio, traces and backup policy. No sensitive payloads in production logs.
- Prompt injection, unsupported factual assertions and relevant paired scoring tests
  pass the recorded release criteria; feedback is coaching, not hiring validation.
- Clean checkout builds/tests/migrates/starts; CI gates changes and documented rollback
  and backup restoration are exercised. Secrets are configured outside Git.
- Load/restart/provider-outage tests show no cross-session leakage, lost accepted turns
  or duplicate progression at the declared single-instance/deployment limits.
- Quality/latency/error/cost budgets and safe correlated diagnostics are recorded.
  Choose one tracing backend only if needed; missing usage remains explicitly unknown.
- Owner can explain storage, access, deletion, release and recovery end to end.

## Agent execution and handoff protocol

1. Read AGENTS, onboarding, this plan and the current handoff. Inspect branch, worktree
   and actual implementation; prior assessment findings may already have changed.
2. Select the smallest unfinished slice whose prerequisites are satisfied. State the
   expected invariant, affected files, owner exercise and verification before editing.
3. Default to learning-oriented collaboration: the owner writes the core exercise;
   the agent explains, reviews, scaffolds and handles repetition. Do not implement an
   entire milestone unasked. Explicit owner delegation of a slice authorizes completing
   it without repeated confirmations; record any remaining owner exercise separately.
4. Reproduce the defect and use real internal components at integration seams. Mock
   LLM/STT/TTS/Chroma and use temporary storage; never mutate real runtime data for tests.
5. Keep changes small and working. Before M3's benchmark exists, add deterministic
   contract/regression cases and record that model-quality delta is not yet measured.
   Afterwards, behavior changes include benchmark/version delta or an explicit reason
   and unresolved verification status; never invent a measured improvement.
6. Run relevant checks from CONTRIBUTING. Update implemented docs only for actual
   behavior; record design alternatives/tradeoffs in short `docs/decisions/` records.
7. Update the handoff with files, commands/results, evidence, owner explanation status,
   open risks and the exact next action. No secrets or raw user payloads in handoffs.
8. Mark a gate complete only with linked evidence. Do not infer human understanding,
   data provenance, release approval or optional-track selection from agent test success.

## Non-goals and portfolio deliverables

No rewrite, extra model providers, multi-agent swarm, single-resume vector RAG,
Kubernetes, Kafka, microservices, graph databases, RL, facial/personality inference or
3D avatars. A GPU server, queue or broad model-routing layer needs a real measured need.

Fine-tuning is a separate future decision requiring a narrow demonstrated failure,
traceable consented/synthetic human-reviewed data, stable benchmark, held-out evaluation,
API baseline and reproducible model/data cards. No arbitrary label count guarantees
readiness. Existing inference/tokenizer artifacts do not demonstrate model training.

Portfolio evidence: synthetic-data demo, architecture diagram, a few decision records,
human-reviewed benchmark with failures, and a recovery demonstration. Claim only results
that the repository reproduces. This project targets Applied AI product engineering;
training-heavy ML roles may be better served by a separate focused modeling project.
