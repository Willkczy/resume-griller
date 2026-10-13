# Shared interview commands and active turns

Date: 2026-09-20. Scope: M1, local synthetic-data correctness.
Implementation was explicitly delegated. Owner learning remains pending/not assessed.

## Problem and decision

Initial-state defaults overwrote the start action with null. Route-level fake-graph
tests missed the defect. Follow-up answers were evaluated against the indexed main
question, and reconnect returned that main prompt. Graph and engine policy disagreed;
REST and WS accepted different lifecycle commands.

Both adapters now call `backend/app/graph/commands.py::execute_command`. It reads the
latest checkpoint, validates through the pure `transitions.command_error`, and injects
services only for start/answer. It owns no workflow state. LangGraph nodes remain the
only progression mechanism, with `session_id` as `thread_id`. New-state defaults are
merged before the command so `action=None` cannot overwrite `start`.

## Sequential transition contract

| Checkpoint phase | Start | Answer | Skip | End |
|---|---|---|---|---|
| Pending | Generate and ask | Reject 400 | Reject 400 | Cancel |
| Asking with active prompt | Read-only resume | Evaluate active turn | Record skip and advance | Cancel |
| Asking without recoverable prompt | Reject 409 | Reject 409 | Reject 409 | Cancel |
| Generating/evaluating | Reject 409 | Reject 409 | Reject 409 | Reject 409 |
| Completed/cancelled | Reject 400 | Reject 400 | Reject 400 | Reject 400 |
| Unknown phase | Reject 409 | Reject 409 | Reject 409 | Reject 409 |

Unknown commands and whitespace/non-string answers are rejected. REST schema errors
may return 422 before command validation; WS errors use an error envelope, with
`data.error_code` and `data.status_code` for shared command rejections. REST exposes
start via session creation; WS also exposes start/resume on an existing session.
Malformed WS messages leave the connection usable. Terminal sessions have no active
prompt; chat/video clients route terminal reconnects to results. Existing response
kinds remain question/follow_up/complete/error.

Provider setup failures return a safe 503 before a graph write; skip/end need no
provider. An invocation failure may leave partial checkpoint progress. Existing engine
fallbacks can also absorb provider/model errors. This change does not establish
recovery, concurrency serialization, exactly-once answers or retry idempotency (M5).

## Turn identity and compatibility

`graph/state.py::QuestionTurn` stores the actual prompt, main index, follow-up count,
question ID and parent ID. IDs are sequential within a session: `session:q1`, then
`session:q1:f1`. Candidate messages carry the matching question ID and
`question_id:answer` as their answer ID. Skip messages identify the skipped prompt
without claiming an answer ID. These identities support sequential correlation, not
concurrent submission protection or an evaluation ledger.

Evaluation reads the active turn, including repeated follow-ups. The next follow-up
is generated from that same prompt. Main-question advance clears the active turn;
asking the next main question creates a new one. Public REST detail and WS connection
and resume responses project the active prompt and additive identity fields.
Backend schemas and frontend types include them; old conversation messages may have
null IDs. Session detail does not claim the list-only `questions_asked` field.

For older checkpoints, reads derive a turn from conversation history without rewriting
runtime data. A main prompt can be recovered by index. A missing follow-up prompt is
not replaced with the main prompt; commands reject it with 409. Accepted answering
upgrades that active turn in the checkpoint. This is compatibility projection, not a
bulk migration or tested process-restart guarantee.

## One follow-up policy

`core/follow_up_policy.py`, version `follow-up-v1`, is pure and used by graph routing
and `GrillingEngine.should_grill`. The cap takes precedence; below-cap first answers
under 0.72 receive a follow-up. Later insufficient answers, scores below 0.58, or
significant gaps before two follow-ups also trigger one. Otherwise advance.

These rules preserve the prior engine policy rather than inventing calibrated
thresholds. A sufficient high-scoring first answer is no longer always followed up.
Evaluation no longer changes sufficiency solely to force routing or eagerly generates
a follow-up that routing might discard. Policy version is attached to evaluation and
follow-up metadata. Existing score parsing, context truncation, custom scalar-derived
scores and count-based reports remain limitations for M2–M4.

## Verification and learning handoff

`tests/test_interview_flow.py` uses real FastAPI adapters, a real LangGraph with
MemorySaver, real GrillingEngine and a temporary synthetic Markdown resume; only the
model and service factory boundary are mocked. It checks startup, repeated follow-ups,
standard/custom targeting, multi-question advance/skip, cancellation, invalid commands,
read-only resume, legacy projection and setup-failure isolation. The pure policy table
checks both graph and engine against the same explicit expected decisions.

Startup, wrong-target/reconnect, forced-follow-up, blank-answer and terminal-end
regressions failed before correction. These deterministic cases measure software
behavior, not model quality; no benchmark delta or quality improvement is claimed.
Real providers, browser interactions and SQLite process-restart/concurrency behavior
are not established by these tests.

Optional learning-agent topics: trace the default/action merge; explain append-only
conversation reducers versus replaceable active state; compare pure transition checks
with orchestration; explain why fake graph adapters missed startup; distinguish turn
identity from command idempotency. No owner learning completion is inferred.
