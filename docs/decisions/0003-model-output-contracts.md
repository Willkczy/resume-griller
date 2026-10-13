# 0003 — Explicit model output contracts

Date: 2026-09-22. Updated: 2026-10-13. Status: stages 1–4 implemented; CI pending.

## Why

A syntactically valid JSON object is not necessarily an evaluation. Before M2, the
engine accepted `{}` as 0.425, accepted dimension scores of 100, and converted a
provider timeout into a heuristic judgment. Stage 1 reproduced these defects;
stage 2 enforces the contracts and removes those fallback scores.

## Contract decisions

`backend/app/models/model_outputs.py` defines internal `model-output-v1` contracts.
Stage 3 adds public schemas in `backend/app/models/interview.py`; checkpoint values
remain dictionaries and are projected through those schemas at the transport boundary.

| Outcome | Meaning | Score permitted? | Who creates it? |
|---|---|---|---|
| valid | Structurally valid judgment; accuracy still unproven | Required explicit dimensions | Model, validated by application |
| insufficient_evidence | Cannot judge from available evidence | No | Model, validated by application |
| invalid_output | Response violates JSON or domain schema | No | Application parser |
| provider_failure | Provider call failed | No | Application adapter |
| unsupported_mode | Execution mode cannot satisfy evaluation contract | No | Application |

All evaluation fields are required, including nullable fields and empty lists.
Scores accept JSON numbers in [0, 1], never strings, booleans or non-finite values.
Unknown fields and gap labels are rejected. Every detected gap needs an explanation;
a priority gap must be among the detected gaps. Duplicate gap labels are rejected.
Only valid/insufficient-evidence outcomes may be supplied by model text: a model
cannot claim that an infrastructure error occurred.

The initial scored contract preserves the existing seven explicit dimensions to
bound the migration. It does not endorse their calibration or permit constructing
them from a scalar. Stage 2 explicitly reports compact evaluation as unsupported.
Prompt/rubric identities are defined beside the changed prompts; full execution
metadata propagation belongs to stage 4. M3 tests judgment quality.

Parsing accepts a complete JSON document, without extracting a convenient substring
or repairing markdown. Duplicate object keys are rejected rather than overwritten.
Invalid output retains a safe code, never the original response or validation error
text. Stage 2 removes engine exception/payload logging. Broader provider-adapter
logging and metadata/timing work remain stage 4.

Questions use an exact requested count of nonblank strings (1–20), with duplicate
questions rejected after trimming/case folding. Follow-ups require one nonblank
question field. These checks establish structure, not relevance or usefulness.
No fallback question is silently inserted by these parsers.

## Staged rollout and behavior boundary

1. Stage 1: contracts, synthetic examples and strict expected-failure engine tests.
2. Wire engine/question outputs; eliminate fallback scoring and scalar fabrication.
3. Integrate explicit outcomes into graph/API/UI, synchronizing public types.
   Unavailable evaluation must show no score and must not trigger score-based
   advancement. Specify allowed subsequent commands before implementing this stage;
   do not promise retry safety that belongs to M5.
4. Version prompts/rubrics and capture provider metadata with safe logging/timings.
5. Add deterministic CI and final M2 acceptance evidence.

Stage 2 may need minimal graph compatibility changes to keep its commit runnable;
never let a new outcome silently enter old score-based policy. Full UI integration
remains stage 3. Each stage stops with a corresponding local commit for owner review.

## Reading exercise

Read `tests/test_model_outputs.py` first: compare `valid` with missing fields,
wrong types, insufficient evidence, and provider failure. Then trace
`parse_evaluation` through JSON syntax checking and discriminated-union validation.
Explain why 0 is a legitimate score while a timeout has none. Change a synthetic
example locally and predict the outcome before running the test.

Stage 1 committed three strict `xfail` tests and verified they failed directly with
`--runxfail -k test_engine_`. Stage 2 removes those markers: all three now pass.
The expanded examples are synthetic contract fixtures (`model-output-cases-v2`),
not human-labeled benchmark evidence. Owner understanding remains pending,
independently of test results.


## Stage 2 runtime decisions and reading path

1. `GrillingEngine.evaluate_answer` calls the model even for a short answer, then
   `_parse_llm_evaluation` validates the complete response. Only a `ScoredEvaluation`
   becomes the compatibility `AnswerEvaluation` used by the graph. Existing weights
   remain unchanged and uncalibrated. No substring repair, scalar expansion or
   length/exception heuristic remains.
2. Resume consistency is part of that validated judgment. The unvalidated second
   call and its mutation of the evaluation are removed. The prompt distinguishes
   missing resume evidence from contradiction; benchmark evidence for this instruction
   is not yet available. Context still has an explicit existing truncation limit.
3. Graph `evaluate_answer` stores a non-scored outcome without appending an accepted
   candidate answer, retains the active question, and emits the existing error
   envelope. `route_after_evaluate` checks outcome before applying follow-up caps:
   a failed evaluation cannot complete the last question even at a zero cap.
4. Non-scored outcomes allow read/resume, a new explicit answer submission, skip or
   end. Insufficient evidence gets a distinct message/code but uses the existing
   error envelope until stage 3. An explicit submission is not an automatic retry
   or an idempotency guarantee. The current answer remains in checkpoint state;
   full draft/history restoration is not implemented here.
5. If evaluation succeeds but follow-up generation fails, the accepted answer and
   evaluation remain checkpointed. No interviewer turn or follow-up count is added.
   A recorded `follow_up_failure` prevents another answer to that same turn (409);
   read/resume and skip/end remain allowed. This bounded guard is not M5 recovery.
6. Questions (including prepared Hybrid questions and the standalone resume endpoint)
   use the same complete-JSON, exact-count contract. No default-question padding or
   punctuation repair remains. Failed startup stays pending without an active turn.
   Graph failures retain the existing HTTP 200 error envelope; the standalone
   endpoint returns 502 for invalid output and 503 for unavailable generation.
7. Follow-ups use either the validated suggested question or a validated JSON
   generation result. Compact helpers return unsupported without model calls.
   API-mode provider adapters still must satisfy the same contract regardless of
   configured provider; Custom/Hybrid session mode disables compact evaluation.

Question timestamp entropy was removed while replacing the question prompt.
Follow-up policy has a new identity for its outcome guard; thresholds are unchanged.
Provider failure classification stores safe categories, not exception messages.
Stage 4 still owns complete provider/model/usage/finish-reason/timing/version metadata.
Stage 3 owns the explicit public outcome union and dedicated user experience; stage 2
only added the valid tag to the existing TypeScript evaluation shape.

Read `test_unavailable_evaluation_never_advances_even_at_cap` and
`test_failed_follow_up_does_not_create_turn_or_duplicate_answer` in
`tests/test_interview_flow.py`, then trace `nodes.evaluate_answer`,
`edges.route_after_evaluate`, and `transitions.command_error`. Explain why a low
valid score may trigger a follow-up while a timeout cannot trigger any progression.

## Stage 3 public outcomes and command rules

Implemented on 2026-10-10. Public contract identity: `interview-outcome-v1`.
`graph/public.py` projects one checkpoint into public response data for REST commands,
session reads, WebSocket commands and connections. The public evaluation union keeps
the existing valid score/dimension shape and adds the four non-scored branches. The
generation failure is separate: a failed follow-up can coexist with a valid evaluation.
Question/answer identities associate feedback with the evaluated turn rather than the
next question. Older valid checkpoints gain their missing tag in memory; no migration
or runtime-data rewrite is performed.

| Current condition | Answer text and score | Allowed next commands |
|---|---|---|
| Active question, no failed outcome | Local draft; no current score | Resume, answer, skip, end |
| Insufficient evidence | Checkpointed submitted draft; no score | Resume, new answer, skip, end |
| Invalid output or provider failure during evaluation | Checkpointed submitted draft; no score | Resume, new answer, skip, end |
| Unsupported evaluation mode | Checkpointed submitted draft; no score | Resume, skip, end |
| Follow-up generation failed after evaluation | Accepted answer in conversation; valid score | Resume, skip, end |
| Startup generation failed | No active question or score | Explicit start, end |
| Completed or cancelled | Terminal conversation; final valid feedback on completion | Read only |

`allowed_actions` derives from the same command guard that validates execution; it is
an advisory snapshot, not a concurrency lock. `answer_status` refers to the active
question. `draft_answer` contains only a submitted answer whose evaluation is unscored.
Conversation projection lets the UI distinguish accepted messages from local drafts.
Successful answer feedback is carried in that command's response even when advancing
clears the current evaluation. Reads and read-only resume retain the latest feedback;
the next mutating command clears it. Final valid feedback remains after completion,
while cancellation clears it. This does not replace M4's ledger or make reports
reconstructible.

Both retained rooms show outcome-specific messages, preserve submitted text until the
checkpoint confirms acceptance, and expose manual reconnect. A restored startup failure
does not automatically invoke the model. Reconnect reads state and never replays an
answer, skip, end or failed start. Unsent drafts are local UI state; durable draft storage,
unknown-completion recovery, process restart and idempotent retries remain M5.

Owner exercise: predict the public outcome, score availability and allowed actions for
a valid zero score, insufficient evidence, malformed JSON and timeout. Then trace
`interview_snapshot`, the command guard and the frontend outcome helper. Implementation
was delegated; owner understanding remains pending/not assessed.


## Stage 4 execution metadata and safe diagnostics

Implemented on 2026-10-13. Public snapshot identity is now `interview-outcome-v2`;
the additive metadata contract is `model-execution-v1`. Model-output JSON remains
`model-output-v1`: a model cannot supply execution metadata. Existing prompt/rubric
identities remain unchanged; first metadata identities are assigned to the unchanged
resume Markdown, Hybrid summary and question-context prompts.

`BaseLLMService.generate` returns a request-local `ModelResult`. `call_model` measures
one attempt using a monotonic clock, catches provider failures into safe categories,
and creates an application UUID. The caller parses text and finalizes/logs its outcome.
No singleton "last response" state is used. Legacy injected string-only adapters have
unknown provider fields rather than inferred usage. Active production adapters return
typed results. Hybrid preprocessing records Groq; generic Hybrid routing identifies
the actual selected adapter, including on failure. Unsupported compact evaluation and
reuse of a validated suggested follow-up create no invented call record.

The graph stores only the latest command's `model_executions`. Evaluation and a separate
follow-up call can coexist in this list; successful Hybrid preparation carries its
summary/questions/context records into startup. Read/resume/reconnect preserves IDs
and timings. The next mutating command replaces the list, including skip/end with no
calls. Earlier checkpoints project an empty list. This is not M4's ledger, session cost
aggregation, an external tracing system or M5 idempotency/recovery. Resume parsing and
the standalone question endpoint emit safe execution logs but do not create an
interview checkpoint. Failed Hybrid preparation emits its diagnostics without claiming
that a session was created.

- `configured_model` is the requested model label; `model` is the SDK-reported label,
  which may still be an alias rather than an immutable revision. Missing provider
  request IDs, token counts and totals stay null. Reported zero stays zero. No local
  token estimates or invented totals/costs are substituted. Cost and external trace
  identity are null until a trustworthy source is implemented.
- OpenAI/Groq/Custom map prompt/completion/total usage. Anthropic's input counter
  excludes its separately reported cache-read/cache-creation counts; those counters
  are preserved separately and total remains null. Gemini's protobuf message presence
  distinguishes absent usage from reported default-zero counters. Its installed SDK
  exposes model version and finish reason but no public request ID here. SDK-shaped
  mocked fixtures verify these mappings. The OpenAI field mapping follows the
  [Chat Completions response reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
- Explicit finish reasons outside the supported normal stops reject the output,
  including length limits, content filters, safety blocks and tool calls. Missing
  finish reason remains unknown and still requires strict parsing. The six synthetic
  complete-looking evaluation cases with incomplete/blocked reasons are now rejected;
  the previous adapters discarded those reasons before parsing. This is a contract
  behavior delta, not evidence of improved model judgment.
- `LLM_REQUEST_TIMEOUT_SECONDS` bounds each active non-streaming call. SDK automatic
  retries are disabled; Gemini also receives an RPC timeout and `retry=None`. Cancelling
  the await cannot prove the remote provider stopped or immediately terminate Gemini's
  synchronous worker thread. No automatic retry or safe-resubmission promise is made.
  Unused streaming helpers do not emit this non-streaming metadata contract.
- Structured diagnostics contain application-owned call ID, operation, safe outcome/
  error category, token counts, latency and version identities. Free-form model/request
  labels are kept out of logs as well as prompts, resume/answer/audio content, SDK
  response bodies and exception text. SDK debug logging is suppressed at configuration
  and before calls, including late imports. Retained voice/WS exception diagnostics
  now use fixed messages; voice behavior and lack of device verification are unchanged.

Owner exercise: predict null versus zero usage, compare configured and reported model
identity, then trace one timeout and one truncated response from `ModelResult` through
`call_model`, the engine, graph checkpoint and both public transports. Explain why the
call UUID is not an idempotency key and why a timeout has no score. Implementation was
delegated; owner understanding remains pending/not assessed. Review this local stage
before authorizing stage 5 CI. Do not start M3/Jev yet.
