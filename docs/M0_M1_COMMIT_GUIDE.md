# M0–M1 commit reading guide

Recorded 2026-09-20 after the owner authorized local commits. These commits package
completed work into coherent review units; the handoff records the actual development
slices and failing-before/passing-after evidence. This is a guided reading path, not
required exercises or evidence that owner learning is complete.

## Commit map

| Order | Commit | Purpose | Start reading |
|---|---|---|---|
| 1 | `af60e12` | Safe local baseline: synthetic fixture, runtime ignores, text-only defaults | `tests/test_parser.py`, `tests/test_voice_baseline.py`, decision 0001 |
| 2 | `c7c1d2a` | Correct sequential interview: actual turns, shared policy and command validation | Decision 0002, then the flow below |
| 3 | `docs(refactor): record M0-M1 evidence and commit reading guide` | Current docs, data-remediation record, acceptance and this map | `REFACTOR_HANDOFF.md` |

The M1 commit keeps coupled state, graph, API and frontend contracts together. Splitting
those changes into independent commits would obscure dependencies and risk broken
intermediate states. Tests accompany their implementation instead of appearing in a
separate final test commit. Existing `.claude/settings.local.json` edits are excluded.
Removed historical artifacts are absent from the sanitized parent history; commit 1
adds preventive rules and a safe fixture, not another historical-data removal.

## Read the diff without changing your checkout

```bash
git log -3 --format=fuller
git show --stat af60e12
git show af60e12 -- tests/test_parser.py backend/app/config.py tests/test_voice_baseline.py
git show --stat c7c1d2a
git show c7c1d2a -- backend/app/graph/state.py backend/app/graph/nodes.py
git show c7c1d2a -- backend/app/graph/commands.py backend/app/graph/transitions.py
git show c7c1d2a -- tests/test_interview_flow.py tests/test_follow_up_policy.py
```

A commit's message explains why it exists and what was verified. `git show` exposes
the corresponding patch. These commands are read-only; no reset or checkout is
needed to follow the history while retaining local runtime data and settings.

## Follow one request through the implementation

1. **Creation:** `session.py::create_session` loads synthetic Markdown in integration
   tests and creates initial graph state. `execute_command` merges defaults first,
   then writes the requested action. This order prevents initial `action=None` from
   silently replacing `start`.
2. **Boundary:** `commands.py::execute_command` reads the checkpoint and calls
   `transitions.py::command_error`. Both HTTP and WS use this path. Invalid commands
   stop before writes or provider initialization; the adapters only map errors into
   their own wire format. There is no second session-state store.
3. **Presentation:** `nodes.py::ask_question` creates a `QuestionTurn` containing the
   exact prompt and a stable ID such as `session:q1`. LangGraph merges the returned
   partial update. Conversation uses an append reducer; active_question is replaced.
4. **Answering:** `evaluate_answer` uses `active_question.prompt`, not merely
   `questions[current_question_index]`. The candidate message links to that question
   and gets an answer ID. This fixes follow-up answers being scored against a parent.
5. **Decision:** `core/follow_up_policy.py::should_follow_up` chooses follow-up/advance
   from evaluation, count and cap. It has no I/O and does not mutate evaluation.
   Both engine and graph use it, so transport choice cannot change this policy.
6. **Follow-up or advance:** `generate_follow_up` creates `session:q1:f1` with parent
   `session:q1`. A repeated follow-up has its own prompt/ID. `advance_question` resets
   the current turn and the next ask creates `session:q2`. Completion clears it.
7. **Read/reconnect:** `get_active_question` and `active_question_response` project
   the same prompt for REST and WS. Start while asking is read-only. Legacy checkpoints
   derive a prompt from existing history without a migration; missing follow-up text
   rejects commands instead of guessing the indexed main question.

The relevant tests make these concepts observable: startup tests use the real graph;
repeated-follow-up tests inspect actual mocked-model prompts and correlation IDs;
lifecycle tests compare REST/WS rejection and unchanged checkpoints; legacy cases
check safe recovery/refusal. The custom compact path is covered too.

## Evidence and limits

M0's exported committed snapshot passes 29 Python tests. M1's integrated implementation
passed 73 Python tests, Python lint/format checks, graph smoke, frontend types/lint
and production build. The handoff records commands and the committed-snapshot reruns.
Tests mock external models and use synthetic temporary data; internal adapters, graph
and engine execute normally in the new integration suite.

This demonstrates sequential software behavior. Stable turn IDs do not serialize
concurrent requests or make duplicate answers safe. Existing model parsing/fallbacks,
score calibration, evaluation retention and partial-node recovery remain limitations.
No model-quality benchmark delta is claimed. M2 is the next eligible milestone and
requires separate authorization. Deferred GitHub PR cleanup remains a release issue;
public beta and real-user processing stay blocked until M9 passes.
