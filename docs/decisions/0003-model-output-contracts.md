# 0003 — Explicit model output contracts

Date: 2026-09-22. Status: M2 stage 1 implemented; runtime integration pending.

## Why

A syntactically valid JSON object is not necessarily an evaluation. The current
engine accepts `{}` as 0.425, accepts dimension scores of 100, and converts a
provider timeout into a heuristic judgment. Stage 1 records these defects without
claiming to repair production behavior.

## Contract decisions

`backend/app/models/model_outputs.py` defines internal `model-output-v1` contracts.
These do not yet replace public API schemas or checkpoint dictionaries.

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
them from a scalar. Stage 2 must either obtain real dimension judgments in compact
modes or explicitly report unsupported evaluation. Rubric/prompt revisions and their
identities belong to the later wiring/versioning stages; M3 tests judgment quality.

Parsing accepts a complete JSON document, without extracting a convenient substring
or repairing markdown. Duplicate object keys are rejected rather than overwritten.
Invalid output retains a safe code, never the original response or validation error
text. This does not fix existing engine exception logging, which remains later M2 work.

Questions use an exact requested count of nonblank strings (1–20), with duplicate
questions rejected after trimming/case folding. Follow-ups require one nonblank
question field. These checks establish structure, not relevance or usefulness.
No fallback question is silently inserted by these parsers.

## Staged rollout and behavior boundary

1. This commit: contracts, synthetic examples and strict expected-failure engine tests.
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

The three strict `xfail` tests are tracked production defects, not successful fixes.
Run with `--runxfail -k test_engine_` to see them fail directly. Stage 2 must remove
those markers as each defect is fixed. The examples are synthetic contract fixtures
(`model-output-cases-v1`), not human-labeled benchmark evidence. Owner understanding
remains pending, independently of test results.
