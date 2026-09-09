# Experimental ML Area

This directory is a placeholder for future model experiments. It is not the active
Resume Griller runtime and it does not currently contain a training or evaluation
pipeline.

The production prototype uses hosted API models or the separately hosted Custom/
Hybrid execution path implemented in `backend/app/services/llm_service.py`. Although
`LLM_MODE=local` and `LOCAL_MODEL_*` settings exist, automatic local LoRA loading is
not connected to the active service factory.

## Current contents

```text
ml/
├── configs/                         # placeholder only
├── data/
│   ├── resumes/                     # ignored/private data location
│   ├── interview_qa/                # ignored/private data location
│   └── processed/                   # ignored/private data location
├── evaluation/__init__.py           # no evaluator implementation yet
├── training/__init__.py             # no training implementation yet
└── models/
    ├── checkpoints/                 # ignored checkpoint location
    ├── exported/                    # ignored export location
    └── interview-coach-lora/        # tokenizer/config metadata; no adapter weights
```

There is currently no `train.py`, dataset loader, benchmark runner, dataset card, or
validated exported model. The files under `scripts/ml/` are legacy duplicates, not a
second ML package.

## Fine-tuning gate

Fine-tuning is intentionally outside the core learning-oriented roadmap. It may start only
after all of the following exist:

- a stable, versioned evaluation benchmark;
- consented and traceable labels with dataset lineage;
- periodic human calibration and documented slice performance;
- a repeated failure that prompt, schema, grounding, or routing work cannot fix;
- a hosted-model baseline for quality, latency, and cost;
- a model card, data card, privacy review, rollback, and shadow-release plan.

The learning curve, held-out results and label quality determine whether an experiment
is justified; no arbitrary label count is a readiness gate.

The first reasonable fine-tuning target is a narrow structured task such as gap
classification, competency classification, or evidence verification. Do not begin
by fine-tuning the complete interviewer.

## Required experiment outputs

Any future experiment must record:

- dataset and split versions;
- base model and adapter configuration;
- prompt/rubric/schema versions used to produce labels;
- training configuration and random seed;
- overall and sliced evaluation results;
- comparison with the current API baseline;
- latency, throughput, hardware, and estimated cost;
- known limitations, PII handling, and allowed uses.

An adapter must not be integrated into the backend merely because training loss
decreased. It must pass the same release benchmark as API models in shadow mode.

## Environment

The root `pyproject.toml` is authoritative. The optional `ml` extra currently adds
Transformers, PEFT, and Accelerate; it does not constitute a complete training
pipeline.

```bash
uv sync --extra ml --extra dev
```

Add future ML dependencies through `uv add --optional ml package-name` and commit the
updated root `uv.lock`.

See `DEVELOPMENT_ROADMAP.md` for the current non-goals and optional-track and model-experiment decisions.
