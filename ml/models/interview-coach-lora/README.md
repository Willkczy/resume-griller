---
base_model: mistralai/Mistral-7B-Instruct-v0.2
library_name: peft
pipeline_tag: text-generation
tags:
  - experimental
  - lora
  - peft
---

# Interview Coach LoRA — Incomplete Experimental Artifact

## Status

This directory is **not a runnable or validated model release**. It contains tokenizer
and adapter configuration metadata, but no adapter weight file such as
`adapter_model.safetensors` or `adapter_model.bin`.

Do not load it from the backend, publish it as a completed model, or use it to score
interview candidates.

## Known metadata

- Declared base model: `mistralai/Mistral-7B-Instruct-v0.2`
- Adapter library: PEFT
- Intended task family: text generation
- Intended project context: experimental interview coaching

## Missing release evidence

The repository does not provide:

- adapter weights;
- training code or immutable training configuration;
- dataset or data card;
- dataset consent, provenance, filtering, or PII policy;
- train/validation/test split definitions;
- training metrics or hardware record;
- benchmark results against the current API models;
- slice, bias, safety, or prompt-injection evaluation;
- intended-use validation or a production integration path.

Because those items are absent, quality, language support, limitations, license
compatibility, and reproducibility are unknown.

## Allowed use

The current files may be inspected as historical experiment metadata only. Any future
model work must create a new, versioned experiment with a complete model card and pass
the fine-tuning gate in `ml/README.md` and `DEVELOPMENT_ROADMAP.md`.

## Out-of-scope use

- hiring or employment decisions;
- candidate ranking;
- unsupervised scoring of real people;
- processing real resumes without consent and an approved data policy;
- production inference or fallback routing.

## Future release requirements

A replacement model card must document at least:

1. developer, license, base-model revision, adapter revision, and code commit;
2. intended and prohibited uses;
3. dataset lineage, consent, redaction, language, occupation, and seniority slices;
4. training configuration, hardware, duration, seed, and checkpoint selection;
5. benchmark protocol and comparisons with API baselines;
6. quality, calibration, safety, bias, latency, and cost results;
7. known limitations, monitoring, rollback, and deletion procedures.
