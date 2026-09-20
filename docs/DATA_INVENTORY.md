# Data inventory and M0 disposition

Current disposition: the dated sections below preserve the audit/remediation sequence.
Local and remote branch sanitation is complete; GitHub PR-reference cleanup is
explicitly deferred. The owner authorized local synthetic-data M1 work and then local
implementation commits. See [the current handoff](REFACTOR_HANDOFF.md) for superseding
status; old pending-authorization statements below are historical, not current gates.

Inspected 2026-09-12. This is a code/storage inventory, not a completed historical
PII audit. No resume content was sent to providers, and unrelated runtime data was preserved.
The owner explicitly requested removal of the artifacts below and separately approved
removal of their derived prompt export. They are removed from the working tree; their
historical copies remain blocked for use/distribution pending provenance or separately
authorized remediation. Removal is not evidence of synthetic origin.

## Tracked artifacts

| Artifact | Repository evidence | Provenance / disposition |
|---|---|---|
| `data/sample_resumes/Resume1.txt` | Tracked sample | Unknown origin; removed at owner request |
| `data/sample_resumes/Resume2.txt` | Tracked sample | Unknown origin; removed at owner request |
| `data/sample_resumes/Resume3.txt` | Tracked sample | Unknown origin; removed at owner request |
| `data/sample_resumes/Resume4.txt` | Tracked sample | Unknown origin; removed at owner request |
| `data/sample_resumes/resume_sp.pdf` | Tracked PDF | Unknown origin; removed at owner request |
| `frontend/data/uploads/resume_sp_20251217_144220_fd7dbd5e.pdf` | Byte-identical copy of the sample PDF | Unknown origin; removed at owner request |
| `LLM_Inference.ipynb` | Four cells; zero saved outputs in current file | Unknown source/input provenance; removed at owner request |

The PDFs have matching SHA-256 digests (computed locally without extracting contact
information). Relevant path history includes `8adc190`, `dd70f27`, and `4fec1b6`.
Commit names, plausible sample names and lack of notebook outputs do not establish
synthetic provenance. History content has not been comprehensively audited. Root and
backend `requirements.txt` are dependency documents, not resume fixtures.

New `tests/test_parser.py` input is authored synthetic text: an explicitly synthetic
candidate, a reserved `.invalid` email and generic skills. The test creates a PDF in
pytest temporary storage, exercises real PDF extraction and asserts results without
printing contact fields. Other automated tests use in-code synthetic inputs and
mock provider/retrieval boundaries; no tracked sample is needed by this parser test.

`data/exported_prompts.json` was also tracked and referenced the sample PDF with
derived prompts. It was removed with separate explicit owner authorization. No
unrelated runtime copies were removed.

## Copies, processors and deletion boundaries

Paths below are defaults; environment settings can redirect storage. Relative paths
resolve from the backend working directory, which explains why nested data folders
need ignore coverage. Existing local payloads were not opened for this inventory.

| Data | Location / flow | Retention and deletion gaps |
|---|---|---|
| Original resume | `UPLOAD_DIR`, normally `data/uploads/`; saved by resume upload route | Retained until explicit deletion; upload failure attempts original cleanup |
| Extracted text | Process memory in `InterviewRetriever._extract_text` | PDF/TXT extraction is local; raw extraction is not separately versioned |
| Normalized resume | Groq receives raw extraction; Markdown saved under `PARSED_RESUME_DIR` | Resume deletion removes Markdown but does not remove checkpoint copies |
| Legacy chunks/embeddings | `CHROMA_PERSIST_DIR`; local Chroma and embedding model | Used when Markdown absent; deletion takes legacy branch only when no parsed copy exists; no guaranteed cascade |
| Resume, questions, answers, evaluations | Graph state; `CHECKPOINT_DB_PATH`, normally `data/interview_checkpoints.db` | Full resume and conversation can occur in multiple checkpoints; session deletion is separate from resume deletion; no TTL/backup deletion policy |
| Provider prompts/results | Selected Groq/Gemini/OpenAI/Anthropic API; Custom endpoint; Hybrid Groq preprocessing plus custom endpoint | Provider-side retention/consent is not enforced or established by repository code; hosted custom service is also a trust boundary |
| Hybrid context | In-process service context cache | Derived resume/question context; no coordinated resume-deletion contract |
| Audio/transcript | Browser recording buffers and base64 WS/HTTP payloads; Deepgram receives audio; ElevenLabs receives text when enabled | No dedicated audio file store in active voice services; transcripts become answers in graph; no end-to-end retention/export guarantee |
| Browser copies | Component state, rendered conversation, audio playback; WS client console logs full payloads | Refresh/closing UI is not server deletion; developer tools/log exports may preserve content |
| Server logs/errors | stdout/structured logs; parser IDs/paths and provider error paths | No unified payload redaction or retention policy; provider errors may contain sensitive content |
| Experimental artifacts | Notebook, `ml/` data/model paths | No provenance or trained-model quality claim inferred; model downloads may use external registries |
| Backups / Git history | Operator copies and historical tracked blobs | No implemented coordinated erasure; ignore rules do not remove tracked files/history |

Request flow: upload -> original file -> local extraction -> Groq normalization ->
Markdown -> session loads full resume -> graph invokes selected model -> checkpoint
persists conversation/state -> REST/WS exposes response. An answer enters the same
graph thread and can be copied into model prompts, checkpoints and browser logs.
LangGraph remains the sole workflow-state owner; this inventory adds no new store.

## Ignore verification and unresolved gate

`.gitignore` now covers root and nested `data/uploads/`, `data/parsed_resumes/`,
and `data/chromadb/`. Database extensions are ignored. `git check-ignore --no-index`
checks synthetic path names without writing payloads. Previously nested uploads and
parsed Markdown were not covered. Ignore changes are preventive, not historical remediation; the named tracked
artifacts were removed separately under explicit authorization. Audio/log files outside these paths are not covered
by a blanket rule, so never stage them without inspection.

Next: resolve historical provenance/distribution disposition for removed artifacts.
Origin remains unknown, and current-tree removal alone does not clear Git history.
If real PII is confirmed,
prepare a separately authorized removal/history/distribution remediation scope,
including copies; do not delete files or rewrite history implicitly. M0 cannot be
closed by this inventory alone. Public beta and real-user processing remain blocked
until M9. Learning status is pending/not assessed, separate from engineering gates.

### Historical review status and remediation scope

- Scope starts with all historical versions of the eight removed paths (four TXT,
  two PDF, notebook and prompt export), including renamed/copied blobs in reachable
  local refs. Inspect locally with sanitized findings only; never echo contact data,
  prompts or credentials. A full secret/PII audit remains a separate claim.
- Record categories and affected blob/commit identities, not personal details. If
  credentials are discovered, credential rotation is separate from deleting blobs.
- Before rewriting, establish remote/fork/distribution scope and agree backup,
  collaborator coordination and recovery procedures. A local rewrite cannot erase
  third-party clones or provider-retained copies.
- Current authorization covers only current-tree removal of the eight paths. It
  does not cover history rewriting, force pushes, deleting runtime stores or backups.
  Keep historical distribution blocked until provenance or remediation is resolved.

## 2026-09-15 historical review update

The bounded local review is complete; see [HISTORICAL_DATA_REVIEW.md](HISTORICAL_DATA_REVIEW.md)
for blob identities, sanitized findings, reachability and a concrete remediation scope.
Contact-like data remains reachable in historical artifacts, including notebook source.
Origin is still unknown. Historical distribution remains blocked; no rewrite occurred.

## 2026-09-20 remediation update

The owner authorized isolated local removal and reports being the only user.
The isolated rewrite is verified; see [HISTORY_REMEDIATION.md](HISTORY_REMEDIATION.md).
This supersedes earlier statements that no rewrite was authorized: only the isolated
copy was rewritten. Original checkout, remote and unrelated runtime data are unchanged.
Adoption/remote cleanup remain pending; synthetic provenance was not inferred.

## 2026-09-20 adoption and remote verification update

The owner authorized local adoption and five remote branch updates; both are complete.
Active local history (including three stashes) and the five remote branch histories
are sanitized. A fresh mirror still retrieves the seven blobs through GitHub PR refs
1–5, all closed/merged. M0 remains open for that host-side issue, not branch adoption.
See [HISTORY_REMEDIATION.md](HISTORY_REMEDIATION.md) and the prepared
[GitHub request](GITHUB_HISTORY_CLEANUP_REQUEST.md). No request has been sent.
