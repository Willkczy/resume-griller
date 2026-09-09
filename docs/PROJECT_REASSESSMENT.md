# Project reassessment and refactor rationale

Assessment date: 2026-09-09. Plan adopted: 2026-09-10.
This historical assessment supports [the roadmap](../DEVELOPMENT_ROADMAP.md).
Recheck findings against current code before acting; current progress lives in
[the handoff](REFACTOR_HANDOFF.md).

## Product and architecture

Resume Griller is a FastAPI + Next.js mock-interview prototype. PDF/TXT uploads are
saved locally, extracted with deterministic tools and normalized to Markdown by Groq.
Session creation loads the complete parsed resume into LangGraph state. REST and WS
invoke one deterministic interview graph; SQLite checkpoints use session ID as thread
ID. GrillingEngine scores answers and creates follow-ups through provider adapters.
ChromaDB is a legacy resume fallback, not active per-question retrieval. Recorded audio
uses batch Deepgram STT and ElevenLabs TTS; webcam preview is local, not video analysis.

Keep the modular monolith, graph, injected services, full-document context and simple
local persistence. The core issue is correctness and evidence, not a missing framework.

## Findings and reproduction targets

| Finding | Code to inspect | Consequence |
|---|---|---|
| Initial state overwrites start action | `backend/app/api/routes/session.py::_invoke_graph`, `graph/state.py::create_initial_state`, `graph/nodes.py::handle_error` | Real graph with mocked services fails response validation before a model call |
| Follow-up answers use original main question | `backend/app/graph/nodes.py::evaluate_answer` | Wrong primary scoring target |
| Follow-up policies disagree | `backend/app/graph/edges.py::route_after_evaluate`, `core/grilling_engine.py::should_grill` | Sufficient answers can receive forced follow-ups |
| Empty/unbounded evaluation accepted | `core/grilling_engine.py::_parse_llm_evaluation` | `{}` produced 0.425; dimensions of 100 produced score 100 |
| Scalar expanded into detailed scores | `core/grilling_engine.py::_parse_compact_evaluation` | Custom mode does not independently assess seven dimensions |
| Evaluation cleared on progression | `graph/nodes.py::advance_question`, `ask_question` | Checkpoint history is not a stable reporting ledger |
| REST/WS lifecycle checks differ | `api/routes/session.py`, `websocket.py` | Shared graph does not guarantee shared validation |
| Reconnect reports indexed main question | Session detail and WS connection/start handlers | Active follow-up and browser history can be lost |
| Context truncated despite full state | `_llm_evaluate`, `_llm_evaluate_compact`, `check_resume_consistency` | Later resume evidence may not reach evaluator |
| Preset short-answer and English keyword scoring | `core/grilling_engine.py::evaluate_answer`, `_fallback_evaluation` | Concise/cross-language answers can be unfairly penalized |
| Provider strings discard metadata | `services/llm_service.py` | No normalized usage, timing, finish reason or failure contract |
| Count-based quality claims | `frontend/src/app/result/[sessionId]/page.tsx` | Skips/disabled follow-ups can produce unsupported praise |
| Duplicated client controllers, unused Zustand store | `frontend/src/components/interview/`, `src/stores/` | Drift and weak reconnect handling |
| Read-aloud is a timer | `VideoInterviewRoom.tsx::readQuestion` | Control does not perform advertised action |
| Unsupported fine-tuned model claims | `ResumeUploader.tsx`, `ml/models/interview-coach-lora/` | Metadata/inference artifacts are not a validated trained-model release |
| Voice defaults/dependencies disagree | `backend/app/config.py`, `.env.example`, `pyproject.toml` | Default capabilities cannot be trusted |
| No resource ownership/cascade data lifecycle | Resume/session routes, checkpoint storage | Resume deletion leaves checkpoint copies; no cross-user isolation |
| Full payload browser logs | `frontend/src/lib/websocket.ts` | Sensitive content may appear in diagnostics |

Further debt: synchronous extraction inside async upload paths, client-supplied IDs in
filesystem paths without a dedicated containment contract, heavy legacy imports in root
dependencies, and unused duplicate schemas/scaffolding. Address by milestone, not a rewrite.

The notebook inspected has four cells, no saved outputs and loading/generation code,
not a training run. Two tracked PDFs are byte-identical. Provenance of resume-shaped
artifacts is unresolved; the review did not audit all history or validate synthetic status.
Do not publish contact details in issue reports or send fixtures to model providers.

## Validation limits

The existing suite passed 21 tests, and TypeScript/frontend lint passed. Most graph tests
mock GrillingEngine and API tests often mock the graph, missing important seams. The
legacy PDF test prints extracted fields without substantive assertions. The startup and
parser reproductions used synthetic inputs and in-memory state, with no live provider.
Coverage, production build, live voice/models and browser usability were not rerun.

## Recruiting evidence and interpretation

The following U.S. employer/ATS postings informed the plan. They were available or
indexed at assessment time; some 2026 start deadlines had passed. Crawl dates are not
publication dates, and presence in search does not guarantee an open vacancy. This is a
small directional sample weighted toward AI product companies, not a national survey.
Do not reuse it as current hiring availability without checking again.

| Employer / source | Level and location | Relevant signals |
|---|---|---|
| [Notion new grad AI](https://jobs.ashbyhq.com/notion/7e6dc7fe-7ddd-42c1-8928-13f7bddb9ec9/) / [early career AI](https://jobs.ashbyhq.com/notion/85947779-6b87-466a-98bc-30a640448c28/) | New grad / separately 0–2 years; SF | Product integration, relational data, evals, reliability, communication |
| [FurtherAI Software/AI Engineer](https://jobs.ashbyhq.com/furtherai/282de7b6-0117-4e7a-b1f6-c2d256af4a47) | New grad; SF | Production Python/TypeScript, document pipelines, evaluation, APIs, deployment |
| [NewsBreak Applied AI Engineer](https://job-boards.greenhouse.io/newsbreak/jobs/4700278006) | New grad; Mountain View | Testability, async/concurrency, schemas, APIs, observability and measurable quality |
| [Quora/Poe AI Engineer](https://jobs.ashbyhq.com/quora/6df58d3e-855a-423e-99fd-a56ac8824b34/) | 2025–2026 graduates; U.S.-eligible remote | Python/TypeScript, ML foundations, LLM apps, RAG, evals and reliable deployment |
| [Netic ML Engineer](https://jobs.ashbyhq.com/netic/f645d611-ae5f-40cc-bfa8-613286060a76/) | New grad; SF | Evals, data models, APIs, customer feedback and production ownership |
| [Root Access ML Engineer](https://jobs.ashbyhq.com/root-access/dfa466b4-5851-4f78-9ca7-70d76e3abb5f) | 1–3+ years; NYC; comparison role | Reliable Python, messy data, PyTorch/transformers, AWS and product integration |
| [TikTok content understanding](https://lifeattiktok.com/search/7534982208201984274) | 2026 BS/MS; San Jose | Multimodal/LLM knowledge, deep-learning frameworks and model development |
| [TikTok logistics](https://lifeattiktok.com/search/7675843332462872885) | 2027 graduate; Seattle | ML/statistics/coding; preferences include benchmarks, agent workflows and post-training |

Common foundations: programming, decomposition, communication, hands-on AI understanding;
ML depth depends on role. Evals, data processing and production iteration recur as work
expectations even when not minimum entry requirements. Internships/projects, framework
experience, RAG/tool concepts and cloud/container familiarity are often preferences or
role-dependent requirements. Fine-tuning/GPU/distributed training matter more for specific
model-focused tracks than for every AI product role.

The project already demonstrates integration and stateful workflow work. It demonstrates
little calibrated evaluation, relational domain modeling, recovery or production operation.
Those gaps solve real product problems and therefore deserve priority. Keeping dormant
RAG or LoRA artifacts does not demonstrate expertise. A separate modeling project may
serve training-heavy roles better than forcing training into this interview coach.

## Portfolio claims and tradeoffs

Deliver a synthetic demo, clear flow diagram, a few design decisions, benchmark results
including failures and a recovery demonstration. Explain how an answer is linked to the
question asked, why a rubric is credible, and how retries behave. Avoid claiming validated
hiring scores, production RAG or successful fine-tuning without reproducible evidence.

A ledger and schemas make mistakes inspectable; they do not establish evaluator quality.
PostgreSQL does not make a multi-step graph transition atomic. More dimensions do not
mean more accurate feedback. Human calibration and failure experiments remain necessary.
Optional JD/adaptive work should follow a useful and measurable core product.
