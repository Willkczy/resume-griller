# AI Onboarding Guide - Resume Griller

> **For AI Assistants**: This document helps you quickly understand the current state of the Resume Griller project.
>
> **Last Updated**: 2026-01-16 (after Quick Wins Phase 1 completion)

---

## 🎯 TL;DR - What You Need to Know

**Project**: Full-stack AI interview simulator with sophisticated grilling engine
**Status**: Working prototype → Production readiness 6/10 (improved from 3/10)
**Stack**: FastAPI + Next.js + ChromaDB + LangGraph (incoming)
**Current Focus**: Phase 1 remaining tasks (session persistence, tests, auth)

---

## 📊 Current State (2026-01-16)

### ✅ What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| **Grilling Engine** | ✅ Production-ready | 18 gap types, 7D scoring, hybrid mode support |
| **LLM Service** | ✅ Stable | 6 providers (Groq, Gemini, OpenAI, Claude, Custom, Hybrid) |
| **RAG Pipeline** | ✅ Complete | ChromaDB, semantic chunking, working parser |
| **WebSocket** | ✅ Working | Real-time interview, voice integration |
| **Voice Services** | ✅ Working | Deepgram STT, ElevenLabs TTS |
| **Frontend** | ✅ Working | React 19, Next.js 16, video/voice integration |
| **Docker Setup** | ✅ NEW | docker-compose with 4 services |
| **Logging** | ✅ NEW | Structured logging with structlog |
| **Rate Limiting** | ✅ NEW | SlowAPI middleware (5-10 req/min) |
| **Health Check** | ✅ Enhanced | Custom model probe, dependency monitoring |

### 🔴 Critical Issues (Blocking Production)

| Issue | Priority | Impact | Location |
|-------|----------|--------|----------|
| **In-memory Sessions** | CRITICAL | Data loss on restart | `backend/app/db/session_store.py` |
| **No Tests** | HIGH | Can't safely refactor | `tests/` (<5% coverage) |
| **No Auth** | HIGH | Security risk | Missing JWT system |
| **No Session Expiration** | MEDIUM | Memory leak | `session_store.py` |

### ⏳ Recently Added (Phase 1 Quick Wins)

**Completed on 2026-01-16**:
1. ✅ Structured Logging (structlog)
2. ✅ Rate Limiting (SlowAPI)
3. ✅ Enhanced Health Check
4. ✅ Docker Compose
5. ✅ LangGraph Dependencies
6. ✅ Code Cleanup (-350 lines)
7. ✅ Development Roadmap

See `QUICK_WINS_COMPLETED.md` for details.

---

## 🗺️ Project Architecture

### High-Level Flow

```
User → Frontend (Next.js)
  ↓
Upload Resume → Backend API → RAG Pipeline (ChromaDB)
  ↓
Start Interview → WebSocket Connection
  ↓
Answer Question → Grilling Engine → LLM Service
  ↓
Evaluation (7D scores + 18 gap types) → Follow-up Question
  ↓
Repeat → Interview Complete → Results Page
```

### Key Files to Know

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `backend/app/core/grilling_engine.py` | Answer evaluation, gap detection | 973 | ⭐ Core logic |
| `backend/app/services/llm_service.py` | Multi-provider LLM abstraction | 828 | Stable |
| `backend/app/api/routes/websocket.py` | Real-time interview handler | 776 | Working |
| `backend/app/core/interview_agent.py` | Interview orchestration | 596 | API mode only |
| `backend/app/db/session_store.py` | Session storage | 235 | ⚠️ In-memory |
| `backend/app/core/logging_config.py` | Structured logging | 56 | NEW |
| `backend/app/middleware/rate_limit.py` | Rate limiting | 37 | NEW |
| `rag/resume_parser.py` | Resume parsing | 464 | Working |
| `rag/retriever.py` | RAG retrieval | 268 | Working |
| `DEVELOPMENT_ROADMAP.md` | 12-month plan | 2,780 | Reference |

---

## 🚀 Quick Start for Development

### Using Docker (Recommended)

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with API keys

# 2. Start everything
docker-compose up -d

# 3. Check health
curl http://localhost:8000/health

# 4. View logs
docker-compose logs -f backend
```

### Manual Setup

```bash
# Backend
uv sync
source .venv/bin/activate
PYTHONPATH=. uv run uvicorn backend.app.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

---

## 📋 Common Tasks

### 1. Understanding the Grilling Engine

**Location**: `backend/app/core/grilling_engine.py:973`

**Key Concepts**:
- **18 Gap Types**: Detects deficiencies in answers
  - `no_specific_example`, `no_metrics`, `unclear_role`, `no_outcome`, etc.
- **7-Dimensional Scoring**: Relevancy, Clarity, Informativeness, Specificity, Quantification, Depth, Completeness
- **Forced First Follow-up**: Always asks at least one follow-up
- **Hybrid Mode**: Uses compact prompts for custom models (reduces load by 80%)

**Important Methods**:
```python
async def evaluate_answer(
    question: str,
    answer: str,
    resume_context: str,
    conversation_history: List[Dict],
    follow_up_count: int
) -> AnswerEvaluation
```

### 2. Adding a New LLM Provider

**Location**: `backend/app/services/llm_service.py:828`

**Steps**:
1. Create new provider class inheriting `BaseLLMService`
2. Implement `generate()` and `generate_structured()` methods
3. Add to `LLMServiceFactory`
4. Update `backend/app/config.py` with new settings
5. Test with `tests/unit/test_llm_service.py` (TODO: create)

### 3. Debugging an Interview Session

**Check logs** (structured):
```bash
# With Docker
docker-compose logs -f backend | grep "session_id"

# Manual
PYTHONPATH=. DEBUG=true uv run uvicorn backend.app.main:app --reload
```

**Inspect session state**:
```python
# In backend/app/db/session_store.py
session = session_store.get_session(session_id)
print(session.model_dump_json(indent=2))
```

**WebSocket test**:
- Open `backend/test_websocket.html` in browser
- Connect to `ws://localhost:8000/ws/interview/{session_id}`

### 4. Working with RAG Pipeline

**Location**: `rag/` directory

**Flow**:
```python
# 1. Parse resume
from rag.resume_parser import ResumeParser
parser = ResumeParser()
parsed = parser.parse_file("resume.pdf")

# 2. Chunk
from rag.chunker import Chunker
chunker = Chunker()
chunks = chunker.chunk_resume(parsed)

# 3. Embed
from rag.embedder import ChromaDBEmbedder
embedder = ChromaDBEmbedder()
embedder.embed_chunks(resume_id, chunks)

# 4. Retrieve
from rag.retriever import InterviewRetriever
retriever = InterviewRetriever()
context = retriever.retrieve(resume_id, focus="python", top_k=3)
```

---

## 🎯 Next Development Priorities

### Phase 1 Remaining (CRITICAL)

**Must complete before moving to Phase 2**:

#### 1. Session Persistence (CRITICAL)
**Current**: In-memory dictionary
**Target**: PostgreSQL
**Effort**: ~1 week
**Files to modify**:
- `backend/app/db/session_store.py` - Add PostgreSQL backend
- `backend/app/models/schemas.py` - Add SQLAlchemy models
- Add Alembic for migrations

**Implementation sketch**:
```python
# New file: backend/app/db/models.py
from sqlalchemy import Column, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class SessionModel(Base):
    __tablename__ = "interview_sessions"

    id = Column(String, primary_key=True)
    resume_id = Column(String, nullable=False)
    state = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
```

#### 2. Test Coverage (HIGH)
**Current**: <5% (only `tests/test_parser.py`)
**Target**: 70%+
**Effort**: ~2 weeks

**Priority tests**:
```python
# tests/unit/test_grilling_engine.py
- Test all 18 gap types
- Test 7D scoring
- Test forced first follow-up
- Test hybrid mode prompts

# tests/unit/test_llm_service.py
- Test provider switching
- Test error handling
- Mock LLM calls

# tests/integration/test_websocket_flow.py
- Test complete interview flow
- Test voice integration
```

#### 3. Authentication (HIGH)
**Current**: No auth
**Target**: JWT-based auth
**Effort**: ~1 week

**Files to create**:
```
backend/app/core/auth.py
backend/app/core/security.py
backend/app/api/routes/auth.py
backend/app/api/deps.py (update)
```

---

### Phase 2: LangGraph Refactor (Next)

**Status**: Dependencies added, not yet implemented
**Estimated Start**: After Phase 1 complete
**Effort**: 3-4 weeks

**Goal**: Replace current interview flow with multi-agent architecture

**Agents**:
1. QuestionGeneratorAgent
2. EvaluatorAgent (wraps GrillingEngine)
3. FollowUpAgent
4. SummaryAgent (NEW)

**Benefits**:
- Persistent state via LangGraph checkpoints (solves session persistence!)
- Visual debugging (LangGraph Studio)
- Better error recovery
- Scalable agent architecture

See `DEVELOPMENT_ROADMAP.md` for full plan.

---

## 🐛 Known Issues & Workarounds

### Issue 1: Session Lost on Restart
**Problem**: All sessions stored in-memory
**Workaround**: Don't restart server during active interviews
**Fix**: Phase 1 task #1 (PostgreSQL persistence)

### Issue 2: No Rate Limit Storage
**Problem**: Rate limits reset on restart (in-memory)
**Workaround**: Currently acceptable for prototype
**Fix**: Upgrade to Redis storage (change 1 line in `rate_limit.py`)

### Issue 3: ChromaDB Not Auto-Created
**Problem**: First run fails if `data/chromadb/` doesn't exist
**Workaround**: Already fixed! (auto-creates on startup)
**Status**: ✅ Resolved (2026-01-16)

### Issue 4: Frontend WebSocket Reconnection
**Problem**: On disconnect, must manually refresh
**Workaround**: Stable WebSocket connection usually works
**Fix**: Add reconnection logic (medium priority)

---

## 📚 Important Documents

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `CLAUDE.md` | Complete project documentation | First time setup |
| `AI_ONBOARDING.md` | This file - quick start for AIs | Always read first |
| `DEVELOPMENT_ROADMAP.md` | 12-month development plan | Planning new features |
| `QUICK_WINS_COMPLETED.md` | Phase 1 completion summary | Understanding recent changes |
| `README.md` | User-facing documentation | Understanding user flow |
| `CONTRIBUTING.md` | Development guidelines | Before contributing |

---

## 🔧 Development Workflow

### Git Branching Strategy

**Follow Feature Branch Workflow**:

```bash
# Start new feature
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

# Develop with atomic commits
git add .
git commit -m "feat: add user authentication"

# Push and create PR
git push origin feature/your-feature-name
# Create PR on GitHub

# After merge, delete branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

**Branch naming**:
- `feature/` - New features
- `fix/` - Bug fixes
- `refactor/` - Code refactoring
- `docs/` - Documentation
- `test/` - Testing

**Commit message format**:
```
type(scope): short description

[optional body]

[optional footer]
```

Examples:
- `feat(backend): add JWT authentication`
- `fix(grilling): resolve gap detection edge case`
- `test(rag): add retriever integration tests`

---

## 🚨 Things to Avoid

❌ **Don't restart the backend during active interviews** (session data loss)
❌ **Don't modify `grilling_engine.py` without understanding the paper** (core logic)
❌ **Don't use `backend/app/core/resume_parser.py`** (deleted - use `rag/resume_parser.py`)
❌ **Don't commit without tests** (we need to improve coverage, not worsen it)
❌ **Don't create long-lived branches** (merge within 1 week)
❌ **Don't skip rate limiting on new endpoints** (security)

✅ **Do read existing code before modifying**
✅ **Do write tests for new features**
✅ **Do use structured logging** (not print())
✅ **Do follow the roadmap** (priorities are set)
✅ **Do ask questions** (via comments/PRs)

---

## 💡 Quick Reference

### Environment Variables (`.env`)

**Required for basic functionality**:
```bash
# LLM Provider (pick one)
GROQ_API_KEY=your_key_here          # Recommended (free, fast)
# OR
GOOGLE_API_KEY=your_key_here        # Gemini
ANTHROPIC_API_KEY=your_key_here     # Claude
OPENAI_API_KEY=your_key_here        # GPT

# LLM Configuration
LLM_MODE=api                        # "api" or "local"
LLM_PROVIDER=groq                   # "groq", "gemini", "openai", "anthropic"
```

**Optional**:
```bash
# Voice Services
VOICE_ENABLED=true
DEEPGRAM_API_KEY=your_key
ELEVENLABS_API_KEY=your_key

# Custom Model (GCP vLLM)
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_URL=http://localhost:8001/v1
```

### Useful Commands

```bash
# Start everything with Docker
docker-compose up -d

# View logs
docker-compose logs -f backend

# Run tests
PYTHONPATH=. uv run pytest tests/ --cov=backend

# Health check
curl http://localhost:8000/health

# Install dependencies
uv sync

# Format code
uv run black backend/ rag/

# Lint
uv run ruff check backend/ rag/

# Type check
uv run mypy backend/ rag/
```

---

## 🎓 Learning the Codebase

### Recommended Reading Order

1. **Start here**: `AI_ONBOARDING.md` (this file)
2. **Architecture**: `CLAUDE.md` - Project Structure section
3. **Core logic**: `backend/app/core/grilling_engine.py` (read comments)
4. **API flow**: `backend/app/api/routes/websocket.py`
5. **LLM integration**: `backend/app/services/llm_service.py`
6. **RAG pipeline**: `rag/retriever.py`
7. **Future plans**: `DEVELOPMENT_ROADMAP.md`

### Key Concepts to Understand

1. **Grilling Engine**: 18 gap types + 7D scoring
2. **Hybrid Mode**: Groq preprocessing + Custom model execution
3. **RAG Pipeline**: Parse → Chunk → Embed → Retrieve
4. **WebSocket Flow**: Connect → Question → Answer → Evaluate → Follow-up
5. **Session Management**: In-memory (TODO: PostgreSQL)

---

## ✨ Final Tips

1. **Always check `DEVELOPMENT_ROADMAP.md` before starting new work**
   - Priorities are clearly defined
   - Don't duplicate planned work

2. **Use Docker for development**
   - Consistent environment
   - All services configured
   - Easy to reset

3. **Test your changes**
   - Even though coverage is low, don't make it worse
   - Write tests for new features

4. **Follow the logging pattern**
   - Use `logger = get_logger(__name__)`
   - Use structured logging: `logger.info("event", key=value)`
   - Don't use `print()`

5. **When in doubt, ask**
   - Create a GitHub issue
   - Comment on PR
   - Check existing documentation first

---

**Welcome to Resume Griller! 🎉**

**Last Updated**: 2026-01-16 by Claude Code
**Next Review**: After Phase 1 completion (target: 2026-03-01)
