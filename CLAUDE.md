# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 🚀 Current Status (Updated 2026-01-16)

**Production Readiness**: 6/10 (improved from 3/10)

### ✅ Recently Completed (Quick Wins - Phase 1)
- ✅ **Structured Logging** with structlog (JSON + console modes)
- ✅ **Rate Limiting** middleware (SlowAPI - 5 req/min sessions, 10 req/min answers)
- ✅ **Enhanced Health Check** (custom model probe, dependency monitoring)
- ✅ **Docker Compose** setup (PostgreSQL, Redis, Backend, Frontend)
- ✅ **LangGraph Dependencies** added (preparation for Phase 2 refactor)
- ✅ **Code Cleanup** (removed 350 lines of duplicate code)
- ✅ **Development Roadmap** (12-month plan in DEVELOPMENT_ROADMAP.md)

### ⏳ Next Priorities (Phase 1 Remaining)
1. 🔴 **Session Persistence** (CRITICAL) - Migrate from in-memory to PostgreSQL
2. 🟡 **Test Coverage** (HIGH) - Currently <5%, target 70%+
3. 🟡 **Authentication** (HIGH) - Implement JWT-based auth

**See DEVELOPMENT_ROADMAP.md for full 4-phase plan (Q1-Q4 2026)**

---

## Project Overview

**Resume Griller** is a full-stack AI-powered interview simulator that analyzes resumes and conducts realistic mock interviews with intelligent follow-up questions. The system features a sophisticated "grilling engine" based on academic research that evaluates answers across 7 dimensions and detects 18 types of gaps to generate targeted follow-up questions.

**Current Branch**: `main`

**Note**: For the standalone RAG pipeline branch (offline question generation only), see the `rag-pipeline` branch.

## Quick Start

### Option 1: Docker Compose (Recommended) 🐳

The easiest way to run the entire stack:

```bash
# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start all services (PostgreSQL, Redis, Backend, Frontend)
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop all services
docker-compose down
```

**Services**:
- Backend API: `http://localhost:8000` (with auto-reload)
- Frontend: `http://localhost:3000`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

**Health Check**: `curl http://localhost:8000/health`

---

### Option 2: Local Development (Manual)

#### Backend

```bash
# Install dependencies (using uv - recommended)
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or .venv\Scripts\activate  # Windows

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Start backend server
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend available at: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

#### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend available at: `http://localhost:3000`

---

### Running Tests

```bash
# Backend tests
PYTHONPATH=. uv run pytest tests/

# With coverage
PYTHONPATH=. uv run pytest tests/ --cov=backend --cov=rag
```

## Architecture

### System Overview

Resume Griller is a **full-stack web application** with three main components:

1. **Backend** (FastAPI + Python 3.11+): API server, interview logic, LLM integration, voice services
2. **Frontend** (Next.js 16 + React 19 + TypeScript): Web UI, WebSocket client, voice recorder
3. **RAG Pipeline** (ChromaDB + sentence-transformers): Resume parsing, semantic chunking, vector embeddings

### Application Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │ Upload Page  │→ │ Interview    │→ │ Results Page            │  │
│  │ (Resume)     │  │ Room (WS)    │  │ (Score + Feedback)      │  │
│  └──────────────┘  └──────────────┘  └─────────────────────────┘  │
│         ↓                  ↕                                        │
│    HTTP Upload      WebSocket Messages                             │
└─────────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────────┐
│                          Backend (FastAPI)                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    WebSocket Handler                         │  │
│  │  (Real-time interview session management)                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│         ↓                    ↓                    ↓                 │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     │
│  │ Interview   │     │ Grilling     │     │ LLM Service     │     │
│  │ Agent       │ ←→  │ Engine       │ ←→  │ (Multi-provider)│     │
│  │ (API Mode)  │     │ (18 gap types│     │                 │     │
│  │             │     │  7D scoring) │     │ • Groq          │     │
│  └─────────────┘     └──────────────┘     │ • Gemini        │     │
│         ↓                                  │ • OpenAI        │     │
│  ┌─────────────┐     ┌──────────────┐     │ • Claude        │     │
│  │ Voice       │     │ RAG Pipeline │     │ • Custom (vLLM) │     │
│  │ Services    │     │ (ChromaDB)   │     │ • Hybrid Mode   │     │
│  │ • STT       │     └──────────────┘     └─────────────────┘     │
│  │ • TTS       │                                                   │
│  └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
                             ↕
┌─────────────────────────────────────────────────────────────────────┐
│                      External Services                              │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │ Groq API │  │ Deepgram │  │ ElevenLabs│  │ GCP vLLM Server  │  │
│  │ (Primary)│  │ (STT)    │  │ (TTS)     │  │ (Custom Model)   │  │
│  └──────────┘  └──────────┘  └───────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Hybrid Model Architecture

The Hybrid mode provides the best interview quality by combining API and custom models:

```
Phase 1: Preprocessing (Groq API)
  ├─ Generate condensed resume summary (~200 tokens)
  ├─ Generate targeted interview questions
  └─ Prepare per-question evaluation context

Phase 2: Interview Execution (Custom LoRA Model via vLLM)
  ├─ Evaluate answers using compact prompts
  ├─ Detect gaps in responses (18 gap types)
  ├─ Score across 7 dimensions
  └─ Generate contextual follow-up questions
```

This architecture reduces custom model load by 80% while maintaining high-quality grilling.

## Project Structure

```
resume-griller/
├── backend/                          # FastAPI Backend (Python 3.11+)
│   ├── app/
│   │   ├── api/routes/               # API Endpoints
│   │   │   ├── resume.py             # Resume upload & processing (322 lines)
│   │   │   ├── session.py            # Session management (679 lines)
│   │   │   ├── voice.py              # STT/TTS endpoints (282 lines)
│   │   │   └── websocket.py          # WebSocket interview handler (776 lines)
│   │   │
│   │   ├── core/                     # Core Business Logic
│   │   │   ├── grilling_engine.py    # Gap detection & scoring (973 lines)
│   │   │   ├── interview_agent.py    # Interview orchestration (596 lines)
│   │   │   └── logging_config.py     # Structured logging config (NEW - 56 lines)
│   │   │
│   │   ├── middleware/               # Middleware (NEW)
│   │   │   └── rate_limit.py         # Rate limiting with SlowAPI (37 lines)
│   │   │
│   │   ├── services/                 # External Service Integrations
│   │   │   ├── llm_service.py        # LLM provider abstraction (828 lines)
│   │   │   │                         # Supports: Groq, Gemini, OpenAI, Claude, Custom, Hybrid
│   │   │   ├── stt_service.py        # Deepgram Speech-to-Text (167 lines)
│   │   │   └── tts_service.py        # ElevenLabs Text-to-Speech (212 lines)
│   │   │
│   │   ├── db/
│   │   │   └── session_store.py      # In-memory session storage (⚠️ TODO: PostgreSQL)
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py            # Pydantic models
│   │   │
│   │   ├── config.py                 # Settings from .env (98 lines)
│   │   └── main.py                   # FastAPI app entry point (enhanced logging)
│   │
│   └── test_websocket.html           # WebSocket testing interface
│
├── frontend/                         # Next.js Frontend (TypeScript)
│   ├── src/
│   │   ├── app/                      # App Router Pages
│   │   │   ├── page.tsx              # Home page
│   │   │   ├── upload/page.tsx       # Resume upload page
│   │   │   ├── interview/
│   │   │   │   └── [sessionId]/page.tsx  # Interview session page
│   │   │   └── result/
│   │   │       └── [sessionId]/page.tsx  # Results page
│   │   │
│   │   ├── components/               # React Components
│   │   │   ├── interview/
│   │   │   │   ├── VideoInterviewRoom.tsx  # Main interview UI
│   │   │   │   ├── VoiceRecorder.tsx       # Voice input component
│   │   │   │   ├── ChatMessage.tsx         # Message display
│   │   │   │   └── InterviewRoom.tsx       # Interview container
│   │   │   ├── upload/
│   │   │   │   └── ResumeUploader.tsx      # Resume upload UI
│   │   │   ├── ui/                         # Reusable UI components
│   │   │   │   ├── button.tsx
│   │   │   │   ├── card.tsx
│   │   │   │   ├── input.tsx
│   │   │   │   └── progress.tsx
│   │   │   └── layout/
│   │   │       └── Header.tsx
│   │   │
│   │   ├── stores/
│   │   │   └── interviewStore.ts     # Zustand state management
│   │   │
│   │   ├── lib/
│   │   │   ├── api.ts                # API client
│   │   │   └── websocket.ts          # WebSocket client
│   │   │
│   │   └── types/
│   │       └── index.ts              # TypeScript definitions
│   │
│   ├── package.json                  # Frontend dependencies
│   ├── .env.local                    # Frontend environment variables
│   └── README.md
│
├── rag/                              # RAG Pipeline (Shared)
│   ├── resume_parser.py              # PDF/TXT parser (464 lines)
│   ├── chunker.py                    # Semantic chunking (224 lines)
│   ├── embedder.py                   # ChromaDB operations (282 lines)
│   ├── retriever.py                  # Context retrieval (268 lines)
│   └── generator.py                  # LoRA model inference (185 lines)
│
├── ml/                               # Machine Learning Models
│   ├── models/
│   │   └── interview-coach-lora/     # Fine-tuned LoRA adapter
│   │       ├── adapter_config.json
│   │       ├── tokenizer.json
│   │       └── README.md
│   └── README.md                     # ML documentation (258 lines)
│
├── data/
│   ├── sample_resumes/               # 5 sample resumes (1 PDF + 4 TXT)
│   ├── chromadb/                     # Vector database (3.2MB)
│   ├── uploads/                      # User-uploaded resumes (30+ files)
│   └── exported_prompts.json         # Prompts for offline inference
│
├── tests/
│   └── test_parser.py                # RAG parser tests (47 lines) (⚠️ Need more coverage)
│
├── scripts/
│   └── start_iap_tunnel.sh           # GCP IAP tunnel helper
│
├── docs/
│   └── CONTRIBUTING.md               # Development guidelines (228 lines)
│
├── pyproject.toml                    # Python project config (uv)
├── uv.lock                           # Locked dependencies
├── .env.example                      # Environment template (48 lines)
├── .gitignore
├── .dockerignore                     # Docker ignore patterns (NEW)
├── docker-compose.yml                # Multi-service orchestration (NEW - 116 lines)
├── Dockerfile.backend                # Backend container (NEW - 36 lines)
├── README.md                         # User documentation
├── CLAUDE.md                         # This file
├── DEVELOPMENT_ROADMAP.md            # 12-month development plan (NEW - 2,780 lines)
└── QUICK_WINS_COMPLETED.md           # Phase 1 completion summary (NEW - 300 lines)
```

## Key Components

### Backend

#### 1. Grilling Engine (`backend/app/core/grilling_engine.py`)

**Purpose**: Evaluates candidate answers and generates intelligent follow-up questions.

**Based on**: "Requirements Elicitation Follow-Up Question Generation" research paper.

**Key Features**:
- **18 Gap Types**: Detects deficiencies in answers
  - `no_specific_example` - Answer lacks concrete examples
  - `no_metrics` - Missing quantifiable results
  - `unclear_role` - Personal contribution unclear
  - `no_outcome` - Missing results/impact
  - `no_tech_depth` - Insufficient technical detail
  - `unclear_statement` - Vague or ambiguous
  - `too_generic` - Not specific to resume
  - `unexplained_jargon` - Technical terms without explanation
  - And 10 more...

- **7-Dimensional Scoring**:
  - Relevancy (to question)
  - Clarity (understandability)
  - Informativeness (useful content)
  - Specificity (concrete details)
  - Quantification (metrics/numbers)
  - Depth (technical detail)
  - Completeness (full answer)

- **Forced First Follow-up**: Always asks at least one follow-up to encourage depth
- **Resume Consistency Checking**: Validates answers against resume content
- **Adaptive Questioning**: Follow-ups target the most significant gaps

**Line count**: 973 lines

#### 2. LLM Service (`backend/app/services/llm_service.py`)

**Purpose**: Unified interface for multiple LLM providers.

**Supported Providers**:
1. **Groq** (Primary, recommended)
   - Model: `llama-3.3-70b-versatile`
   - Fast, free, excellent quality
   - Best for API mode

2. **Gemini** (Google)
   - Model: `gemini-2.5-flash`
   - Fast, cost-effective

3. **OpenAI**
   - Model: `gpt-4o`
   - High quality, higher cost

4. **Claude** (Anthropic)
   - Model: `claude-sonnet-4-20250514`
   - Excellent reasoning

5. **Custom Model** (vLLM on GCP)
   - Model: `mistralai/Mistral-7B-Instruct-v0.2` + LoRA adapter
   - Fine-tuned for interview coaching
   - Requires GCP VM with GPU

6. **Hybrid Mode**
   - Groq for preprocessing (summaries, questions)
   - Custom Model for interview execution (evaluation, follow-ups)
   - Best quality with reduced custom model load

**Key Methods**:
- `generate()`: Generate text from prompt
- `generate_structured()`: Generate JSON responses
- `chat()`: Multi-turn conversation

**Line count**: 828 lines

#### 3. Interview Agent (`backend/app/core/interview_agent.py`)

**Purpose**: Orchestrates the interview flow in API mode.

**Responsibilities**:
- Generate interview questions from resume
- Manage interview state (current question, progress)
- Coordinate with Grilling Engine for evaluation
- Handle mode-specific question generation (HR/Tech/Mixed)

**Interview Modes**:
- **HR Mode**: Behavioral STAR questions (soft skills, leadership, teamwork)
- **Technical Mode**: Technical deep-dives (architecture, implementation, trade-offs)
- **Mixed Mode**: Combination of both

**Line count**: 596 lines

#### 4. Voice Services

**Speech-to-Text** (`backend/app/services/stt_service.py`):
- Provider: Deepgram
- Model: `nova-2`
- Streaming and batch transcription
- Line count: 167 lines

**Text-to-Speech** (`backend/app/services/tts_service.py`):
- Provider: ElevenLabs
- Model: `eleven_flash_v2_5`
- Configurable voice ID
- Streaming audio generation
- Line count: 212 lines

#### 5. WebSocket Handler (`backend/app/api/routes/websocket.py`)

**Purpose**: Real-time interview session management.

**Features**:
- Bidirectional communication (question/answer flow)
- Session state persistence
- Answer evaluation and follow-up generation
- Progress tracking
- Interview completion handling

**Message Types**:
- `answer`: User's answer to current question
- `question`: New question from interviewer
- `follow_up`: Follow-up question based on gap analysis
- `complete`: Interview finished
- `error`: Error occurred

**Line count**: 776 lines

### Frontend

#### 1. Video Interview Room (`frontend/src/components/interview/VideoInterviewRoom.tsx`)

**Purpose**: Main interview UI component.

**Features**:
- Real-time message display (questions, answers, follow-ups)
- Voice input toggle
- Sound on/off control
- Text input fallback
- Progress tracking
- Score display

**State Management**: Uses Zustand store for interview state.

#### 2. Voice Recorder (`frontend/src/components/interview/VoiceRecorder.tsx`)

**Purpose**: Voice input component.

**Features**:
- Browser MediaRecorder API
- Real-time recording indication
- Audio upload to backend
- Transcription via Deepgram
- Error handling

#### 3. Resume Uploader (`frontend/src/components/upload/ResumeUploader.tsx`)

**Purpose**: Resume upload and interview configuration.

**Features**:
- Drag-and-drop file upload
- PDF/TXT support
- Interview mode selection (HR/Tech/Mixed)
- Model type selection (API/Custom)
- Number of questions configuration
- Max follow-ups configuration

### RAG Pipeline

The RAG pipeline is shared across both the full-stack app and the standalone `rag-pipeline` branch.

#### 1. Resume Parser (`rag/resume_parser.py`)

**Purpose**: Extract structured data from resumes.

**Supported Formats**: PDF (via pdfplumber), TXT

**Extracted Sections**:
- Contact information
- Professional summary
- Skills
- Work experience (company, role, dates, responsibilities)
- Education (institution, degree, dates)
- Projects
- Certifications

**Returns**: `ParsedResume` dataclass with structured fields.

**Line count**: 464 lines

#### 2. Chunker (`rag/chunker.py`)

**Purpose**: Convert parsed resume into semantic chunks for RAG.

**Chunking Strategy**:
- One chunk for overview (contact + summary + skills)
- One chunk per job experience
- One chunk per education entry
- One chunk per project

**Returns**: List of `Chunk` objects with metadata (section type, chunk index).

**Line count**: 224 lines

#### 3. Embedder (`rag/embedder.py`)

**Purpose**: Generate and store vector embeddings.

**Vector Database**: ChromaDB (persistent SQLite at `data/chromadb/`)

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)

**Operations**:
- `embed_chunks()`: Store resume chunks with embeddings
- `search()`: Retrieve relevant chunks by semantic similarity
- `get_all_chunks()`: List all chunks for a resume
- `delete()`: Remove resume from database
- `clear_collection()`: Wipe all data

**Metadata Structure**:
```python
{
    "resume_id": "resume_xyz123",
    "section": "experience",  # or "skills", "education", "projects", "overview"
    "chunk_index": 0
}
```

**Line count**: 282 lines

#### 4. Retriever (`rag/retriever.py`)

**Purpose**: End-to-end RAG pipeline orchestration.

**Key Methods**:
- `process_resume(path, resume_id)`: Parse → Chunk → Embed
- `retrieve(resume_id, focus, top_k)`: Get relevant chunks by semantic search
- `build_prompt(resume_id, question_type, focus)`: Format LLM prompt with context
- `get_resume_summary(resume_id)`: Get condensed overview

**Question Types**: `"technical"`, `"behavioral"`, `"mixed"`

**Line count**: 268 lines

#### 5. Generator (`rag/generator.py`)

**Purpose**: Local LoRA model inference (for offline use).

**Model**: `mistralai/Mistral-7B-Instruct-v0.2` + `shubhampareek/interview-coach-lora`

**Hardware Requirements**:
- GPU: 12GB+ VRAM (T4/L4/A100)
- CPU: ~14GB RAM (very slow, not recommended)

**Note**: In the full-stack app, this is replaced by the Custom Model via vLLM on GCP.

**Line count**: 185 lines

## Dependencies

### Backend (`pyproject.toml`)

**Total**: 35+ packages (managed with uv)

**Categories**:

1. **Web Framework**:
   - `fastapi` - Web framework
   - `uvicorn[standard]` - ASGI server
   - `python-multipart` - File upload support
   - `pydantic` - Data validation
   - `pydantic-settings` - Settings management

2. **Async Support**:
   - `httpx` - Async HTTP client
   - `aiofiles` - Async file I/O

3. **Database**:
   - `sqlalchemy` - ORM
   - `aiosqlite` - Async SQLite

4. **LLM Providers**:
   - `anthropic` - Claude API
   - `openai` - GPT API
   - `groq` - Groq API (Llama 3.3)
   - `google-generativeai` - Gemini API

5. **RAG Pipeline**:
   - `torch` - PyTorch
   - `pdfplumber` - PDF parsing
   - `PyMuPDF` - PDF parsing (alternative)
   - `chromadb` - Vector database
   - `sentence-transformers` - Embeddings
   - `langchain` - LLM framework

6. **Multi-Agent Framework** (NEW):
   - `langgraph>=1.0.6` - Multi-agent orchestration
   - `langgraph-checkpoint-sqlite>=3.0.2` - Persistent state checkpoints

7. **LoRA Model**:
   - `transformers` - HuggingFace Transformers
   - `peft` - LoRA fine-tuning
   - `accelerate` - Model optimization

8. **Voice Services**:
   - `deepgram-sdk` - Deepgram STT (implied in code)
   - `elevenlabs` - ElevenLabs TTS (implied in code)

9. **Logging & Monitoring** (NEW):
   - `structlog>=24.1.0` - Structured logging

10. **Security & Rate Limiting** (NEW):
    - `slowapi>=0.1.9` - Rate limiting middleware

11. **Utilities**:
    - `python-dotenv` - Environment variables

### Frontend (`frontend/package.json`)

**Total**: 10 dependencies

**Framework**:
- `next@16.0.10` - Next.js framework
- `react@19.2.1` - React library
- `react-dom@19.2.1` - React DOM

**Styling**:
- `tailwindcss@^4` - Tailwind CSS
- `@tailwindcss/postcss@^4` - PostCSS plugin

**State Management**:
- `zustand@^5.0.9` - State management

**UI Components**:
- `lucide-react@^0.561.0` - Icons
- `sonner@^2.0.7` - Toast notifications
- `clsx@^2.1.1` - Class name utility
- `tailwind-merge@^3.4.0` - Tailwind utilities

**Development**:
- `typescript@^5` - TypeScript
- `@types/node`, `@types/react`, `@types/react-dom` - Type definitions
- `eslint@^9` - Linting
- `eslint-config-next@16.0.10` - Next.js ESLint config

### Installation

**Using uv (Recommended)**:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Or install specific groups
uv sync --extra dev   # Include dev dependencies
uv sync --extra ml    # Include ML dependencies
```

**Using pip**:
```bash
pip install -r backend/requirements.txt
```

## Configuration

### Environment Variables

Create a `.env` file in the project root. See `.env.example` for the template (48 lines total).

#### App Settings

```bash
DEBUG=true                           # Enable debug mode
DATABASE_URL=sqlite+aiosqlite:///./data/resume_griller.db
```

#### LLM Configuration

```bash
# Mode: "api" (cloud providers) or "local" (requires GPU)
LLM_MODE=api

# Provider: anthropic, openai, gemini, groq
LLM_PROVIDER=groq

# API Keys (get from respective providers)
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here        # Optional
ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
OPENAI_API_KEY=your_openai_key_here        # Optional

# Model names (customize if needed)
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.5-flash
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_MODEL=gpt-4o
```

#### Custom Model (GCP vLLM)

```bash
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_URL=http://localhost:8001/v1    # IAP tunnel endpoint
CUSTOM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
CUSTOM_MODEL_TIMEOUT=120
```

#### Voice Services

```bash
VOICE_ENABLED=true
STT_PROVIDER=deepgram
TTS_PROVIDER=elevenlabs

DEEPGRAM_API_KEY=your_deepgram_key_here
ELEVENLABS_API_KEY=your_elevenlabs_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Default voice
```

#### RAG Settings

```bash
CHROMA_PERSIST_DIR=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

#### File Upload

```bash
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE=10485760  # 10MB
```

### Quick Setup Options

#### Option 1: Groq API Mode (Recommended for Getting Started)

```bash
LLM_MODE=api
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
VOICE_ENABLED=false  # Disable voice for simplicity
```

**Pros**: Fast, free, easy setup
**Cons**: Cloud-based, no fine-tuned model

#### Option 2: Hybrid Mode (Best Quality)

```bash
LLM_MODE=api
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_URL=http://localhost:8001/v1
```

**Pros**: Best interview quality, reduced custom model load
**Cons**: Requires GCP setup (see [Custom Model Deployment](#custom-model-deployment))

#### Option 3: Custom Model Only

```bash
LLM_MODE=local
CUSTOM_MODEL_URL=http://localhost:8001/v1
```

**Pros**: Full control, fine-tuned for interviews
**Cons**: Requires GPU, slower preprocessing

## Usage

### Starting the Application

#### 1. Start Backend

```bash
# Using uv (recommended)
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --reload --port 8000

# Or with activated venv
PYTHONPATH=. python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend runs at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

#### 2. Start Frontend

```bash
cd frontend
npm run dev
```

Frontend runs at: `http://localhost:3000`

#### 3. (Optional) Start IAP Tunnel for Custom Model

```bash
gcloud compute start-iap-tunnel YOUR_INSTANCE_NAME 8000 \
    --local-host-port=localhost:8001 \
    --zone=YOUR_ZONE
```

### Using the Application

1. **Upload Resume**: Navigate to `http://localhost:3000/upload`
2. **Configure Interview**:
   - Mode: HR / Technical / Mixed
   - Model: API (cloud) or Custom (fine-tuned)
   - Number of questions: 3-10
   - Max follow-ups: 1-5
3. **Start Interview**: Click "Start Interview"
4. **Answer Questions**:
   - Type answers in text box OR
   - Enable voice and speak (requires voice services configured)
5. **Receive Follow-ups**: Based on gaps detected in your answer
6. **Review Results**: See scores, feedback, and full transcript

### Interview Modes

| Mode | Focus | Question Examples |
|------|-------|-------------------|
| **HR** | Behavioral, soft skills, leadership | "Tell me about a time when you faced a conflict with a team member..." |
| **Tech** | Technical depth, architecture, implementation | "Walk me through how you implemented the authentication system. What trade-offs did you consider?" |
| **Mixed** | Balanced combination | Both behavioral and technical questions |

## Custom Model Deployment

### GCP Setup

#### 1. Create GPU VM Instance

```bash
gcloud compute instances create interview-model \
    --zone=us-central1-a \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE
```

#### 2. SSH into Instance

```bash
gcloud compute ssh interview-model --zone=us-central1-a
```

#### 3. Install vLLM and Start Server

```bash
pip install vllm

# Start vLLM with LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-lora \
    --lora-modules interview-lora=shubhampareek/interview-coach-lora \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

#### 4. Create IAP Tunnel (Local Machine)

```bash
# In a separate terminal
gcloud compute start-iap-tunnel interview-model 8000 \
    --local-host-port=localhost:8001 \
    --zone=us-central1-a
```

#### 5. Test Connection

```bash
curl http://localhost:8001/v1/models
```

#### 6. Configure Application

In your `.env`:
```bash
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_URL=http://localhost:8001/v1
```

### Using Custom Model

In the frontend upload page, select:
- **Model Type**: "Custom Model (Fine-tuned)"

For Hybrid mode, the system automatically:
1. Uses Groq for preprocessing (resume summary, question generation)
2. Uses Custom Model for interview execution (answer evaluation, follow-ups)

## Development Workflow

### Git Branching

**Main branch**: Full-stack application (this documentation)
**rag-pipeline branch**: Standalone RAG pipeline (see `rag-pipeline` branch for details)

**Recent commits on main**:
```
6540cbf - chore: clean up some unnecessary files and alter gitignore
877d776 - fix(backend): add model_type parameter to GrillingEngine for Hybrid mode
76354fa - chore: update the Readme again
e1c8090 - chore: update latest Readme
```

### Commit Message Format

```
type(scope): short description

[optional body]
```

**Types**: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

**Scopes**: `backend`, `frontend`, `rag`, `ml`, `docs`

**Examples**:
- `feat(backend): add Groq provider to LLM service`
- `fix(frontend): resolve WebSocket reconnection issue`
- `refactor(rag): optimize ChromaDB query performance`
- `docs(readme): update deployment instructions`

### Code Quality

**Python**:
- **Line length**: 88 characters (Black default)
- **Target version**: Python 3.11+
- **Type hints**: Required for public APIs
- **Linting**: `ruff check backend/ rag/`
- **Formatting**: `black backend/ rag/`
- **Type checking**: `mypy backend/ rag/`

**TypeScript**:
- **Config**: `tsconfig.json`
- **Linting**: `npm run lint` (ESLint)
- **Formatting**: Prettier (via ESLint)

**Testing**:
- **Framework**: pytest
- **Coverage**: Aim for >70% on core logic
- **Run**: `PYTHONPATH=. uv run pytest tests/ --cov=backend --cov=rag`

### Adding Dependencies

**Backend (using uv)**:
```bash
# Add production dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Add ML dependency
uv add --extra ml package-name
```

**Frontend**:
```bash
cd frontend
npm install package-name
# or for dev
npm install --save-dev package-name
```

## API Reference

### REST Endpoints

#### Health Check

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "llm_mode": "api",
  "llm_provider": "groq"
}
```

#### Upload Resume

```http
POST /api/v1/resumes/upload
Content-Type: multipart/form-data

file: <resume.pdf>
```

Response:
```json
{
  "resume_id": "resume_xyz123",
  "filename": "resume.pdf",
  "parsed": true
}
```

#### Create Interview Session

```http
POST /api/v1/sessions
Content-Type: application/json

{
  "resume_id": "resume_xyz123",
  "mode": "tech",           // "hr", "tech", "mixed"
  "model_type": "api",      // "api" or "custom"
  "num_questions": 5,
  "max_follow_ups": 3
}
```

Response:
```json
{
  "session_id": "session_abc456",
  "resume_id": "resume_xyz123",
  "mode": "tech",
  "status": "initialized"
}
```

#### Submit Answer (HTTP)

```http
POST /api/v1/sessions/{session_id}/answer
Content-Type: application/json

{
  "answer": "I built a microservices architecture using..."
}
```

Response:
```json
{
  "evaluation": {
    "scores": {
      "relevancy": 0.85,
      "clarity": 0.78,
      ...
    },
    "gaps": ["no_metrics", "unclear_role"],
    "overall_score": 0.75
  },
  "follow_up": "You mentioned microservices, but what specific metrics did you use to measure performance improvements?",
  "is_complete": false
}
```

### WebSocket API

**Endpoint**: `ws://localhost:8000/ws/interview/{session_id}`

#### Client → Server Messages

**Answer**:
```json
{
  "type": "answer",
  "content": "I built a microservices architecture..."
}
```

#### Server → Client Messages

**Question**:
```json
{
  "type": "question",
  "content": "Tell me about a challenging technical project you worked on.",
  "data": {
    "question_number": 1,
    "total_questions": 5
  }
}
```

**Follow-up**:
```json
{
  "type": "follow_up",
  "content": "What specific metrics did you use to measure performance?",
  "data": {
    "evaluation": {
      "scores": { "relevancy": 0.85, "clarity": 0.78, ... },
      "gaps": ["no_metrics", "unclear_role"],
      "overall_score": 0.75
    },
    "follow_up_count": 1
  }
}
```

**Complete**:
```json
{
  "type": "complete",
  "content": "Interview completed!",
  "data": {
    "summary": {
      "total_questions": 5,
      "total_follow_ups": 8,
      "average_score": 0.78,
      "strengths": ["Clear communication", "Technical depth"],
      "areas_for_improvement": ["Add more metrics", "Clarify personal role"]
    }
  }
}
```

### Voice Endpoints

#### Speech-to-Text

```http
POST /api/v1/voice/stt
Content-Type: multipart/form-data

audio: <audio.wav>
```

Response:
```json
{
  "transcript": "I built a microservices architecture using..."
}
```

#### Text-to-Speech

```http
POST /api/v1/voice/tts
Content-Type: application/json

{
  "text": "Tell me about your experience with microservices."
}
```

Response: Audio stream (audio/mpeg)

## Troubleshooting

### Common Issues

#### Backend won't start

**Error**: `ModuleNotFoundError: No module named 'backend'`

**Solution**: Set `PYTHONPATH`:
```bash
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --reload
```

#### Custom model not available

**Symptoms**: Errors when selecting "Custom Model" in frontend

**Solutions**:
1. Ensure IAP tunnel is running: `gcloud compute start-iap-tunnel ...`
2. Check vLLM server is running on GCP: `ssh` to instance and verify
3. Test connection: `curl http://localhost:8001/v1/models`
4. Verify `CUSTOM_MODEL_URL` in `.env`

#### HR mode asking technical questions

**Cause**: Retriever doesn't handle `"hr"` mode properly

**Solution**: Ensure you're using latest `rag/retriever.py` (handles both `"hr"` and `"behavioral"` values)

#### Voice features not working

**Symptoms**: Microphone button grayed out, no transcription

**Solutions**:
1. Check `VOICE_ENABLED=true` in `.env`
2. Verify Deepgram API key: `echo $DEEPGRAM_API_KEY`
3. Verify ElevenLabs API key: `echo $ELEVENLABS_API_KEY`
4. Check browser microphone permissions (browser console)
5. Test STT endpoint: `curl -X POST http://localhost:8000/api/v1/voice/stt -F "audio=@test.wav"`

#### ChromaDB errors

**Error**: `chromadb.errors.InvalidCollectionException`

**Solutions**:
1. Delete and rebuild:
   ```bash
   rm -rf data/chromadb/
   PYTHONPATH=. python -m rag.retriever
   ```
2. Check disk space: `df -h`
3. Check permissions: `ls -la data/`

#### WebSocket connection failed

**Symptoms**: Frontend shows "Connecting..." indefinitely

**Solutions**:
1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check WebSocket endpoint: Browser DevTools → Network → WS
3. Verify CORS settings in `backend/app/config.py`
4. Check session ID is valid

#### Frontend build errors

**Error**: `Module not found: Can't resolve 'zustand'`

**Solution**:
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Debug Mode

**Enable detailed logging**:

In `.env`:
```bash
DEBUG=true
```

**Run with debug logging**:
```bash
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --port 8000 --log-level debug
```

**Frontend debug** (browser console):
- Network tab: Check API calls
- Console tab: Check WebSocket messages
- Application tab: Check local storage

### Testing WebSocket Locally

Use the included test page:

```bash
# Start backend
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --port 8000

# Open in browser
open backend/test_websocket.html
```

## Tech Stack Summary

### Languages
- **Backend**: Python 3.11+
- **Frontend**: TypeScript 5

### Frameworks
- **Backend**: FastAPI
- **Frontend**: Next.js 16 (App Router), React 19

### Databases
- **Vector DB**: ChromaDB (SQLite backend)
- **Session Store**: In-memory (can be extended to PostgreSQL)

### LLM Providers
- Groq (Primary)
- Gemini, OpenAI, Claude (Alternatives)
- Custom vLLM (Fine-tuned)

### Voice Services
- Deepgram (STT)
- ElevenLabs (TTS)

### Infrastructure
- **Package Manager**: uv (Python), npm (Node.js)
- **Cloud**: Google Cloud Platform (optional, for custom model)
- **GPU**: T4 / L4 / A100 (for vLLM)

## What's Different from rag-pipeline Branch

The `rag-pipeline` branch is a **standalone RAG pipeline** for offline question generation (no web app, no real-time interview).

| Feature | main (This Branch) | rag-pipeline |
|---------|-------------------|--------------|
| **Architecture** | Full-stack web app | Standalone RAG pipeline |
| **Backend** | FastAPI + WebSocket + Voice | None |
| **Frontend** | Next.js 16 + React 19 | None |
| **Interview Mode** | Real-time interactive | Offline batch generation |
| **LLM Providers** | 6 providers (Groq, Gemini, OpenAI, Claude, Custom, Hybrid) | Local LoRA only |
| **Voice** | Deepgram STT + ElevenLabs TTS | None |
| **Grilling Engine** | 973 lines with 18 gap types, 7D scoring | Not included |
| **Deployment** | Web application (Frontend + Backend) | Google Colab notebook |
| **Dependencies** | 32+ backend, 10 frontend | 6 RAG-focused |
| **Use Case** | Live mock interviews with follow-ups | Generate question sets for study |

**When to use rag-pipeline branch**: For generating interview question prompts in bulk for offline use (e.g., export to Colab, run batch inference).

**When to use main branch**: For interactive interview simulation with real-time grilling and voice support.

## Sample Data

### Resumes

Located in `data/sample_resumes/`:

1. `resume_sp.pdf` - Shubham Pareek (PDF, embedded in ChromaDB)
2. `Resume1.txt` - Software Engineer
3. `Resume2.txt` - Data Scientist
4. `Resume3.txt` - Product Manager
5. `Resume4.txt` - DevOps Engineer

### User Uploads

Located in `data/uploads/`:
- 30+ PDF files from actual usage (ignored by git)

### ChromaDB

Located in `data/chromadb/`:
- Size: ~3.2MB
- Collection: "resumes"
- Embedding model: all-MiniLM-L6-v2 (384 dimensions)
- Current chunks: Varies based on uploaded resumes

## Contributing

See `docs/CONTRIBUTING.md` for development guidelines (228 lines).

Key points:
- Follow commit message format
- Write tests for new features
- Update documentation
- Run linters before committing

## Acknowledgments

- **LoRA Model**: [shubhampareek/interview-coach-lora](https://huggingface.co/shubhampareek/interview-coach-lora)
- **Base Model**: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- **Research**: "Requirements Elicitation Follow-Up Question Generation" paper
- **Voice Services**: Deepgram, ElevenLabs
- **LLM Providers**: Groq, Google, OpenAI, Anthropic

## License

MIT License - see LICENSE file for details.

---

**Made with 🔥 by the Resume Griller team**
