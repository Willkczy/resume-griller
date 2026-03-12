# Resume Griller 🔥

> **AI-powered interview simulator that grills candidates with resume-specific questions**

An intelligent interview preparation platform that analyzes your resume and conducts realistic mock interviews with AI. Features voice support, adaptive follow-up questions based on a Mistake-Guided Framework, and multiple interview modes (HR, Technical, Mixed).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Custom Model Deployment](#custom-model-deployment)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Core Interview Experience

- **Resume Analysis**: Upload PDF/TXT resumes for AI-powered parsing and semantic analysis
- **Multiple Interview Modes**:
  - **HR Mode**: Behavioral STAR questions focused on soft skills, teamwork, and leadership
  - **Technical Mode**: Deep technical questions about architecture, implementation, and trade-offs
  - **Mixed Mode**: Comprehensive interview combining both HR and technical questions
- **Intelligent Grilling Engine**: Based on "Requirements Elicitation Follow-Up Question Generation" paper
  - 18 gap types for answer evaluation (no_specific_example, no_metrics, unclear_role, etc.)
  - 7-dimensional scoring (relevancy, clarity, informativeness, specificity, quantification, depth, completeness)
  - Forced first follow-up to ensure depth
  - Resume consistency checking
- **Voice Support**:
  - Speech-to-Text (STT) via Deepgram
  - Text-to-Speech (TTS) via ElevenLabs
  - Real-time voice interview simulation with sound on/off control
- **Real-time Communication**: WebSocket-based interactive interview sessions
- **Persistent Sessions**: Interview state checkpointed to SQLite via LangGraph — survives server restarts

### LLM Backend Options

| Mode | Provider | Use Case |
|------|----------|----------|
| **API Mode** | Groq (Llama 3.3 70B) | Recommended - Fast and free |
| **API Mode** | Gemini, OpenAI, Claude | Alternative cloud providers |
| **Hybrid Mode** | Groq + Custom Model | Best quality with fine-tuned model |
| **Custom Mode** | Self-hosted vLLM on GCP | Full control, uses fine-tuned LoRA |

### Hybrid Model Architecture

The Hybrid mode provides the best interview quality by combining:
- **Groq (Preprocessing)**: Handles resume summarization, question generation, and context preparation
- **Custom Model (Execution)**: Fine-tuned Mistral-7B for answer evaluation and follow-up generation

```
Phase 1 (Groq - Preprocessing):
  ├─ Generate condensed resume summary (~200 tokens)
  ├─ Generate targeted interview questions
  └─ Prepare per-question evaluation context

Phase 2 (Custom Model - Interview Execution):
  ├─ Evaluate answers using compact prompts
  ├─ Detect gaps in responses
  └─ Generate contextual follow-up questions
```

---

## Architecture

### Full-Stack Application Flow

```
┌─────────────────┐
│    Frontend     │  Next.js 16 + React 19 + TypeScript + Tailwind CSS
│   (Port 3000)   │
└────────┬────────┘
         │ HTTP/WebSocket
         ▼
┌─────────────────┐
│    Backend      │  FastAPI + Python 3.11+ + Async I/O
│   (Port 8000)   │
└────────┬────────┘
         │
         ├──────► LangGraph Interview Graph
         │         ├─► StateGraph with 9 nodes + 3 routers
         │         ├─► SQLite checkpoint persistence
         │         └─► GraphServices dependency injection
         │
         ├──────► LLM Service
         │         ├─► API Mode: Groq / Gemini / OpenAI / Claude
         │         ├─► Hybrid Mode: Groq + Custom Model
         │         └─► Custom Mode: vLLM on GCP (via IAP Tunnel)
         │
         ├──────► Grilling Engine
         │         ├─► Gap Detection (18 types)
         │         ├─► Multi-dimensional Scoring
         │         └─► Follow-up Generation
         │
         ├──────► Voice Services
         │         ├─► Deepgram (STT)
         │         └─► ElevenLabs (TTS)
         │
         └──────► RAG Pipeline
                   ├─► Resume Parser (pdfplumber/PyMuPDF)
                   ├─► Semantic Chunker
                   ├─► ChromaDB Embedder (all-MiniLM-L6-v2)
                   └─► Context Retriever
```

### Custom Model on GCP

```
┌─────────────────┐     IAP Tunnel      ┌─────────────────────────┐
│  Local Machine  │ ◄─────────────────► │  GCP Compute Engine     │
│  (Port 8001)    │    gcloud tunnel    │  (GPU: T4/L4/A100)      │
└─────────────────┘                     │                         │
                                        │  vLLM Server (Port 8000)│
                                        │  ├─ Mistral-7B-Instruct │
                                        │  └─ LoRA Adapter        │
                                        └─────────────────────────┘
```

---

## Tech Stack

### Frontend
- **Framework**: Next.js 16 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **State Management**: Zustand
- **UI Components**: Custom components + Lucide React icons
- **Real-time**: WebSocket API

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip
- **Async Runtime**: uvicorn + asyncio
- **Orchestration**: LangGraph (StateGraph + SQLite checkpointing)
- **LLM Clients**: groq, anthropic, openai, google-generativeai

### RAG Pipeline
- **Resume Parsing**: pdfplumber, PyMuPDF
- **Vector Database**: ChromaDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM Framework**: Transformers, PEFT (for LoRA)

### Voice Services
- **STT**: Deepgram (nova-2 model)
- **TTS**: ElevenLabs (eleven_flash_v2_5 model)

### Custom Model Infrastructure
- **Cloud**: Google Cloud Platform (Compute Engine)
- **GPU**: T4 / L4 / A100
- **Inference Server**: vLLM
- **Tunnel**: IAP (Identity-Aware Proxy)

---

## Project Structure

```
resume-griller/
├── backend/                      # FastAPI backend
│   └── app/
│       ├── graph/               # LangGraph interview orchestration
│       │   ├── state.py         # InterviewState TypedDict
│       │   ├── nodes.py         # 9 graph node functions
│       │   ├── edges.py         # 3 routing functions
│       │   ├── builder.py       # StateGraph construction
│       │   ├── services.py      # GraphServices dependency injection
│       │   └── checkpointer.py  # SQLite checkpoint persistence
│       ├── api/routes/
│       │   ├── resume.py        # Resume upload & processing
│       │   ├── session.py       # Interview session endpoints (via graph)
│       │   ├── voice.py         # STT/TTS endpoints
│       │   └── websocket.py     # Real-time interview handler (via graph)
│       ├── core/
│       │   └── grilling_engine.py   # Gap detection & follow-ups
│       ├── services/
│       │   ├── llm_service.py   # LLM provider abstraction
│       │   │                    # (Groq, Gemini, OpenAI, Custom, Hybrid)
│       │   ├── stt_service.py   # Deepgram integration
│       │   └── tts_service.py   # ElevenLabs integration
│       ├── config.py            # Settings from .env
│       └── main.py              # FastAPI app entry
│
├── frontend/                     # Next.js frontend
│   └── src/
│       ├── app/                 # App router pages
│       ├── components/
│       │   ├── interview/       # Interview UI components
│       │   │   ├── VideoInterviewRoom.tsx
│       │   │   └── VoiceRecorder.tsx
│       │   └── upload/
│       │       └── ResumeUploader.tsx
│       └── stores/              # Zustand stores
│
├── rag/                          # RAG Pipeline
│   ├── resume_parser.py         # PDF/TXT parsing
│   ├── chunker.py               # Semantic chunking
│   ├── embedder.py              # ChromaDB operations
│   └── retriever.py             # Context retrieval
│
├── data/
│   ├── sample_resumes/          # Test resumes
│   ├── chromadb/                # Vector database
│   ├── interview_checkpoints.db # LangGraph session state (SQLite)
│   └── uploads/                 # User uploads
│
├── scripts/
│   ├── test_graph.py            # Interactive graph test (mock services)
│   └── start_iap_tunnel.sh      # GCP IAP tunnel helper
│
├── pyproject.toml               # Python project config (uv)
├── uv.lock                      # Locked dependencies
├── .env.example                 # Environment template
└── README.md
```

---

## Prerequisites

### Required
- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher
- **uv**: Python package manager ([install guide](https://github.com/astral-sh/uv))

### Optional
- **GCP Account**: For custom model deployment
- **GPU Instance**: T4/L4/A100 for vLLM inference

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Willkczy/resume-griller.git
cd resume-griller
```

### 2. Backend Setup (using uv)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Create Data Directories

```bash
mkdir -p data/uploads data/chromadb
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ============== LLM Configuration ==============
# Provider: anthropic, openai, gemini, groq
LLM_PROVIDER=groq

# API Keys
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here        # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # Optional
OPENAI_API_KEY=your_openai_api_key_here        # Optional

# Model names
GROQ_MODEL=llama-3.3-70b-versatile
GEMINI_MODEL=gemini-2.0-flash-exp

# ============== Custom Model (GCP vLLM) ==============
CUSTOM_MODEL_URL=http://localhost:8001/v1
CUSTOM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
CUSTOM_MODEL_TIMEOUT=120

# ============== Voice Services ==============
VOICE_ENABLED=true
DEEPGRAM_API_KEY=your_deepgram_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# ============== RAG Settings ==============
CHROMA_PERSIST_DIR=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ============== File Upload ==============
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE=10485760
```

### Quick Setup Options

#### Option 1: API Mode with Groq (Recommended)
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```
Fast, free, and easy to set up.

#### Option 2: Hybrid Mode (Best Quality)
```bash
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
CUSTOM_MODEL_URL=http://localhost:8001/v1
```
Requires GCP setup (see [Custom Model Deployment](#custom-model-deployment)).

---

## Usage

### Starting the Application

#### 1. Start Backend Server

```bash
# Using uv (recommended)
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --port 8000

# Or with auto-reload for development
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --reload --port 8000
```

Backend available at: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

#### 2. Start Frontend Server

```bash
cd frontend
npm run dev
```

Frontend available at: `http://localhost:3000`

#### 3. (Optional) Start IAP Tunnel for Custom Model

```bash
gcloud compute start-iap-tunnel YOUR_INSTANCE_NAME 8000 \
    --local-host-port=localhost:8001 \
    --zone=YOUR_ZONE
```

### Using the Application

1. Navigate to `http://localhost:3000`
2. Upload your resume (PDF or TXT)
3. Choose interview settings:
   - **Mode**: HR / Technical / Mixed
   - **Model**: API (cloud) or Custom (fine-tuned)
   - **Questions**: 3-10 questions
4. Start the interview
5. Answer questions via text or voice
6. Receive real-time feedback and follow-up questions
7. Review your interview summary

### Interview Modes

| Mode | Focus | Question Style |
|------|-------|----------------|
| **HR** | Soft skills, leadership, teamwork | "Tell me about a time when..." |
| **Tech** | Architecture, implementation, debugging | "Walk me through how you built..." |
| **Mixed** | Balanced combination | Both styles |

---

## Custom Model Deployment

### GCP Setup

1. **Create a GPU VM Instance**

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

2. **SSH into the Instance**

```bash
gcloud compute ssh interview-model --zone=us-central1-a
```

3. **Install vLLM and Start Server**

```bash
pip install vllm

# Start vLLM with LoRA adapter
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --enable-lora \
    --lora-modules interview-lora=shubhampareek/interview-coach-lora \
    --port 8000
```

4. **Create IAP Tunnel (Local Machine)**

```bash
gcloud compute start-iap-tunnel interview-model 8000 \
    --local-host-port=localhost:8001 \
    --zone=us-central1-a
```

5. **Test Connection**

```bash
curl http://localhost:8001/v1/models
```

### Using Custom Model

In the frontend, select "Custom Model" when starting an interview. The system will:
1. Use Groq for preprocessing (resume summary, question generation)
2. Use Custom Model for answer evaluation and follow-ups

---

## API Reference

### Session Endpoints

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

#### Submit Answer
```http
POST /api/v1/sessions/{session_id}/answer
Content-Type: application/json

{
  "answer": "I built a microservices architecture using..."
}
```

### WebSocket Interview

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/interview/{session_id}');

// Send answer
ws.send(JSON.stringify({
  type: "answer",
  content: "My response..."
}));

// Receive question/follow-up
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // data.type: "question" | "follow_up" | "complete"
  // data.content: The interviewer's response
  // data.data.evaluation: Score and gap analysis
};
```

---

## Development

### Running Tests

```bash
# Backend tests
PYTHONPATH=. uv run pytest tests/

# With coverage
PYTHONPATH=. uv run pytest tests/ --cov=backend --cov=rag
```

### Code Quality

```bash
# Format
uv run black backend/ rag/

# Lint
uv run ruff check backend/ rag/

# Type check
uv run mypy backend/ rag/
```

### Adding Dependencies

```bash
# Add a new package
uv add package-name

# Add dev dependency
uv add --dev package-name
```

---

## Troubleshooting

### Common Issues

#### "Custom model not available"
- Ensure IAP tunnel is running: `gcloud compute start-iap-tunnel ...`
- Check if vLLM server is running on GCP instance
- Verify `CUSTOM_MODEL_URL` in `.env`

#### HR mode asking technical questions
- Ensure you're using the latest `retriever.py` that handles both `"hr"` and `"behavioral"` values

#### Voice features not working
- Check `VOICE_ENABLED=true` in `.env`
- Verify Deepgram and ElevenLabs API keys
- Check browser microphone permissions

#### ChromaDB errors
- Delete `data/chromadb/` and restart to reinitialize
- Check disk space

#### WebSocket connection failed
- Ensure backend is running on port 8000
- Check CORS settings in `.env`

### Debug Mode

```bash
# Enable debug logging in .env
DEBUG=true

# View detailed logs
PYTHONPATH=. uv run python -m uvicorn backend.app.main:app --port 8000 --log-level debug
```

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- Fine-tuned LoRA model: [shubhampareek/interview-coach-lora](https://huggingface.co/shubhampareek/interview-coach-lora)
- Base model: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Grilling methodology: "Requirements Elicitation Follow-Up Question Generation" paper

---

Made with 🔥 by the Resume Griller team
