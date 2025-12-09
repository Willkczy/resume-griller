# 🔥 Resume Griller

AI-powered interview simulator that "grills" candidates with deep, resume-specific questions.

## 🎯 What We're Building

An application that:
1. **Parses resumes** (PDF/DOCX) and extracts structured information
2. **Generates targeted interview questions** based on resume content
3. **Conducts mock interviews** with voice interaction and video feed
4. **"Grills" candidates** by asking follow-up questions when answers are vague

### Two Interview Modes
- **HR Mode**: Behavioral questions, STAR method deep-dives
- **Tech Mode**: Technical verification, system design, implementation details

---

## 📁 Project Structure

```
resume-griller/
│
├── backend/                    # FastAPI backend (Python)
│   ├── app/
│   │   ├── api/routes/        # API endpoints
│   │   ├── core/              # Core business logic
│   │   │   └── resume_parser.py   # ⭐ PDF/DOCX parsing
│   │   ├── services/          # External service integrations
│   │   │   └── llm_service.py     # LLM abstraction (API + Local)
│   │   ├── models/            # Pydantic schemas
│   │   └── db/                # Database
│   └── tests/
│
├── ml/                         # Machine Learning module
│   ├── data/
│   │   ├── resumes/           # Resume datasets
│   │   ├── interview_qa/      # Interview Q&A datasets
│   │   └── processed/         # Processed training data
│   ├── models/
│   │   ├── checkpoints/       # Training checkpoints
│   │   └── exported/          # Exported models for inference
│   ├── training/              # LoRA fine-tuning scripts
│   ├── evaluation/            # Model evaluation & benchmarks
│   └── configs/               # Training configurations
│
├── frontend/                   # Next.js frontend (TypeScript)
│   └── src/
│       ├── app/               # Pages
│       ├── components/        # React components
│       └── hooks/             # Custom hooks
│
├── docs/                       # Documentation
├── pyproject.toml             # Python dependencies (using uv)
└── .env.example               # Environment variables template
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 18+ (for frontend)

### 1. Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/resume-griller.git
cd resume-griller

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### 2. Backend Setup (using uv)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# For ML work, also install ML dependencies
uv pip install -e ".[ml]"

# Run the backend
cd backend
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 👥 Team Responsibilities

| Area | Owner | Key Files |
|------|-------|-----------|
| PDF Parser | TBD | `backend/app/core/resume_parser.py` |
| LoRA Fine-tuning | TBD | `ml/training/`, `ml/data/` |
| Frontend | TBD | `frontend/src/` |
| Interview Agent | TBD | `backend/app/core/interview_agent.py` |

---

## 🔧 Architecture Decisions

### LLM Strategy: API vs Fine-tuned Model

We support **both approaches** and can switch between them:

```
┌─────────────────────────────────────────────────┐
│              LLM Service Abstraction            │
├─────────────────────────────────────────────────┤
│                                                 │
│   ┌─────────────┐      ┌─────────────────────┐ │
│   │  API Mode   │      │    Local Mode       │ │
│   │             │      │                     │ │
│   │ Claude API  │      │  Fine-tuned Model   │ │
│   │ GPT-4o API  │      │  (LoRA on Llama/    │ │
│   │             │      │   Mistral/etc)      │ │
│   └─────────────┘      └─────────────────────┘ │
│         │                       │              │
│         └───────────┬───────────┘              │
│                     │                          │
│            Unified Interface                   │
│                     │                          │
│              Interview Agent                   │
└─────────────────────────────────────────────────┘
```

**Configuration** (in `.env`):
```bash
LLM_MODE=api          # "api" or "local"
LLM_PROVIDER=anthropic  # "anthropic" or "openai" (when mode=api)
LOCAL_MODEL_PATH=./ml/models/exported/resume-griller  # (when mode=local)
```

### Why This Design?
1. **Flexibility**: Easy to compare API vs fine-tuned model performance
2. **Cost optimization**: Use local model for high-volume, API for complex tasks
3. **Independent development**: ML team can work on fine-tuning while others use API

---

## 📋 Development Phases

### Phase 1: Core MVP (Current)
- [ ] Resume PDF/DOCX parsing
- [ ] Basic question generation (using API)
- [ ] Text-based Q&A interface
- [ ] Grilling logic (follow-up detection)

### Phase 2: Voice Integration
- [ ] Speech-to-Text (Deepgram)
- [ ] Text-to-Speech (ElevenLabs)
- [ ] Real-time audio streaming

### Phase 3: Video & Polish
- [ ] WebRTC video feed
- [ ] HR/Tech mode switching
- [ ] UI/UX improvements

### Phase 4: ML Integration
- [ ] LoRA fine-tuning pipeline
- [ ] Model evaluation & benchmarking
- [ ] Hybrid API + Local strategy

---

## 🔀 Git Workflow

### Branching

```
main
  └── feature/your-feature-name
```

### Branch Naming
```
feature/pdf-parser
feature/lora-training
feature/interview-agent
fix/resume-encoding
```

### Commit Messages
```
feat(parser): add PDF text extraction
feat(ml): setup LoRA training pipeline
fix(api): handle timeout errors
docs(readme): update setup instructions
```

### Pull Request Process
1. Create feature branch from `main`
2. Make your changes
3. Push and create PR
4. Get at least 1 review
5. Squash and merge

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI, Python 3.11 |
| Frontend | Next.js 14, React, TypeScript |
| Package Manager | uv (Python), npm (Node.js) |
| LLM (API) | Claude 3.5 / GPT-4o |
| LLM (Local) | Llama 3 / Mistral + LoRA |
| STT | Deepgram |
| TTS | ElevenLabs |
| Database | SQLite (dev), PostgreSQL (prod) |

---

## 📚 Useful Resources

### For PDF Parsing
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
- [pdfplumber Documentation](https://github.com/jsvine/pdfplumber)

### For LoRA Fine-tuning
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Fine-tuning LLMs Guide](https://huggingface.co/docs/transformers/training)

### Resume Datasets
- [Resume Dataset (Kaggle)](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- [Resume Corpus](https://github.com/florex/resume_corpus)

---

## ❓ Questions?

Open an issue or reach out to the team!
