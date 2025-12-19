# Resume Griller 🔥

> **AI-powered interview simulator that grills candidates with resume-specific questions**

An intelligent interview preparation platform that analyzes your resume and conducts realistic mock interviews with AI. Features voice support, adaptive follow-up questions, and multiple interview modes (HR, Technical, Mixed).

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
- [RAG Pipeline](#rag-pipeline-standalone-usage)
- [API Reference](#api-reference)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Full-Stack Web Application

- **Resume Analysis**: Upload PDF/TXT resumes for AI-powered parsing and semantic analysis
- **Multiple Interview Modes**:
  - **HR Mode**: Behavioral questions focused on soft skills and experiences
  - **Technical Mode**: Deep technical questions about projects, tools, and skills
  - **Mixed Mode**: Comprehensive interview combining both HR and technical questions
- **Intelligent Follow-ups**: Vague or incomplete answers trigger adaptive follow-up questions that dig deeper
- **Voice Support**:
  - Speech-to-Text (STT) via Deepgram
  - Text-to-Speech (TTS) via ElevenLabs
  - Real-time voice interview simulation
- **Real-time Communication**: WebSocket-based interactive interview sessions
- **Session Management**: Track interview progress, conversation history, and results
- **Multiple LLM Backends**:
  - **API Mode**: Claude (Anthropic), GPT-4 (OpenAI), Gemini (Google), Llama (Groq)
  - **Custom Model**: Self-hosted vLLM on GCP
  - **Local Mode**: Fine-tuned LoRA model (Mistral-7B)

### RAG Pipeline (Standalone)

- **Resume Parsing**: Extract structured data from PDF/TXT files (contact, skills, experience, education, projects)
- **Semantic Chunking**: Intelligent resume segmentation for optimal retrieval
- **Vector Embeddings**: ChromaDB with sentence-transformers (all-MiniLM-L6-v2)
- **Context Retrieval**: RAG-based relevant information extraction
- **Question Generation**: Fine-tuned LoRA model for interview questions
- **Batch Processing**: Export prompts for GPU inference on Google Colab

---

## Architecture

### Full-Stack Application Flow

```
┌─────────────┐
│   Frontend  │  Next.js 16 + React 19 + TypeScript + Tailwind CSS
│  (Port 3000)│
└──────┬──────┘
       │ HTTP/WebSocket
       ▼
┌─────────────┐
│   Backend   │  FastAPI + SQLAlchemy + Async I/O
│  (Port 8000)│
└──────┬──────┘
       │
       ├──────► LLM Service (API/Custom/Local)
       │         └─► Claude / GPT-4 / Gemini / Groq / vLLM / LoRA
       │
       ├──────► Voice Services
       │         ├─► Deepgram (STT)
       │         └─► ElevenLabs (TTS)
       │
       └──────► RAG Pipeline
                 ├─► Resume Parser (pdfplumber/PyMuPDF)
                 ├─► Chunker (semantic segmentation)
                 ├─► Embedder (ChromaDB + sentence-transformers)
                 └─► Retriever (context extraction)
```

### RAG Pipeline (Standalone)

```
Resume (PDF/TXT)
    ↓
Resume Parser
    ├─► Extract contact, skills, experience, education, projects
    └─► ParsedResume dataclass
    ↓
Chunker
    ├─► Semantic chunking (overview, skills, jobs, education, projects)
    └─► Chunk objects with metadata
    ↓
Embedder (ChromaDB)
    ├─► sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
    └─► Vector embeddings + metadata storage
    ↓
Retriever
    ├─► Query-based retrieval (top-k similarity search)
    └─► Context extraction for specific focus areas
    ↓
Prompt Builder
    ├─► Format context for LLM
    └─► Technical / Behavioral / Mixed prompts
    ↓
Generator (LoRA Model)
    ├─► Base: mistralai/Mistral-7B-Instruct-v0.2
    ├─► Adapter: shubhampareek/interview-coach-lora
    └─► Interview question generation
```

---

## Tech Stack

### Frontend
- **Framework**: Next.js 16 (React 19)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **State Management**: Zustand
- **UI Components**: Custom components + lucide-react icons
- **Real-time**: WebSocket API

### Backend
- **Framework**: FastAPI
- **Language**: Python 3.11+
- **Async Runtime**: uvicorn + asyncio
- **Database**: SQLAlchemy + aiosqlite
- **API Clients**: anthropic, openai, google-generativeai, groq
- **File Processing**: aiofiles, python-multipart

### RAG Pipeline
- **Resume Parsing**: pdfplumber, PyMuPDF
- **Vector Database**: ChromaDB (SQLite backend)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2, 384 dimensions)
- **LLM Framework**: PyTorch, Transformers, PEFT
- **Fine-tuning**: LoRA on Mistral-7B-Instruct-v0.2

### Voice Services
- **STT (Speech-to-Text)**: Deepgram (nova-2 model)
- **TTS (Text-to-Speech)**: ElevenLabs (eleven_flash_v2 model)

### Development Tools
- **Testing**: pytest + pytest-asyncio
- **Code Quality**: black, ruff, mypy
- **Package Management**: pip, npm
- **Pre-commit Hooks**: pre-commit

---

## Project Structure

```
resume-griller/
├── backend/                      # FastAPI backend server
│   ├── app/
│   │   ├── api/                 # API routes
│   │   │   ├── routes/
│   │   │   │   ├── resume.py   # Resume upload & processing
│   │   │   │   ├── session.py  # Interview session management
│   │   │   │   ├── voice.py    # STT/TTS endpoints
│   │   │   │   └── websocket.py # WebSocket interview handler
│   │   │   └── deps.py          # Dependency injection
│   │   ├── core/                # Core business logic
│   │   │   ├── interview_agent.py    # Interview orchestration
│   │   │   ├── grilling_engine.py    # Follow-up question logic
│   │   │   └── resume_parser.py      # Resume parsing wrapper
│   │   ├── services/            # External service integrations
│   │   │   ├── llm_service.py  # LLM provider abstraction
│   │   │   ├── stt_service.py  # Speech-to-text service
│   │   │   └── tts_service.py  # Text-to-speech service
│   │   ├── models/              # Data models
│   │   │   └── schemas.py      # Pydantic schemas
│   │   ├── db/                  # Database layer
│   │   │   └── session_store.py # In-memory session storage
│   │   ├── config.py            # Configuration settings
│   │   └── main.py              # FastAPI application entry point
│   ├── requirements.txt
│   └── tests/
│
├── frontend/                     # Next.js frontend
│   ├── src/
│   │   ├── app/                 # Next.js app router
│   │   │   ├── page.tsx         # Landing page
│   │   │   ├── upload/          # Resume upload page
│   │   │   ├── interview/       # Interview room page
│   │   │   └── result/          # Interview results page
│   │   ├── components/
│   │   │   ├── ui/              # Reusable UI components
│   │   │   ├── interview/       # Interview-specific components
│   │   │   │   ├── InterviewRoom.tsx
│   │   │   │   ├── VideoInterviewRoom.tsx
│   │   │   │   ├── VoiceRecorder.tsx
│   │   │   │   └── ChatMessage.tsx
│   │   │   ├── upload/
│   │   │   │   └── ResumeUploader.tsx
│   │   │   └── layout/
│   │   │       └── Header.tsx
│   │   ├── stores/              # Zustand state management
│   │   ├── lib/                 # Utilities
│   │   └── types/               # TypeScript types
│   ├── package.json
│   └── tsconfig.json
│
├── rag/                          # RAG Pipeline (standalone)
│   ├── __init__.py
│   ├── resume_parser.py          # PDF/TXT resume parser (464 lines)
│   ├── chunker.py                # Semantic chunking (224 lines)
│   ├── embedder.py               # ChromaDB embeddings (282 lines)
│   ├── retriever.py              # RAG retrieval (214 lines)
│   └── generator.py              # LoRA model inference (185 lines)
│
├── data/
│   ├── sample_resumes/           # Test resumes (5 samples)
│   ├── chromadb/                 # ChromaDB vector database
│   ├── uploads/                  # User-uploaded resumes
│   └── exported_prompts.json     # Prompts for Colab inference
│
├── ml/                           # ML training artifacts (optional)
│   ├── models/
│   │   └── interview-coach-lora/ # LoRA model checkpoints
│   ├── training/                 # Training scripts
│   └── evaluation/               # Model evaluation
│
├── tests/                        # Test suite
│   └── test_parser.py
│
├── docs/                         # Documentation
│   └── CONTRIBUTING.md
│
├── LLM_Inference.ipynb           # Google Colab inference notebook
├── export_prompts.py             # Export prompts for Colab
├── main.py                       # Unified entry point (optional)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
├── CLAUDE.md                     # Claude Code assistant instructions
└── README.md                     # This file
```

---

## Prerequisites

### Required
- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher
- **npm**: 9.x or higher

### Optional (for specific features)
- **CUDA-capable GPU**: For local LoRA model inference (12GB+ VRAM recommended)
- **Google Colab**: For cloud-based GPU inference (free tier available)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resume-griller.git
cd resume-griller
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Or install with optional ML dependencies for local LoRA inference
pip install -r requirements.txt
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

---

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# ============== App Settings ==============
APP_NAME="Resume Griller"
DEBUG=false

# ============== LLM Configuration ==============
# Mode: "api" (API providers) or "local" (LoRA model)
LLM_MODE=api

# API Mode - Choose provider: anthropic, openai, gemini, groq
LLM_PROVIDER=groq

# API Keys (only needed for API mode)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Model names (customize as needed)
ANTHROPIC_MODEL=claude-sonnet-4-20250514
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.5-flash
GROQ_MODEL=llama-3.3-70b-versatile

# Custom Model (self-hosted vLLM on GCP)
CUSTOM_MODEL_ENABLED=false
CUSTOM_MODEL_URL=http://localhost:8001/v1
CUSTOM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2

# Local LoRA Model (for LLM_MODE=local)
LOCAL_MODEL_BASE=mistralai/Mistral-7B-Instruct-v0.2
LOCAL_MODEL_LORA=shubhampareek/interview-coach-lora

# ============== Voice Services ==============
VOICE_ENABLED=true

# Deepgram (Speech-to-Text)
DEEPGRAM_API_KEY=your_deepgram_api_key_here
DEEPGRAM_MODEL=nova-2

# ElevenLabs (Text-to-Speech)
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice
ELEVENLABS_MODEL=eleven_flash_v2

# ============== RAG Settings ==============
CHROMA_PERSIST_DIR=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ============== File Upload ==============
UPLOAD_DIR=./data/uploads
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes

# ============== CORS ==============
CORS_ORIGINS=["http://localhost:3000"]
```

### Quick Setup for Different LLM Modes

#### Option 1: API Mode (Recommended for most users)

```bash
# .env
LLM_MODE=api
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```

#### Option 2: Local LoRA Model (Requires GPU)

```bash
# .env
LLM_MODE=local
LOCAL_MODEL_BASE=mistralai/Mistral-7B-Instruct-v0.2
LOCAL_MODEL_LORA=shubhampareek/interview-coach-lora
```

#### Option 3: Custom vLLM Server (Advanced)

```bash
# .env
LLM_MODE=api
CUSTOM_MODEL_ENABLED=true
CUSTOM_MODEL_URL=http://your-vllm-server:8001/v1
CUSTOM_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.2
```

---

## Usage

### Running the Full Application

#### 1. Start Backend Server

```bash
cd backend
python -m backend.app.main

# Or with auto-reload for development
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend will be available at: `http://localhost:8000`
API Documentation: `http://localhost:8000/docs`

#### 2. Start Frontend Development Server

```bash
cd frontend
npm run dev
```

Frontend will be available at: `http://localhost:3000`

#### 3. Use the Application

1. Navigate to `http://localhost:3000`
2. Click "Start Interview"
3. Upload your resume (PDF or TXT)
4. Choose interview mode (HR / Technical / Mixed)
5. Start the interview session
6. Answer questions via text or voice
7. Review your interview results

### Interview Modes

- **HR Mode**: Focuses on behavioral questions, soft skills, leadership, teamwork, and experiences
- **Technical Mode**: Deep-dive into technical skills, projects, tools, algorithms, and problem-solving
- **Mixed Mode**: Balanced mix of HR and technical questions

### Voice Interview

1. Enable microphone when prompted
2. Click the microphone button to record your answer
3. Speak naturally
4. Click stop when finished
5. Your speech will be transcribed and sent to the interviewer
6. The AI response will be read aloud (if TTS is enabled)

---

## RAG Pipeline (Standalone Usage)

The RAG pipeline can be used independently without running the full web application.

### Quick Start

```bash
# Test resume parser
python -m rag.resume_parser

# Test full RAG pipeline
python -m rag.retriever

# Export prompts for Colab inference
python export_prompts.py
```

### Parse a Resume

```python
from rag.resume_parser import ResumeParser

parser = ResumeParser()
parsed = parser.parse("data/sample_resumes/resume_sp.pdf")

print(f"Name: {parsed.contact.name}")
print(f"Skills: {parsed.skills}")
print(f"Experience: {len(parsed.experience)} jobs")
print(f"Education: {len(parsed.education)} degrees")
```

### Process and Embed Resume

```python
from rag.retriever import InterviewRetriever

retriever = InterviewRetriever()

# Process resume (parse, chunk, embed)
resume_id = retriever.process_resume(
    file_path="data/sample_resumes/resume_sp.pdf",
    resume_id="resume_sp"
)
print(f"Processed: {resume_id}")
```

### Retrieve Context for Questions

```python
# Retrieve relevant chunks for specific focus
chunks = retriever.retrieve(
    resume_id="resume_sp",
    focus_area="Python programming",
    n_chunks=5
)

# Build prompt for question generation
prompt = retriever.build_prompt(
    resume_id="resume_sp",
    question_type="technical",  # or "behavioral", "mixed"
    focus_area="machine learning projects",
    n_questions=5
)
print(prompt)
```

### Generate Questions (Local GPU)

```python
from rag.generator import InterviewGenerator

# Requires GPU (12GB+ VRAM recommended)
generator = InterviewGenerator()

question = generator.generate(
    prompt=prompt,
    temperature=0.7,
    max_tokens=256
)
print(question)
```

### Generate Questions (Google Colab)

1. Export prompts to JSON:
```bash
python export_prompts.py
```

2. Upload to Google Colab:
   - Upload `LLM_Inference.ipynb` to Colab
   - Upload `data/exported_prompts.json` to Colab
   - Run the notebook cells (uses free T4 GPU)

3. Download generated questions

### ChromaDB Operations

```python
from rag.embedder import ResumeEmbedder

embedder = ResumeEmbedder()

# Get all chunks for a resume
chunks = embedder.get_all_chunks(resume_id="resume_sp")
print(f"Total chunks: {len(chunks)}")

# Search for specific content
results = embedder.search(
    query="Python machine learning experience",
    n_results=5,
    resume_id="resume_sp"
)

# Clear all data
embedder.clear_collection()
```

---

## API Reference

### Resume Endpoints

#### Upload Resume
```http
POST /api/v1/resume/upload
Content-Type: multipart/form-data

file: <resume.pdf>
```

Response:
```json
{
  "resume_id": "resume_sp_20241218_123456_abc123",
  "filename": "resume_sp.pdf",
  "chunks_created": 15,
  "sections": ["overview", "skills", "experience", "education", "projects"],
  "message": "Resume processed successfully"
}
```

#### Get Resume Summary
```http
GET /api/v1/resume/{resume_id}/summary
```

Response:
```json
{
  "resume_id": "resume_sp_20241218_123456_abc123",
  "name": "John Doe",
  "total_chunks": 15,
  "sections": ["overview", "skills", "experience", "education"],
  "skills": ["Python", "Machine Learning", "FastAPI"],
  "experience_count": 3,
  "education_count": 1
}
```

### Session Endpoints

#### Create Interview Session
```http
POST /api/v1/session/create
Content-Type: application/json

{
  "resume_id": "resume_sp_20241218_123456_abc123",
  "mode": "mixed",
  "focus_areas": ["Python", "leadership"],
  "num_questions": 5
}
```

Response:
```json
{
  "session_id": "session_xyz789",
  "resume_id": "resume_sp_20241218_123456_abc123",
  "mode": "mixed",
  "status": "in_progress",
  "created_at": "2024-12-18T12:00:00Z",
  "focus_areas": ["Python", "leadership"],
  "total_questions": 5,
  "questions_asked": 0
}
```

#### Get Session Details
```http
GET /api/v1/session/{session_id}
```

### Voice Endpoints

#### Transcribe Audio (STT)
```http
POST /api/v1/voice/transcribe
Content-Type: multipart/form-data

audio: <audio.webm>
```

Response:
```json
{
  "text": "I have five years of experience in Python development...",
  "confidence": 0.95
}
```

#### Generate Speech (TTS)
```http
POST /api/v1/voice/speak
Content-Type: application/json

{
  "text": "Tell me about your experience with Python."
}
```

Response: Audio file (audio/mpeg)

### WebSocket Endpoint

#### Interview Session WebSocket
```
WS /ws/interview/{session_id}
```

Send message:
```json
{
  "type": "answer",
  "content": "I have worked with Python for 5 years..."
}
```

Receive message:
```json
{
  "type": "question",
  "content": "Can you elaborate on a specific Python project?",
  "metadata": {
    "question_number": 2,
    "is_followup": true
  }
}
```

---

## Development

### Running Tests

```bash
# Backend tests
pytest tests/

# With coverage
pytest tests/ --cov=backend --cov=rag

# Frontend tests (if configured)
cd frontend
npm test
```

### Code Quality

```bash
# Format code
black backend/ rag/

# Lint
ruff backend/ rag/

# Type check
mypy backend/ rag/
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Project Dependencies

Update dependencies:

```bash
# Backend
pip install -r requirements.txt --upgrade

# Frontend
cd frontend
npm update
```

### Adding New Features

1. Create a new branch
2. Implement feature with tests
3. Update documentation
4. Run code quality checks
5. Submit pull request

---

## Troubleshooting

### Common Issues

#### Backend won't start
- Check if port 8000 is already in use
- Verify Python version (3.11+)
- Ensure all dependencies are installed
- Check environment variables in `.env`

#### Frontend won't start
- Check if port 3000 is already in use
- Verify Node.js version (18+)
- Delete `node_modules` and run `npm install` again
- Check for TypeScript errors

#### Resume upload fails
- Verify file size is under 10MB
- Check file format (PDF or TXT only)
- Ensure `data/uploads` directory exists
- Check backend logs for errors

#### Voice features not working
- Verify `VOICE_ENABLED=true` in `.env`
- Check API keys for Deepgram and ElevenLabs
- Ensure microphone permissions are granted
- Check browser console for errors

#### ChromaDB errors
- Delete `data/chromadb` and let it reinitialize
- Check disk space
- Verify ChromaDB version compatibility

#### LoRA model won't load (Local Mode)
- Requires 12GB+ GPU VRAM
- Check CUDA installation
- Verify model files downloaded from HuggingFace
- Try reducing batch size or using CPU (slow)

### Debug Mode

Enable debug logging:

```bash
# .env
DEBUG=true
```

View logs:
```bash
# Backend logs
tail -f backend.log

# Frontend logs
Check browser console (F12)
```

---

## License

MIT License

Copyright (c) 2024 Resume Griller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for development guidelines.

---

## Acknowledgments

- Fine-tuned LoRA model: [shubhampareek/interview-coach-lora](https://huggingface.co/shubhampareek/interview-coach-lora)
- Base model: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- Embedding model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

Made with 🔥 by the Resume Griller team
