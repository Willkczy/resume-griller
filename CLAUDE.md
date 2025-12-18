# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Branch: rag-pipeline**

Resume Griller (RAG Pipeline) is a standalone RAG (Retrieval-Augmented Generation) system for generating resume-specific interview questions using a fine-tuned LoRA model. This branch implements a complete pipeline for parsing resumes, chunking them semantically, embedding them in a vector database, retrieving relevant context, and generating interview questions via GPU inference on Google Colab.

**Important**: This branch is a significant pivot from the `main` branch. The full-stack interview simulator (FastAPI backend + Next.js frontend) planned on `main` is NOT implemented here. This is purely a RAG pipeline for offline question generation.

## Commands

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test the RAG pipeline
python -m rag.retriever              # Full pipeline test
python -m rag.resume_parser          # Test parser only
python export_prompts.py             # Export prompts for Colab

# Run tests
pytest tests/test_parser.py
```

### RAG Pipeline Workflow

```bash
# 1. Parse a resume (PDF or TXT)
python -c "from rag.resume_parser import ResumeParser; \
  parser = ResumeParser(); \
  result = parser.parse('data/sample_resumes/resume_sp.pdf'); \
  print(result)"

# 2. Process resume and embed in ChromaDB
python -c "from rag.retriever import RAGRetriever; \
  retriever = RAGRetriever(); \
  retriever.process_resume('data/sample_resumes/resume_sp.pdf', 'resume_sp')"

# 3. Retrieve context and build prompts
python -c "from rag.retriever import RAGRetriever; \
  retriever = RAGRetriever(); \
  prompt = retriever.build_prompt('resume_sp', 'technical', 'Python programming'); \
  print(prompt)"

# 4. Export prompts for GPU inference (Google Colab)
python export_prompts.py

# 5. Upload LLM_Inference.ipynb + data/exported_prompts.json to Colab
# Run the notebook cells to generate questions on GPU
```

### Development Commands

```bash
# Run specific modules
python -m rag.chunker               # Test chunker
python -m rag.embedder              # Test embedder
python -m rag.generator             # Test generator (requires GPU)

# Check ChromaDB contents
python -c "from rag.embedder import Embedder; \
  embedder = Embedder(); \
  chunks = embedder.get_all_chunks(); \
  print(f'Total chunks: {len(chunks)}')"

# Clear ChromaDB
python -c "from rag.embedder import Embedder; \
  embedder = Embedder(); \
  embedder.clear_collection()"
```

## Architecture

### RAG Pipeline Flow

```
Resume (PDF/TXT)
    ↓
[resume_parser.py] → ParsedResume (contact, skills, experience, education, projects)
    ↓
[chunker.py] → Chunks (overview, skills, each job, each education, each project)
    ↓
[embedder.py] → ChromaDB (all-MiniLM-L6-v2 embeddings, 384 dimensions)
    ↓
[retriever.py] → Relevant chunks based on focus area
    ↓
[retriever.py] → Formatted prompt (technical/behavioral/mixed)
    ↓
[generator.py] → LoRA model inference (Mistral-7B + interview-coach-lora)
    ↓
Interview Questions
```

### Project Structure

```
resume-griller/
├── rag/                            # RAG pipeline (IMPLEMENTED)
│   ├── resume_parser.py            # 464 lines - PDF/TXT parser
│   ├── chunker.py                  # 224 lines - Semantic chunking
│   ├── embedder.py                 # 282 lines - ChromaDB embeddings
│   ├── retriever.py                # 214 lines - RAG retrieval & prompt building
│   └── generator.py                # 185 lines - LoRA model inference
│
├── data/
│   ├── sample_resumes/             # 5 sample resumes (4 TXT + 1 PDF)
│   ├── chromadb/                   # ChromaDB SQLite (280KB)
│   └── exported_prompts.json       # 117+ prompts for Colab inference
│
├── tests/
│   └── test_parser.py              # Basic parser tests
│
├── LLM_Inference.ipynb             # Google Colab notebook for GPU inference
├── export_prompts.py               # Export prompts to JSON for Colab
├── requirements.txt                # 6 dependencies (torch, chromadb, etc.)
└── README.md
```

### Key Components

**Resume Parser** (`rag/resume_parser.py`):
- Parses PDF (via pdfplumber) and TXT files
- Extracts structured sections: contact, summary, skills, experience, education, projects, certifications
- Uses regex patterns to identify section headers
- Handles multiple resume formats with bullet points, dates, job entries
- Returns `ParsedResume` dataclass with structured data (464 lines, fully implemented)

**Chunker** (`rag/chunker.py`):
- Converts parsed resume into semantic chunks for RAG
- One chunk per meaningful unit: overview, skills, each job, each education entry, each project
- Returns `Chunk` objects with content, section type, and metadata (224 lines, fully implemented)

**Embedder** (`rag/embedder.py`):
- Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings
- Stores vectors in ChromaDB (persistent at `data/chromadb/`)
- Supports search with filters (by resume_id, section, focus area)
- CRUD operations: embed, search, get_all_chunks, delete, clear (282 lines, fully implemented)

**Retriever** (`rag/retriever.py`):
- End-to-end pipeline: PDF → Parse → Chunk → Embed → Retrieve
- `process_resume()`: Full resume ingestion
- `retrieve()`: Get relevant chunks based on focus area
- `build_prompt()`: Formats prompts for LLM with resume context (technical/behavioral/mixed)
- `get_resume_summary()`: Returns resume overview (214 lines, fully implemented)

**Generator** (`rag/generator.py`):
- Loads fine-tuned LoRA model from HuggingFace
- Base model: `mistralai/Mistral-7B-Instruct-v0.2`
- LoRA adapter: `shubhampareek/interview-coach-lora`
- Generates interview questions from prompts
- Supports CUDA/MPS/CPU (requires ~14GB RAM, GPU recommended) (185 lines, fully implemented)

**LLM Inference Notebook** (`LLM_Inference.ipynb`):
- Google Colab notebook for GPU inference
- Loads LoRA model and processes batched prompts
- Reads from `data/exported_prompts.json`
- Generates questions with custom temperature

## Dependencies

**Current (requirements.txt)**:
```
torch>=2.0.0
pdfplumber>=0.10.0
PyMuPDF>=1.23.0
langchain>=0.1.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
```

Only 6 dependencies - focused on RAG/ML pipeline.

**Installation**:
```bash
pip install -r requirements.txt
```

## Development Workflow

### Git Branching

This is the `rag-pipeline` branch, which diverged from `main` at commit `7518948`.

**Recent commits**:
- `dd70f27` - "Added the inference colab file"
- `e5b50c3` - "RAG file and updated README"
- `8adc190` - "RAG Pipeline for interview coach"

### Commit Message Format

```
type(scope): short description

[optional body]
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`

Examples:
- `feat(rag): add semantic chunking for projects`
- `fix(parser): handle UTF-8 encoding in resumes`
- `refactor(embedder): optimize ChromaDB query performance`

## Code Quality

**Python**:
- Line length: 88 characters (Black default recommended)
- Target version: Python 3.11+
- Type hints: Used extensively (ParsedResume, Chunk dataclasses)
- Test framework: pytest

## How to Use

### 1. Parse a Resume

```python
from rag.resume_parser import ResumeParser

parser = ResumeParser()
parsed = parser.parse("data/sample_resumes/resume_sp.pdf")

print(f"Name: {parsed.contact.name}")
print(f"Skills: {parsed.skills}")
print(f"Jobs: {len(parsed.experience)}")
```

### 2. Process and Embed

```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever()
retriever.process_resume("data/sample_resumes/resume_sp.pdf", "resume_sp")
```

### 3. Retrieve Context and Build Prompts

```python
# Get resume summary
summary = retriever.get_resume_summary("resume_sp")
print(summary)

# Retrieve relevant chunks for specific focus
chunks = retriever.retrieve("resume_sp", focus="Python programming", top_k=5)

# Build prompt for question generation
prompt = retriever.build_prompt(
    resume_id="resume_sp",
    question_type="technical",  # or "behavioral", "mixed"
    focus="leadership experience"
)
print(prompt)
```

### 4. Generate Questions (GPU Required)

```python
from rag.generator import Generator

# Local inference (requires GPU, ~14GB RAM)
generator = Generator()
question = generator.generate(prompt, temperature=0.7, max_tokens=256)
print(question)
```

**OR**

```bash
# Export prompts and use Colab (recommended)
python export_prompts.py
# Upload LLM_Inference.ipynb + data/exported_prompts.json to Google Colab
# Run notebook cells for batched GPU inference
```

## Sample Resumes

The `data/sample_resumes/` directory contains 5 sample resumes:

1. `resume_sp.pdf` - Shubham Pareek (PDF format, embedded in ChromaDB)
2. `resume1.txt` - Software Engineer
3. `resume2.txt` - Data Scientist
4. `resume3.txt` - Product Manager
5. `resume4.txt` - DevOps Engineer

## ChromaDB Details

**Location**: `data/chromadb/`
**Size**: 280KB SQLite database
**Collection**: "resumes"
**Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
**Current Chunks**: 15 chunks from resume_sp.pdf

**Metadata Structure**:
```python
{
    "resume_id": "resume_sp",
    "section": "experience",  # or "skills", "education", "projects", "overview"
    "chunk_index": 0
}
```

## LoRA Model Details

**Base Model**: `mistralai/Mistral-7B-Instruct-v0.2`
**LoRA Adapter**: `shubhampareek/interview-coach-lora`
**Hosted on**: HuggingFace

**Hardware Requirements**:
- **Inference (GPU)**: 12GB+ VRAM, 16GB RAM (Colab T4/A100 recommended)
- **Inference (CPU)**: ~14GB RAM, very slow (not recommended)

**LoRA Configuration** (used during training):
- r=16
- lora_alpha=32
- target_modules=[q_proj, k_proj, v_proj, o_proj]

## Prompt Types

The retriever supports three question types:

1. **Technical**: Focus on technical skills, projects, tools
   - "Based on the resume, ask a technical question about [focus area]"

2. **Behavioral**: Focus on experiences, leadership, soft skills
   - "Based on the resume, ask a behavioral question about [focus area]"

3. **Mixed**: Combination of technical and behavioral
   - "Based on the resume, ask an interview question about [focus area]"

## Focus Areas

When retrieving context, you can specify focus areas:
- "Python programming"
- "leadership experience"
- "machine learning projects"
- "cloud infrastructure"
- Or leave empty for general questions

## What's NOT Implemented

The following features from the `main` branch vision are NOT implemented on `rag-pipeline`:

- FastAPI backend server
- REST API endpoints
- Next.js frontend
- Voice integration (Deepgram STT, ElevenLabs TTS)
- Video feed (WebRTC)
- Interactive interview agent with follow-up questions
- "Grilling" logic for vague answers
- HR vs Tech mode switching
- WebSocket real-time communication
- SQLAlchemy database models
- API-based LLM service (Claude/GPT)
- Authentication/sessions
- Web UI

This branch is purely a RAG pipeline for offline question generation, not a full-stack web application.

## Tech Stack

- **Language**: Python 3.11+
- **PDF Parsing**: pdfplumber, PyMuPDF
- **Vector Database**: ChromaDB (SQLite backend)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Mistral-7B-Instruct-v0.2 + LoRA fine-tuning (shubhampareek/interview-coach-lora)
- **Inference**: Google Colab (GPU), local inference possible but slow
- **Dependencies**: torch, transformers, peft, accelerate

## Comparison: main vs rag-pipeline

| Aspect | main Branch | rag-pipeline Branch |
|--------|-------------|---------------------|
| Architecture | Full-stack web app | Standalone RAG pipeline |
| Backend | FastAPI + SQLAlchemy | No backend server |
| Frontend | Next.js + React | No frontend |
| LLM Strategy | API (Claude/GPT) + Local LoRA | Local LoRA only |
| Deployment | Web application | Colab notebook |
| Purpose | Interactive interview simulator | Batch question generation |
| Dependencies | 20+ packages | 6 packages |
| Database | PostgreSQL/SQLite | ChromaDB (vector DB) |

## Current State

**What Works**:
- Resume parsing (PDF/TXT) with full section extraction ✓
- Semantic chunking for RAG ✓
- Vector embeddings with ChromaDB ✓
- RAG retrieval pipeline ✓
- Prompt generation for interview questions ✓
- LoRA fine-tuned model inference (via Colab) ✓
- Export prompts for GPU inference ✓

**What Doesn't Work**:
- FastAPI server (doesn't exist)
- Frontend (doesn't exist)
- Local model inference without GPU (too slow, not recommended)
- Interactive interview simulation (not implemented)

## Next Steps (Suggestions)

1. Add more sample resumes to test edge cases
2. Improve chunking strategy (e.g., sliding windows, hierarchical)
3. Add evaluation metrics (question quality, relevance)
4. Compare different embedding models
5. Fine-tune retrieval hyperparameters (top_k, similarity thresholds)
6. Add batched processing for multiple resumes
7. Experiment with different LoRA configurations
8. Add question deduplication logic

## Troubleshooting

**ChromaDB errors**:
```bash
# Clear and rebuild ChromaDB
python -c "from rag.embedder import Embedder; Embedder().clear_collection()"
python -m rag.retriever
```

**Parser errors**:
- Ensure PDF is text-based (not scanned image)
- Check file encoding (UTF-8 recommended)
- Try with sample resumes first

**Generator errors**:
- Requires GPU for reasonable performance
- Use Colab for inference instead of local
- Check HuggingFace model availability
