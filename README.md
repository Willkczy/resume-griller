# 🔥 Resume Griller

AI-powered interview simulator that "grills" candidates with deep, resume-specific questions.

## Project Structure

```
resume-griller/
├── backend/          # FastAPI backend
├── frontend/         # Next.js frontend  
├── ml/               # LoRA fine-tuning module
└── docs/             # Documentation
```

## Team Members

| Member | Responsibility |
|--------|---------------|
| [Name] | PDF Parser / Backend |
| [Name] | LoRA Fine-tuning |
| [Name] | Frontend |
| [Name] | TBD |

## Quick Start

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

## Development Status

- [ ] Phase 1: Core MVP
  - [ ] Resume PDF parsing
  - [ ] Basic question generation
  - [ ] Text-based Q&A
- [ ] Phase 2: Voice Integration
- [ ] Phase 3: Video Interface
- [ ] Phase 4: Production

## Git Workflow

See [CONTRIBUTING.md](docs/CONTRIBUTING.md)
