# Contributing Guide

## 🛠️ Development Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager
- Node.js 18+ (for frontend)
- Git

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### Setup Project

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/resume-griller.git
cd resume-griller

# Create virtual environment
uv venv

# Activate (choose your OS)
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install dependencies
uv pip install -e ".[dev]"     # Core + dev tools
uv pip install -e ".[ml]"      # Add ML dependencies (optional)
```

---

## 🔀 Git Workflow

### Branch Naming

```
feature/description    # New features
fix/description        # Bug fixes
refactor/description   # Code improvements
docs/description       # Documentation
```

**Examples:**
```
feature/pdf-parser
feature/lora-training
feature/interview-agent
fix/unicode-handling
docs/api-documentation
```

### Commit Message Format

```
type(scope): short description

[optional body]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `refactor` - Code refactoring
- `test` - Adding tests
- `chore` - Maintenance

**Examples:**
```
feat(parser): add PDF text extraction using PyMuPDF
fix(parser): handle UTF-8 encoding in resumes
docs(readme): add setup instructions for uv
refactor(api): simplify error handling
```

---

## 📝 Development Workflow

### 1. Start New Work

```bash
# Make sure you're on main and up to date
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Work on your code...

# Check code style
ruff check .
black --check .

# Fix formatting
black .
ruff check --fix .

# Run tests
pytest
```

### 3. Commit and Push

```bash
git add .
git commit -m "feat(scope): your message"
git push -u origin feature/your-feature-name
```

### 4. Create Pull Request

1. Go to GitHub repo
2. Click "Compare & pull request"
3. Fill in PR description
4. Request review from teammate
5. Wait for approval, then merge

---

## 📁 Where to Work

| Task | Directory | Key Files |
|------|-----------|-----------|
| PDF Parsing | `backend/app/core/` | `resume_parser.py` |
| API Routes | `backend/app/api/routes/` | `resume.py`, `interview.py` |
| LLM Service | `backend/app/services/` | `llm_service.py` |
| LoRA Training | `ml/training/` | `train.py`, `dataset.py` |
| ML Evaluation | `ml/evaluation/` | `evaluate.py`, `benchmark.py` |
| Frontend | `frontend/src/` | Components, pages |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest backend/tests/test_parser.py

# Run with coverage
pytest --cov=backend/app

# Run tests matching pattern
pytest -k "test_pdf"
```

---

## 🎨 Code Style

### Python

We use **Black** for formatting and **Ruff** for linting.

```bash
# Format code
black .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

Configuration is in `pyproject.toml`.

### TypeScript

```bash
cd frontend

# Format
npm run format

# Lint
npm run lint
```

---

## 📦 Adding Dependencies

### Python (using uv)

```bash
# Add runtime dependency
uv pip install package-name

# Then add to pyproject.toml under [project.dependencies]
```

### Node.js

```bash
cd frontend
npm install package-name
```

---

## ❓ Questions?

- Open a GitHub issue
- Tag relevant team members
- Provide context and code examples
