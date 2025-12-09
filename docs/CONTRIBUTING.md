# Contributing Guide

## Git Workflow

We use **feature branches** with PRs to `main`.

### Branch Naming

```
feature/description    # New features
fix/description        # Bug fixes
refactor/description   # Code improvements
```

Examples:
- `feature/pdf-parser`
- `feature/lora-training`
- `fix/resume-encoding`

### Commit Messages

Format: `type(scope): description`

```
feat(parser): add PDF text extraction
fix(parser): handle unicode characters
docs(readme): update setup instructions
```

## Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature

# 2. Make changes and commit
git add .
git commit -m "feat(scope): description"

# 3. Push and create PR
git push -u origin feature/your-feature
```

## Code Style

- Python: Use `black` and `ruff`
- TypeScript: Use `prettier` and `eslint`

```bash
# Format Python code
cd backend
black .
ruff check .
```
