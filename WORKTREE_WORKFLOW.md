# Git Worktree Workflow Guide

**For**: Resume Griller Project
**Last Updated**: 2026-01-17
**Purpose**: Help developers understand Claude Code's worktree workflow

---

## 🌳 What is a Git Worktree?

A **worktree** allows you to check out multiple branches of the same repository into different directories **simultaneously**.

### Traditional Git Workflow (Single Working Directory)
```
/Users/willkczy/Projects/resume-griller/
├── .git/                    # Git database
└── [files for current branch]

To switch branches:
git checkout other-branch    # Changes files in place
```

### Worktree Workflow (Multiple Working Directories)
```
Main Repository:
/Users/willkczy/Projects/resume-griller/
├── .git/                    # Git database (shared)
└── [files for feature/docker-deployment]

Worktree #1:
~/.claude-worktrees/resume-griller/heuristic-swirles/
└── [files for main branch]

Worktree #2:
~/.claude-worktrees/resume-griller/mystifying-robinson/
└── [files for mystifying-robinson branch]

All share the same .git database!
```

---

## 🤖 Why Claude Code Uses Worktrees

### Benefits

1. **Non-Intrusive**: Claude Code works in a separate directory, doesn't touch your main workspace
2. **Safe Experimentation**: Changes in worktree won't affect your current work
3. **Parallel Work**: You can keep coding while Claude works in the background
4. **Clean Separation**: Easy to review Claude's changes before merging

### How It Works

```
User starts Claude Code session
         ↓
Claude creates/reuses worktree
         ↓
Claude makes changes in worktree
         ↓
Claude commits to a branch (e.g., mystifying-robinson)
         ↓
User reviews changes in main repo
         ↓
User merges branch when ready
```

---

## 📊 Current Worktree Structure

As of 2026-01-17:

```
Main Repository
├─ Location: /Users/willkczy/Projects/resume-griller/
├─ Current Branch: feature/docker-deployment
└─ Contains: Your daily work

Worktree: heuristic-swirles
├─ Location: ~/.claude-worktrees/resume-griller/heuristic-swirles/
├─ Branch: main
└─ Purpose: Previous Claude Code session (2026-01-16)

Worktree: mystifying-robinson
├─ Location: ~/.claude-worktrees/resume-griller/mystifying-robinson/
├─ Branch: mystifying-robinson
├─ Purpose: Current session (2026-01-17)
└─ Status: ✅ Merged to feature/docker-deployment
```

---

## 🔄 Common Workflows

### 1. View All Worktrees

```bash
git worktree list
```

Output:
```
/Users/willkczy/Projects/resume-griller                               c2fec28 [feature/docker-deployment]
/Users/willkczy/.claude-worktrees/resume-griller/heuristic-swirles    27e4146 [main]
/Users/willkczy/.claude-worktrees/resume-griller/mystifying-robinson  a3a01bd [mystifying-robinson]
```

---

### 2. View Claude's Changes

```bash
# In your main repository
cd /Users/willkczy/Projects/resume-griller

# View commit history
git log mystifying-robinson -5 --oneline

# View differences
git diff feature/docker-deployment..mystifying-robinson

# View changed files
git diff feature/docker-deployment..mystifying-robinson --stat
```

---

### 3. Merge Claude's Work

#### Option A: Fast-forward merge (if no conflicts)
```bash
cd /Users/willkczy/Projects/resume-griller
git checkout feature/docker-deployment
git merge mystifying-robinson
git push origin feature/docker-deployment
```

#### Option B: Merge with conflicts
```bash
git merge mystifying-robinson
# Git shows conflicts
# Edit conflicted files
git add <resolved-files>
git commit
git push origin feature/docker-deployment
```

#### Option C: Create Pull Request
```bash
git push origin mystifying-robinson
# Then create PR on GitHub: mystifying-robinson → main
```

---

### 4. Clean Up After Merging

```bash
# Remove the worktree
git worktree remove mystifying-robinson

# Or force remove if needed
git worktree remove --force mystifying-robinson

# Delete the branch (if no longer needed)
git branch -d mystifying-robinson

# Delete remote branch (if pushed)
git push origin --delete mystifying-robinson
```

---

### 5. Navigate to Worktree

```bash
# Go to Claude's working directory
cd ~/.claude-worktrees/resume-griller/mystifying-robinson

# View status
git status

# View files
ls -la

# View commits
git log --oneline -10
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: "Why can't I see Claude's changes in my directory?"

**Reason**: Claude worked in a separate worktree directory.

**Solution**:
```bash
# View Claude's branch from your main repo
git log mystifying-robinson --oneline

# Or merge to see the changes
git merge mystifying-robinson
```

---

### Issue 2: "Merge conflict in .claude/settings.local.json"

**Reason**: Both you and Claude modified local configuration.

**Solution**:
```bash
# Option A: Merge both permissions
cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": [
      "Bash(tree:*)",
      "Bash(grep:*)"
    ]
  }
}
EOF
git add .claude/settings.local.json

# Option B: Add to .gitignore to prevent future conflicts
echo ".claude/settings.local.json" >> .gitignore
git add .gitignore

# Complete merge
git commit
```

---

### Issue 3: "error: Your local changes would be overwritten by merge"

**Reason**: You have uncommitted changes in your working directory.

**Solution**:
```bash
# Option A: Commit your changes first
git add .
git commit -m "Save my changes before merge"
git merge mystifying-robinson

# Option B: Stash your changes
git stash
git merge mystifying-robinson
git stash pop
```

---

### Issue 4: "How do I know which branch Claude worked on?"

**Solution**:
```bash
# List all branches
git branch -a

# See recent branches with activity
git for-each-ref --sort=-committerdate refs/heads/ --format='%(refname:short) - %(committerdate:relative)'

# Check worktree list
git worktree list
```

---

## 📋 Worktree Best Practices

### Do's ✅

1. **Review before merging**: Always check `git diff` before merging
2. **Clean up old worktrees**: Remove worktrees after merging
3. **Use descriptive branch names**: Claude usually generates names like `mystifying-robinson`
4. **Keep .gitignore updated**: Add `.claude/` to prevent config conflicts
5. **Communicate**: Ask Claude which branch it's working on

### Don'ts ❌

1. **Don't delete worktree manually**: Always use `git worktree remove`
2. **Don't modify worktree from outside**: Let Claude manage its worktree
3. **Don't commit .claude/ configs**: These are local-only files
4. **Don't panic on conflicts**: They're usually easy to resolve

---

## 🎓 Understanding Git After Worktree Sessions

### Where are my commits?

All commits are in the **shared .git database** in your main repository.

```bash
# From anywhere in your system
cd /Users/willkczy/Projects/resume-griller

# All commits are visible
git log --all --oneline --graph

# Even commits from worktrees
git log mystifying-robinson --oneline
```

### Why does `git status` show clean?

```bash
# In main repo
git status  # Shows status of feature/docker-deployment

# In worktree
cd ~/.claude-worktrees/resume-griller/mystifying-robinson
git status  # Shows status of mystifying-robinson
```

They're **independent working directories**, so they show different statuses.

---

## 🔍 Debugging Worktree Issues

### Check worktree integrity
```bash
git worktree list

# If worktrees are broken
git worktree prune  # Remove stale references
```

### Find orphaned branches
```bash
# List all branches
git branch -a

# Find branches without worktrees
git branch --no-merged main
```

### Repair a broken worktree
```bash
# If worktree directory deleted manually
git worktree prune

# Re-create if needed
git worktree add ~/.claude-worktrees/resume-griller/new-branch -b new-branch
```

---

## 📝 Example Session Workflow (2026-01-17)

This is what happened in today's session:

```bash
# 1. User started Claude Code session
# 2. Claude created worktree at:
#    ~/.claude-worktrees/resume-griller/mystifying-robinson

# 3. Claude made changes in worktree:
#    - Deleted scripts/backend/
#    - Deleted backend/app/core/resume_parser.py

# 4. Claude committed changes:
git commit -m "chore: complete Quick Wins cleanup - remove remaining duplicates"
# Commit: a3a01bd

# 5. User reviewed changes in main repo:
cd /Users/willkczy/Projects/resume-griller
git log mystifying-robinson -5 --oneline
git diff feature/docker-deployment..mystifying-robinson

# 6. User encountered merge conflict:
git merge mystifying-robinson
# Conflict: .claude/settings.local.json

# 7. User resolved conflict:
# Merged both permissions (tree + grep)
cat > .claude/settings.local.json << 'EOF'
{
  "permissions": {
    "allow": ["Bash(tree:*)", "Bash(grep:*)"]
  }
}
EOF

# 8. User added to .gitignore:
echo ".claude/settings.local.json" >> .gitignore

# 9. User completed merge:
git add .gitignore .claude/settings.local.json
git commit -m "chore: update .gitignore and merge Claude permissions"
git merge mystifying-robinson
git commit --no-edit  # Merge commit

# 10. User pushed to remote:
git push origin feature/docker-deployment

# 11. (Optional) Clean up:
git worktree remove mystifying-robinson
git branch -d mystifying-robinson
```

---

## 🚀 Quick Reference Card

```bash
# View all worktrees
git worktree list

# View Claude's changes
git diff your-branch..claude-branch
git log claude-branch --oneline

# Merge Claude's work
git merge claude-branch

# Clean up
git worktree remove claude-branch
git branch -d claude-branch

# Add to .gitignore
echo ".claude/" >> .gitignore
```

---

## 🤝 Working with Future Claude Sessions

### Before Starting a New Session

```bash
# Check for existing worktrees
git worktree list

# Clean up old ones if needed
git worktree remove old-worktree-name
```

### During the Session

Ask Claude:
- "Which branch are you working on?"
- "Where is your worktree located?"

### After the Session

```bash
# Review changes
git diff main..claude-branch

# Merge when ready
git merge claude-branch

# Clean up
git worktree remove claude-branch
```

---

## 📞 Getting Help

If you encounter issues:

1. **Check worktree list**: `git worktree list`
2. **Check branch status**: `git branch -a`
3. **View diff**: `git diff your-branch..claude-branch`
4. **Ask Claude**: "What's the status of your worktree?"
5. **Git docs**: `git worktree --help`

---

**Created**: 2026-01-17
**Session**: mystifying-robinson worktree
**Purpose**: Document worktree workflow for future reference

---

## 🔗 Related Documentation

- `PROJECT_STATUS.md` - Current project state
- `CLAUDE.md` - AI assistant instructions
- `QUICK_WINS_COMPLETED.md` - Phase 1 completion log
- `.gitignore` - Ignored files (including .claude/)
