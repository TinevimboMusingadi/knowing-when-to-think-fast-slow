---
name: my-commit-style
description: Review and commit changes by coherent purpose, keeping related implementation, tests, configuration, and documentation together. Use when committing code, saving project changes, or wrapping up a coding session.
compatibility: Requires shell access to run git commands (status, diff, add, commit, log).
---

# Production-Ready Atomic Commit Style

Prefer **one commit per coherent purpose**. A complete feature may be one commit across several files. Split unrelated purposes, not file types. Do not target a particular number of commits.

The golden rule of production git history:
> **Every single commit in git history must compile cleanly and pass tests.**

Keep every intermediate commit usable and reviewable. Run relevant checks before each commit; report unavailable checks honestly. Preserve unrelated user changes and never rewrite shared history without authorization.

---

## Workflow

### 1. Get the Real List of Changed Files
Never commit from memory. Always inspect the porcelain status first:

```bash
git status --porcelain
```

Understand the status codes:
- `M` — modified
- `A` — added / staged
- `??` — untracked / new file
- `D` — deleted
- `R` — renamed

---

### 2. Group into Atomic Logical Units

Before staging, categorize your changed files into **atomic units of work**:

#### Rule A: Group by purpose (default)
Keep files together when they explain or implement one change, even if each file
could technically stand alone. Include related documentation and configuration
with the implementation. Standalone documents or cleanup deserve separate commits
only when they represent a separate purpose. Never default to one commit per file.

#### Rule B: Tightly Coupled Files Commit Together (Atomic Co-Commit)
If separating files would cause the build to fail or tests to break on an intermediate commit, commit them **together in one atomic commit**:
- **Implementation + Unit Test**: e.g., `desktop-bridge.ts` and `desktop-bridge.test.ts`
- **Interface/Signature Change + Callers**: e.g., renaming a function and updating its call sites
- **Manifest + Lockfile**: e.g., `package.json` + `package-lock.json` or `pyproject.toml` + `uv.lock`
- **Component + Dedicated Stylesheet**: e.g., `dialog.tsx` + `dialog.module.css`

---

### 3. Review Diffs & Commit Unit by Unit

For each atomic unit in dependency order (foundational dependencies first):

1. **Review the diff**:
   ```bash
   git diff -- <file1> <file2>
   ```
2. **Stage only the files in that atomic unit**:
   ```bash
   git add <file1> <file2>
   ```
3. **Run relevant checks and review the staged diff**, then commit with a clear Conventional Commit message:
   ```bash
   git commit -m "<type>(<scope>): <what changed and why>"
   ```
4. Move to the next unit and repeat until `git status --porcelain` is clean.

---

### 4. Commit Message Format

Use the Conventional Commits specification:

```text
<type>(<scope>): <what changed, and why if not self-evident>
```

- **Types**:
  - `feat`: A new feature or capability
  - `fix`: A bug fix
  - `refactor`: Code change that neither fixes a bug nor adds a feature
  - `docs`: Documentation only changes
  - `test`: Adding or correcting tests
  - `chore`: Build tasks, package managers, configs, dotfiles
  - `perf`: Code change that improves performance

- **Scope**: The file or module name being touched (e.g., `bridge`, `tpu-vm`, `sft-trainer`, `config`).

#### Examples:

- **Single Independent File**:
  ```bash
  feat(scripts): add Cloud TPU v6e-8 VM provisioning script
  chore(gitignore): ignore environment secrets and training checkpoints
  docs(readme): document financial SFT dataset and training runner
  ```

- **Coupled Implementation + Test (Atomic Co-Commit)**:
  ```bash
  feat(desktop-bridge): add event listener with unit tests
  test(auth): add integration tests for session refresh
  ```

- **Refactor Across Call Sites**:
  ```bash
  refactor(storage): rename loadCheckpoint to fetchCheckpoint (+ callers)
  ```

---

### 5. Verify the Payoff

Once all units are committed, show the user the readable, bisectable result:

```bash
git log --oneline -n <number of commits just made>
```

Reviewers get a pristine git log where each commit represents an understandable, green-building, bisectable step.
