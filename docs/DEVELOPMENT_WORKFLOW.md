# Development workflow

## New Codex session checklist

Ask Codex to read:

1. `AGENTS.md`
2. `docs/PROJECT_STATE.md`
3. `docs/EXPERIMENTS.md`
4. `docs/COMMANDS.md`

Then ask it to inspect the relevant source files and current Git diff before editing.

## Recommended task template

```text
Goal:
[one concrete task]

Current behavior:
[what happens now]

Expected behavior:
[what should happen]

Constraints:
- preserve dataset schema
- preserve checkpoint compatibility unless explicitly changing it
- do not touch generated experiment outputs

Before editing:
1. inspect relevant implementation
2. identify the likely root cause
3. propose the smallest change

After editing:
1. run a smoke test or static check
2. summarize changed files
3. report verification results and remaining risks
```

## Branch and commit rhythm

- Use `main` for stable project state.
- Use short feature branches for larger work, for example `feat/rnn-conditioning` or `exp/early-dense-transformer`.
- Keep one task close to one commit when possible.
- Use commit messages such as:
  - `docs: add project state notes`
  - `feat: add temporal r2 checkpoint selection`
  - `fix: preserve visualized sample indices`

## Review habit

After a meaningful change, ask for a review-style pass:

```text
Review the current git diff as if you did not write it.
Look for correctness issues, tensor-shape bugs, data leakage, checkpoint incompatibility, and unnecessary complexity.
Do not modify files yet.
```
