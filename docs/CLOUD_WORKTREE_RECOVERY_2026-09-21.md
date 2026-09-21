# Cloud Worktree Recovery (2026-09-21)

The previous cloud checkout was frozen as a legacy worktree after all active
RNN benchmark jobs completed. It was at commit `1fafb7f`, while GitHub main
contained the subsequent project history.

## Recovery record

- Legacy source snapshot and patches:
  `/mnt/Luna.jl-master/cloud_git_worktree_backups/20260921_104547/`
- Legacy checkout retained as:
  `/mnt/AI-AR-HCF-UPPE-Luna-Modeling-legacy-20260921`
- Canonical clean checkout:
  `/mnt/AI-AR-HCF-UPPE-Luna-Modeling`

The comparison found no cloud-only source files. Thirty-eight formerly
untracked source files were byte-identical to the clean clone; tracked source
scripts were also already represented in GitHub history. The old differences
were only stale project documentation and an older `.gitignore`.

## Environment notes

- `scripts/cloud_luna_env.sh` now derives the checkout root from its own path.
- Source it before Python or Julia work so CoreX Python packages and GPU
  libraries are available.
- The clean Julia project was instantiated with the shared depot. PyCall is
  bound to `/usr/local/bin/python3`, matching the Python 3.10 CoreX packages
  exposed by `cloud_luna_env.sh`; do not bind it to the legacy Python 3.7 Keras
  environment.
- The previous one-thread FFTW wisdom file was archived with the recovery
  backup and rebuilt after an import failure.

## Operating rule

Before a new cloud task, run `git status --short --branch` in the canonical
checkout. Implement source changes locally, commit and push them, then pull
them into this clean checkout. Cloud-only changes are limited to environments,
data, models, logs, and visualization artifacts.
