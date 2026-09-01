# Project instructions for Codex

## Project boundary

The outer project directory is:

`Z:\H3I空芯光纤小组组会\Luna.jl-master`

The actual Julia package root is nested at:

`Z:\H3I空芯光纤小组组会\Luna.jl-master\Luna.jl-master`

Run Julia package commands from the nested package root when they need `Project.toml`. Run project-management and Git commands from the outer root unless the user asks otherwise.

## Main project goal

This project studies AI surrogate modeling for UPPE/Luna-simulated ultrafast pulse propagation in gas-filled antiresonant hollow-core fibers. The main modeling task is forward prediction from pulse, gas, and fiber parameters to:

- final output spectra;
- full spectral-evolution maps `S(z, lambda)`;
- later, inverse-design candidates verified by Luna/UPPE.

## Engineering rules

- Do not delete or revert user-generated experimental results unless explicitly requested.
- Do not silently change dataset schemas, normalization strategy, wavelength-grid size, split logic, or checkpoint compatibility.
- Read the relevant scripts and existing result files before making non-trivial changes.
- Keep training and inference preprocessing consistent.
- Prefer small, testable changes over broad refactors.
- Report changed files, verification commands, verification results, and remaining risks at the end of each implementation task.
- Large data, model checkpoints, generated figures, Office documents, and PDFs are not tracked in ordinary Git by default.

## Important code areas

- Luna package source: `Luna.jl-master/src/`
- AR-HCF simulation and visualization: `Luna.jl-master/examples/simple_interface/*.jl`
- ML preprocessing and training: `Luna.jl-master/examples/simple_interface/*.py`
- RNN baseline adaptation: `rnnnonlinear-master/rnnnonlinear-master/*.py`
- Project state and experiment notes: `docs/`

## Current known modeling context

- Current paper/PPT narrative focuses on a Transformer forward surrogate for gas-filled AR-HCF propagation.
- Transformer/CNN models directly predict propagation maps rather than rolling forward one z slice at a time.
- Luna RNN work is a baseline and diagnostic line: high stepwise accuracy does not guarantee stable autoregressive rollout.
- Early-dense data should be used to revisit early-z dynamics.
- Absolute-intensity modeling should use global-log style targets and identity output activation, not old per-sample min-max assumptions.

## Preferred task workflow

For non-trivial work:

1. Inspect relevant files and current outputs.
2. Identify root cause or concrete implementation target.
3. Make the smallest compatible change.
4. Run a smoke test or static check.
5. Summarize outcome and residual risk.

For review tasks, lead with bugs, risks, regressions, and missing tests.
