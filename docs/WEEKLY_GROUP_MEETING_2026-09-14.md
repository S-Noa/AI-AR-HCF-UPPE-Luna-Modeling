# Weekly Group Meeting: RNN Baseline Attribution and Next Steps

## Purpose

This meeting should answer one focused question:

> Why does the RNN achieve high teacher-forced one-step accuracy on the Luna AR-HCF data but weak autoregressive rollout, whereas the original RNN works well on its published supercontinuum task?

The conclusion to defend is deliberately narrow: the RNN implementation is valid on the original task, but the current Luna recurrent target representation is substantially more difficult for local autoregressive rollout. A short-horizon RNN baseline is usable; the direct CNN/Transformer models remain the main full-map route.

## Recommended Slide Sequence (7 Slides, 10--12 Minutes)

### Slide 1. Question and This Week's Goal (1 min)

**Title:** `Can the RNN Autoregressively Model Luna AR-HCF Propagation?`

**Put on the slide:**

- Original RNN paper: excellent one-step and autoregressive spectral-evolution prediction.
- Earlier Luna result: `stepwise R2 about 0.994`, but full autoregressive `R2 about -0.410`.
- This week's work: controlled attribution across code, preprocessing, horizon, and data dynamics.

**Say:**

"The issue is not whether an RNN can fit the next spectrum. It clearly can. The issue is whether a local spectral window defines a stable propagation state once the model must feed back its own prediction."

### Slide 2. Controlled Attribution Matrix (1.5 min)

**Title:** `Separate Code Effects from Data Effects`

Use a compact table:

| Training/evaluation combination | Stepwise R2 | Autoregressive R2 | Interpretation |
|---|---:|---:|---|
| Original Keras RNN + original SC data | 0.9990 | 0.8559 | Published workflow is reproducible. |
| PyTorch RNN + original SC data | 0.9989 | 0.9633 | PyTorch migration is sound on the original task. |
| Original Keras RNN + Luna, 10 cm/51 z/251 lambda | 0.9743 | 0.3161 | Original implementation is also weak on Luna. |
| Original Keras RNN + Luna raw power + original dBm | 0.9794 | 0.3071 | Matching original preprocessing alone does not solve it. |

**Source:** `rnn_visual_diagnostics/rollout_experiments/summary_metrics.csv`.

**Say:**

"This is the key attribution result. Both implementations work on original SC data, while the unmodified Keras workflow also degrades on reduced Luna data. Therefore code migration is not the dominant explanation."

### Slide 3. Short-Horizon Luna Baseline: What Works (1.5 min)

**Title:** `A Usable RNN Baseline Exists Only at Short Horizon`

Use a simple line or table:

| First 10 cm task | Recurrent prediction steps | Autoregressive R2 | Final-slice R2 |
|---|---:|---:|---:|
| 51 z points, 1000 lambda | 41 | 0.4873 | 0.3301 |
| 101 z points, 1000 lambda | 91 | 0.4339 | 0.2839 |
| 201 z points, 1000 lambda | 191 | 0.3716 | 0.2430 |

Footnote: same target (`per_sample_minmax`), `features_z`, direct sigmoid prediction, low scheduled-sampling feedback, no-detach feedback, and gradient clipping.

**Say:**

"When all training choices are held fixed, performance declines as the number of recurrent feedback steps increases. This isolates rollout horizon as a material limitation even within the first 10 centimeters. We should present the 51-point configuration as the current RNN baseline, not as a replacement for direct full-map prediction."

### Slide 4. Matched dBm Data: Target Sparsity and Early Restructuring (2 min)

**Title:** `Why the Same dBm Mapping Is Less RNN-Friendly for Luna`

**Main figure:**
`rnn_visual_diagnostics/data_dynamics_comparison/data_dynamics_ppt_overview.png`

Point to these panels:

- Target value distribution: Luna has `61.45%` zero-clipped values; original SC has `26.41%`.
- Mean adjacent-z change: Luna is largest at normalized propagation position `0`; original SC peaks near `0.207`.
- Rollout R2: original task remains stable; Luna degrades over the recurrent horizon.

**Say:**

"This does not claim that dBm scaling is physically wrong. It shows that the original global maximum and -55 dB floor create a much sparser state space for the Luna spectra. At the same time, the strongest average restructuring occurs immediately at the input, precisely where autoregressive rollout starts."

**Important caveat on the slide:**
`Propagation is normalized for comparison; the two physical systems do not share the same length scale.`

### Slide 5. Representative Target Maps and Local Predictability (1.5 min)

**Title:** `Luna Trajectories Are More Diverse Under the Same Recurrent Representation`

**Main figure:**
`rnn_visual_diagnostics/data_dynamics_comparison/representative_targets_original_vs_luna.png`

Optional supporting figure:
`rnn_visual_diagnostics/data_dynamics_comparison/local_window_ambiguity_original_vs_luna.png`

**Put on the slide:**

- In every heatmap, x is the spectral-bin index and y is the propagation-step index.
- The four rows are P10/P50/P90/P99 samples ranked by early-z change; they are examples, not matched physical cases.
- Shared-PCA local-window diagnostic:
  - median nearest-history distance: `1.764` (Luna) versus `0.498` (original);
  - median next-step difference: `0.00893` (Luna) versus `0.00339` (original).

**Say:**

"After standardizing both datasets and embedding histories into a shared PCA space, Luna windows are more dispersed and their nearest local histories lead to less similar next spectra. This is a descriptive predictability diagnostic, not proof of a unique physical mechanism."

### Slide 6. What We Learned and Model Positioning (1.5 min)

**Title:** `Model Roles After This Week's Experiments`

Use a three-row diagram:

| Model | Appropriate role | Current limitation |
|---|---|---|
| RNN | Local/short-horizon autoregressive baseline; study error accumulation | Long rollout degrades, especially for early-dense full maps. |
| Temporal CNN | Strong direct propagation-map engineering baseline | Requires matched early-dense retraining for fair comparison. |
| Transformer | Main paper architecture; direct full-map prediction with global z relations | Narrow UV features remain smoothed. |

**Say:**

"The RNN result is still useful. It supplies a controlled negative result: high one-step R2 is not evidence of stable long-range propagation. For the full `S(z, lambda)` task, direct models avoid the feedback loop by construction."

### Slide 7. Decisions Requested and Next Week (1 min)

**Title:** `Recommended Next Decisions`

**Recommended decisions:**

1. Freeze `z <= 10 cm, 51 z points, 1000 lambda` as the current reportable RNN autoregressive baseline.
2. Do not spend the next iteration on more dB-floor sweeps; matching the original pipeline did not resolve the core problem.
3. Start matched early-dense training/evaluation for temporal CNN and Transformer using temporal metrics.
4. Keep the RNN only for local-rollout diagnostics; reserve inverse design for a verified direct forward surrogate.

**Suggested next experiments:**

- CNN versus Transformer on the same early-dense split and target representation.
- A planned, limited RNN ablation only if needed: state augmentation with input spectrum plus physical features, or a multi-step latent-state RNN. Do not claim this is guaranteed to solve full rollout.
- Use the direct surrogate for constrained parameter optimization, then validate candidates with Luna.

## Figure Inventory

| Slide | File | Use |
|---|---|---|
| 4 | `rnn_visual_diagnostics/data_dynamics_comparison/data_dynamics_ppt_overview.png` | Main six-panel evidence figure. |
| 5 | `rnn_visual_diagnostics/data_dynamics_comparison/representative_targets_original_vs_luna.png` | Ground-truth trajectory comparison. |
| 5 | `rnn_visual_diagnostics/data_dynamics_comparison/local_window_ambiguity_original_vs_luna.png` | Optional quantitative support. |
| Backup | `rnn_visual_diagnostics/rawpower_dbm_failure/per_step_autoreg_r2_original_vs_luna.png` | Direct per-step R2 evidence. |
| Backup | `rnn_visual_diagnostics/rawpower_dbm_failure/sample_r2_distribution_original_vs_luna.png` | Show heterogeneous Luna sample failures. |
| Backup | `rnn_visual_diagnostics/rawpower_dbm_failure/target_value_hist_original_vs_luna.png` | Show dBm-induced sparsity only. |

## Likely Questions and Concise Answers

**Why not simply use the original preprocessing?**  We did: Luna raw linear power was passed through the unmodified original dBm path. The original Keras RNN still achieved only `0.3071` autoregressive R2, so preprocessing mismatch is not the sole cause.

**Does the result prove that RNNs cannot model AR-HCF propagation?**  No. It shows that this local-window autoregressive formulation is weak for the current early-dense Luna representation at long horizons. It remains usable as a short-horizon baseline.

**Why is the PyTorch result on original SC even higher than the original Keras result?**  The implementations and training selections are not perfectly identical, so this is a validation of qualitative competence, not a benchmark claim of superiority.

**Are the two data systems physically comparable after normalizing z?**  Not in absolute physical length. Normalized z is used only to compare recurrent dynamics and where change occurs along each trajectory.

**Why not optimize RNN indefinitely?**  The research objective is accurate full-map forward prediction. CNN/Transformer directly predict the entire map and avoid recursive feedback accumulation; they are the higher-value path.

## One-Sentence Closing

"This week's controlled cross-checks show that the RNN rollout bottleneck is mainly associated with the Luna AR-HCF recurrent target dynamics rather than a simple code-port or dBm-preprocessing error; we will retain it as a short-horizon diagnostic baseline and prioritize direct full-map surrogates."
