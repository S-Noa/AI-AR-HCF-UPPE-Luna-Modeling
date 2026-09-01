# Physics-Guided Inverse Design of Ultrafast Nonlinear Pulse Propagation in Gas-Filled Hollow-Core Fibers

## Full Project Design Document

**Date:** 2026-03-25
**Status:** Design phase

---

## 1. Problem Statement

### 1.1 The Physical System

Gas-filled hollow-core fibers (HCF) are a versatile platform for generating tailored ultrafast light sources. The system consists of:

- **Fiber:** AR-HCF with core diameter ~10-50 μm
- **Gas:** Argon at variable pressure (0-50 bar)
- **Pump laser:** Yb-based, 1030/800 nm center wavelength, ultrafast pulses (5-50 fs)
- **Fiber length:** 0.03-0.5 m

When an intense ultrafast pulse propagates through the gas-filled fiber, it undergoes complex nonlinear dynamics including self-phase modulation (SPM), soliton formation and fission, resonant dispersive wave (RDW) emission, and ionization of the gas. These processes are governed by the interplay of dispersion and nonlinearity, and are accurately modeled by the Unidirectional Pulse Propagation Equation (UPPE).

### 1.2 Why UPPE, Not GNLSE

The Generalized Nonlinear Schrödinger Equation (GNLSE) is the standard simulation tool for fiber nonlinear optics. However, GNLSE cannot capture:

- **Ionization dynamics:** Argon ionization creates free electrons that modify the refractive index and absorb energy. This is significant at high peak powers.
- **Plasma effects:** Free-electron plasma causes blue-shifting and energy loss.
- **Full-bandwidth propagation:** UPPE handles the entire spectral range without the slowly-varying envelope approximation.

The current data (1 m fiber, 200 μm core, 3 bar argon, 1030 nm pump) shows ~10% energy loss (output/input ratio = 0.90), confirming that ionization is active and UPPE is necessary.

### 1.3 The Inverse Design Problem

**The practical question:** "I want light at wavelength X with spectral property Y. What gas pressure, fiber length, and pulse parameters should I use?"

**Why this is hard:**

1. **Forward simulation is expensive.** A single UPPE run takes minutes to hours. Scanning the parameter space requires thousands of runs.
2. **No closed-form inverse solution.** The mapping from parameters to output spectrum is highly nonlinear, involving soliton dynamics, dispersive wave emission, and ionization thresholds.
3. **Non-intuitive parameter interactions.** Changing gas pressure shifts the zero-dispersion wavelength, which alters soliton dynamics, which changes RDW emission wavelength, which redistributes spectral energy. Small parameter changes can produce qualitatively different outputs.
4. **Regime boundaries.** The system transitions abruptly between propagation regimes (linear, soliton compression, soliton fission, ionization-dominated) as parameters change.

### 1.4 What We Build

A physics-guided machine learning system consisting of:

1. **A forward surrogate model** that predicts the output spectrum from input parameters in milliseconds (replacing minutes-long UPPE simulations)
2. **An inverse design tool** that, given a desired output spectrum, predicts the optimal fiber and pulse parameters
3. **Physics embedding** at the input and loss level to ensure physical consistency

---

## 2. Data Generation

### 2.1 Parameter Space

Phase A uses analytical pulse shapes (sech² or Gaussian), so scalar parameters fully define the input electric field including spectral phase.

| Parameter | Symbol | Range | Sampling | Justification |
|-----------|--------|-------|----------|---------------|
| Pulse energy | E | 0.3 - 3 μJ | Log-scale | Spans sub-soliton to ionization regime |
| Pulse duration | τ | 5 - 50 fs | Log-scale | Short = high peak power, long = low |
| Chirp parameter | C | -3 to +3 | Linear | Pre-chirp affects compression dynamics |
| Center wavelength | λ₀ | 1030 nm or 800nm | Bool | Yb laser |
| Gas pressure | P | 0 - 50 bar | Linear | Practical experimental range |
| Fiber length | L | 0.03 - 0.5 m | Linear | Short to long propagation |
| Core diameter | d | 10 - 50 μm | Linear | Range of available capillaries |

### 2.2 Why Scalar Parameters Are Correct for Phase A

For an analytical pulse shape such as sech²:

```
E(t) = √(P_peak) · sech(t/τ) · exp(i·C·t²/(2τ²)) · exp(iω₀t)
```

The parameters [E, τ, C, λ₀] **uniquely and completely** define the complex electric field E(t), including the spectral phase φ(ω). This means:

- Scalar parameters contain ALL the information the UPPE needs about the input pulse
- Using the power spectrum |Ê_in(ω)|² alone would be INCORRECT because it drops the spectral phase (chirp information)
- Scalar parameters are the more physically complete representation for analytical pulses

This changes in Phase C (real experimental pulses), where the full measured spectral field [|Ê_in(ω)|², φ_in(ω)] must be used as input. Phase C requires a different network architecture (CNN-based encoder for spectral inputs) and is a separate extension.

### 2.3 Sampling Strategy

- **Latin Hypercube Sampling (LHS)** across the 6 variable parameters
- LHS provides better coverage than grid or random sampling for the same number of points
- Log-scale sampling for E and τ (these span orders of magnitude)
- Target dataset size: ~10,000 simulations
- Each simulation stores: input parameters, output power spectrum, metadata (convergence, ionization fraction, total output energy)

### 2.4 Spectral Representation

- Raw UPPE output has 8,193 frequency points spanning 100-5936 nm
- Only ~600 points have significant power (above -60 dB of peak)
- Crop to useful range: 200-2500 nm (covers UV/RDW through near-IR)
- Downsample to 500 evenly-spaced wavelength points (resolution ~4.6 nm)
- Convert to log scale: log₁₀(|Ê_out(ω)|² + ε), normalized to [0, 1]
- Log transform is critical because spectral intensities span ~15 orders of magnitude

### 2.5 Data Split

- 80% training / 10% validation / 10% test
- Stratified by soliton order N to ensure all propagation regimes are represented in each split

---

## 3. Forward Surrogate Model

### 3.1 Purpose

Replace expensive UPPE simulations with a fast, differentiable neural network that predicts the output spectrum from input parameters.

### 3.2 Input Features (15 total)

**Raw parameters (7):**

| Feature | Description |
|---------|-------------|
| E | Pulse energy |
| τ | Pulse duration |
| C | Chirp parameter |
| λ₀ | Center wavelength (fixed at 1030 nm) |
| P | Gas pressure |
| L | Fiber length |
| d | Core diameter |

**Pre-computed physics features (8):**

| Feature | Formula | Physical meaning |
|---------|---------|-----------------|
| β₂ | From Marcatili model + argon Sellmeier | Group velocity dispersion at pump |
| ZDW | Where β₂(ω) = 0 | Zero-dispersion wavelength |
| γ | n₂(P)·ω₀/(c·A_eff) | Nonlinear coefficient |
| N | √(L_D/L_NL) where L_D=τ²/|β₂|, L_NL=1/(γ·P_peak) | Soliton order (governs dynamics) |
| L/L_fiss | L/(L_D/N) | Normalized propagation length |
| λ_RDW | From β(ω_RDW) = β(ω_s) + γP_s/2 | Predicted RDW wavelength |
| γ_K | ω₀√(2m_eI_p)/(eE₀) | Keldysh parameter (ionization indicator) |
| P_peak/P_cr | Peak power / critical power for self-focusing | Self-focusing indicator |

All physics features are computed from known analytical formulas at negligible computational cost. The network receives physically meaningful derived quantities rather than having to learn these relationships from data.

### 3.3 Output

500-point log-power spectrum: log₁₀(|Ê_out(ω)|² + ε), normalized to [0, 1], sampled at 500 wavelength points spanning 200-2500 nm.

### 3.4 Architecture

```
Input: 15 features (7 raw + 8 physics)
  │
  ├── Normalize all inputs to [0, 1] via min-max scaling
  │
  Dense(256)  → BatchNorm → LeakyReLU(0.01)
  Dense(512)  → BatchNorm → LeakyReLU(0.01)
  Dense(512)  → BatchNorm → LeakyReLU(0.01) → Dropout(0.1)
  Dense(256)  → BatchNorm → LeakyReLU(0.01)
  Dense(500)  → Output (log-power spectrum)
```

- **BatchNorm** after each dense layer for training stability
- **LeakyReLU** instead of ReLU to avoid dead neurons
- **Dropout(0.1)** on the widest layer for regularization
- Total parameters: ~800K (manageable for 10K training samples)

### 3.5 Loss Function

```
L = L_spectral + α·L_energy + β·L_RDW

L_spectral:  MSE(predicted_log_spectrum, true_log_spectrum)
             Primary loss. Matches the full output spectrum.

L_energy:    ReLU(E_out_predicted - E_in)²
             Penalizes predictions where output energy exceeds input energy.
             E_out < E_in is allowed (ionization absorbs energy).
             E_out is computed by integrating the predicted spectrum.

L_RDW:       MSE(detected_peak_λ, λ_RDW_analytical) · 𝟙(N > 1)
             Soft constraint on RDW peak position.
             Only active when soliton order N > 1 (RDW emission is expected).
             λ_RDW_analytical is the phase-matching prediction from inputs.
```

Hyperparameters α and β are tuned on the validation set. Suggested starting values: α = 0.1, β = 0.01.

### 3.6 Training

- **Optimizer:** Adam, learning rate 1e-3
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=10 epochs)
- **Epochs:** 300-500
- **Early stopping:** patience 20 epochs on validation loss
- **Batch size:** 256
- **Target performance:** R² > 0.95 on test set

### 3.7 Validation Protocol

1. R² score on held-out test set (overall accuracy)
2. Per-regime accuracy (separate metrics for low-N, high-N, ionized regimes)
3. Visual comparison of predicted vs. true spectra for representative cases
4. Sharp feature test: accuracy on RDW peak position and amplitude specifically

---

## 4. Inverse Design

### 4.1 Problem Formulation

Given a desired output spectrum S_target(ω), find the parameters θ = [E, τ, C, P, L, d] such that UPPE(θ) ≈ S_target.

Two practical sub-cases:

**Case 1: Fixed laser, optimize fiber only**
- Known (measured): E, τ, C, λ₀
- Unknown (to predict): P_gas, L_fiber, d_core (3 parameters)

**Case 2: Optimize everything**
- Known: λ₀ only
- Unknown: E, τ, C, P, L, d (6 parameters)

### 4.2 Step 1: Gradient Optimization Through Frozen Surrogate (Baseline)

No inverse network needed. Directly optimize parameters by backpropagating through the frozen forward model.

```python
forward_model.eval()  # freeze weights

params = initialize_randomly(within_physical_bounds)
params.requires_grad_(True)

optimizer = Adam([params], lr=0.01)

for step in range(1000):
    predicted_spectrum = forward_model(params, physics_features(params))
    loss = MSE(predicted_spectrum, target_spectrum)
    loss.backward()       # gradient w.r.t. params (not model weights)
    optimizer.step()
    params.data.clamp_(min_bounds, max_bounds)  # enforce physical bounds
```

- Speed: ~1-5 seconds per target spectrum
- No additional training required
- Serves as the accuracy baseline for all inverse methods
- May find local minima; run multiple random initializations

### 4.3 Step 2: Tandem Inverse Network

Train a neural network that directly maps a desired spectrum to parameters, verified through the frozen forward model.

**Architecture:**

```
Input: desired_spectrum (500 points)
  │
  1D-CNN Encoder:
    Conv1D(1→32, kernel=7, stride=1) → ReLU → MaxPool(2)
    Conv1D(32→64, kernel=5, stride=1) → ReLU → MaxPool(2)
    Conv1D(64→128, kernel=3, stride=1) → ReLU → AdaptiveAvgPool(1)
    Flatten → Dense(128)
  │
  (If Case 1: concatenate with known [E, τ, C, λ₀])
  │
  Dense(256) → ReLU
  Dense(128) → ReLU
  Dense(K) → Sigmoid → scale to physical bounds
  │
  Output: predicted_params (K = 3 for Case 1, 6 for Case 2)
```

**Training (tandem configuration):**

The inverse network is trained with the forward model frozen and connected:

```
target_spectrum → [Inverse Net] → predicted_params
                                        │
                        ┌───────────────┘
                        ↓
                  [Forward Model (frozen)]
                        │
                        ↓
                  reconstructed_spectrum
                        │
                        ↓
            Loss = L_recon + α(t)·L_params

L_recon:    MSE(reconstructed_spectrum, target_spectrum)
L_params:   MSE(predicted_params, true_params)

α(t) schedule:
  Epochs 1-50:    α = 1.0   (parameter supervision dominates)
  Epochs 50-100:  α = 0.1   (shift toward reconstruction)
  Epochs 100+:    α = 0.01  (reconstruction dominates)
```

The decaying α is important:
- **Early:** direct parameter supervision stabilizes training
- **Late:** reconstruction loss allows the network to find ANY valid parameter set, not just the one in the training data. This naturally handles the one-to-many problem.

**Training details:**
- Optimizer: Adam, lr=5e-4
- Epochs: 300-500
- Early stopping on validation reconstruction loss
- Speed at inference: ~1 ms per target spectrum

### 4.4 Upgrade Path: cVAE (Only If Needed)

If the tandem network shows poor accuracy due to severe one-to-many degeneracy (multiple parameter sets producing similar spectra), upgrade to a Conditional Variational Autoencoder:

```
Encoder (training only):
  [target_spectrum, true_params] → μ(32), σ(32) → sample z

Decoder (training + inference):
  [target_spectrum, z] → predicted_params

At inference: sample z multiple times → get multiple valid solutions
```

This should only be implemented if the tandem network demonstrably fails. For 3-6 output parameters, the tandem approach is likely sufficient.

---

## 5. Validation Strategy

### 5.1 Forward Model Validation

| Test | Metric | Target |
|------|--------|--------|
| Overall accuracy | R² on test set | > 0.95 |
| Per-regime accuracy | R² in low-N, high-N, ionized regimes | > 0.90 each |
| RDW peak position | Mean absolute error | < 10 nm |
| RDW peak amplitude | Relative error | < 20% |
| Energy conservation | |E_out/E_in - true_ratio| | < 5% |

### 5.2 Inverse Model Validation (Critical)

The ground truth validation is NOT accuracy against the surrogate. It is accuracy against actual UPPE:

1. Choose 100 test target spectra
2. Inverse model predicts parameters for each
3. Run actual UPPE simulation with predicted parameters
4. Compare UPPE output spectrum with target spectrum
5. Report: spectral similarity (cosine similarity, MSE), RDW wavelength error, peak power error

This is the only validation that matters. If the inverse predictions produce correct spectra when verified through UPPE, the system works.

### 5.3 Comparison Benchmarks

| Method | Speed | Requires |
|--------|-------|----------|
| UPPE parameter scan (brute force) | Hours-days | Cluster computing |
| PSO + UPPE (Hofmann approach) | Hours per target | UPPE in the loop |
| Gradient optimization through surrogate | 1-5 seconds per target | Trained forward model |
| Tandem inverse network | ~1 ms per target | Trained forward + inverse models |

---

## 6. Physics Embedding Summary

Physics enters the model at three levels:

### Level 1: Physics-Informed Inputs

Eight pre-computed features derived from known analytical relationships for hollow-core fiber dispersion and gas nonlinearity (Marcatili model, Sellmeier equation, soliton theory). These give the network a head start by providing physically meaningful derived quantities that it would otherwise need to learn from data.

### Level 2: Physics-Constrained Loss

Energy conservation (output cannot exceed input) and RDW phase-matching (dispersive wave peak position should approximately match the analytically predicted wavelength). These soft constraints improve physical consistency without making training unstable.

### Level 3: Physics-Embedded Architecture (Neural Split-Step) — Alternate Design Direction

Instead of embedding UPPE in the loss (PINN), embed the tractable part of UPPE directly in the network architecture. See Section 8A for full design.

---

## 7. Publication Strategy

### 7.1 Two Design Directions and Their Publication Targets

**Direction A: MLP Surrogate (Sections 3-4)**

- Title: "Physics-guided inverse design of ultrafast nonlinear pulse propagation in gas-filled hollow-core fibers"
- Target: Optics Express, Optics Letters, APL Photonics (Tier 3)

**Direction B: Neural Split-Step (Section 8A) — Recommended for higher impact**

- Title: "Neural split-step propagation: physics-embedded deep learning for inverse design of nonlinear pulse dynamics in gas-filled hollow-core fibers"
- Target: Optica, Photonics Research, Light: Science & Applications (Tier 2)

**Direction B+: Neural Split-Step + Pressure Gradient — Highest impact**

- Title: "Machine-learning-accelerated discovery of optimal pressure profiles for ultrafast light generation in gas-filled hollow-core fibers"
- Target: Optica, Light: Science & Applications, Nature Communications candidate (Tier 1-2)
- Requires: UPPE code modified to accept non-uniform pressure profiles

### 7.2 Novelty Claims

**Core claims (all directions):**

1. **First inverse design tool for gas-filled HCF including ionization physics (UPPE).** Prior ML work in fiber optics uses GNLSE, which cannot capture ionization.
2. **Tandem inverse network** with built-in verification through the forward surrogate, providing instant parameter prediction with confidence checking.
3. **Parameter space mapping** revealing accessible spectral regions and regime boundaries as a scientific contribution beyond the tool itself.

**Additional claims (Direction B: neural split-step):**

4. **Novel physics-embedded architecture** where the UPPE's linear dispersion operator is applied exactly within the network, and only the nonlinear+ionization response is learned from data.
5. **Interpretable intermediate predictions** showing the spatio-spectral pulse evolution at each propagation step.
6. **Principled separation** of known physics (exact) from unknown physics (learned), demonstrating that embedding tractable PDE components in the architecture outperforms both black-box surrogates and full PINN approaches.

**Additional claims (Direction B+: pressure gradient):**

7. **Discovery of non-intuitive optimal pressure profiles** for targeted spectral output, enabled by ML-accelerated exploration of a high-dimensional parameter space.
8. **Direct parallel to temperature-gradient dispersion engineering** in liquid-core fibers (Hofmann et al., Nat. Commun. 2025), extended to gas-filled systems with ionization.

### 7.3 Paper Results Structure

1. Forward model accuracy across the full parameter space (heat maps of R² vs parameter combinations)
2. MLP vs neural split-step comparison (accuracy, interpretability, generalization)
3. Inverse design demonstrations: "target RDW at 350 nm" → model predicts parameters → UPPE verification confirms
4. Speed comparison: UPPE scan vs gradient optimization vs tandem network (orders-of-magnitude speedup)
5. Physical insight: maps showing which output spectra are achievable, regime boundaries
6. (Direction B) Intermediate evolution visualization from neural split-step
7. (Direction B+) Discovered optimal pressure gradient profiles and their physical interpretation

### 7.4 Impact Assessment

| Direction | Novelty | Effort | Target Venue | Confidence |
|-----------|---------|--------|-------------|-----------|
| A: MLP surrogate | Moderate | Low | Optics Express | 85% |
| B: Neural split-step | High | Medium | Optica / Photonics Research | 70% |
| B+: Split-step + pressure gradient | Very high | Medium-High | Optica / Light:S&A / Nat. Commun. | 55% |

---

## 8. Justification for Not Including UPPE in the Model (PINN Analysis)

### 8.1 What a PINN Approach Would Require

A Physics-Informed Neural Network (PINN) embeds the governing PDE directly into the loss function. For our system, this means the UPPE:

```
∂Ê/∂z = i·k_z(ω)·Ê + i·(ω/c)·FT{P_NL(t) + P_plasma(t)}
```

where:
- k_z(ω) = √(k²(ω) - k_⊥²) is the z-component of the wavevector (known analytically)
- P_NL(t) = ε₀·χ³·|E(t)|²·E(t) is the Kerr nonlinear polarization
- P_plasma(t) includes the ionization-induced current and plasma response

The PINN loss would be:

```
L_PINN = L_data + λ · L_physics

L_physics = || ∂Ê_pred/∂z - [i·k_z·Ê_pred + i·(ω/c)·FT{P_NL + P_plasma}] ||²
```

evaluated at collocation points across the (ω, z) domain.

### 8.2 Technical Barriers

**Barrier 1: Domain switching in the nonlinear term**

The UPPE's nonlinear term requires computing |E(t)|²E(t) in the time domain, then Fourier-transforming back to the frequency domain. This means the PINN loss must include FFT/IFFT operations at every collocation point:

```
P_NL(ω) = FT{ ε₀·χ³ · |IFFT{Ê(ω)}|² · IFFT{Ê(ω)} }
```

While FFT is differentiable in PyTorch, computing it at every collocation point during every training step is computationally expensive and numerically sensitive. The cubic nonlinearity |E|²E also produces steep gradients in the loss landscape.

**Barrier 2: Ionization is a threshold function**

The ionization rate W(E) in argon follows the ADK (Ammosov-Delone-Krainov) or PPT (Perelomov-Popov-Terent'ev) model:

```
W(E) ∝ exp(-2(2I_p)^(3/2) / (3|E|))
```

This is an exponentially steep function of the electric field. It is essentially zero below a threshold and rises explosively above it. Key problems:

- **Extreme gradient magnitudes:** The exponential creates gradients that span many orders of magnitude, causing numerical instability during backpropagation.
- **Near-discontinuity:** The ionization threshold acts as a near-step function. PINNs are known to struggle with sharp transitions in the PDE solution.
- **Stiffness:** The ionization dynamics are stiff (fast timescale of ionization vs. slow timescale of propagation), requiring implicit time-stepping in traditional solvers. PINNs handle stiff systems poorly.

**Barrier 3: Scale of the collocation problem**

The UPPE physics loss must be evaluated across a 2D grid of (ω, z) points. For adequate resolution:

- Frequency: ~8,000 points (to capture narrow spectral features)
- Propagation: ~200 z-steps (to capture dynamics)
- Total: 1.6 million collocation points per training sample

Each evaluation requires FFT, nonlinear computation, and ionization rate calculation. For a training set of 10,000 samples, this becomes computationally prohibitive.

**Barrier 4: Multi-scale dynamics**

The UPPE simultaneously resolves:
- Carrier oscillations at ~1 fs period (optical cycle)
- Pulse envelope at ~10-300 fs
- Soliton fission events at specific propagation distances
- Slow spectral evolution over the full fiber length

PINNs must balance loss contributions from all these scales simultaneously. This multi-scale challenge is an active research problem even for simpler PDEs. For UPPE with its combination of dispersion, Kerr nonlinearity, and ionization, no successful PINN implementation exists in the literature.

**Barrier 5: Comparison with existing PINN work in optics**

The closest precedents are PINNs for the standard Nonlinear Schrödinger Equation (NLSE):

| System | Equation complexity | PINN success? |
|--------|-------------------|---------------|
| Linear Schrödinger | Low | Yes, well-demonstrated |
| NLSE (Kerr only) | Moderate | Demonstrated for simple cases (single soliton), struggles with soliton fission |
| GNLSE (Kerr + Raman + self-steepening) | High | Very limited demonstrations, often inaccurate |
| **UPPE (Kerr + ionization + plasma + full-bandwidth)** | **Very high** | **No successful demonstration exists** |

Each step up in equation complexity makes PINN training significantly harder. UPPE represents a substantial jump beyond what has been achieved.

### 8.3 The Speed Argument

Even if a PINN could be trained for the UPPE (overcoming all barriers above), it would not serve the inverse design goal efficiently:

- A PINN learns to satisfy the PDE at specific input conditions. For each new set of parameters, the PINN must effectively re-solve the PDE.
- This is fundamentally different from a data-driven surrogate, which learns a direct mapping from parameters to output and evaluates in a single forward pass (~1 ms).
- The PINN approach trades data generation cost for training complexity, without gaining speed at inference for the inverse design use case.

### 8.4 What We Do Instead

Rather than embedding the raw UPPE equation (impractical), we extract and use the physical knowledge that the UPPE implies:

| UPPE component | What we extract | How we use it |
|----------------|----------------|---------------|
| Linear term: i·k_z(ω)·Ê | Dispersion relation β(ω, P, d) | Pre-compute β₂, ZDW as input features |
| Kerr nonlinearity | Nonlinear coefficient γ(P, d) | Pre-compute γ, soliton order N, fission length as input features |
| Soliton + dispersive wave dynamics | RDW phase-matching condition | Pre-compute λ_RDW as input feature + loss constraint |
| Ionization threshold | Keldysh parameter γ_K | Pre-compute as input feature (regime indicator) |
| Energy conservation | ∫|Ê_out|²dω ≤ ∫|Ê_in|²dω | Energy conservation loss term |

This approach embeds the physics that CAN be computed analytically and lets the network learn from data only what CANNOT be solved in closed form (complex multi-soliton dynamics, exact ionization-modified spectral shapes, interplay of all effects simultaneously).

### 8.5 Conclusion on PINN

The PINN approach for UPPE is currently infeasible due to the combination of domain-switching nonlinear terms, threshold ionization dynamics, multi-scale physics, and the sheer scale of the collocation problem. No successful PINN implementation exists for equations of this complexity. Our physics-guided data-driven approach achieves the same goal (physically informed predictions) through a more practical route: pre-computed analytical features and physics-constrained loss functions, trained on UPPE simulation data.

However, a middle ground exists that embeds the UPPE's structure without the PINN training difficulties. See Section 8A.

---

## 8A. Alternate Design Direction: Neural Split-Step Propagation

### 8A.1 Motivation

The PINN approach (embedding the full UPPE in the loss) is infeasible. The MLP surrogate approach (Sections 3-4) is feasible but methodologically incremental. The neural split-step architecture offers a middle ground that embeds UPPE physics directly in the network architecture, avoiding PINN training difficulties while achieving higher novelty than a plain MLP.

### 8A.2 Key Insight

The UPPE has two parts:

```
∂Ê/∂z = i·k_z(ω)·Ê           +  i·(ω/c)·FT{P_NL(t) + P_plasma(t)}
         ├── LINEAR TERM ──┤      ├──── NONLINEAR TERM ──────────────┤
         Known EXACTLY              Hard (Kerr + ionization)
         Just phase rotation        No closed-form solution
         in frequency domain
```

The linear term applies dispersion, which is analytically known from the Marcatili model and argon Sellmeier equation. The nonlinear term includes Kerr effect and ionization, which have no tractable closed-form solution for the general case.

**Embed the linear part exactly. Learn the nonlinear part from data.**

This mirrors the split-step Fourier method (SSFM) used to numerically solve the UPPE, where each propagation step alternates between a linear (dispersion) half-step and a nonlinear half-step.

### 8A.3 Architecture

```
Input: [E, τ, C, λ₀]  →  construct Ê_in(ω)  (or use measured spectrum in Phase C)
       [P_gas, L_fiber, d_core]  →  compute β(ω, P, d) and Δz = L/K

For step k = 1 to K (K = 5-10 coarse propagation steps):
  ┌────────────────────────────────────────────────────────────┐
  │ DISPERSION LAYER (exact, zero learnable parameters)        │
  │                                                            │
  │   Ê(ω) ← Ê(ω) · exp(i · β(ω, P, d) · Δz/2)              │
  │                                                            │
  │   β(ω) is computed analytically from the Marcatili model   │
  │   and argon Sellmeier equation. This layer applies the     │
  │   exact linear propagation operator of the UPPE.           │
  │   No learning needed. No approximation.                    │
  ├────────────────────────────────────────────────────────────┤
  │ NONLINEAR LAYER (learned from data)                        │
  │                                                            │
  │   E(t) ← IFFT{Ê(ω)}                                      │
  │   [|E(t)|², phase(E(t)), P_gas, γ_K]  →  small MLP        │
  │   → ΔE_NL(t)   (predicted nonlinear field change)         │
  │   E(t) ← E(t) + ΔE_NL(t)                                 │
  │   Ê(ω) ← FFT{E(t)}                                       │
  │                                                            │
  │   This layer learns the COMBINED effect of Kerr            │
  │   nonlinearity and ionization. It replaces the explicit    │
  │   computation of |E|²E and the ADK ionization rate.        │
  ├────────────────────────────────────────────────────────────┤
  │ DISPERSION LAYER (exact, second half-step)                 │
  │                                                            │
  │   Ê(ω) ← Ê(ω) · exp(i · β(ω, P, d) · Δz/2)              │
  └────────────────────────────────────────────────────────────┘

Output: |Ê_out(ω)|² = |Ê(ω) after K steps|²
```

### 8A.4 Why This Works and Why It Is Novel

1. **Dispersion is handled exactly.** The linear UPPE operator is applied as-is, with zero approximation and zero learnable parameters. The network never needs to learn dispersion from data.

2. **The nonlinear layer learns only what analytics cannot solve.** The Kerr effect combined with ionization in the general case has no closed-form solution. The network learns this from UPPE training data.

3. **Architecture mirrors established numerical method.** The split-step Fourier method is the standard algorithm for solving pulse propagation equations. Each network "layer" corresponds to a physical propagation segment, making the model interpretable.

4. **Avoids all PINN barriers.** The ionization rate is never differentiated analytically (it is learned). No collocation points are needed. No PDE residual is computed. Training is standard supervised learning on UPPE input-output data.

5. **FFT/IFFT operations are differentiable in PyTorch.** `torch.fft.fft` and `torch.fft.ifft` support autograd, enabling standard backpropagation through the entire architecture.

6. **Coarse stepping is sufficient.** The network uses K = 5-10 steps (vs. 200+ in UPPE). Each nonlinear layer learns to approximate the cumulative nonlinear effect over a larger Δz. The exact dispersion layers between steps maintain accuracy for the linear dynamics.

### 8A.5 Comparison of Forward Model Design Directions

| Aspect | MLP Surrogate (Sec. 3) | Neural Split-Step (Sec. 8A) |
|--------|----------------------|---------------------------|
| Physics in architecture | None (physics in inputs/loss only) | Exact dispersion operator embedded |
| Physics in inputs | 8 pre-computed features | Fewer needed (dispersion is in the architecture) |
| Physics in loss | Energy conservation, RDW position | Same, plus possible per-step constraints |
| Interpretability | Black box | Each layer = physical propagation step |
| What network learns | Full input→output map | Only the nonlinear + ionization response |
| Methodological novelty | Low (standard MLP surrogate) | High (novel physics-structured architecture) |
| Implementation complexity | Low | Medium (FFT operations, multi-step architecture) |
| Training difficulty | Easy | Moderate (must tune K, nonlinear layer capacity) |
| Inference speed | ~1 ms | ~5-10 ms (K FFT/IFFT pairs) |
| Expected accuracy | Good with enough data | Potentially better (dispersion is exact) |
| Generalizability | Specific to trained parameter range | Dispersion layer generalizes; only nonlinear part needs retraining |
| Publication impact | Tier 3 (Optics Express) | **Tier 2 (Optica, Photonics Research, Light:S&A)** |

### 8A.6 Inverse Design With Neural Split-Step

The inverse design pipeline remains the same:

- **Gradient optimization (baseline):** Backpropagate through the frozen neural split-step model to optimize parameters. The exact dispersion layers provide physically meaningful gradients.
- **Tandem inverse network:** Train a CNN-based inverse network verified through the frozen neural split-step forward model.

The neural split-step model is fully differentiable (dispersion layers are analytic functions, nonlinear layers are standard neural networks, FFT/IFFT are differentiable), so both inverse approaches work without modification.

### 8A.7 Additional Advantage: Intermediate Predictions

Unlike the MLP surrogate, the neural split-step produces the spectrum at every intermediate step:

```
Ê(ω, z=Δz), Ê(ω, z=2Δz), ..., Ê(ω, z=L)
```

This gives the full spatio-spectral evolution for free, enabling:
- Visualization of where soliton fission occurs
- Identification of where RDW is emitted along the fiber
- Analysis of where ionization becomes significant

This is a valuable scientific output, not just a tool feature.

### 8A.8 Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| K=5-10 coarse steps may miss dynamics | Medium | Increase K; validate against UPPE with fine z-resolution |
| Nonlinear layer may not generalize across regimes | Medium | Train across full parameter space; regime-aware data sampling |
| FFT/IFFT memory overhead during training | Low | Use mixed precision; batch size reduction if needed |
| Longer training time than MLP | Low | Still hours on single GPU, not days |

### 8A.9 Recommendation

The neural split-step architecture is recommended as the primary design direction if the goal is Tier 2 publication. The MLP surrogate (Section 3) remains as a simpler fallback and baseline for comparison. The paper can present both, showing that the physics-embedded architecture outperforms the black-box MLP in accuracy and interpretability while maintaining millisecond-scale inference speed.

---

## 9. Project Roadmap

### Phase A: Simulation-Based Inverse Design (This Paper)

1. Generate UPPE training dataset (~10,000 runs via Latin Hypercube Sampling)
2. Train and validate MLP forward surrogate (Direction A baseline)
3. Implement and train neural split-step forward model (Direction B)
4. Compare MLP vs neural split-step accuracy and interpretability
5. Implement gradient-based inverse design through both forward models
6. Train and validate tandem inverse network
7. Demonstrate inverse design examples with UPPE verification
8. Write and submit paper

### Phase A+ (Optional, for higher impact): Pressure Gradient Extension

1. Modify UPPE code to accept non-uniform pressure profile P(z)
2. Parameterize pressure as N=10-20 piecewise-constant segments
3. Regenerate training data with variable pressure profiles
4. Train neural split-step model (dispersion layer now varies per step)
5. Use inverse model to discover optimal pressure profiles for target spectra
6. Verify discovered profiles with UPPE and interpret physically

### Phase C: Extension to Real Experimental Pulses (Future Work)

1. Replace scalar pulse params with measured spectral field [|Ê_in(ω)|², φ_in(ω)]
2. Redesign input stage with CNN-based pulse encoder
3. Transfer-learn from Phase A forward model
4. Validate with experimental measurements (FROG/SPIDER for input, spectrometer for output)

### Future Extensions

- Multi-objective inverse design (simultaneously optimize RDW wavelength + efficiency + bandwidth)
- Integration with experimental feedback loop (real-time adaptive control)
- Extension to other gas species (neon, helium for shorter UV wavelengths)

---

## 10. Technical Requirements

- **UPPE simulation code:** Available (MATLAB-based, from current data format)
- **Computing:** Cluster access for parallel UPPE data generation
- **ML framework:** PyTorch
- **Libraries:** NumPy, SciPy, scikit-learn (data processing), matplotlib (visualization)
- **Hardware for ML:** Single GPU sufficient (model is small, ~800K parameters)
