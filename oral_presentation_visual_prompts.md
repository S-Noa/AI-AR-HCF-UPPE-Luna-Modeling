# Oral Presentation Visual Prompts

This file contains the image-generation prompts and final-slide text for `oral presentation.pptx`.
Use a 16:9 output ratio, preferably 1920x1080 or 2560x1440. Keep the visual style consistent with the deck: white background, blue headings, cyan/orange scientific accents, and minimal text inside generated images.

## Slide 2: AI Predicts Pulse Propagation

```text
Create a clean scientific conference-slide illustration in 16:9 aspect ratio, white background, blue and cyan accent colors, minimal professional style.

Topic: AI-assisted ultrafast pulse propagation prediction in gas-filled antiresonant hollow-core fibers.

Composition:
Left side: an ultrashort input pulse entering a gas-filled hollow-core fiber, with a small input spectrum curve below it.
Center: a compact AI model block labeled "AI Surrogate Model" with subtle neural-network nodes or Transformer-like attention lines inside.
Right side: two outputs: (1) a colorful spectral-evolution heatmap labeled "S(z, lambda)" with axes "Distance z" and "Wavelength lambda"; (2) a final output spectrum curve labeled "Final spectrum".
Add arrows from left to center and from center to right.

Visual details:
Show the fiber as a transparent hollow tube with light confined in the gas core. Use a blue pulse at the input and a broadened rainbow spectrum at the output. The AI block should look like a scientific model, not a cartoon robot. Include small icons for "pulse / gas / fiber parameters" feeding into the AI block.

Text labels should be short and readable:
"Input pulse"
"Gas-filled AR-HCF"
"Pulse, gas, fiber parameters"
"AI Surrogate Model"
"Spectral evolution S(z, lambda)"
"Final spectrum"

Avoid dense paragraphs, avoid decorative background, avoid photorealistic people, avoid misleading equations.
```

Recommended PPT placement:
- Use this as the main visual on the right or center of Slide 2.
- Add one short caption outside the image: `AI surrogate: physical parameters -> full propagation map and final spectrum`.
- If GPT misspells symbols, regenerate without text labels and add labels manually in PowerPoint.

## Slide 8: z-token + Loss Module Explanation

```text
Create a detailed but clean scientific diagram for a conference presentation, 16:9 aspect ratio, white background, blue/cyan/orange accent colors, vector-like style.

Title at top: "z-token modeling and physics-aware loss"

Layout: two main panels side by side.

Left panel title: "Propagation-position z tokens"
Show six physical input parameters as small rounded boxes:
Energy E, pulse duration tau, gas pressure p, propagation length L, core diameter d, wall thickness t.
These feed into a "Parameter Encoder" block.
Below or beside it, show a vertical sequence of propagation positions:
z0, z1, z2, ..., zN.
Each z position passes through a "z projection + learned z embedding" block.
Then combine parameter encoding + z embedding to form a sequence of tokens:
token(z0), token(z1), ..., token(zN).
These tokens enter a "Transformer Encoder" block with multi-head self-attention arrows between different z positions.
Output: "z-context vectors".

Right panel title: "Loss modules"
Show predicted map S_hat(z,lambda) and UPPE target S(z,lambda) side by side.
From them draw arrows into five loss components:
L_map: full evolution map MSE
L_final: final spectrum MSE
L_UV: 200-700 nm UV-region MSE
L_early: first 10 cm early-propagation MSE
L_lambda-grad: wavelength-gradient consistency
Then combine them into:
L = L_map + L_final + w_UV L_UV + w_early L_early + w_lambda L_lambda-grad

Add small visual highlights:
- shade the early z region on a heatmap as "early z <= 10 cm"
- shade the 200-700 nm wavelength band as "UV band"
- draw a small sharp peak/valley curve near L_lambda-grad to imply preserving narrow spectral features.

Use readable labels, simple arrows, and avoid excessive text. Make the diagram suitable for a technical optics + AI audience.
```

Recommended PPT placement:
- Use this as a large full-slide figure on Slide 8.
- Keep only 1-2 external bullets beside or below the figure:
  - `z tokens let attention model nonlocal propagation relationships.`
  - `Auxiliary loss terms emphasize UV, early-z, and narrow spectral features.`
- For final slide polish, add math labels manually if GPT renders formulas poorly.

## Slide 9: Overall Workflow

```text
Create a polished scientific workflow diagram for a conference slide, 16:9 aspect ratio, white background, clean blue/cyan style with orange highlights.

Title: "Workflow of Transformer-based pulse-propagation prediction"

Use a left-to-right pipeline with five stages:

1. "Input parameter space"
Show six parameter boxes:
E, tau, p, L, d, t
Label: pulse / gas / fiber parameters

2. "UPPE simulation with Luna"
Show a simplified gas-filled antiresonant hollow-core fiber and a small label "High-fidelity simulation".
Add output files/icons labeled "HDF5 dataset".

3. "Preprocessing"
Show interpolation over "200-2500 nm", normalized log-power map, and train/validation/test split.

4. "Temporal Transformer"
Show parameter encoder, z tokens, Transformer encoder, wavelength-conditioned decoder.
Keep this compact, like a simplified version of the architecture.

5. "Fast prediction and design"
Show outputs:
- spectral evolution map S(z,lambda)
- final spectrum
- rapid parameter screening / inverse design

Add arrows between stages. Include one small note under the pipeline:
"Direct full-map prediction, no recursive rollout"

The figure should be suitable for academic optics conference slides. Avoid cartoon robots, avoid busy backgrounds, avoid unreadable tiny text.
```

Recommended PPT placement:
- Use as the core visual of Slide 9.
- Good spoken transition: `After simulation data are generated once, the trained model gives fast full-map predictions for new operating conditions.`

## Final Slide: Conclusion and Outlook

Recommended title:

```text
Conclusion and Outlook
```

Recommended three-bullet version:

```text
• We built a Transformer-based forward surrogate for UPPE-simulated pulse propagation in gas-filled AR-HCFs.

• The model directly predicts both the final spectrum and the full spectral-evolution map, achieving R² = 0.9635 and R² = 0.9650 on the test set.

• Next steps: improve fine spectral features, extend toward absolute-intensity prediction, and use the trained surrogate for inverse design and experimental parameter screening.
```

More conversational version for oral delivery:

```text
This work shows that a Transformer surrogate can learn the forward UPPE map from physical operating parameters to full spectral evolution. It provides fast prediction of both output spectra and propagation dynamics, making it useful for AR-HCF source design. Future work will focus on sharper UV features, real-intensity modeling, and inverse design validated by Luna simulations and experiments.
```

Recommended PPT placement:
- Put the three bullets on the left.
- Put a compact visual loop on the right if space allows:
  `Parameters -> Transformer surrogate -> Candidate design -> Luna/experiment validation`.
- Keep contact information/QR code visually separate at the bottom right.

