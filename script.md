# Oral Presentation Script

## Slide 1. Title and Opening [~40 seconds]

Good afternoon everyone. My name is Meng Gu, from the Sustainable Photonics Research Center at Hangzhou International Innovation Institute, Beihang University.

Today I will present our work on Transformer-based predictive modeling of ultrafast pulse propagation in gas-filled antiresonant hollow-core fibers.

The main motivation is straightforward. High-fidelity pulse propagation simulations are accurate, but they are too slow for dense parameter scanning and rapid design. So we ask whether an AI model can learn the forward propagation behavior and quickly predict not only the final spectrum, but also the full spectral evolution inside the fiber.

## Slide 2. Background: AI x Optics [~1 minute 10 seconds]

Let me start from the broader background. In recent years, artificial intelligence has been rapidly integrated into many fields. In medical imaging, AI supports image reconstruction, segmentation, and diagnosis assistance. In autonomous driving, AI helps understand complex visual scenes and make real-time decisions. In industrial manufacturing, AI is used for defect detection, process monitoring, and optimization.

The same trend is also appearing in optics and photonics. AI can assist optical fiber design, optimize photonic devices, analyze sensing signals, and predict nonlinear optical responses. This is attractive because many optical systems have a large parameter space, and the mapping from design parameters to optical output is often highly nonlinear.

Our work is located in this direction, but with a more specific target: we use AI to predict pulse propagation in hollow-core fibers. The goal is to estimate the final spectrum and the full propagation map from physical operating parameters.

## Slide 3. Background: Supercontinuum and Ultraviolet Generation [~1 minute 25 seconds]

Now let us move to the physical system. Supercontinuum generation is important because it provides broadband and coherent light sources. These sources are useful in optical coherence tomography, hyperspectral imaging, remote sensing, spectroscopy, and other applications.

For these applications, spectral coverage and beam quality are both important. Extending the spectrum toward the ultraviolet or mid-infrared can open additional opportunities. Ultraviolet light is useful for spectroscopy and photochemical processes, while mid-infrared light can access molecular fingerprint regions.

Gas-filled hollow-core fibers are a promising platform for generating supercontinuum. Because the optical field is mainly guided in a gas-filled core, the overlap with silica is strongly reduced. This gives high power-handling capability, good beam quality, and pressure-tunable dispersion. In antiresonant hollow-core fibers, the core diameter and wall thickness provide additional control over the propagation dynamics.

So our goal is to support the design of hollow-core-fiber systems for broadband supercontinuum and ultraviolet generation by predicting how the pulse spectrum evolves during propagation.

## Slide 4. Background: Current Research [~1 minute 5 seconds]

There has already been important work on applying machine learning to nonlinear fiber optics. For example, recurrent neural networks have been used to predict ultrafast nonlinear dynamics in fiber systems. Neural networks have also been used to optimize supercontinuum bandwidth in gas-filled hollow-core fibers.

Most of these studies are based on the nonlinear Schrodinger equation or the generalized nonlinear Schrodinger equation, usually called the GNLSE. These models are very useful. They are efficient and can describe many dispersion and nonlinear effects in optical fibers.

However, our regime is more demanding. We are interested in broadband strong-field propagation in gas-filled antiresonant hollow-core fibers. In this situation, the spectrum can become very broad, dispersion should be handled over a wide frequency range, and ionization or plasma effects can become important. This motivates us to use a more general propagation model for data generation.

## Slide 5. Why UPPE not GNLSE [~1 minute 20 seconds]

This slide compares the physical modeling choices. The GNLSE is very efficient, but in many implementations it relies on a Taylor-expanded dispersion model around a carrier frequency. Ionization is also not always included naturally. For many fiber problems this is acceptable, but it can become limiting for broadband strong-field propagation in gas-filled hollow-core fibers.

In this work, we use the unidirectional pulse propagation equation, or UPPE. UPPE can handle exact dispersion over a broad spectral range and can include strong-field effects such as ionization and plasma response. Therefore, it is a suitable high-fidelity model for generating training data.

The challenge is computational cost. A single UPPE simulation can be expensive, and dense scans over pulse, gas, and fiber parameters become prohibitive. This is where an AI surrogate becomes useful. We still rely on UPPE as the physical teacher, but after training, the neural network gives fast predictions, on the order of milliseconds, and enables rapid forward screening.

## Slide 6. Classic Transformer [~1 minute]

The model we use is based on the Transformer architecture. The key idea of the Transformer is self-attention. Instead of processing a sequence only step by step, self-attention allows the model to compare all positions in the sequence and focus on the most relevant relationships.

This is useful for our problem because spectral evolution can be viewed as a sequence along the propagation coordinate. The spectral state at one position may depend on features generated at earlier positions. Early spectral broadening, for example, can influence later ultraviolet or long-wavelength components.

Compared with common baselines such as MLP, CNN, and LSTM, the Transformer is attractive because it can model long-range spectral and propagation correlations. It also predicts all propagation positions directly, rather than relying on a recursive rollout.

## Slide 7. Workflow and Model Architecture [~1 minute 15 seconds]

This slide summarizes the overall workflow and the model architecture.

We start from an input parameter space. The six operating parameters are pulse energy E, pulse duration tau, gas pressure p, propagation length L, core diameter d, and wall thickness t. For each parameter set, we run a high-fidelity UPPE simulation using Luna. The simulation produces the spectral evolution along the fiber and the final output spectrum.

Next, the data are preprocessed. The spectral target is interpolated over the wavelength range from 200 to 2500 nanometers and represented in normalized log-power space. The dataset is then split into training, validation, and test sets.

Finally, the Transformer learns the forward map from physical parameters to the complete spectral evolution S of z and lambda. The final spectrum is simply the last distance slice of this predicted map.

## Slide 8. Model Architecture [~1 minute 15 seconds]

Let me describe the model architecture in more detail.

The six physical parameters first enter a parameter encoder. This encoder maps the input condition into a latent representation. For each saved propagation position, the model also builds a propagation-position representation. These two pieces of information are combined to form a sequence of tokens along the z direction.

The token sequence is processed by a Transformer encoder. In our implementation, the model uses four encoder blocks, four attention heads, and a model dimension of 192. The attention mechanism acts along the propagation-distance tokens, so the model can learn relationships among different positions inside the fiber.

After the Transformer encoder, each z-context vector is combined with wavelength information. A wavelength-conditioned decoder then predicts the spectral intensity at each distance and wavelength. The wavelength range is divided into four bands, from 200 to 700, 700 to 1200, 1200 to 1800, and 1800 to 2500 nanometers. This band-specific design helps reduce averaging over spectrally different regions.

## Slide 9. z-token [~1 minute]

This slide focuses on the z-token module.

Here, z denotes the propagation coordinate along the fiber. Instead of treating each propagation position as an independent output row, we encode each saved z position into a token. Each token contains both the physical operating condition and the propagation-position information.

This design is important because propagation is not a collection of independent spectra. It is a continuous physical evolution. Through self-attention, the Transformer can learn nonlocal relationships between different z positions. For example, spectral features generated near the fiber input can influence structures that appear later in the propagation.

So the z-token design allows the model to learn the trajectory of the spectral evolution, not only the final output plane.

## Slide 10. Loss Function [~1 minute 25 seconds]

The training loss is also designed around the full propagation task.

The first term is L_map. This is the mean-squared error over the complete predicted spectral-evolution map. It is the primary objective because our main target is the full z by wavelength prediction.

The second term is L_final. This term evaluates the last propagation slice, so it keeps the final output spectrum accurate.

The third term is L_UV. It focuses on the 200 to 700 nanometer region, because ultraviolet and short-wavelength features are important for this work.

The fourth term is L_early. It emphasizes the early propagation segment, where rapid spectral restructuring often occurs.

The last term is L_lambda-grad. It compares wavelength-direction gradients between prediction and simulation. This helps preserve narrow peaks and valleys and reduces excessive smoothing.

Together, these terms guide the model to learn the global propagation map, the final spectrum, and physically important local structures.

## Slide 11. Results [~1 minute 40 seconds]

Now let us look at the results.

The dataset contains 10,000 UPPE simulations. We use 7,000 samples for training, 1,500 for validation, and 1,500 for testing. The wavelength range is 200 to 2500 nanometers.

On the test set, the final-spectrum R squared is 0.9635. For the complete spectral-evolution map, the R squared is 0.9650. This means that the model is not only matching the output spectrum, but also learning the propagation trajectory.

In the figure, each row shows one representative test sample. The left column is the UPPE ground truth, the middle column is the Transformer prediction, and the right column compares the final spectrum. We can see that the model captures the main broadband evolution, the major spectral bands, and the relative positions of important spectral features.

At the same time, there are still limitations. Some narrow ultraviolet structures and fine interference-like textures are smoother in the prediction than in the UPPE result. This is one of the main directions for future improvement.

## Slide 12. Conclusion and Outlook [~1 minute]

To conclude, we built a Transformer-based forward surrogate for UPPE-simulated pulse propagation in gas-filled antiresonant hollow-core fibers.

The model directly predicts both the final spectrum and the full spectral-evolution map. On the test set, it achieves an R squared of 0.9635 for final spectra and 0.9650 for the complete evolution map.

The main value of this approach is speed. It provides a fast approximation to high-fidelity UPPE simulations and can support rapid forward screening of operating conditions.

For future work, we will focus on improving fine spectral features, especially in the ultraviolet region. We also want to extend the model toward absolute-intensity prediction and use the trained surrogate for inverse design, with final validation by Luna simulations and experiments.

Thank you very much for your attention. I would be happy to take questions.

