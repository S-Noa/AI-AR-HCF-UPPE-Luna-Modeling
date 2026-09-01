# Oral Presentation Script

Target duration: about 13-14 minutes.  
Topic: Transformer-Based Predictive Modeling of Ultrafast Pulse Propagation in Gas-filled Antiresonant Hollow-core Fibers.

## Slide 1. Title [~40 seconds]

Good morning everyone. My name is Meng Gu, from the Sustainable Photonics Research Center at Hangzhou International Innovation Institute, Beihang University.

Today I will present our work on **Transformer-based predictive modeling of ultrafast pulse propagation in gas-filled antiresonant hollow-core fibers**.

The central question of this work is simple: high-fidelity pulse propagation simulations are accurate, but they are often too slow for dense parameter scanning. So we ask whether an AI model can learn the forward propagation behavior and quickly predict not only the final spectrum, but also the full spectral evolution inside the fiber.

## Slide 2. Background: AI x Optics [~1 minute 10 seconds]

Let me start from the broader background. In recent years, artificial intelligence has been rapidly integrated into many fields. In medical imaging, AI is used for image reconstruction, segmentation, and diagnosis assistance. In autonomous driving, AI helps understand complex visual scenes and make real-time decisions. In industrial manufacturing, AI supports defect detection, process monitoring, and optimization.

The same trend is also appearing in optics and photonics. AI has been used to assist optical fiber design, optimize photonic devices, analyze sensing signals, and predict nonlinear optical responses. This is attractive because many optical systems have a large parameter space, and the mapping from design parameters to optical output can be highly nonlinear.

Our work is located in this direction, but with a more specific target: we want to use AI to predict **pulse propagation** in hollow-core fibers. Instead of only predicting one final scalar metric, we want the model to predict the final spectrum and the complete propagation map.

## Slide 3. Background: Supercontinuum and Ultraviolet Generation [~1 minute 30 seconds]

Now let me move to the physical system. Supercontinuum generation is important because it can provide broadband, coherent light sources. These sources are useful in many applications, such as optical coherence tomography, hyperspectral imaging, remote sensing, spectroscopy, and frequency metrology.

For these applications, the spectral coverage and beam quality are both important. In particular, extending the spectrum toward the ultraviolet or mid-infrared can open additional applications. For example, ultraviolet light is useful for spectroscopy and photochemical processes, while mid-infrared light can access molecular fingerprint regions.

Hollow-core fibers provide a very promising platform for supercontinuum generation. Because most of the light is guided in a gas-filled core, the overlap with silica is strongly reduced. This enables a high damage threshold, low material nonlinearity from the glass, and pressure-tunable dispersion through the gas. In antiresonant hollow-core fibers, the core geometry and wall thickness also provide additional control over the dispersion.

So our goal is to support the design of hollow-core-fiber systems that generate broadband supercontinuum, especially with ultraviolet extension, by predicting how the pulse spectrum evolves during propagation.

## Slide 4. Background: Current Research [~1 minute 10 seconds]

There has already been important work on applying machine learning to nonlinear fiber optics. For example, recurrent neural networks have been used to predict ultrafast nonlinear dynamics in fiber systems, and neural networks have also been used to optimize supercontinuum bandwidth in gas-filled hollow-core fibers.

Most of these studies are based on the nonlinear Schrodinger equation or the generalized nonlinear Schrodinger equation. These models are very useful. They are efficient, and they can describe many dispersion and nonlinear effects in optical fibers.

However, our regime is more demanding. We are interested in broadband strong-field propagation in gas-filled antiresonant hollow-core fibers. In this situation, the spectrum can become very broad, the dispersion should be treated carefully over a wide frequency range, and ionization or plasma effects can become important.

This motivates us to use a more general physical model for data generation.

## Slide 5. Why UPPE, Why AI [~1 minute 20 seconds]

Here we compare the physical modeling choices. The generalized nonlinear Schrodinger equation usually relies on a Taylor-expanded dispersion model around a carrier frequency, and ionization is not always included naturally. This can be sufficient for many fiber problems, but it can become limiting for broadband strong-field gas-filled hollow-core-fiber propagation.

In this work, we use the **unidirectional pulse propagation equation**, or UPPE. UPPE can handle exact dispersion over a broad spectral range and can include strong-field effects such as ionization and plasma response. So it gives us a high-fidelity way to generate training data.

But the cost is high. A single UPPE simulation can take a long time, and if we want to scan many combinations of pulse energy, pulse duration, gas pressure, fiber diameter, wall thickness, and propagation length, direct simulation becomes expensive.

This is exactly where the AI surrogate becomes useful. We still rely on UPPE as the physical teacher, but after training, the neural network can provide fast predictions, on the order of milliseconds, for rapid forward screening.

## Slide 6. Classic Transformer [~1 minute 5 seconds]

The model we use is based on the Transformer architecture. The key idea of the Transformer is self-attention. Instead of processing a sequence only step by step, self-attention allows the model to compare all positions in the sequence and focus on the most relevant relationships.

This is useful for our problem because spectral evolution can be viewed as a sequence along the propagation coordinate. The spectral state at one position may be related to features that appear earlier or later along the fiber. For example, early spectral broadening can influence later ultraviolet or long-wavelength components.

Compared with a purely recursive model, our Transformer does not need to predict one step and then feed that prediction back repeatedly. Instead, it directly outputs the full propagation map. This helps avoid error accumulation during long autoregressive rollouts.

In addition to the Transformer, we also considered common baselines such as MLP, CNN, and LSTM. But the Transformer is especially attractive here because it can model long-range correlations among propagation positions.

## Slide 7. Architecture Overview [~1 minute 15 seconds]

This slide shows the architecture of our predictive model.

The input contains six physical operating parameters: pulse energy, pulse duration, gas pressure, propagation length, core diameter, and wall thickness. We denote them as \(E\), \(\tau\), \(p\), \(L\), \(d\), and \(t\). These parameters are first passed into a parameter encoder, which maps them into a latent representation.

Then, for every saved propagation position \(z\), we build a propagation-position token. This token combines the parameter encoding with the coordinate information of that position. The sequence of \(z\)-tokens is sent into a Transformer encoder.

After the Transformer encoder, each propagation-position context is combined with wavelength information. A wavelength-conditioned decoder then predicts the spectral intensity at each pair of propagation distance and wavelength.

So the final output is a two-dimensional map, \(S(z,\lambda)\). The final spectrum is simply the last propagation slice of this map.

## Slide 8. z-token and Physics-aware Loss [~1 minute 45 seconds]

Let me explain two important parts of the model in more detail: the \(z\)-token design and the loss function.

First, the \(z\)-token. In our problem, \(z\) is the propagation coordinate along the fiber. Instead of treating propagation positions as independent output rows, we encode each saved \(z\) position into a token. Each token carries both the physical operating condition and the propagation-position information. Through self-attention, the Transformer can learn nonlocal relationships between different positions along the fiber.

This means the model is not just fitting isolated spectra. It is learning how the spectral evolution is structured along the propagation direction.

Second, the loss function. The main term is \(L_{\mathrm{map}}\), which is the mean-squared error over the complete spectral-evolution map. This is our primary objective because we care about the full \(z \times \lambda\) prediction.

We also include \(L_{\mathrm{final}}\), the loss on the final output spectrum, to make sure the final spectrum remains accurate. \(L_{\mathrm{UV}}\) focuses on the 200 to 700 nanometer region, because ultraviolet and short-wavelength features are important in this work. \(L_{\mathrm{early}}\) gives additional weight to the early propagation region, where rapid spectral restructuring can occur. Finally, \(L_{\lambda\text{-grad}}\) compares wavelength-direction gradients, helping the model preserve narrow peaks and valleys rather than producing an overly smooth spectrum.

Together, these terms guide the model to learn both the global propagation map and the physically important local structures.

## Slide 9. Workflow [~1 minute 25 seconds]

This slide summarizes the full workflow.

We start from the input parameter space. The six operating parameters define the pulse, gas, and fiber conditions. For each sample, we run a high-fidelity UPPE simulation using Luna. The simulation produces the spectral evolution along the fiber and the final output spectrum.

Next, we preprocess the data. The spectral target is interpolated over the wavelength range from 200 to 2500 nanometers, and represented in normalized log-power space. The dataset is then split into training, validation, and test sets.

The Transformer is trained to learn the forward map from the physical parameters to the complete spectral evolution. Once trained, it can predict the full map and the final spectrum directly.

The key point is that the model performs **direct full-map prediction**. It does not recursively predict one propagation step after another. This makes the prediction faster and avoids the drift that can occur in recursive propagation models.

## Slide 10. Results [~1 minute 40 seconds]

Now let us look at the results.

The dataset contains 10,000 UPPE simulations. We use 7,000 samples for training, 1,500 for validation, and 1,500 for testing. The wavelength range is 200 to 2500 nanometers.

On the test set, the coefficient of determination for the final spectrum is \(R^2 = 0.9635\). For the complete spectral-evolution map, the score is \(R^2 = 0.9650\). This means that the model is not only matching the output spectrum, but also learning the propagation trajectory.

In the figure, each row shows one representative test sample. The left column is the UPPE ground truth, the middle column is the Transformer prediction, and the right column compares the final spectrum. We can see that the model captures the main broadband evolution, the major spectral bands, and the relative positions of important spectral features.

At the same time, there are still limitations. Some narrow ultraviolet structures and fine interference-like textures are smoother in the prediction than in the UPPE result. This is one of the main directions for future improvement.

## Slide 11. Conclusion and Outlook [~1 minute 10 seconds]

To conclude, we built a Transformer-based forward surrogate for UPPE-simulated pulse propagation in gas-filled antiresonant hollow-core fibers.

The model directly predicts both the final spectrum and the full spectral-evolution map. On the test set, it achieves \(R^2 = 0.9635\) for final spectra and \(R^2 = 0.9650\) for the complete evolution map.

The main value of this approach is speed. It provides a fast approximation to high-fidelity UPPE simulations and can support rapid forward screening of operating conditions.

For future work, we will focus on three directions. First, we want to improve the prediction of fine spectral features, especially in the ultraviolet region. Second, we plan to extend the model toward absolute-intensity prediction. Third, we want to use the trained surrogate for inverse design and experimental parameter screening, with final validation by Luna simulations and experiments.

Thank you very much for your attention. I would be happy to take questions.

