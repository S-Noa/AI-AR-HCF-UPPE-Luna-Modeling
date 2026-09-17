# Software Copyright Application Checklist

## 1. Recommended Software Scope

Apply for the locally developed application software rather than the entire
Luna package or the upstream RNN repository.

Suggested name:

> `UPPE-Based Gas-Filled Antiresonant Hollow-Core Fiber Ultrafast Pulse Prediction and Optimization Software V1.0`

Suggested functional scope:

1. AR-HCF parameter sampling, Luna/UPPE data generation, and HDF5 management.
2. Spectral preprocessing, feature construction, and normalization.
3. Transformer/CNN forward prediction of final spectra and spectral-evolution maps.
4. Surrogate-based inverse design and reinforcement-learning optimization.
5. Luna verification, final-spectrum comparison, and spectral-evolution visualization.

Do not present the following as self-developed code:

- The upstream `Luna.jl-master/src/` package.
- The upstream `rnnnonlinear` repository.
- PyTorch, NumPy, SciPy, Gymnasium, Stable-Baselines3, or other dependency source code.
- Large raw data, model checkpoints, generated figures, and third-party figures.

Use a `third_party_notices.md` file to list dependencies, sources, versions, and
licenses. Luna is MIT licensed, so its copyright and license notice must be
retained when its source or substantial portions are distributed.

## 2. Decide Registration Basics

Confirm these items with the supervisor and the institution before preparing
the formal material:

1. **Copyright owner**: institution, institution plus collaborator, or an
   individual. For work developed with institutional resources and under a
   research task, confirm whether it is a work-for-hire or institutional asset.
2. **Developer list**: keep it consistent with the development record and
   project documentation.
3. **Version**: freeze the first filing as `V1.0`.
4. **Completion date**: use a date when the stated V1.0 functions were usable.
5. **First publication date**: use "unpublished" if the software has not been
   publicly released.
6. **Software name**: use exactly the same name and version in the application
   form, source-code headers, manual cover, page headers, and screenshots.

## 3. Release Package

Create a clean release snapshot. Do not use the working research root directly.

```text
softcopyright_release/
  README.md
  LICENSES/
    Luna_MIT_LICENSE.txt
    third_party_notices.md
  src/
    ar_hcf_simulation/
    preprocessing/
    forward_model/
    inverse_design/
    visualization/
    gui/
  docs/
    software_design_specification.docx
    user_manual.docx
    test_report.docx
  demo/
    demo_input/
    demo_output/
    run_demo.md
```

Recommended self-developed source areas:

```text
data_generation.jl
anti_resonant_simulation.jl
data_preprocessing.py
train_mlp.py
create_raw4_feature_view.py
create_raw4_zslice_view.py
rl_inverse_env.py
train_rl_inverse.py
evaluate_rl_luna_validation.py
plot_rl_luna_evolution_montage.py
hcf_predictor_gui.py
```

Before filing, create a Git tag such as `softcopyright-v1.0` and record its
commit hash in the release README and test report.

## 4. Required Material Checklist

### 4.1 Registration application form

Prepare the standard software copyright registration application form with:

- Full software name, short name, and `V1.0`.
- Copyright owner and developer information.
- Development completion date and publication status.
- Programming languages: Julia and Python.
- Operating environment: Linux or Windows, Julia, Python, PyTorch, optional
  CUDA, and required scientific Python packages.
- Purpose, functions, and technical features.
- Rights acquisition and ownership information.

### 4.2 Source-program identification material

Prepare a source-code printout from the self-developed modules:

- Normally provide the first consecutive 30 pages and last consecutive 30
  pages of the source program.
- If the full source program is under 60 pages, provide all pages.
- A program page normally contains at least 50 lines.
- Add the software name and version to each page header.
- Add continuous page numbers at the top-right corner.
- Use the same black-and-white A4 formatting consistently.

Prioritize code that demonstrates the project-specific implementation:

1. Parameterized AR-HCF simulation and data handling.
2. Spectral preprocessing and feature construction.
3. Forward surrogate model, loss, training, and evaluation.
4. Inverse-design/RL environment and candidate export.
5. Luna validation and spectral-evolution visualization.

Do not fill the source material with imports, dependency code, or upstream
Luna/RNN source.

### 4.3 Document identification material

Prepare either a *Software Design Specification* or a *User Manual*. A design
specification is recommended for this research software. Prepare at least 60
pages where practical, following the same first-30/last-30-page rule used for
documentation identification material.

Suggested outline:

```text
Cover page and version record
1. Software overview
2. Application scope and users
3. Runtime environment and dependencies
4. Overall architecture
5. Module design
6. Input, output, and HDF5 data formats
7. Simulation-data generation workflow
8. Spectral preprocessing workflow
9. Forward-model training and prediction workflow
10. Inverse-design and RL workflow
11. Luna verification workflow
12. Visualization and result interpretation
13. Installation and deployment
14. Operating instructions
15. Demonstration examples
16. Test results
17. Limitations and error handling
18. Third-party software and license notice
```

Recommended figures:

- `parameter -> UPPE/Luna -> preprocessing -> Transformer/CNN -> prediction`
  workflow.
- Transformer z-token and wavelength-conditioned decoder diagram.
- RL loop: `surrogate -> agent -> candidate -> Luna verification`.
- Representative physical spectral-evolution map.
- Prediction-versus-ground-truth comparison.
- GUI screenshot or command-line workflow screenshot.

### 4.4 Ownership and supporting documents

Prepare the applicable evidence:

- Natural-person or organization identification.
- Institutional task assignment, project approval, or ownership statement.
- Cooperation-development agreement, if applicable.
- Permission evidence if an existing copyrighted program was modified under a
  proprietary license.
- Third-party open-source license notices and dependency statement.

Useful supplementary material, although not always mandatory:

- Test report.
- Project task book or approval record.
- Git tag, commit hash, and development record.
- Runnable demo instructions.
- Screenshots or a short demonstration video.

## 5. Suggested Function Description

The following text can be adapted for the application form:

> The software is designed for ultrafast nonlinear pulse propagation in
> gas-filled antiresonant hollow-core fibers. It integrates parameterized
> UPPE simulation-data management, spectral preprocessing, deep-learning
> forward prediction, spectral-evolution visualization, and surrogate-model
> inverse optimization. Given physical operating parameters including pulse
> energy, pulse duration, gas pressure, core diameter, and wall thickness, the
> software predicts output spectra and distance-wavelength spectral-evolution
> maps, and supports physical verification of candidate parameters through
> Luna/UPPE simulations.

## 6. Filing Workflow

1. Confirm owner, developer list, title, completion date, and V1.0 scope.
2. Freeze the code and create the `softcopyright-v1.0` Git tag.
3. Build `softcopyright_release/`; exclude large data, weights, and upstream
   source repositories.
4. Prepare the 60-page source-code identification material.
5. Write the design specification or user manual and prepare identification
   pages.
6. Prepare the third-party notice and ownership evidence.
7. Prepare a small reproducible demo, screenshots, and test report.
8. Complete the official registration form through the institution's research
   administration process and the relevant copyright-registration channel.
9. Verify that software name, version, dates, owners, and page headers match
   across every material.
10. Archive the submitted PDF files, source snapshot, Git tag, and receipt.
11. If a correction is requested, update all related documents consistently;
   do not change the frozen V1.0 source without recording the change.

## 7. Immediate Actions for This Project

1. Confirm whether the owner should be the university/research institute or
   another entity.
2. Decide whether V1.0 includes the experimental RL module. A conservative
   V1.0 can include data generation, preprocessing, Transformer/CNN forward
   prediction, and visualization; RL can be described as an optional module
   only after its verification workflow is stable.
3. Create the clean release directory and third-party notices.
4. Produce a small reproducible demo dataset and output figures.
5. Start the design specification before selecting the final 60 code pages.

## 8. Official Basis

According to the Chinese *Measures for the Registration of Computer Software
Copyright*, a registration application includes the application form, program
and document identification materials, and relevant supporting evidence.
Ordinary identification material consists of the first and last 30 consecutive
pages of source program and documentation; if the complete material is under
60 pages, submit the whole material. Except in special cases, program pages
should contain at least 50 lines and document pages at least 30 lines.

Reference: [National Copyright Administration, Measures for the Registration
of Computer Software Copyright](https://www.ncac.gov.cn/xxfb/flfg/bmgz/202410/t20241015_869486.html).

This checklist is a project-preparation guide, not legal advice. Confirm the
latest institutional and registration-channel requirements before submission.
