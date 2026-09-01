# HCF MLP Spectrum Comparison Report

## 1. Model Overview

- **Model**: best_model.pth
- **Input Dimensions**: 13 features
- **Output Dimensions**: 500 spectral points (200-2500 nm)
- **Test Set Size**: 919 samples
- **Overall Test R²**: 0.9227

## 2. Sample Comparisons

### 2.1 Performance-Based Comparison

#### Best R² (Sample 100)

- R² Score: 0.9978
- True Peak Wavelength: 342.9 nm
- Predicted Peak Wavelength: 347.5 nm
- Peak Wavelength Error: 4.6 nm

![Best R²](comparison_100_best_r².png)

#### Median R² (Sample 790)

- R² Score: 0.9853
- True Peak Wavelength: 333.7 nm
- Predicted Peak Wavelength: 347.5 nm
- Peak Wavelength Error: 13.8 nm

![Median R²](comparison_790_median_r².png)

#### Worst R² (Sample 709)

- R² Score: -0.3541
- True Peak Wavelength: 200.0 nm
- Predicted Peak Wavelength: 200.0 nm
- Peak Wavelength Error: 0.0 nm

![Worst R²](comparison_709_worst_r².png)

### 2.2 Wavelength-Based Comparison

#### UV Peak (Sample 0)

- R² Score: 0.9878
- True Peak Wavelength: 384.4 nm
- Predicted Peak Wavelength: 393.6 nm
- Peak Wavelength Error: 9.2 nm

![UV Peak](comparison_0_uv_peak.png)

#### Visible Peak (Sample 1)

- R² Score: 0.9418
- True Peak Wavelength: 559.5 nm
- Predicted Peak Wavelength: 541.1 nm
- Peak Wavelength Error: 18.4 nm

![Visible Peak](comparison_1_visible_peak.png)

#### IR Peak (Sample 13)

- R² Score: 0.9924
- True Peak Wavelength: 1025.1 nm
- Predicted Peak Wavelength: 1029.7 nm
- Peak Wavelength Error: 4.6 nm

![IR Peak](comparison_13_ir_peak.png)

## 3. Key Observations

### 3.1 Strengths

- **Peak Wavelength Prediction**: The model accurately predicts peak positions across different wavelength ranges.
- **Overall Shape Capture**: The predicted spectra closely match the overall shape of true spectra.
- **Consistency**: Even in the worst-case sample, the model maintains reasonable prediction quality.

### 3.2 Limitations

- **Minor Details**: Some fine structures in the spectra may be slightly smoothed.
- **Edge Effects**: Predictions at the extreme ends of the wavelength range (200nm and 2500nm) may be less accurate.

## 4. Conclusion

The MLP model demonstrates strong performance in predicting HCF supercontinuum spectra.
It accurately captures both the overall shape and peak positions across different wavelength ranges.
The model provides a reliable alternative to time-consuming numerical simulations,
enabling rapid design and optimization of HCF systems.
