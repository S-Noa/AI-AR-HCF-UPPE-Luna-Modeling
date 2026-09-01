#!/usr/bin/env python3
"""
Model Evaluation Script: TemporalLSTM vs TemporalMLP

对两个训练好的时序模型进行系统性评估：
1. 加载 models_t650_lstm 和 models_t650_temporal 中的最佳模型
2. 使用 sample_000050.h5 的输入参数执行完整预测流程
3. 生成脉冲传输全过程的热力图可视化
4. 生成并对比两个模型输出的最终光谱结果
5. 保存高质量图像文件

输出文件：
- model_comparison_heatmap.png : 两个模型的时序演化热力图对比
- model_comparison_spectrum.png : 两个模型的最终光谱对比
- model_evaluation_report.txt : 详细的评估指标报告

Usage:
    python evaluate_models_heatmap.py
"""

import os
import sys
import json
import re
import warnings

import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(SCRIPT_DIR, "sample_000050.h5")
LSTM_DIR = os.path.join(SCRIPT_DIR, "models_t650_lstm_v3")
TEMPORAL_DIR = os.path.join(SCRIPT_DIR, "models_t650_temporal_v3")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "evaluation_output_v3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature configuration (must match training)
# ---------------------------------------------------------------------------
# Base features (13 dimensions - must match data_preprocessing.py extract_features)
# Order: [energy, tau, pressure, diameter, wallthickness, beta2, gamma, N, L0, gamma_K, P_ratio, Aeff, neff]
FEATURE_NAMES = [
    'energy', 'tau', 'pressure', 'diameter', 'wallthickness',
    'beta2', 'gamma', 'N', 'L0',
    'gamma_K', 'P_ratio', 'Aeff', 'neff'
]

# Extended features with polynomial interactions (15 dimensions after preprocessing)
# energy_pressure = energy * pressure, diameter_pressure = diameter * pressure
FEATURE_NAMES_15 = FEATURE_NAMES + ['energy_pressure', 'diameter_pressure']

# Extended features with one-hot thickness encoding (legacy, not used in v4.1)
FEATURE_NAMES_20 = FEATURE_NAMES + ['T650', 'T360', 'T200', 'T080']

PARAM_SPACE_RANGES = {
    'energy': (0.3, 3.0),
    'tau': (5.0, 50.0),
    'pressure': (0.5, 50.0),
    'diameter': (100.0, 200.0),
    'beta2': (-1000.0, 1000.0),
    'gamma': (0.001, 10.0),
    'N': (0.5, 5.0),
    'L0': (0.1, 100.0),
    'gamma_K': (0.1, 5.0),
    'P_ratio': (0.1, 10.0),
    'Aeff': (100.0, 2000.0),
    'neff': (1.0001, 1.001),
}

# ---------------------------------------------------------------------------
# Model Architectures (must exactly match train_mlp.py)
# ---------------------------------------------------------------------------

class TemporalMLP(nn.Module):
    """TemporalMLP v2.1 - matches train_mlp.py exactly."""
    def __init__(self, input_dim, output_dim, n_z_steps=20):
        super(TemporalMLP, self).__init__()
        self.n_z_steps = n_z_steps
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
        )

        # Temporal decoder: predicts spectrum at each z step
        self.temporal_decoder = nn.Sequential(
            nn.Linear(512 + 1, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Constrain to [0, 1]
        )

        # Final decoder: receives [latent + last_temporal]
        self.final_decoder = nn.Sequential(
            nn.Linear(512 + output_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Constrain to [0, 1]
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        latent = self.encoder(x)

        if z_positions is None:
            # Fallback: use latent directly when no z_positions provided
            final_output = self.final_decoder(
                torch.cat([latent, torch.zeros(batch_size, self.output_dim, device=x.device)], dim=1)
            )
            return None, final_output

        n_z = z_positions.size(1)
        temporal_output = []
        for i in range(n_z):
            z_i = z_positions[:, i:i+1]
            z_i_norm = z_i / (z_i.max() + 1e-8)
            latent_z = torch.cat([latent, z_i_norm], dim=1)
            spectrum_i = self.temporal_decoder(latent_z)
            temporal_output.append(spectrum_i)

        temporal_output = torch.stack(temporal_output, dim=1)

        # Improved final output: use last temporal step concatenated with latent
        last_temporal = temporal_output[:, -1, :]
        final_input = torch.cat([latent, last_temporal], dim=1)
        final_output = self.final_decoder(final_input)

        return temporal_output, final_output


class TemporalLSTM(nn.Module):
    """TemporalLSTM - matches train_mlp.py exactly."""
    def __init__(self, input_dim, output_dim, n_z_steps=20, hidden_dim=512):
        super(TemporalLSTM, self).__init__()
        self.n_z_steps = n_z_steps
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.01),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Sigmoid-constrained output for normalized log power
        self.spectrum_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim),
            nn.Sigmoid()  # Constrain to [0, 1]
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        param_latent = self.param_encoder(x)

        if z_positions is None:
            output = self.spectrum_decoder(param_latent)
            return None, output

        n_z = z_positions.size(1)
        lstm_input = []
        for i in range(n_z):
            z_i = z_positions[:, i:i+1]
            z_i_norm = z_i / (z_i.max() + 1e-8)
            step_input = torch.cat([param_latent, z_i_norm], dim=1)
            lstm_input.append(step_input)

        lstm_input = torch.stack(lstm_input, dim=1)
        lstm_output, _ = self.lstm(lstm_input)

        temporal_output = []
        for i in range(n_z):
            spectrum_i = self.spectrum_decoder(lstm_output[:, i, :])
            temporal_output.append(spectrum_i)

        temporal_output = torch.stack(temporal_output, dim=1)
        final_output = temporal_output[:, -1, :]
        return temporal_output, final_output


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def normalize_value(name, value):
    if name in PARAM_SPACE_RANGES:
        vmin, vmax = PARAM_SPACE_RANGES[name]
        normalized = (value - vmin) / (vmax - vmin)
        return float(max(0.0, min(1.0, normalized)))
    return float(value)


def parse_log_file(log_path):
    """Parse data_generation.log to extract sample parameters."""
    if not os.path.isfile(log_path):
        return {}

    params = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract parameter ranges
    range_pattern = r'(\w+)\s*:\s*([\d.]+)\s*-\s*([\d.]+)'
    for match in re.finditer(range_pattern, content):
        param_name = match.group(1)
        if param_name in ['energy', 'tau', 'pressure', 'diameter']:
            params[param_name] = {
                'range': (float(match.group(2)), float(match.group(3)))
            }

    return params


def extract_parameters_from_hdf5(file_path):
    """Extract input features from HDF5 sample file with error handling."""
    params = {}
    sample_name = os.path.basename(file_path)

    with h5py.File(file_path, 'r') as f:
        # Read input parameters from HDF5 (input group should be safe)
        if 'input' in f:
            inp = f['input']
            for key in inp.keys():
                val = safe_read_dataset(inp[key])
                if val is None:
                    continue
                if isinstance(val, (int, float)):
                    params[key] = float(val)
                elif isinstance(val, np.ndarray) and val.size == 1:
                    params[key] = float(val)

        # Read physics features
        if 'physics_features' in f:
            pf = f['physics_features']
            for key in pf.keys():
                try:
                    val = safe_read_dataset(pf[key])
                    if val is None:
                        continue
                    if isinstance(val, (int, float)):
                        params[key] = float(val)
                    elif isinstance(val, np.ndarray) and val.size == 1:
                        params[key] = float(val)
                except (TypeError, OSError, ValueError) as e:
                    print(f"  ⚠ Skipping physics_features/{key}: {e}")
                    continue

        # Read grid info
        if 'grid' in f:
            grid = f['grid']
            for key in ['ω', 'λ']:
                if key in grid:
                    val = safe_read_dataset(grid[key])
                    if val is not None and isinstance(val, np.ndarray):
                        params[f'grid_{key}'] = val

        # Read stats
        if 'stats' in f:
            stats = f['stats']
            for key in ['z']:
                if key in stats:
                    val = safe_read_dataset(stats[key])
                    if val is not None and isinstance(val, np.ndarray):
                        params[f'stats_{key}'] = val

    return params


def safe_read_dataset(ds):
    """Safely read an HDF5 dataset, handling Julia-specific types."""
    try:
        return ds[()]
    except (TypeError, OSError) as e:
        # TypeBitfieldID and other unsupported types
        return None


def convert_value(val):
    """Convert HDF5 value to Python scalar."""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    if isinstance(val, str):
        return val
    if isinstance(val, (np.ndarray, np.generic)):
        if val.size == 1:
            return float(val)
        return val
    return val


def build_input_tensor(sample_params, feature_names, target_dim=None):
    """Build input tensor from sample parameters.

    Supports:
      - 13 base features (FEATURE_NAMES)
      - 15 features with polynomial interactions (FEATURE_NAMES_15)
      - Legacy 20 features with one-hot encoding (FEATURE_NAMES_20)
    """
    features = []
    for name in feature_names:
        if name == 'energy_pressure':
            # Polynomial interaction: energy * pressure
            energy = sample_params.get('energy', 0.0)
            pressure = sample_params.get('pressure', 0.0)
            features.append(normalize_value('energy', energy) * normalize_value('pressure', pressure))
        elif name == 'diameter_pressure':
            # Polynomial interaction: diameter * pressure
            diameter = sample_params.get('diameter', 0.0)
            pressure = sample_params.get('pressure', 0.0)
            features.append(normalize_value('diameter', diameter) * normalize_value('pressure', pressure))
        elif name in sample_params:
            val = sample_params[name]
            if isinstance(val, (int, float)):
                features.append(normalize_value(name, val))
            else:
                features.append(0.0)
        elif name.startswith('T'):
            # Thickness one-hot encoding (legacy)
            features.append(1.0 if name == 'T650' else 0.0)
        else:
            features.append(0.0)

    input_tensor = torch.FloatTensor(features).unsqueeze(0)

    # Pad or trim to target dimension if specified
    if target_dim is not None and input_tensor.shape[1] != target_dim:
        current_dim = input_tensor.shape[1]
        if current_dim < target_dim:
            padding = torch.zeros(1, target_dim - current_dim)
            input_tensor = torch.cat([input_tensor, padding], dim=1)
        else:
            input_tensor = input_tensor[:, :target_dim]

    return input_tensor


def load_model_and_config(model_dir, model_type):
    """Load model, config, and metrics from directory."""
    model_path = os.path.join(model_dir, 'best_model.pth')
    config_path = os.path.join(model_dir, 'config.json')
    metrics_path = os.path.join(model_dir, 'metrics.json')
    history_path = os.path.join(model_dir, 'history.json')

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load config
    config = {}
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Load metrics
    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    # Load history
    history = {}
    if os.path.isfile(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)

    # Load state dict first to infer dimensions
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)

    # Infer input_dim from first Linear layer weight shape: (out_features, in_features)
    if model_type == 'lstm':
        first_key = 'param_encoder.0.weight'
    else:
        first_key = 'encoder.0.weight'

    if first_key in state_dict:
        input_dim = state_dict[first_key].shape[1]
    else:
        input_dim = config.get('input_dim', len(FEATURE_NAMES))

    # Infer output_dim from last Linear layer (decoder output)
    # For LSTM: spectrum_decoder.6.weight (last layer in Sequential with Sigmoid)
    # For TemporalMLP: temporal_decoder.6.weight or final_decoder.6.weight
    output_dim = config.get('output_dim', 500)
    if model_type == 'lstm':
        # Find the last Linear layer in spectrum_decoder
        last_keys = [k for k in state_dict.keys() if k.startswith('spectrum_decoder.') and 'weight' in k]
        if last_keys:
            last_key = sorted(last_keys, key=lambda x: int(x.split('.')[1]))[-1]
            output_dim = state_dict[last_key].shape[0]
    else:
        # For TemporalMLP, check temporal_decoder or final_decoder
        last_keys = [k for k in state_dict.keys() if k.startswith('temporal_decoder.') and 'weight' in k]
        if last_keys:
            last_key = sorted(last_keys, key=lambda x: int(x.split('.')[1]))[-1]
            output_dim = state_dict[last_key].shape[0]

    n_z_steps = config.get('n_z_steps', 20)

    # Create model instance with inferred dimensions
    if model_type == 'lstm':
        model = TemporalLSTM(input_dim, output_dim, n_z_steps=n_z_steps)
    elif model_type == 'temporal':
        model = TemporalMLP(input_dim, output_dim, n_z_steps=n_z_steps)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(state_dict)
    model.eval()

    return model, config, metrics, history


def predict_with_model(model, input_tensor, z_positions, model_type):
    """Run prediction with a model."""
    with torch.no_grad():
        temporal_output, final_output = model(input_tensor, z_positions)
    return temporal_output, final_output


def create_wavelength_axis():
    """Create wavelength axis for 500-point spectra (200-2500 nm)."""
    return np.linspace(200, 2500, 500)


def load_ground_truth_from_hdf5(file_path):
    """Load and process ground truth temporal evolution from HDF5.

    This function replicates the exact same processing as data_preprocessing.py
    to ensure consistency between training and evaluation:
      1. Read Eω and frequency grid ω
      2. Determine layout (freq_z or z_freq)
      3. For each z-step: convert ω to λ, mask 200-2500nm, interpolate to 500 points
      4. Apply log10 transformation
      5. Per-sample Min-Max normalization
      6. Return temporal_evolution and final_spectrum

    Returns:
        temporal_evolution: np.ndarray, shape (n_z, 500), normalized to [0, 1]
        final_spectrum: np.ndarray, shape (500,), same normalization as temporal[-1]
        wavelengths: np.ndarray, shape (500,), uniform 200-2500nm grid
        z_positions: np.ndarray or None
    """
    C = 299792458.0  # Speed of light (m/s)
    # CRITICAL: WAVELENGTH_RANGE must be in METERS to match data_preprocessing.py
    # data_preprocessing.py uses (200e-9, 2500e-9) = (200nm, 2500nm)
    WAVELENGTH_RANGE = (200e-9, 2500e-9)  # meters
    TARGET_POINTS = 500
    EPSILON = 1e-15

    def estimate_wavelength(n_points):
        """Estimate wavelength array as fallback."""
        freq = np.linspace(C / WAVELENGTH_RANGE[1], C / WAVELENGTH_RANGE[0], n_points)
        return C / freq

    def process_single_spectrum(Eω_spectrum, omega=None):
        """Process single spectrum - replicates data_preprocessing.py process_spectrum()."""
        n_freq = len(Eω_spectrum)

        if omega is not None and len(omega) == n_freq:
            valid_mask = np.abs(omega) > 1e10
            wavelength = np.full(n_freq, np.nan)
            wavelength[valid_mask] = 2 * np.pi * C / omega[valid_mask]
            nan_mask = np.isnan(wavelength)
            if np.any(nan_mask):
                valid_idx = np.where(~nan_mask)[0]
                nan_idx = np.where(nan_mask)[0]
                if len(valid_idx) >= 2:
                    wavelength[nan_idx] = np.interp(nan_idx, valid_idx, wavelength[valid_idx])
                else:
                    wavelength = estimate_wavelength(n_freq)
        else:
            wavelength = estimate_wavelength(n_freq)

        power = np.abs(Eω_spectrum) ** 2
        mask = (wavelength >= WAVELENGTH_RANGE[0]) & (wavelength <= WAVELENGTH_RANGE[1])
        valid_points = np.sum(mask)

        if valid_points < 10:
            wl_min, wl_max = np.nanmin(wavelength), np.nanmax(wavelength)
            mask = (wavelength >= max(WAVELENGTH_RANGE[0], wl_min)) & \
                   (wavelength <= min(WAVELENGTH_RANGE[1], wl_max))

        spectrum_cropped = power[mask]
        wavelength_cropped = wavelength[mask]

        sort_idx = np.argsort(wavelength_cropped)
        wavelength_cropped = wavelength_cropped[sort_idx]
        spectrum_cropped = spectrum_cropped[sort_idx]

        if len(spectrum_cropped) < 10:
            return np.zeros(TARGET_POINTS)

        target_wavelength = np.linspace(WAVELENGTH_RANGE[0], WAVELENGTH_RANGE[1], TARGET_POINTS)
        spectrum_interp = np.interp(target_wavelength, wavelength_cropped, spectrum_cropped)

        log_spectrum = np.log10(spectrum_interp + EPSILON)
        min_val, max_val = log_spectrum.min(), log_spectrum.max()
        if max_val > min_val:
            normalized = (log_spectrum - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(log_spectrum)

        return normalized

    with h5py.File(file_path, 'r') as f:
        # Read Eω
        if 'Eω' in f:
            E_omega = f['Eω'][()]
        else:
            E_omega = None

        # Read z positions
        if 'stats' in f and 'z' in f['stats']:
            z_positions = f['stats']['z'][()]
        else:
            z_positions = None

        # Read frequency grid ω
        omega = None
        if 'grid' in f:
            grid = f['grid']
            for candidate in ['ω', 'omega', 'w']:
                if candidate in grid:
                    try:
                        omega = grid[candidate][:]
                        if omega is not None and len(omega) > 0:
                            break
                    except:
                        continue
            if omega is None and 'sidx' in grid and 'ωo' in grid:
                try:
                    sidx = grid['sidx'][:]
                    omegao = grid['ωo'][:]
                    omega = omegao[sidx]
                except:
                    pass

    if E_omega is not None:
        # Determine layout: (z, freq) or (freq, z)
        if E_omega.shape[0] <= E_omega.shape[1] and E_omega.shape[0] < 50:
            # Likely (z, freq) layout
            n_z = E_omega.shape[0]
            Eω_per_z = E_omega
        elif E_omega.shape[1] <= E_omega.shape[0] and E_omega.shape[1] < 50:
            # Likely (freq, z) layout
            n_z = E_omega.shape[1]
            Eω_per_z = E_omega.T
        else:
            # Fallback: try to determine from z_positions length
            if z_positions is not None and len(z_positions) > 0:
                n_z = len(z_positions)
                if n_z == E_omega.shape[0]:
                    Eω_per_z = E_omega
                else:
                    Eω_per_z = E_omega.T
            else:
                # Assume first dim is z
                n_z = E_omega.shape[0]
                Eω_per_z = E_omega

        # Process each z-step using the same logic as data_preprocessing.py
        temporal_spectra = []
        for i in range(n_z):
            processed = process_single_spectrum(Eω_per_z[i], omega=omega)
            temporal_spectra.append(processed)

        temporal_spectra = np.array(temporal_spectra)  # shape: (n_z, 500)

        # Create uniform wavelength axis for visualization (in nm for display)
        wavelengths = np.linspace(200, 2500, TARGET_POINTS)

        # Downsample z steps to 20 if needed
        if temporal_spectra.shape[0] > 20:
            indices = np.linspace(0, temporal_spectra.shape[0] - 1, 20, dtype=int)
            temporal_evolution = temporal_spectra[indices]
            if z_positions is not None and len(z_positions) > 20:
                z_indices = np.linspace(0, len(z_positions) - 1, 20, dtype=int)
                z_positions = z_positions[z_indices]
        else:
            temporal_evolution = temporal_spectra

        # final_spectrum is the last z-step (same as temporal_evolution[-1])
        final_spectrum = temporal_evolution[-1]

        return temporal_evolution, final_spectrum, wavelengths, z_positions

    return None, None, create_wavelength_axis(), z_positions


def plot_heatmap(ax, data, wavelengths, z_positions, title, cmap='hot'):
    """Plot a heatmap of temporal evolution."""
    if data is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Ensure data is 2D
    if len(data.shape) == 3:
        data = data[0]  # Take first batch

    extent = [wavelengths.min(), wavelengths.max(),
              z_positions.min(), z_positions.max()]

    im = ax.imshow(data, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Propagation distance Z (m)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Normalized Log Power')


def plot_spectrum(ax, wavelengths, spectrum, label, color, title=None):
    """Plot a single spectrum."""
    if spectrum is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
        if title:
            ax.set_title(title)
        return

    if len(spectrum.shape) > 1:
        spectrum = spectrum.flatten()

    ax.plot(wavelengths, spectrum, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Log Power')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_spectrum_comparison(ax, wavelengths, true_spectrum,
                             lstm_spectrum, temporal_spectrum,
                             title="Final Spectrum Comparison"):
    """Plot comparison of predicted and true spectra."""
    if true_spectrum is not None:
        if len(true_spectrum.shape) > 1:
            true_spectrum = true_spectrum.flatten()
        ax.plot(wavelengths, true_spectrum, label='Ground Truth',
                color='black', linewidth=2, linestyle='--')

    if lstm_spectrum is not None:
        if len(lstm_spectrum.shape) > 1:
            lstm_spectrum = lstm_spectrum.flatten()
        ax.plot(wavelengths, lstm_spectrum, label='TemporalLSTM',
                color='#1f77b4', linewidth=1.5)

    if temporal_spectrum is not None:
        if len(temporal_spectrum.shape) > 1:
            temporal_spectrum = temporal_spectrum.flatten()
        ax.plot(wavelengths, temporal_spectrum, label='TemporalMLP',
                color='#ff7f0e', linewidth=1.5)

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Log Power')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def create_evaluation_figure(lstm_temporal, lstm_final,
                             temporal_temporal, temporal_final,
                             true_spectrum, true_temporal,
                             wavelengths, z_positions,
                             sample_params,
                             lstm_metrics, temporal_metrics):
    """Create comprehensive evaluation figure."""
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: Heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    if lstm_temporal is not None:
        plot_heatmap(ax1, lstm_temporal, wavelengths, z_positions,
                     'TemporalLSTM Prediction', cmap='hot')

    ax2 = fig.add_subplot(gs[0, 1])
    if temporal_temporal is not None:
        plot_heatmap(ax2, temporal_temporal, wavelengths, z_positions,
                     'TemporalMLP Prediction', cmap='hot')

    ax3 = fig.add_subplot(gs[0, 2])
    if true_temporal is not None:
        plot_heatmap(ax3, true_temporal, wavelengths, z_positions,
                     'Ground Truth', cmap='hot')

    # Row 2: Difference heatmaps
    ax4 = fig.add_subplot(gs[1, 0])
    if lstm_temporal is not None and true_temporal is not None:
        diff = lstm_temporal - true_temporal
        plot_heatmap(ax4, diff, wavelengths, z_positions,
                     'LSTM - Ground Truth', cmap='RdBu_r')

    ax5 = fig.add_subplot(gs[1, 1])
    if temporal_temporal is not None and true_temporal is not None:
        diff = temporal_temporal - true_temporal
        plot_heatmap(ax5, diff, wavelengths, z_positions,
                     'TemporalMLP - Ground Truth', cmap='RdBu_r')

    ax6 = fig.add_subplot(gs[1, 2])
    if lstm_temporal is not None and temporal_temporal is not None:
        diff = lstm_temporal - temporal_temporal
        plot_heatmap(ax6, diff, wavelengths, z_positions,
                     'LSTM - TemporalMLP', cmap='RdBu_r')

    # Row 3: Spectra comparison
    ax7 = fig.add_subplot(gs[2, :])
    plot_spectrum_comparison(
        ax7, wavelengths, true_spectrum,
        lstm_final, temporal_final,
        title="Final Spectrum Comparison"
    )

    # Add metrics text
    metrics_text = "Model Performance:\n"
    if lstm_metrics:
        metrics_text += f"LSTM - MSE: {lstm_metrics.get('MSE', 'N/A'):.6f}, R²: {lstm_metrics.get('R2', 'N/A'):.4f}\n"
    if temporal_metrics:
        metrics_text += f"MLP  - MSE: {temporal_metrics.get('MSE', 'N/A'):.6f}, R²: {temporal_metrics.get('R2', 'N/A'):.4f}"

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Add sample parameters
    param_text = "Sample Parameters:\n"
    for key in ['energy', 'tau', 'pressure', 'diameter']:
        if key in sample_params:
            param_text += f"{key}: {sample_params[key]:.3f}  "

    fig.text(0.02, 0.98, param_text, ha='left', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
             transform=fig.transFigure)

    return fig


def save_evaluation_report(lstm_metrics, temporal_metrics,
                           sample_params, report_path):
    """Save evaluation report to text file."""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Model Evaluation Report\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sample Parameters:\n")
        for key, val in sample_params.items():
            if isinstance(val, (int, float)):
                f.write(f"  {key:15s}: {val:12.6f}\n")
            else:
                f.write(f"  {key:15s}: {str(val):>12s}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("TemporalLSTM Metrics:\n")
        for key, val in lstm_metrics.items():
            if isinstance(val, (int, float)):
                f.write(f"  {key:20s}: {val:12.6f}\n")
            else:
                f.write(f"  {key:20s}: {str(val):>12s}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("TemporalMLP Metrics:\n")
        for key, val in temporal_metrics.items():
            if isinstance(val, (int, float)):
                f.write(f"  {key:20s}: {val:12.6f}\n")
            else:
                f.write(f"  {key:20s}: {str(val):>12s}\n")

        f.write("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Model Evaluation: TemporalLSTM vs TemporalMLP")
    print("=" * 70)

    # Step 1: Extract sample parameters
    print("\n[Step 1] Extracting sample parameters from HDF5...")
    sample_params = extract_parameters_from_hdf5(SAMPLE_PATH)
    print(f"  Found {len(sample_params)} parameters")
    for key in ['energy', 'tau', 'pressure', 'diameter', 'beta2', 'gamma', 'N']:
        if key in sample_params:
            print(f"    {key}: {sample_params[key]:.4f}")

    # Step 2: Load ground truth
    print("\n[Step 2] Loading ground truth from HDF5...")
    true_temporal, true_spectrum, wavelengths, z_positions = load_ground_truth_from_hdf5(SAMPLE_PATH)
    if true_temporal is not None:
        print(f"  Temporal shape: {true_temporal.shape}")
        print(f"  Wavelength range: [{wavelengths.min():.1f}, {wavelengths.max():.1f}] nm")
        if z_positions is not None:
            print(f"  Z range: [{z_positions.min():.4f}, {z_positions.max():.4f}] m")
    else:
        print("  ⚠ Could not load ground truth temporal evolution")
        # Create default z_positions for prediction
        z_positions = np.linspace(0, 0.5, 20)
        wavelengths = create_wavelength_axis()

    # Step 3: Build input tensor
    print("\n[Step 3] Building input tensor...")

    # Try different feature configurations
    input_tensor_13 = build_input_tensor(sample_params, FEATURE_NAMES)
    print(f"  13-feature input shape: {input_tensor_13.shape}")

    input_tensor_15 = build_input_tensor(sample_params, FEATURE_NAMES_15)
    print(f"  15-feature input shape: {input_tensor_15.shape}")

    input_tensor_20 = build_input_tensor(sample_params, FEATURE_NAMES_20)
    print(f"  20-feature input shape: {input_tensor_20.shape}")

    # Step 4: Load LSTM model
    print("\n[Step 4] Loading TemporalLSTM model...")
    try:
        lstm_model, lstm_config, lstm_metrics, lstm_history = load_model_and_config(LSTM_DIR, 'lstm')
        print(f"  Model loaded: input_dim={lstm_config.get('input_dim', '?')}, output_dim={lstm_config.get('output_dim', '?')}")
        print(f"  Metrics: MSE={lstm_metrics.get('MSE', 'N/A')}, R²={lstm_metrics.get('R2', 'N/A')}")

        # Determine correct input tensor based on model's expected input_dim
        lstm_input_dim = lstm_config.get('input_dim', input_tensor_15.shape[1])
        if lstm_input_dim == 20:
            lstm_input = input_tensor_20
        elif lstm_input_dim == 15:
            lstm_input = input_tensor_15
        else:
            lstm_input = input_tensor_13

        # Pad if needed
        if lstm_input.shape[1] != lstm_input_dim:
            lstm_input = build_input_tensor(sample_params, FEATURE_NAMES_15, target_dim=lstm_input_dim)

        print(f"  Using input shape: {lstm_input.shape}")

        # Create z_positions tensor
        z_tensor = torch.FloatTensor(z_positions).unsqueeze(0)
        if z_tensor.shape[1] != 20:
            # Interpolate to 20 steps
            z_old = np.linspace(0, 1, z_tensor.shape[1])
            z_new = np.linspace(0, 1, 20)
            z_interp = np.interp(z_new, z_old, z_positions)
            z_tensor = torch.FloatTensor(z_interp).unsqueeze(0)

        print(f"  Z positions shape: {z_tensor.shape}")

        print("\n[Step 5] Running TemporalLSTM prediction...")
        lstm_temporal, lstm_final = predict_with_model(
            lstm_model, lstm_input, z_tensor, 'lstm'
        )
        print(f"  Temporal output shape: {lstm_temporal.shape if lstm_temporal is not None else 'None'}")
        print(f"  Final output shape: {lstm_final.shape}")
        print(f"  Final output range: [{lstm_final.min():.4f}, {lstm_final.max():.4f}]")
    except Exception as e:
        print(f"  ERROR loading/running LSTM model: {e}")
        import traceback
        traceback.print_exc()
        lstm_temporal, lstm_final = None, None
        lstm_metrics = {}

    # Step 5: Load TemporalMLP model
    print("\n[Step 6] Loading TemporalMLP model...")
    try:
        temporal_model, temporal_config, temporal_metrics, temporal_history = load_model_and_config(TEMPORAL_DIR, 'temporal')
        print(f"  Model loaded: input_dim={temporal_config.get('input_dim', '?')}, output_dim={temporal_config.get('output_dim', '?')}")
        print(f"  Metrics: MSE={temporal_metrics.get('MSE', 'N/A')}, R²={temporal_metrics.get('R2', 'N/A')}")

        # Determine correct input tensor based on model's expected input_dim
        temporal_input_dim = temporal_config.get('input_dim', input_tensor_15.shape[1])
        if temporal_input_dim == 20:
            temporal_input = input_tensor_20
        elif temporal_input_dim == 15:
            temporal_input = input_tensor_15
        else:
            temporal_input = input_tensor_13

        # Pad if needed
        if temporal_input.shape[1] != temporal_input_dim:
            temporal_input = build_input_tensor(sample_params, FEATURE_NAMES_15, target_dim=temporal_input_dim)

        print(f"  Using input shape: {temporal_input.shape}")

        # Build input tensor with correct dimension
        print(f"  TemporalMLP input shape: {temporal_input.shape}")

        print("\n[Step 7] Running TemporalMLP prediction...")
        temporal_temporal, temporal_final = predict_with_model(
            temporal_model, temporal_input, z_tensor, 'temporal'
        )
        print(f"  Temporal output shape: {temporal_temporal.shape if temporal_temporal is not None else 'None'}")
        print(f"  Final output shape: {temporal_final.shape}")
        print(f"  Final output range: [{temporal_final.min():.4f}, {temporal_final.max():.4f}]")
    except Exception as e:
        print(f"  ERROR loading/running TemporalMLP model: {e}")
        import traceback
        traceback.print_exc()
        temporal_temporal, temporal_final = None, None
        temporal_metrics = {}

    # Step 6: Generate visualizations
    print("\n[Step 8] Generating visualizations...")

    # Main comparison figure
    fig = create_evaluation_figure(
        lstm_temporal, lstm_final,
        temporal_temporal, temporal_final,
        true_spectrum, true_temporal,
        wavelengths, z_positions,
        sample_params,
        lstm_metrics, temporal_metrics
    )

    heatmap_path = os.path.join(OUTPUT_DIR, "model_comparison_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  Saved heatmap comparison: {heatmap_path}")
    plt.close(fig)

    # Spectrum-only comparison figure
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    plot_spectrum_comparison(
        ax2, wavelengths, true_spectrum,
        lstm_final, temporal_final,
        title="Final Spectrum Comparison: TemporalLSTM vs TemporalMLP"
    )

    spectrum_path = os.path.join(OUTPUT_DIR, "model_comparison_spectrum.png")
    fig2.savefig(spectrum_path, dpi=300, bbox_inches='tight',
                 facecolor='white', edgecolor='none')
    print(f"  Saved spectrum comparison: {spectrum_path}")
    plt.close(fig2)

    # Step 7: Save evaluation report
    print("\n[Step 9] Saving evaluation report...")
    report_path = os.path.join(OUTPUT_DIR, "model_evaluation_report.txt")
    save_evaluation_report(
        lstm_metrics, temporal_metrics,
        sample_params, report_path
    )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - model_comparison_heatmap.png")
    print(f"  - model_comparison_spectrum.png")
    print(f"  - model_evaluation_report.txt")


if __name__ == '__main__':
    main()
