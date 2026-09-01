#!/usr/bin/env python3
"""
Model Training Script v4.0 - Multi-thickness Anti-resonant Fiber

Features:
  - Comparative experiment framework (MLP / TemporalMLP / TemporalLSTM)
  - Comprehensive logging and metric tracking
  - Visualization of training curves and predictions
  - Support for thickness-stratified evaluation

Usage:
  python train_mlp.py --model lstm --input-dir processed_data --output-dir models
  python train_mlp.py --model mlp --epochs 300 --batch-size 128
  python train_mlp.py --compare-all  # Run all architectures and compare
"""

import os
import sys
import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import json
import logging
import time
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRAIN_WAVELENGTH_RANGE_NM = (200.0, 2500.0)
LUNA_STYLE_DB_MIN = -40.0
LUNA_STYLE_CMAP = 'viridis'

# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Train anti-resonant fiber pulse propagation prediction models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_mlp.py --model lstm --epochs 500
  python train_mlp.py --compare-all --output-dir comparison_results
  python train_mlp.py --model temporal --lr 0.0005 --batch-size 128
        """
    )
    
    # Model selection
    parser.add_argument('--model', '-m', type=str, default='lstm',
                        choices=['mlp', 'temporal', 'lstm', 'transformer', 'linear', 'shallow', 'banded_mlp', 'temporal_cnn'],
                        help='Model architecture (default: lstm)')
    parser.add_argument('--compare-all', action='store_true',
                        help='Run all architectures and generate comparison report')
    
    # Data paths
    parser.add_argument('--input-dir', '-i', type=str, default='processed_data',
                        help='Input data directory')
    parser.add_argument('--output-dir', '-o', type=str, default='models',
                        help='Output directory for models and results')
    parser.add_argument('--eval-only', action='store_true',
                        help='Load --checkpoint and evaluate the input directory test split without training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint path used with --eval-only')
    
    # Training parameters
    parser.add_argument('--epochs', '-e', type=int, default=500,
                        help='Training epochs (default: 500)')
    parser.add_argument('--batch-size', '-b', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3,
                        help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='L2 regularization (default: 1e-5)')
    parser.add_argument('--selection-metric', type=str, default='val_loss',
                        choices=[
                            'val_loss', 'val_r2', 'val_uv_r2',
                            'val_temporal_r2', 'val_temporal_uv_r2',
                            'val_early_z_r2', 'val_early_uv_r2'
                        ],
                        help='Checkpoint restored for final metrics/plots (default: val_loss)')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Final loss weight (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.01,
                        help='Smoothness loss weight (default: 0.01)')
    parser.add_argument('--spectral-gradient-weight', type=float, default=0.0,
                        help='Weight for spectral-gradient matching loss (default: 0.0)')
    parser.add_argument('--short-wavelength-weight', type=float, default=1.0,
                        help='Extra loss weight for wavelengths below cutoff (default: 1.0, disabled)')
    parser.add_argument('--short-wavelength-cutoff-nm', type=float, default=700.0,
                        help='Cutoff wavelength for short-wavelength loss weighting (default: 700 nm)')
    parser.add_argument('--uv-min-nm', type=float, default=200.0,
                        help='UV band lower bound for metrics/loss in nm (default: 200)')
    parser.add_argument('--uv-max-nm', type=float, default=700.0,
                        help='UV band upper bound for metrics/loss in nm (default: 700)')
    parser.add_argument('--uv-loss-weight', type=float, default=0.0,
                        help='Extra MSE loss weight inside UV band (default: 0.0)')
    parser.add_argument('--uv-gradient-weight', type=float, default=0.0,
                        help='Extra first-derivative loss weight inside UV band (default: 0.0)')
    parser.add_argument('--uv-r2-weight', type=float, default=0.0,
                        help='Stable batch UV R2-style loss weight (default: 0.0)')
    parser.add_argument('--num-bands', type=int, default=4,
                        help='Number of wavelength submodels for banded_mlp (default: 4)')
    parser.add_argument('--band-boundaries-nm', type=float, nargs='+', default=None,
                        help='Custom wavelength band edges for banded_mlp, e.g. 200 700 1200 1800 2500')
    parser.add_argument('--band-head-hidden', type=int, default=128,
                        help='Hidden width for regular banded_mlp heads (default: 128)')
    parser.add_argument('--uv-band-head-hidden', type=int, default=256,
                        help='Hidden width for the first/UV banded_mlp head when custom bands are used (default: 256)')
    parser.add_argument('--band-boundary-weight', type=float, default=0.0,
                        help='Continuity loss weight at banded_mlp wavelength boundaries (default: 0.0)')
    parser.add_argument('--uv-second-derivative-weight', type=float, default=0.0,
                        help='Weight for UV second-derivative matching loss (default: 0.0)')
    parser.add_argument('--long-wavelength-weight', type=float, default=1.0,
                        help='Extra loss weight for wavelengths above long cutoff (default: 1.0, disabled)')
    parser.add_argument('--long-wavelength-cutoff-nm', type=float, default=1350.0,
                        help='Cutoff wavelength for long-wavelength loss weighting (default: 1350 nm)')
    parser.add_argument('--z-gradient-weight', type=float, default=0.0,
                        help='Weight for temporal z-gradient matching loss (default: 0.0)')
    parser.add_argument('--z-gradient-coordinate', type=str, default='cm',
                        choices=['index', 'cm', 'm'],
                        help='Coordinate scale for z-gradient losses: index, per cm, or per m (default: cm)')
    parser.add_argument('--lambda-gradient-weight', type=float, default=0.0,
                        help='Weight for wavelength-gradient matching loss on temporal output (default: 0.0)')
    parser.add_argument('--temporal-late-weight', type=float, default=1.0,
                        help='Extra temporal loss weight for second half of propagation (default: 1.0, disabled)')
    parser.add_argument('--early-z-loss-weight', type=float, default=0.0,
                        help='Extra temporal MSE weight for early propagation z <= early-z-max-cm (default: 0.0)')
    parser.add_argument('--early-z-max-cm', type=float, default=2.0,
                        help='Early propagation range for extra temporal loss in cm (default: 2.0)')
    parser.add_argument('--early-z-gradient-weight', type=float, default=0.0,
                        help='Extra z-gradient loss weight inside early propagation range (default: 0.0)')
    parser.add_argument('--temporal-cnn-channels', type=int, default=128,
                        help='Base channel count for temporal_cnn decoder (default: 128)')
    parser.add_argument('--temporal-cnn-lambda-init', type=int, default=64,
                        help='Initial low-resolution wavelength bins for temporal_cnn (default: 64)')
    parser.add_argument('--temporal-cnn-z-init', type=int, default=16,
                        help='Initial low-resolution z bins for temporal_cnn (default: 16)')
    parser.add_argument('--temporal-cnn-architecture', type=str, default='basic',
                        choices=['basic', 'multiscale'],
                        help='Temporal CNN architecture: basic keeps checkpoint compatibility; multiscale adds band heads')
    parser.add_argument('--temporal-cnn-band-boundaries-nm', type=float, nargs='+',
                        default=[200.0, 700.0, 1200.0, 1800.0, 2500.0],
                        help='Band edges for multiscale temporal_cnn heads')
    parser.add_argument('--temporal-cnn-coordinate-channels', action='store_true',
                        help='Add normalized z/lambda coordinate channels to multiscale temporal_cnn heads')
    parser.add_argument('--transformer-mode', type=str, default='final',
                        choices=['final', 'temporal'],
                        help='Transformer task mode: final spectrum or full temporal evolution (default: final)')
    parser.add_argument('--transformer-d-model', type=int, default=128,
                        help='Transformer hidden dimension (default: 128)')
    parser.add_argument('--transformer-heads', type=int, default=4,
                        help='Transformer attention heads (default: 4)')
    parser.add_argument('--transformer-layers', type=int, default=3,
                        help='Transformer encoder layers (default: 3)')
    parser.add_argument('--transformer-use-z-embedding', action='store_true',
                        help='Use explicit z-position embedding in temporal transformer')
    parser.add_argument('--transformer-band-boundaries-nm', type=float, nargs='+', default=None,
                        help='Optional wavelength band edges for temporal transformer decoder heads')
    parser.add_argument('--output-activation', type=str, default='auto',
                        choices=['auto', 'sigmoid', 'identity'],
                        help='Output activation. auto uses sigmoid for per_sample_minmax and identity for log-power data')
    parser.add_argument('--log-integral-weight', type=float, default=0.0,
                        help='Optional loss weight for matching linear-power spectral integral in log-power mode')
    parser.add_argument('--log-peak-weight', type=float, default=0.0,
                        help='Optional loss weight for matching log-power peak in log-power mode')
    
    # Other
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable GPU')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='DataLoader worker processes (default: 0)')
    parser.add_argument('--pin-memory', action='store_true',
                        help='Enable DataLoader pinned memory')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--visualize-indices', type=int, nargs='*', default=None,
                        help='Specific test split indices to show in predictions.png')
    parser.add_argument('--finetune-from', type=str, default=None,
                        help='Load model weights from an existing .pth file before training')
    parser.add_argument('--freeze-shared', action='store_true',
                        help='Freeze BandedMLP.shared during fine-tuning')
    parser.add_argument('--freeze-non-uv-heads', action='store_true',
                        help='For BandedMLP fine-tuning, train only the first/UV band head')
    parser.add_argument('--finetune-strict', type=str, default='true',
                        choices=['true', 'false'],
                        help='Use strict state_dict loading for --finetune-from (default: true)')
    
    return parser.parse_args()

def set_random_seed(seed):
    """Set all relevant random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """Seed NumPy/Python RNGs inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_processing_params(input_dir):
    """Load preprocessing metadata when available."""
    params_path = os.path.join(input_dir, 'processing_params.json')
    if not os.path.exists(params_path):
        return {}
    try:
        with open(params_path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        logging.warning(f"Failed to read processing params: {exc}")
        return {}

def get_wavelength_range_nm(processing_params, default=TRAIN_WAVELENGTH_RANGE_NM):
    """Return wavelength range for plots in nanometers."""
    wl_range = processing_params.get('wavelength_range_nm')
    if wl_range is not None and len(wl_range) == 2:
        return (float(wl_range[0]), float(wl_range[1]))
    wl_range_m = processing_params.get('wavelength_range_m')
    if wl_range_m is not None and len(wl_range_m) == 2:
        return (float(wl_range_m[0]) * 1e9, float(wl_range_m[1]) * 1e9)
    return default

def is_expected_training_range(wavelength_range, expected=TRAIN_WAVELENGTH_RANGE_NM, atol=1e-6):
    """Check whether processed spectra use the intended ML training wavelength range."""
    return (abs(float(wavelength_range[0]) - expected[0]) <= atol and
            abs(float(wavelength_range[1]) - expected[1]) <= atol)

def safe_r2_score(y_true, y_pred, eps=1e-12):
    """Global R2 with finite-value filtering and near-constant-target guard."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    dropped = y_true.size - int(np.sum(mask))
    if dropped > 0:
        logging.warning(f"R2 calculation dropped {dropped} non-finite points")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 2:
        return float('nan')
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if not np.isfinite(ss_res) or not np.isfinite(ss_tot) or ss_tot <= eps:
        logging.warning("R2 calculation skipped because target variance is too small or non-finite")
        return float('nan')
    return float(1.0 - ss_res / ss_tot)

def finite_metric_arrays(y_true, y_pred, name):
    """Return finite flattened arrays for scalar regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    dropped = y_true.size - int(np.sum(mask))
    if dropped > 0:
        logging.warning(f"{name} metrics dropped {dropped} non-finite points")
    return y_true[mask], y_pred[mask]

def resolve_output_activation(requested, spectrum_normalization):
    """Resolve auto output activation from preprocessing normalization."""
    if requested != 'auto':
        return requested
    if spectrum_normalization == 'per_sample_minmax':
        return 'sigmoid'
    return 'identity'

def apply_output_activation(values, mode):
    """Apply model output activation."""
    if mode == 'sigmoid':
        return torch.sigmoid(values)
    if mode == 'identity':
        return values
    raise ValueError(f"Unsupported output activation: {mode}")

class OutputActivation(nn.Module):
    """Small module wrapper so Sequential heads can use configurable activation."""
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def forward(self, values):
        return apply_output_activation(values, self.mode)

def inverse_spectrum_normalization(values, output_normalization):
    """Return log10(power)-space spectra when preprocessing retained absolute scale."""
    if output_normalization is None:
        return None
    mode = output_normalization.get('mode')
    values = np.asarray(values, dtype=np.float64)
    if mode == 'global_log_standard':
        log_mean = float(output_normalization.get('log_mean', 0.0))
        log_std = float(output_normalization.get('log_std', 1.0))
        return values * log_std + log_mean
    if mode == 'none_log':
        return values
    return None

def relative_db_from_log_power(log_power, dBmin=LUNA_STYLE_DB_MIN):
    """Convert log10(power) spectra to per-sample relative dB display."""
    log_power = np.asarray(log_power, dtype=np.float64)
    peak = np.nanmax(log_power, axis=-1, keepdims=True)
    rel_db = 10.0 * (log_power - peak)
    return np.clip(rel_db, dBmin, 0.0)

def add_log_power_metrics(metrics, y_true, y_pred, output_normalization,
                          wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                          uv_min_nm=200.0, uv_max_nm=700.0,
                          prefix=''):
    """Add metrics in inverse-normalized log10(power) space when available."""
    true_log = inverse_spectrum_normalization(y_true, output_normalization)
    pred_log = inverse_spectrum_normalization(y_pred, output_normalization)
    if true_log is None or pred_log is None:
        return metrics

    name = f'{prefix}LogPower' if prefix else 'LogPower'
    metric_targets, metric_preds = finite_metric_arrays(true_log, pred_log, name)
    if metric_targets.size > 0:
        metrics[f'{name}_RMSE'] = float(np.sqrt(mean_squared_error(metric_targets, metric_preds)))
        metrics[f'{name}_MAE'] = float(mean_absolute_error(metric_targets, metric_preds))
        metrics[f'{name}_R2'] = safe_r2_score(true_log, pred_log)

    uv_mask = uv_mask_for_output(true_log.shape[-1], wavelength_range, uv_min_nm, uv_max_nm)
    if np.any(uv_mask):
        uv_true = true_log[..., uv_mask]
        uv_pred = pred_log[..., uv_mask]
        uv_targets, uv_preds = finite_metric_arrays(uv_true, uv_pred, f'{prefix}UV_LogPower')
        if uv_targets.size > 0:
            uv_prefix = f'{prefix}UV_LogPower' if prefix else 'UV_LogPower'
            metrics[f'{uv_prefix}_RMSE'] = float(np.sqrt(mean_squared_error(uv_targets, uv_preds)))
            metrics[f'{uv_prefix}_R2'] = safe_r2_score(uv_true, uv_pred)
    return metrics

def wavelength_axis(output_dim, wavelength_range=TRAIN_WAVELENGTH_RANGE_NM):
    """Return the wavelength axis used by the processed spectra."""
    return np.linspace(float(wavelength_range[0]), float(wavelength_range[1]), int(output_dim))

def uv_mask_for_output(output_dim, wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                       uv_min_nm=200.0, uv_max_nm=700.0):
    """Boolean mask for the UV band on the model output axis."""
    wavelengths = wavelength_axis(output_dim, wavelength_range)
    return (wavelengths >= float(uv_min_nm)) & (wavelengths <= float(uv_max_nm))

def make_band_indices(output_dim, wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                      num_bands=4, band_boundaries_nm=None):
    """Return contiguous wavelength band index arrays."""
    output_dim = int(output_dim)
    if band_boundaries_nm is None:
        return [np.asarray(idx, dtype=int) for idx in np.array_split(np.arange(output_dim), max(1, int(num_bands)))]

    boundaries = [float(v) for v in band_boundaries_nm]
    if len(boundaries) < 2:
        raise ValueError("--band-boundaries-nm must contain at least two values")
    if any(boundaries[i] >= boundaries[i + 1] for i in range(len(boundaries) - 1)):
        raise ValueError("--band-boundaries-nm must be strictly increasing")

    wl_min, wl_max = float(wavelength_range[0]), float(wavelength_range[1])
    if boundaries[0] > wl_min + 1e-6 or boundaries[-1] < wl_max - 1e-6:
        raise ValueError(
            f"--band-boundaries-nm must cover the full training range {wl_min:g}-{wl_max:g} nm"
        )

    wavelengths = wavelength_axis(output_dim, wavelength_range)
    edge_indices = [0]
    for boundary in boundaries[1:-1]:
        edge_indices.append(int(np.searchsorted(wavelengths, boundary, side='left')))
    edge_indices.append(output_dim)

    if any(edge_indices[i] >= edge_indices[i + 1] for i in range(len(edge_indices) - 1)):
        raise ValueError("Band boundaries produced an empty wavelength band")
    return [np.arange(edge_indices[i], edge_indices[i + 1], dtype=int)
            for i in range(len(edge_indices) - 1)]

def band_metrics(y_true, y_pred, mask, prefix):
    """Compute scalar metrics for a wavelength band."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or y_pred.size == 0 or mask is None or not np.any(mask):
        return {
            f'{prefix}_MSE': float('nan'),
            f'{prefix}_RMSE': float('nan'),
            f'{prefix}_MAE': float('nan'),
            f'{prefix}_R2': float('nan'),
        }
    band_true = y_true[..., mask]
    band_pred = y_pred[..., mask]
    metric_targets, metric_preds = finite_metric_arrays(band_true, band_pred, prefix)
    if metric_targets.size == 0:
        return {
            f'{prefix}_MSE': float('nan'),
            f'{prefix}_RMSE': float('nan'),
            f'{prefix}_MAE': float('nan'),
            f'{prefix}_R2': float('nan'),
        }
    mse = mean_squared_error(metric_targets, metric_preds)
    return {
        f'{prefix}_MSE': float(mse),
        f'{prefix}_RMSE': float(np.sqrt(mse)),
        f'{prefix}_MAE': float(mean_absolute_error(metric_targets, metric_preds)),
        f'{prefix}_R2': safe_r2_score(band_true, band_pred),
    }

def wavelength_band_metrics(y_true, y_pred, output_dim, wavelength_range, num_bands=4, band_indices=None):
    """Compute metrics for equal-width contiguous wavelength bands."""
    metrics = {}
    wavelengths = wavelength_axis(output_dim, wavelength_range)
    if band_indices is None:
        band_indices = np.array_split(np.arange(output_dim), max(1, int(num_bands)))
    for band_idx, indices in enumerate(band_indices, start=1):
        if len(indices) == 0:
            continue
        mask = np.zeros(output_dim, dtype=bool)
        mask[indices] = True
        prefix = f'Band{band_idx}'
        metrics[f'{prefix}_Lambda_Min_Nm'] = float(wavelengths[indices[0]])
        metrics[f'{prefix}_Lambda_Max_Nm'] = float(wavelengths[indices[-1]])
        metrics.update(band_metrics(y_true, y_pred, mask, prefix))
    return metrics

def gradient_mse_np(y_true, y_pred, axis, name):
    """MSE between finite first differences along a chosen axis."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0 or y_true.shape[axis] < 2:
        return float('nan')
    true_grad = np.diff(y_true, axis=axis)
    pred_grad = np.diff(y_pred, axis=axis)
    metric_targets, metric_preds = finite_metric_arrays(true_grad, pred_grad, name)
    if metric_targets.size == 0:
        return float('nan')
    return float(mean_squared_error(metric_targets, metric_preds))

def dtw_distance_1d(a, b, window=None):
    """Simple DTW distance for 1D arrays, used as an offline diagnostic metric."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.any(finite):
        return float('nan')
    a = a[finite]
    b = b[finite]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float('nan')
    if window is None:
        window = max(n, m)
    window = max(int(window), abs(n - m))
    prev = np.full(m + 1, np.inf)
    curr = np.full(m + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[:] = np.inf
        j_start = max(1, i - window)
        j_end = min(m, i + window)
        for j in range(j_start, j_end + 1):
            cost = abs(a[i - 1] - b[j - 1])
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / (n + m))

def uv_shape_metrics(y_true, y_pred, mask, max_samples=64):
    """Compute UV DTW/DDTW diagnostics on a bounded sample subset."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim < 2 or y_true.size == 0 or mask is None or not np.any(mask):
        return {'UV_DTW': float('nan'), 'UV_DDTW': float('nan')}
    true_uv = y_true[..., mask].reshape(-1, int(np.sum(mask)))
    pred_uv = y_pred[..., mask].reshape(-1, int(np.sum(mask)))
    n = min(max_samples, true_uv.shape[0])
    if n == 0:
        return {'UV_DTW': float('nan'), 'UV_DDTW': float('nan')}
    dtw_vals = []
    ddtw_vals = []
    window = max(5, int(true_uv.shape[1] * 0.1))
    for i in range(n):
        dtw_vals.append(dtw_distance_1d(true_uv[i], pred_uv[i], window=window))
        ddtw_vals.append(dtw_distance_1d(np.diff(true_uv[i]), np.diff(pred_uv[i]), window=window))
    return {
        'UV_DTW': float(np.nanmean(dtw_vals)),
        'UV_DDTW': float(np.nanmean(ddtw_vals)),
    }

def count_parameters(model):
    """Count trainable model parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def load_finetune_weights(model, checkpoint_path, device='cpu', strict=True):
    """Load model weights for fine-tuning, optionally skipping shape mismatches."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return {'loaded': len(state_dict), 'skipped': []}

    current_state = model.state_dict()
    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in current_state and current_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)
    current_state.update(compatible)
    model.load_state_dict(current_state, strict=True)
    return {'loaded': len(compatible), 'skipped': skipped}

def split_stats(name, X, y, uv_mask):
    """Compact distribution summary for one split."""
    return {
        f'{name}_n': int(X.shape[0]),
        f'{name}_X_mean': np.mean(X, axis=0).astype(float).tolist(),
        f'{name}_X_std': np.std(X, axis=0).astype(float).tolist(),
        f'{name}_X_min': np.min(X, axis=0).astype(float).tolist(),
        f'{name}_X_max': np.max(X, axis=0).astype(float).tolist(),
        f'{name}_y_mean': float(np.mean(y)),
        f'{name}_y_std': float(np.std(y)),
        f'{name}_y_uv_mean': float(np.mean(y[..., uv_mask])) if np.any(uv_mask) else float('nan'),
        f'{name}_y_uv_std': float(np.std(y[..., uv_mask])) if np.any(uv_mask) else float('nan'),
    }

def hash_rows(array):
    """Exact row hashes for split leakage checks."""
    arr = np.ascontiguousarray(array)
    return {hash(arr[i].tobytes()) for i in range(arr.shape[0])}

def nearest_distance_summary(reference, query, max_query=256):
    """Nearest-neighbor distance summary from query rows to reference rows."""
    if reference.size == 0 or query.size == 0:
        return {}
    q = query[:min(max_query, query.shape[0])]
    distances = []
    for row in q:
        d = np.sqrt(np.sum((reference - row) ** 2, axis=1))
        distances.append(float(np.min(d)))
    distances = np.asarray(distances)
    return {
        'min': float(np.min(distances)),
        'median': float(np.median(distances)),
        'mean': float(np.mean(distances)),
        'max': float(np.max(distances)),
        'n_query': int(q.shape[0]),
    }

def nearest_label_r2_summary(reference_y, query_y, max_query=64):
    """How well the nearest training label can match validation/test labels."""
    if reference_y.size == 0 or query_y.size == 0:
        return {}
    q = query_y[:min(max_query, query_y.shape[0])]
    scores = []
    for row in q:
        d = np.mean((reference_y - row) ** 2, axis=1)
        nearest = reference_y[int(np.argmin(d))]
        scores.append(safe_r2_score(row, nearest))
    scores = np.asarray(scores, dtype=np.float64)
    return {
        'median_r2': float(np.nanmedian(scores)),
        'mean_r2': float(np.nanmean(scores)),
        'max_r2': float(np.nanmax(scores)),
        'n_query': int(q.shape[0]),
    }

def write_split_distribution_report(output_dir, X_train, X_val, X_test, y_train, y_val, y_test,
                                    wavelength_range, uv_min_nm, uv_max_nm):
    """Write diagnostics for val/test curve similarity and leakage checks."""
    uv_mask = uv_mask_for_output(y_train.shape[-1], wavelength_range, uv_min_nm, uv_max_nm)
    train_hash = hash_rows(X_train)
    val_hash = hash_rows(X_val)
    test_hash = hash_rows(X_test)
    report = {
        'uv_range_nm': [float(uv_min_nm), float(uv_max_nm)],
        'wavelength_range_nm': [float(wavelength_range[0]), float(wavelength_range[1])],
        'duplicate_X_hash_counts': {
            'train_val': int(len(train_hash & val_hash)),
            'train_test': int(len(train_hash & test_hash)),
            'val_test': int(len(val_hash & test_hash)),
        },
        'nearest_X_distance_to_train': {
            'val': nearest_distance_summary(X_train, X_val),
            'test': nearest_distance_summary(X_train, X_test),
        },
        'nearest_label_r2_to_train': {
            'val': nearest_label_r2_summary(y_train, y_val),
            'test': nearest_label_r2_summary(y_train, y_test),
        },
    }
    report.update(split_stats('train', X_train, y_train, uv_mask))
    report.update(split_stats('val', X_val, y_val, uv_mask))
    report.update(split_stats('test', X_test, y_test, uv_mask))
    with open(os.path.join(output_dir, 'split_distribution_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    return report

def normalized_to_luna_db(values, dBmin=LUNA_STYLE_DB_MIN):
    """Map normalized [0, 1] spectra to Luna-like dB color scale."""
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    return dBmin + (0.0 - dBmin) * values

# ============================================================================
# Model Architectures
# ============================================================================

class StandardMLP(nn.Module):
    """Standard MLP - backward compatible, predicts final spectrum only.
    
    Output constrained to [0, 1] via Sigmoid for normalized log power spectra.
    """
    def __init__(self, input_dim, output_dim, output_activation='sigmoid'):
        super(StandardMLP, self).__init__()
        self.layers = nn.Sequential(
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
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim),
            OutputActivation(output_activation)
        )
    
    def forward(self, x, z_positions=None):
        return None, self.layers(x)

class LinearBaseline(nn.Module):
    """Single linear layer baseline for final-spectrum prediction."""
    def __init__(self, input_dim, output_dim, output_activation='sigmoid'):
        super(LinearBaseline, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            OutputActivation(output_activation)
        )

    def forward(self, x, z_positions=None):
        return None, self.layers(x)

class ShallowMLP(nn.Module):
    """Low-capacity nonlinear baseline for final-spectrum prediction."""
    def __init__(self, input_dim, output_dim, hidden_dim=64, output_activation='sigmoid'):
        super(ShallowMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, output_dim),
            OutputActivation(output_activation)
        )

    def forward(self, x, z_positions=None):
        return None, self.layers(x)

class SpectrumTransformer(nn.Module):
    """Lightweight transformer baseline for final-spectrum prediction.

    The transformer encodes input-feature tokens, then a wavelength-conditioned
    decoder predicts each spectrum point. This avoids full 1000x1000 wavelength
    self-attention while still testing a transformer-style architecture.
    """
    def __init__(self, input_dim, output_dim, d_model=128, nhead=4, num_layers=2,
                 output_activation='sigmoid'):
        super(SpectrumTransformer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.feature_projection = nn.Linear(1, d_model)
        self.feature_position = nn.Parameter(torch.zeros(1, input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.wavelength_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
        self.register_buffer('wavelength_tokens', torch.linspace(0.0, 1.0, output_dim).view(1, output_dim, 1))

    def forward(self, x, z_positions=None):
        tokens = self.feature_projection(x.unsqueeze(-1)) + self.feature_position
        encoded = self.encoder(tokens)
        context = encoded.mean(dim=1, keepdim=True).expand(-1, self.output_dim, -1)
        wl = self.wavelength_projection(self.wavelength_tokens.expand(x.size(0), -1, -1))
        pred = apply_output_activation(
            self.decoder(torch.cat([context, wl], dim=-1)).squeeze(-1),
            self.output_activation
        )
        return None, pred

class TemporalSpectrumTransformer(nn.Module):
    """Factorized transformer for full z-wavelength propagation prediction.

    Parameter features are encoded once, z tokens model propagation dependence,
    and a lightweight wavelength-conditioned decoder emits each spectrum.
    """
    def __init__(self, input_dim, output_dim, n_z_steps=20,
                 d_model=128, nhead=4, num_layers=3, use_z_embedding=False,
                 output_activation='sigmoid',
                 wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                 band_boundaries_nm=None):
        super(TemporalSpectrumTransformer, self).__init__()
        self.output_dim = int(output_dim)
        self.n_z_steps = int(n_z_steps)
        self.d_model = int(d_model)
        self.use_z_embedding = bool(use_z_embedding)
        self.output_activation = output_activation
        self.band_slices = None
        if band_boundaries_nm is not None:
            band_indices = make_band_indices(
                self.output_dim, wavelength_range=wavelength_range,
                num_bands=4, band_boundaries_nm=band_boundaries_nm
            )
            self.band_slices = [(int(idx[0]), int(idx[-1]) + 1) for idx in band_indices if len(idx) > 0]

        self.param_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.z_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.step_position = nn.Parameter(torch.zeros(1, self.n_z_steps, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, activation='gelu'
        )
        self.z_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.wavelength_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2 * d_model, 1)
        )
        if self.band_slices is not None:
            self.band_decoders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * d_model, 2 * d_model),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(2 * d_model, 1)
                )
                for _ in self.band_slices
            ])
        else:
            self.band_decoders = None
        self.register_buffer(
            'wavelength_tokens',
            torch.linspace(0.0, 1.0, self.output_dim).view(1, 1, self.output_dim, 1)
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        param = self.param_encoder(x).unsqueeze(1)

        if z_positions is None:
            z_norm = torch.linspace(0.0, 1.0, self.n_z_steps, device=x.device, dtype=x.dtype)
            z_norm = z_norm.view(1, self.n_z_steps, 1).expand(batch_size, -1, -1)
        else:
            z = z_positions.to(device=x.device, dtype=x.dtype)
            if z.dim() == 1:
                z = z.view(1, -1).expand(batch_size, -1)
            z_max = torch.clamp(torch.max(z, dim=1, keepdim=True).values, min=1e-8)
            z_norm = (z / z_max).unsqueeze(-1)
            if z_norm.size(1) != self.n_z_steps:
                z_norm = F.interpolate(
                    z_norm.transpose(1, 2), size=self.n_z_steps,
                    mode='linear', align_corners=False
                ).transpose(1, 2)

        z_tokens = param + self.z_projection(z_norm)
        if self.use_z_embedding:
            z_tokens = z_tokens + self.step_position[:, :z_tokens.size(1), :]
        z_context = self.z_encoder(z_tokens)

        wl = self.wavelength_projection(
            self.wavelength_tokens.expand(batch_size, self.n_z_steps, -1, -1)
        )
        z_context = z_context.unsqueeze(2).expand(-1, -1, self.output_dim, -1)
        decoder_input = torch.cat([z_context, wl], dim=-1)
        if self.band_decoders is None:
            pred = self.decoder(decoder_input).squeeze(-1)
        else:
            bands = []
            for decoder, (start, end) in zip(self.band_decoders, self.band_slices):
                bands.append(decoder(decoder_input[:, :, start:end, :]).squeeze(-1))
            pred = torch.cat(bands, dim=-1)
        temporal_output = apply_output_activation(pred, self.output_activation)
        final_output = temporal_output[:, -1, :]
        return temporal_output, final_output

class BandedMLP(nn.Module):
    """Independent MLP heads for contiguous wavelength bands."""
    def __init__(self, input_dim, output_dim, num_bands=4,
                 wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                 band_boundaries_nm=None, band_head_hidden=128,
                 uv_band_head_hidden=256, output_activation='sigmoid'):
        super(BandedMLP, self).__init__()
        self.output_dim = output_dim
        indices = make_band_indices(
            output_dim, wavelength_range=wavelength_range,
            num_bands=num_bands, band_boundaries_nm=band_boundaries_nm
        )
        self.num_bands = len(indices)
        self.band_indices = [np.asarray(idx, dtype=int) for idx in indices]
        self.band_boundaries = [int(idx[0]) for idx in self.band_indices[1:] if len(idx) > 0]
        self.band_sizes = [len(idx) for idx in indices]
        self.band_boundaries_nm = list(band_boundaries_nm) if band_boundaries_nm is not None else None
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
        )
        heads = []
        for band_idx, band_size in enumerate(self.band_sizes):
            hidden = int(uv_band_head_hidden) if band_boundaries_nm is not None and band_idx == 0 else int(band_head_hidden)
            heads.append(nn.Sequential(
                nn.Linear(256, hidden),
                nn.LeakyReLU(0.01),
                nn.Linear(hidden, band_size),
                OutputActivation(output_activation)
            ))
        self.band_heads = nn.ModuleList(heads)

    def forward(self, x, z_positions=None):
        latent = self.shared(x)
        bands = [head(latent) for head in self.band_heads]
        return None, torch.cat(bands, dim=1)

class TemporalMLP(nn.Module):
    """Temporal MLP v2.1 - Encoder + temporal decoder with Sigmoid-constrained outputs.
    
    Architecture improvements:
    - final_decoder receives concatenated [latent + temporal_last] for better prediction
    - All outputs constrained to [0, 1] via Sigmoid for normalized log power spectra
    """
    def __init__(self, input_dim, output_dim, n_z_steps=20, output_activation='sigmoid'):
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
        
        # Temporal decoder: predicts spectrum at each z step, constrained to [0,1]
        self.temporal_decoder = nn.Sequential(
            nn.Linear(512 + 1, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, output_dim),
            OutputActivation(output_activation)
        )
        
        # Final decoder: receives [latent + last_temporal], constrained to [0,1]
        self.final_decoder = nn.Sequential(
            nn.Linear(512 + output_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, output_dim),
            OutputActivation(output_activation)
        )
    
    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        latent = self.encoder(x)
        
        if z_positions is None:
            # Fallback: use latent directly when no z_positions provided
            final_output = self.final_decoder(torch.cat([latent, torch.zeros(batch_size, self.output_dim, device=x.device)], dim=1))
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
    """Temporal LSTM - LSTM-based temporal evolution modeling (recommended).
    
    All outputs constrained to [0, 1] via Sigmoid for normalized log power spectra.
    """
    def __init__(self, input_dim, output_dim, n_z_steps=20, hidden_dim=512,
                 output_activation='sigmoid'):
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
            OutputActivation(output_activation)
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

class TemporalConv2D(nn.Module):
    """2D convolutional decoder for full z-wavelength propagation maps."""
    def __init__(self, input_dim, output_dim, n_z_steps=20,
                 base_channels=128, z_init=16, lambda_init=64,
                 output_activation='sigmoid'):
        super(TemporalConv2D, self).__init__()
        self.output_dim = int(output_dim)
        self.n_z_steps = int(n_z_steps)
        self.base_channels = int(base_channels)
        self.output_activation = output_activation
        self.z_init = max(2, min(int(z_init), self.n_z_steps))
        self.lambda_init = max(8, min(int(lambda_init), self.output_dim))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
        )
        self.projection = nn.Linear(512, self.base_channels * self.z_init * self.lambda_init)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.base_channels, max(32, self.base_channels // 2), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(32, self.base_channels // 2)),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(max(32, self.base_channels // 2), max(16, self.base_channels // 4), kernel_size=3, padding=1),
            nn.BatchNorm2d(max(16, self.base_channels // 4)),
            nn.LeakyReLU(0.01),
            nn.Conv2d(max(16, self.base_channels // 4), 1, kernel_size=1),
        )

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        latent = self.encoder(x)
        fmap = self.projection(latent).view(
            batch_size, self.base_channels, self.z_init, self.lambda_init
        )
        decoded = self.decoder(fmap)
        decoded = F.interpolate(
            decoded, size=(self.n_z_steps, self.output_dim),
            mode='bilinear', align_corners=False
        )
        temporal_output = apply_output_activation(decoded.squeeze(1), self.output_activation)
        final_output = temporal_output[:, -1, :]
        return temporal_output, final_output

class TemporalConv2DMultiScale(nn.Module):
    """Coordinate-aware multi-band CNN decoder for z-wavelength propagation maps."""
    def __init__(self, input_dim, output_dim, n_z_steps=20,
                 base_channels=128, z_init=16, lambda_init=128,
                 wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                 band_boundaries_nm=None, coordinate_channels=False,
                 output_activation='sigmoid'):
        super(TemporalConv2DMultiScale, self).__init__()
        self.output_dim = int(output_dim)
        self.n_z_steps = int(n_z_steps)
        self.base_channels = int(base_channels)
        self.z_init = max(2, min(int(z_init), self.n_z_steps))
        self.lambda_init = max(8, min(int(lambda_init), self.output_dim))
        self.coordinate_channels = bool(coordinate_channels)
        self.output_activation = output_activation
        self.wavelength_range = tuple(float(v) for v in wavelength_range)
        self.band_slices = self._make_band_slices(
            self.output_dim, self.wavelength_range, band_boundaries_nm
        )

        mid_channels = max(32, self.base_channels // 2)
        feature_channels = max(16, self.base_channels // 4)
        self.feature_channels = feature_channels

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
        )
        self.projection = nn.Linear(512, self.base_channels * self.z_init * self.lambda_init)
        self.shared_decoder = nn.Sequential(
            nn.Conv2d(self.base_channels, self.base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.base_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(mid_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.LeakyReLU(0.01),
        )
        head_in_channels = feature_channels + (2 if self.coordinate_channels else 0)
        self.band_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(head_in_channels, head_in_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.01),
                nn.Conv2d(head_in_channels, 1, kernel_size=1),
            )
            for _ in self.band_slices
        ])

    @staticmethod
    def _make_band_slices(output_dim, wavelength_range, band_boundaries_nm):
        if band_boundaries_nm is None or len(band_boundaries_nm) < 2:
            return [(int(idx[0]), int(idx[-1]) + 1) for idx in np.array_split(np.arange(output_dim), 4)]
        wavelengths = np.linspace(float(wavelength_range[0]), float(wavelength_range[1]), int(output_dim))
        edges = [float(v) for v in band_boundaries_nm]
        slices = []
        prev_end = 0
        for band_idx in range(len(edges) - 1):
            if band_idx == len(edges) - 2:
                end = int(output_dim)
            else:
                end = int(np.searchsorted(wavelengths, edges[band_idx + 1], side='left'))
                end = max(end, prev_end + 1)
            start = max(0, min(prev_end, output_dim - 1))
            end = max(start + 1, min(end, output_dim))
            slices.append((start, end))
            prev_end = end
        if slices[-1][1] != output_dim:
            slices[-1] = (slices[-1][0], output_dim)
        return slices

    def _coordinate_maps(self, batch_size, device, dtype):
        z = torch.linspace(0.0, 1.0, self.n_z_steps, device=device, dtype=dtype).view(1, 1, self.n_z_steps, 1)
        lam = torch.linspace(0.0, 1.0, self.output_dim, device=device, dtype=dtype).view(1, 1, 1, self.output_dim)
        z_map = z.expand(batch_size, 1, self.n_z_steps, self.output_dim)
        lam_map = lam.expand(batch_size, 1, self.n_z_steps, self.output_dim)
        return z_map, lam_map

    def forward(self, x, z_positions=None):
        batch_size = x.size(0)
        latent = self.encoder(x)
        fmap = self.projection(latent).view(
            batch_size, self.base_channels, self.z_init, self.lambda_init
        )
        features = self.shared_decoder(fmap)
        features = F.interpolate(
            features, size=(self.n_z_steps, self.output_dim),
            mode='bilinear', align_corners=False
        )
        if self.coordinate_channels:
            coords = self._coordinate_maps(batch_size, features.device, features.dtype)
            features = torch.cat([features, coords[0], coords[1]], dim=1)

        bands = []
        for head, (start, end) in zip(self.band_heads, self.band_slices):
            band_full = head(features).squeeze(1)
            bands.append(band_full[:, :, start:end])
        temporal_output = apply_output_activation(torch.cat(bands, dim=2), self.output_activation)
        final_output = temporal_output[:, -1, :]
        return temporal_output, final_output

# ============================================================================
# Loss Functions
# ============================================================================

class TemporalLoss(nn.Module):
    """
    Temporal loss: L = L_temporal + alpha*L_final + gamma*L_smoothness

    Components:
      - temporal_loss: MSE between predicted and target temporal evolution
      - final_loss: MSE between predicted and target final spectrum
      - smoothness_loss: penalizes large differences between consecutive z steps

    Notes:
      - energy_loss was removed in v4.2 because per-sample normalization destroys
        the physical relationship between spectral sum and input energy.
      - Energy information is already encoded in input features (energy, pressure).
    """
    def __init__(self, alpha=0.1, gamma=0.01, spectral_gradient_weight=0.0,
                 short_wavelength_weight=1.0, short_wavelength_cutoff_nm=700.0,
                 wavelength_range=TRAIN_WAVELENGTH_RANGE_NM, output_dim=None,
                 uv_min_nm=200.0, uv_max_nm=700.0, uv_loss_weight=0.0,
                 uv_gradient_weight=0.0, uv_r2_weight=0.0,
                 num_bands=4, band_boundary_weight=0.0,
                 band_boundary_indices=None,
                 uv_second_derivative_weight=0.0,
                 long_wavelength_weight=1.0,
                 long_wavelength_cutoff_nm=1350.0,
                 z_gradient_weight=0.0, lambda_gradient_weight=0.0,
                 temporal_late_weight=1.0,
                 early_z_loss_weight=0.0,
                 early_z_max_cm=2.0,
                 early_z_gradient_weight=0.0,
                 z_gradient_coordinate='cm',
                 output_normalization=None,
                 log_integral_weight=0.0,
                 log_peak_weight=0.0):
        super(TemporalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spectral_gradient_weight = spectral_gradient_weight
        self.uv_loss_weight = uv_loss_weight
        self.uv_gradient_weight = uv_gradient_weight
        self.uv_r2_weight = uv_r2_weight
        self.band_boundary_weight = band_boundary_weight
        self.uv_second_derivative_weight = uv_second_derivative_weight
        self.z_gradient_weight = z_gradient_weight
        self.lambda_gradient_weight = lambda_gradient_weight
        self.temporal_late_weight = temporal_late_weight
        self.early_z_loss_weight = early_z_loss_weight
        self.early_z_max_cm = early_z_max_cm
        self.early_z_gradient_weight = early_z_gradient_weight
        self.z_gradient_coordinate = z_gradient_coordinate
        self.output_normalization = output_normalization or {'mode': 'per_sample_minmax'}
        self.log_integral_weight = log_integral_weight
        self.log_peak_weight = log_peak_weight
        self.band_boundaries = []
        self.mse = nn.MSELoss()
        self.register_buffer('wavelength_weights', torch.ones(1))
        self.register_buffer('uv_mask', torch.zeros(1, dtype=torch.bool))
        self.last_components = {}

        if output_dim is not None:
            wavelengths = torch.linspace(float(wavelength_range[0]), float(wavelength_range[1]), int(output_dim))
            weights = torch.ones(int(output_dim))
            if short_wavelength_weight != 1.0:
                weights[wavelengths <= float(short_wavelength_cutoff_nm)] = float(short_wavelength_weight)
            if long_wavelength_weight != 1.0:
                weights[wavelengths >= float(long_wavelength_cutoff_nm)] = float(long_wavelength_weight)
            self.register_buffer('wavelength_weights', weights)
            uv_mask = (wavelengths >= float(uv_min_nm)) & (wavelengths <= float(uv_max_nm))
            self.register_buffer('uv_mask', uv_mask)
            if band_boundary_indices is not None:
                self.band_boundaries = [int(idx) for idx in band_boundary_indices]
            elif num_bands is not None and int(num_bands) > 1:
                band_indices = np.array_split(np.arange(int(output_dim)), int(num_bands))
                self.band_boundaries = [int(idx[0]) for idx in band_indices[1:] if len(idx) > 0]

    def weighted_mse(self, pred, target):
        weights = self.wavelength_weights.to(device=pred.device, dtype=pred.dtype)
        if weights.numel() == pred.size(-1):
            return torch.mean((pred - target) ** 2 * weights.view(*([1] * (pred.dim() - 1)), -1))
        return self.mse(pred, target)

    def temporal_weighted_mse(self, pred, target):
        weights = self.wavelength_weights.to(device=pred.device, dtype=pred.dtype)
        loss = (pred - target) ** 2
        if weights.numel() == pred.size(-1):
            loss = loss * weights.view(1, 1, -1)
        if self.temporal_late_weight != 1.0 and pred.dim() == 3 and pred.size(1) > 1:
            z_weights = torch.ones(pred.size(1), device=pred.device, dtype=pred.dtype)
            z_weights[pred.size(1) // 2:] = float(self.temporal_late_weight)
            loss = loss * z_weights.view(1, -1, 1)
        return torch.mean(loss)

    def spectral_gradient_loss(self, pred, target):
        if pred is None or target is None or pred.size(-1) < 2:
            device = target.device if target is not None else self.wavelength_weights.device
            return torch.tensor(0.0, device=device)
        pred_grad = pred[..., 1:] - pred[..., :-1]
        target_grad = target[..., 1:] - target[..., :-1]
        return self.mse(pred_grad, target_grad)

    def uv_slice(self, values):
        mask = self.uv_mask.to(device=values.device)
        if mask.numel() != values.size(-1) or not torch.any(mask):
            return None
        return values[..., mask]

    def uv_mse_loss(self, pred, target):
        pred_uv = self.uv_slice(pred)
        target_uv = self.uv_slice(target)
        if pred_uv is None or target_uv is None:
            return torch.tensor(0.0, device=pred.device)
        return self.mse(pred_uv, target_uv)

    def uv_derivative_loss(self, pred, target):
        pred_uv = self.uv_slice(pred)
        target_uv = self.uv_slice(target)
        if pred_uv is None or target_uv is None or pred_uv.size(-1) < 2:
            return torch.tensor(0.0, device=pred.device)
        return self.spectral_gradient_loss(pred_uv, target_uv)

    def uv_second_derivative_loss(self, pred, target):
        pred_uv = self.uv_slice(pred)
        target_uv = self.uv_slice(target)
        if pred_uv is None or target_uv is None or pred_uv.size(-1) < 3:
            return torch.tensor(0.0, device=pred.device)
        pred_second = pred_uv[..., 2:] - 2 * pred_uv[..., 1:-1] + pred_uv[..., :-2]
        target_second = target_uv[..., 2:] - 2 * target_uv[..., 1:-1] + target_uv[..., :-2]
        return self.mse(pred_second, target_second)

    def uv_r2_loss(self, pred, target, eps=1e-8):
        pred_uv = self.uv_slice(pred)
        target_uv = self.uv_slice(target)
        if pred_uv is None or target_uv is None:
            return torch.tensor(0.0, device=pred.device)
        ss_res = torch.sum((pred_uv - target_uv) ** 2)
        ss_tot = torch.sum((target_uv - torch.mean(target_uv)) ** 2)
        return ss_res / (ss_tot + eps)

    def band_boundary_loss(self, pred, target):
        if not self.band_boundaries:
            return torch.tensor(0.0, device=pred.device)
        losses = []
        for boundary in self.band_boundaries:
            if boundary <= 0 or boundary >= pred.size(-1):
                continue
            pred_jump = pred[..., boundary] - pred[..., boundary - 1]
            target_jump = target[..., boundary] - target[..., boundary - 1]
            losses.append(torch.mean((pred_jump - target_jump) ** 2))
        if not losses:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(losses).mean()

    def z_gradient_loss(self, pred, target, z_positions=None):
        if pred is None or target is None or pred.dim() < 3 or pred.size(1) < 2:
            device = pred.device if pred is not None else self.wavelength_weights.device
            return torch.tensor(0.0, device=device)
        pred_grad = pred[:, 1:, :] - pred[:, :-1, :]
        target_grad = target[:, 1:, :] - target[:, :-1, :]
        if z_positions is not None and self.z_gradient_coordinate != 'index':
            z = z_positions.to(device=pred.device, dtype=pred.dtype)
            if z.dim() == 1:
                z = z.view(1, -1).expand(pred.size(0), -1)
            if z.size(1) == pred.size(1):
                dz = torch.clamp(z[:, 1:] - z[:, :-1], min=1e-12)
                if self.z_gradient_coordinate == 'cm':
                    dz = dz * 100.0
                dz = dz.unsqueeze(-1)
                pred_grad = pred_grad / dz
                target_grad = target_grad / dz
        return self.mse(pred_grad, target_grad)

    def early_z_mask(self, pred, z_positions):
        if z_positions is None or pred is None or pred.dim() < 3:
            return None
        z = z_positions.to(device=pred.device, dtype=pred.dtype)
        if z.dim() == 1:
            z = z.view(1, -1).expand(pred.size(0), -1)
        if z.size(1) != pred.size(1):
            return None
        z_cm = z * 100.0 if torch.max(z) <= 5.0 else z
        return z_cm <= float(self.early_z_max_cm)

    def early_z_mse_loss(self, pred, target, z_positions):
        mask = self.early_z_mask(pred, z_positions)
        if mask is None or not torch.any(mask):
            return torch.tensor(0.0, device=pred.device)
        loss = (pred - target) ** 2
        weights = mask.to(dtype=pred.dtype).unsqueeze(-1)
        return torch.sum(loss * weights) / torch.clamp(torch.sum(weights) * pred.size(-1), min=1.0)

    def early_z_gradient_loss_fn(self, pred, target, z_positions):
        mask = self.early_z_mask(pred, z_positions)
        if mask is None or pred.size(1) < 2:
            return torch.tensor(0.0, device=pred.device)
        step_mask = mask[:, 1:] & mask[:, :-1]
        if not torch.any(step_mask):
            return torch.tensor(0.0, device=pred.device)
        pred_grad = pred[:, 1:, :] - pred[:, :-1, :]
        target_grad = target[:, 1:, :] - target[:, :-1, :]
        z = z_positions.to(device=pred.device, dtype=pred.dtype)
        if z.dim() == 1:
            z = z.view(1, -1).expand(pred.size(0), -1)
        if self.z_gradient_coordinate != 'index':
            dz = torch.clamp(z[:, 1:] - z[:, :-1], min=1e-12)
            if self.z_gradient_coordinate == 'cm':
                dz = dz * 100.0
            dz = dz.unsqueeze(-1)
            pred_grad = pred_grad / dz
            target_grad = target_grad / dz
        weights = step_mask.to(dtype=pred.dtype).unsqueeze(-1)
        loss = (pred_grad - target_grad) ** 2
        return torch.sum(loss * weights) / torch.clamp(torch.sum(weights) * pred.size(-1), min=1.0)

    def lambda_gradient_temporal_loss(self, pred, target):
        if pred is None or target is None or pred.size(-1) < 2:
            device = pred.device if pred is not None else self.wavelength_weights.device
            return torch.tensor(0.0, device=device)
        pred_grad = pred[..., 1:] - pred[..., :-1]
        target_grad = target[..., 1:] - target[..., :-1]
        return self.mse(pred_grad, target_grad)

    def inverse_log_power_tensor(self, values):
        mode = self.output_normalization.get('mode')
        if mode == 'global_log_standard':
            log_mean = float(self.output_normalization.get('log_mean', 0.0))
            log_std = float(self.output_normalization.get('log_std', 1.0))
            return values * log_std + log_mean
        if mode == 'none_log':
            return values
        return None

    def log_integral_loss_fn(self, pred, target):
        pred_log = self.inverse_log_power_tensor(pred)
        target_log = self.inverse_log_power_tensor(target)
        if pred_log is None or target_log is None:
            return torch.tensor(0.0, device=pred.device)
        base = torch.tensor(10.0, device=pred.device, dtype=pred.dtype)
        pred_integral = torch.log10(torch.sum(torch.pow(base, pred_log), dim=-1) + 1e-30)
        target_integral = torch.log10(torch.sum(torch.pow(base, target_log), dim=-1) + 1e-30)
        return self.mse(pred_integral, target_integral)

    def log_peak_loss_fn(self, pred, target):
        pred_log = self.inverse_log_power_tensor(pred)
        target_log = self.inverse_log_power_tensor(target)
        if pred_log is None or target_log is None:
            return torch.tensor(0.0, device=pred.device)
        return self.mse(torch.amax(pred_log, dim=-1), torch.amax(target_log, dim=-1))

    def forward(self, pred_temporal, pred_final, target_temporal, target_final, z_positions=None):
        if pred_temporal is not None and target_temporal is not None:
            temporal_loss = self.temporal_weighted_mse(pred_temporal, target_temporal)
        else:
            temporal_loss = torch.tensor(0.0, device=pred_final.device)

        final_loss = self.weighted_mse(pred_final, target_final)

        smoothness_loss = torch.tensor(0.0, device=pred_final.device)
        if pred_temporal is not None and pred_temporal.size(1) > 1:
            diff = pred_temporal[:, 1:, :] - pred_temporal[:, :-1, :]
            smoothness_loss = torch.mean(diff ** 2)

        gradient_loss = torch.tensor(0.0, device=pred_final.device)
        if self.spectral_gradient_weight > 0:
            gradient_loss = self.spectral_gradient_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                gradient_loss = gradient_loss + self.spectral_gradient_loss(pred_temporal, target_temporal)

        uv_loss = torch.tensor(0.0, device=pred_final.device)
        if self.uv_loss_weight > 0:
            uv_loss = self.uv_mse_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                uv_loss = uv_loss + self.uv_mse_loss(pred_temporal, target_temporal)

        uv_gradient_loss = torch.tensor(0.0, device=pred_final.device)
        if self.uv_gradient_weight > 0:
            uv_gradient_loss = self.uv_derivative_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                uv_gradient_loss = uv_gradient_loss + self.uv_derivative_loss(pred_temporal, target_temporal)

        uv_second_derivative_loss = torch.tensor(0.0, device=pred_final.device)
        if self.uv_second_derivative_weight > 0:
            uv_second_derivative_loss = self.uv_second_derivative_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                uv_second_derivative_loss = uv_second_derivative_loss + self.uv_second_derivative_loss(
                    pred_temporal, target_temporal
                )

        uv_r2_loss = torch.tensor(0.0, device=pred_final.device)
        if self.uv_r2_weight > 0:
            uv_r2_loss = self.uv_r2_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                uv_r2_loss = uv_r2_loss + self.uv_r2_loss(pred_temporal, target_temporal)

        boundary_loss = torch.tensor(0.0, device=pred_final.device)
        if self.band_boundary_weight > 0:
            boundary_loss = self.band_boundary_loss(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                boundary_loss = boundary_loss + self.band_boundary_loss(pred_temporal, target_temporal)

        z_gradient_loss = torch.tensor(0.0, device=pred_final.device)
        if self.z_gradient_weight > 0 and pred_temporal is not None and target_temporal is not None:
            z_gradient_loss = self.z_gradient_loss(pred_temporal, target_temporal, z_positions=z_positions)

        lambda_gradient_loss = torch.tensor(0.0, device=pred_final.device)
        if self.lambda_gradient_weight > 0 and pred_temporal is not None and target_temporal is not None:
            lambda_gradient_loss = self.lambda_gradient_temporal_loss(pred_temporal, target_temporal)

        early_z_loss = torch.tensor(0.0, device=pred_final.device)
        if self.early_z_loss_weight > 0 and pred_temporal is not None and target_temporal is not None:
            early_z_loss = self.early_z_mse_loss(pred_temporal, target_temporal, z_positions)

        early_z_gradient_loss = torch.tensor(0.0, device=pred_final.device)
        if self.early_z_gradient_weight > 0 and pred_temporal is not None and target_temporal is not None:
            early_z_gradient_loss = self.early_z_gradient_loss_fn(pred_temporal, target_temporal, z_positions)

        log_integral_loss = torch.tensor(0.0, device=pred_final.device)
        if self.log_integral_weight > 0:
            log_integral_loss = self.log_integral_loss_fn(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                log_integral_loss = log_integral_loss + self.log_integral_loss_fn(pred_temporal, target_temporal)

        log_peak_loss = torch.tensor(0.0, device=pred_final.device)
        if self.log_peak_weight > 0:
            log_peak_loss = self.log_peak_loss_fn(pred_final, target_final)
            if pred_temporal is not None and target_temporal is not None:
                log_peak_loss = log_peak_loss + self.log_peak_loss_fn(pred_temporal, target_temporal)

        total = (temporal_loss + self.alpha * final_loss +
                 self.gamma * smoothness_loss +
                 self.spectral_gradient_weight * gradient_loss +
                 self.uv_loss_weight * uv_loss +
                 self.uv_gradient_weight * uv_gradient_loss +
                 self.uv_second_derivative_weight * uv_second_derivative_loss +
                 self.uv_r2_weight * uv_r2_loss +
                 self.band_boundary_weight * boundary_loss +
                 self.z_gradient_weight * z_gradient_loss +
                 self.lambda_gradient_weight * lambda_gradient_loss +
                 self.early_z_loss_weight * early_z_loss +
                 self.early_z_gradient_weight * early_z_gradient_loss +
                 self.log_integral_weight * log_integral_loss +
                 self.log_peak_weight * log_peak_loss)
        self.last_components = {
            'gradient': float(gradient_loss.detach().cpu()),
            'uv_mse': float(uv_loss.detach().cpu()),
            'uv_gradient': float(uv_gradient_loss.detach().cpu()),
            'uv_second_derivative': float(uv_second_derivative_loss.detach().cpu()),
            'uv_r2_loss': float(uv_r2_loss.detach().cpu()),
            'band_boundary': float(boundary_loss.detach().cpu()),
            'z_gradient': float(z_gradient_loss.detach().cpu()),
            'lambda_gradient': float(lambda_gradient_loss.detach().cpu()),
            'early_z_mse': float(early_z_loss.detach().cpu()),
            'early_z_gradient': float(early_z_gradient_loss.detach().cpu()),
            'log_integral': float(log_integral_loss.detach().cpu()),
            'log_peak': float(log_peak_loss.detach().cpu()),
        }
        return total, temporal_loss, final_loss, smoothness_loss

# ============================================================================
# Training
# ============================================================================

def evaluate_loss_components(model, loader, criterion, use_temporal=True, device='cpu'):
    """Evaluate loss components and final/temporal R2 diagnostics on a data loader."""
    model.eval()
    losses = {
        'total': 0, 'temporal': 0, 'final': 0, 'smoothness': 0,
        'gradient': 0, 'uv_mse': 0, 'uv_gradient': 0, 'uv_r2_loss': 0,
        'uv_second_derivative': 0, 'band_boundary': 0, 'z_gradient': 0, 'lambda_gradient': 0,
        'early_z_mse': 0, 'early_z_gradient': 0, 'log_integral': 0, 'log_peak': 0
    }
    batches = 0
    preds = []
    targets = []
    temporal_preds = []
    temporal_targets = []
    temporal_z_positions = []

    with torch.no_grad():
        for batch_data in loader:
            if use_temporal and len(batch_data) == 4:
                data, target_temporal, target_final, z_pos = batch_data
                data = data.to(device)
                target_temporal = target_temporal.to(device)
                target_final = target_final.to(device)
                z_pos = z_pos.to(device)

                pred_temporal, pred_final = model(data, z_pos)
                loss, t_loss, f_loss, s_loss = criterion(
                    pred_temporal, pred_final, target_temporal, target_final, z_positions=z_pos
                )
                if pred_temporal is not None:
                    temporal_preds.extend(pred_temporal.cpu().numpy())
                    temporal_targets.extend(target_temporal.cpu().numpy())
                    temporal_z_positions.extend(z_pos.cpu().numpy())
            else:
                data, target_final = batch_data
                data = data.to(device)
                target_final = target_final.to(device)

                _, pred_final = model(data)
                loss, t_loss, f_loss, s_loss = criterion(
                    None, pred_final, None, target_final
                )

            losses['total'] += loss.item()
            losses['temporal'] += t_loss.item() if torch.is_tensor(t_loss) else 0
            losses['final'] += f_loss.item()
            losses['smoothness'] += s_loss.item() if torch.is_tensor(s_loss) else 0
            for key, value in getattr(criterion, 'last_components', {}).items():
                if key in losses:
                    losses[key] += float(value)
            batches += 1
            preds.extend(pred_final.cpu().numpy())
            targets.extend(target_final.cpu().numpy())

    if batches == 0:
        nan_metrics = {
            'final_r2': float('nan'),
            'final_uv_r2': float('nan'),
            'temporal_r2': float('nan'),
            'temporal_uv_r2': float('nan'),
            'early_z_r2': float('nan'),
            'early_uv_r2': float('nan'),
        }
        return {k: float('nan') for k in losses}, nan_metrics

    avg_losses = {k: v / batches for k, v in losses.items()}
    preds = np.array(preds)
    targets = np.array(targets)
    mask = criterion.uv_mask.detach().cpu().numpy().astype(bool) if hasattr(criterion, 'uv_mask') else None
    metrics = {
        'final_r2': safe_r2_score(targets, preds) if preds.size else float('nan'),
        'final_uv_r2': band_metrics(targets, preds, mask, 'UV').get('UV_R2', float('nan')) if preds.size else float('nan'),
        'temporal_r2': float('nan'),
        'temporal_uv_r2': float('nan'),
        'early_z_r2': float('nan'),
        'early_uv_r2': float('nan'),
    }

    if temporal_preds:
        temporal_preds = np.asarray(temporal_preds)
        temporal_targets = np.asarray(temporal_targets)
        temporal_z_positions = np.asarray(temporal_z_positions)
        metrics['temporal_r2'] = safe_r2_score(temporal_targets, temporal_preds)
        if mask is not None and np.any(mask):
            metrics['temporal_uv_r2'] = safe_r2_score(temporal_targets[..., mask], temporal_preds[..., mask])

        if temporal_z_positions.size:
            z_cm = temporal_z_positions * 100.0 if np.nanmax(temporal_z_positions) <= 5.0 else temporal_z_positions
            early_mask = z_cm <= float(getattr(criterion, 'early_z_max_cm', 2.0))
            if np.any(early_mask):
                early_targets = temporal_targets[early_mask]
                early_preds = temporal_preds[early_mask]
                metrics['early_z_r2'] = safe_r2_score(early_targets, early_preds)
                if mask is not None and np.any(mask):
                    metrics['early_uv_r2'] = safe_r2_score(early_targets[..., mask], early_preds[..., mask])

    return avg_losses, metrics

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=500, patience=20, use_temporal=True, device='cpu',
                test_loader=None, selection_metric='val_loss', checkpoint_dir=None):
    """Train model and record train/val/test diagnostic losses."""
    best_val_loss = float('inf')
    early_stop_count = 0
    best_scores = {
        'val_loss': float('inf'),
        'val_r2': -float('inf'),
        'val_uv_r2': -float('inf'),
        'val_temporal_r2': -float('inf'),
        'val_temporal_uv_r2': -float('inf'),
        'val_early_z_r2': -float('inf'),
        'val_early_uv_r2': -float('inf'),
    }
    best_state_dicts = {
        'val_loss': copy.deepcopy(model.state_dict()),
        'val_r2': copy.deepcopy(model.state_dict()),
        'val_uv_r2': copy.deepcopy(model.state_dict()),
        'val_temporal_r2': copy.deepcopy(model.state_dict()),
        'val_temporal_uv_r2': copy.deepcopy(model.state_dict()),
        'val_early_z_r2': copy.deepcopy(model.state_dict()),
        'val_early_uv_r2': copy.deepcopy(model.state_dict()),
    }

    history = {
        'config': {
            'model_type': type(model).__name__,
            'num_epochs': num_epochs,
            'patience': patience,
            'batch_size': train_loader.batch_size,
            'use_temporal': use_temporal,
            'optimizer': 'Adam',
            'scheduler': type(scheduler).__name__,
            'has_test_curve': test_loader is not None,
            'selection_metric': selection_metric,
        },
        'epochs': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'best_val_loss_epoch': 0,
        'best_val_r2_epoch': 0,
        'best_val_uv_r2_epoch': 0,
        'best_val_temporal_r2_epoch': 0,
        'best_val_temporal_uv_r2_epoch': 0,
        'best_val_early_z_r2_epoch': 0,
        'best_val_early_uv_r2_epoch': 0,
        'best_val_r2': -float('inf'),
        'best_val_uv_r2': -float('inf'),
        'best_val_temporal_r2': -float('inf'),
        'best_val_temporal_uv_r2': -float('inf'),
        'best_val_early_z_r2': -float('inf'),
        'best_val_early_uv_r2': -float('inf'),
        'total_epochs': 0,
        'early_stopped': False
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = {
            'total': 0, 'temporal': 0, 'final': 0, 'smoothness': 0,
            'gradient': 0, 'uv_mse': 0, 'uv_gradient': 0, 'uv_r2_loss': 0,
            'uv_second_derivative': 0, 'band_boundary': 0, 'z_gradient': 0, 'lambda_gradient': 0,
            'early_z_mse': 0, 'early_z_gradient': 0, 'log_integral': 0, 'log_peak': 0
        }
        train_batches = 0

        for batch_data in train_loader:
            if use_temporal and len(batch_data) == 4:
                data, target_temporal, target_final, z_pos = batch_data
                data = data.to(device)
                target_temporal = target_temporal.to(device)
                target_final = target_final.to(device)
                z_pos = z_pos.to(device)

                optimizer.zero_grad()
                pred_temporal, pred_final = model(data, z_pos)
                loss, t_loss, f_loss, s_loss = criterion(
                    pred_temporal, pred_final, target_temporal, target_final, z_positions=z_pos
                )
            else:
                data, target_final = batch_data
                data = data.to(device)
                target_final = target_final.to(device)

                optimizer.zero_grad()
                _, pred_final = model(data)
                loss, t_loss, f_loss, s_loss = criterion(
                    None, pred_final, None, target_final
                )

            loss.backward()
            optimizer.step()

            train_losses['total'] += loss.item()
            train_losses['temporal'] += t_loss.item() if torch.is_tensor(t_loss) else 0
            train_losses['final'] += f_loss.item()
            train_losses['smoothness'] += s_loss.item() if torch.is_tensor(s_loss) else 0
            for key, value in getattr(criterion, 'last_components', {}).items():
                if key in train_losses:
                    train_losses[key] += float(value)
            train_batches += 1

        train_epoch_losses = {k: v / train_batches for k, v in train_losses.items()}
        val_epoch_losses, val_metrics = evaluate_loss_components(
            model, val_loader, criterion, use_temporal=use_temporal, device=device
        )
        test_epoch_losses = None
        test_metrics = None
        if test_loader is not None:
            test_epoch_losses, test_metrics = evaluate_loss_components(
                model, test_loader, criterion, use_temporal=use_temporal, device=device
            )

        val_r2 = val_metrics.get('final_r2', float('nan'))
        val_uv_r2 = val_metrics.get('final_uv_r2', float('nan'))
        val_temporal_r2 = val_metrics.get('temporal_r2', float('nan'))
        val_temporal_uv_r2 = val_metrics.get('temporal_uv_r2', float('nan'))
        val_early_z_r2 = val_metrics.get('early_z_r2', float('nan'))
        val_early_uv_r2 = val_metrics.get('early_uv_r2', float('nan'))

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_epoch_losses['total'])

        epoch_data = {
            'epoch': epoch + 1,
            'train': train_epoch_losses,
            'val': val_epoch_losses,
            'val_r2': val_r2,
            'val_uv_r2': val_uv_r2,
            'val_temporal_r2': val_temporal_r2,
            'val_temporal_uv_r2': val_temporal_uv_r2,
            'val_early_z_r2': val_early_z_r2,
            'val_early_uv_r2': val_early_uv_r2,
            'lr': current_lr,
            'time': time.time() - epoch_start
        }
        if test_epoch_losses is not None:
            epoch_data['test'] = test_epoch_losses
            epoch_data['test_r2'] = test_metrics.get('final_r2', float('nan'))
            epoch_data['test_uv_r2'] = test_metrics.get('final_uv_r2', float('nan'))
            epoch_data['test_temporal_r2'] = test_metrics.get('temporal_r2', float('nan'))
            epoch_data['test_temporal_uv_r2'] = test_metrics.get('temporal_uv_r2', float('nan'))
            epoch_data['test_early_z_r2'] = test_metrics.get('early_z_r2', float('nan'))
            epoch_data['test_early_uv_r2'] = test_metrics.get('early_uv_r2', float('nan'))
        history['epochs'].append(epoch_data)

        test_msg = ""
        if test_epoch_losses is not None:
            test_msg = f" | Test: {test_epoch_losses['total']:.6f}"
        logging.info(f"Epoch {epoch+1}/{num_epochs} | "
                     f"Train: {train_epoch_losses['total']:.6f} | "
                     f"Val: {val_epoch_losses['total']:.6f}"
                     f"{test_msg} | Temporal_R2: {val_temporal_r2:.4f} | "
                     f"Temporal_UV_R2: {val_temporal_uv_r2:.4f} | "
                     f"Early_Z_R2: {val_early_z_r2:.4f} | "
                     f"Final_R2: {val_r2:.4f} | LR: {current_lr:.6f}")

        if val_epoch_losses['total'] < best_val_loss:
            best_val_loss = val_epoch_losses['total']
            early_stop_count = 0
            history['best_epoch'] = epoch + 1
            history['best_val_loss'] = best_val_loss
            if test_epoch_losses is not None:
                history['best_epoch_test_loss'] = test_epoch_losses
                history['best_epoch_test_r2'] = test_metrics.get('final_r2', float('nan'))
        else:
            early_stop_count += 1

        if val_epoch_losses['total'] < best_scores['val_loss']:
            best_scores['val_loss'] = val_epoch_losses['total']
            best_state_dicts['val_loss'] = copy.deepcopy(model.state_dict())
            history['best_val_loss_epoch'] = epoch + 1
            history['best_val_loss'] = float(val_epoch_losses['total'])

        if np.isfinite(val_r2) and val_r2 > best_scores['val_r2']:
            best_scores['val_r2'] = val_r2
            best_state_dicts['val_r2'] = copy.deepcopy(model.state_dict())
            history['best_val_r2_epoch'] = epoch + 1
            history['best_val_r2'] = float(val_r2)

        if np.isfinite(val_uv_r2) and val_uv_r2 > best_scores['val_uv_r2']:
            best_scores['val_uv_r2'] = val_uv_r2
            best_state_dicts['val_uv_r2'] = copy.deepcopy(model.state_dict())
            history['best_val_uv_r2_epoch'] = epoch + 1
            history['best_val_uv_r2'] = float(val_uv_r2)

        metric_updates = {
            'val_temporal_r2': val_temporal_r2,
            'val_temporal_uv_r2': val_temporal_uv_r2,
            'val_early_z_r2': val_early_z_r2,
            'val_early_uv_r2': val_early_uv_r2,
        }
        for metric_name, metric_value in metric_updates.items():
            if np.isfinite(metric_value) and metric_value > best_scores[metric_name]:
                best_scores[metric_name] = metric_value
                best_state_dicts[metric_name] = copy.deepcopy(model.state_dict())
                history[f'best_{metric_name}_epoch'] = epoch + 1
                history[f'best_{metric_name}'] = float(metric_value)

        if early_stop_count >= patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            history['early_stopped'] = True
            break

    history['total_epochs'] = len(history['epochs'])
    if checkpoint_dir is not None:
        if history.get('best_val_loss_epoch', 0) > 0:
            torch.save(best_state_dicts['val_loss'], os.path.join(checkpoint_dir, 'best_val_loss_model.pth'))
        if history.get('best_val_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_r2'], os.path.join(checkpoint_dir, 'best_val_r2_model.pth'))
        if history.get('best_val_uv_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_uv_r2'], os.path.join(checkpoint_dir, 'best_val_uv_r2_model.pth'))
        if history.get('best_val_temporal_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_temporal_r2'], os.path.join(checkpoint_dir, 'best_val_temporal_r2_model.pth'))
        if history.get('best_val_temporal_uv_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_temporal_uv_r2'], os.path.join(checkpoint_dir, 'best_val_temporal_uv_r2_model.pth'))
        if history.get('best_val_early_z_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_early_z_r2'], os.path.join(checkpoint_dir, 'best_val_early_z_r2_model.pth'))
        if history.get('best_val_early_uv_r2_epoch', 0) > 0:
            torch.save(best_state_dicts['val_early_uv_r2'], os.path.join(checkpoint_dir, 'best_val_early_uv_r2_model.pth'))

    selected_metric = selection_metric if history.get(f'best_{selection_metric}_epoch', 0) > 0 else 'val_loss'
    if selected_metric != selection_metric:
        logging.warning(
            "Requested selection metric %s had no finite improvement; falling back to val_loss",
            selection_metric
        )
    if history['best_val_loss_epoch'] > 0:
        model.load_state_dict(best_state_dicts[selected_metric])
        history['restored_best_epoch'] = True
        history['restored_metric'] = selected_metric
        history['restored_epoch'] = history.get(f'best_{selected_metric}_epoch', history['best_val_loss_epoch'])
        logging.info(
            "Restored best %s model weights from epoch %s",
            selected_metric, history['restored_epoch']
        )
    else:
        history['restored_best_epoch'] = False
        history['restored_metric'] = None
        history['restored_epoch'] = 0
    return model, history

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, test_loader, use_temporal=True, device='cpu',
                   wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                   uv_min_nm=200.0, uv_max_nm=700.0,
                   num_bands=None, band_indices=None,
                   early_z_max_cm=2.0,
                   output_normalization=None):
    """Comprehensive model evaluation."""
    model.eval()
    
    test_preds = []
    test_targets = []
    temporal_preds = []
    temporal_targets = []
    temporal_z_positions = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            if use_temporal and len(batch_data) == 4:
                data, target_temporal, target_final, z_pos = batch_data
                data = data.to(device)
                target_final = target_final.to(device)
                z_pos = z_pos.to(device)
                
                pred_temporal, pred_final = model(data, z_pos)
                test_preds.extend(pred_final.cpu().numpy())
                test_targets.extend(target_final.cpu().numpy())
                
                if pred_temporal is not None:
                    temporal_preds.extend(pred_temporal.cpu().numpy())
                    temporal_targets.extend(target_temporal.numpy())
                    temporal_z_positions.extend(z_pos.cpu().numpy())
            else:
                data, target_final = batch_data
                data = data.to(device)
                target_final = target_final.to(device)
                
                _, pred_final = model(data)
                test_preds.extend(pred_final.cpu().numpy())
                test_targets.extend(target_final.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    metric_targets, metric_preds = finite_metric_arrays(test_targets, test_preds, "Final-spectrum")
    if metric_targets.size == 0:
        logging.warning("No finite final-spectrum points available for test metrics")
        metric_targets = np.array([0.0])
        metric_preds = np.array([0.0])
    
    metrics = {
        'MSE': float(mean_squared_error(metric_targets, metric_preds)),
        'RMSE': float(np.sqrt(mean_squared_error(metric_targets, metric_preds))),
        'MAE': float(mean_absolute_error(metric_targets, metric_preds)),
        'R2': safe_r2_score(test_targets, test_preds)
    }
    add_log_power_metrics(
        metrics, test_targets, test_preds, output_normalization,
        wavelength_range=wavelength_range, uv_min_nm=uv_min_nm, uv_max_nm=uv_max_nm
    )
    uv_mask = uv_mask_for_output(test_targets.shape[-1], wavelength_range, uv_min_nm, uv_max_nm)
    metrics.update(band_metrics(test_targets, test_preds, uv_mask, 'UV'))
    metrics.update(uv_shape_metrics(test_targets, test_preds, uv_mask))
    if num_bands is not None and int(num_bands) > 1:
        metrics.update(wavelength_band_metrics(
            test_targets, test_preds, test_targets.shape[-1], wavelength_range,
            num_bands=num_bands, band_indices=band_indices
        ))
    
    if len(temporal_preds) > 0:
        temporal_preds = np.array(temporal_preds)
        temporal_targets = np.array(temporal_targets)
        temporal_metric_targets, temporal_metric_preds = finite_metric_arrays(
            temporal_targets, temporal_preds, "Temporal"
        )
        if temporal_metric_targets.size > 0:
            metrics['Temporal_MSE'] = float(mean_squared_error(temporal_metric_targets, temporal_metric_preds))
            metrics['Temporal_R2'] = safe_r2_score(temporal_targets, temporal_preds)
            add_log_power_metrics(
                metrics, temporal_targets, temporal_preds, output_normalization,
                wavelength_range=wavelength_range, uv_min_nm=uv_min_nm, uv_max_nm=uv_max_nm,
                prefix='Temporal_'
            )
            metrics['Temporal_Final_R2'] = safe_r2_score(temporal_targets[:, -1, :], temporal_preds[:, -1, :])
            metrics['Temporal_ZGradient_MSE'] = gradient_mse_np(
                temporal_targets, temporal_preds, axis=1, name="Temporal z-gradient"
            )
            metrics['Temporal_LambdaGradient_MSE'] = gradient_mse_np(
                temporal_targets, temporal_preds, axis=2, name="Temporal lambda-gradient"
            )
            temporal_uv = band_metrics(temporal_targets, temporal_preds, uv_mask, 'Temporal_UV')
            metrics['Temporal_UV_R2'] = temporal_uv.get('Temporal_UV_R2', float('nan'))
            if len(temporal_z_positions) == temporal_targets.shape[0]:
                z_arr = np.asarray(temporal_z_positions, dtype=np.float64)
                z_cm = z_arr * 100.0 if np.nanmax(z_arr) <= 5.0 else z_arr
                early_mask = z_cm <= float(early_z_max_cm)
                if np.any(early_mask):
                    early_targets = temporal_targets[early_mask]
                    early_preds = temporal_preds[early_mask]
                    e_targets, e_preds = finite_metric_arrays(early_targets, early_preds, "Early-Z")
                    if e_targets.size > 0:
                        metrics['Early_Z_R2'] = safe_r2_score(early_targets, early_preds)
                        metrics['Early_Z_RMSE'] = float(np.sqrt(mean_squared_error(e_targets, e_preds)))
                        early_uv_targets = temporal_targets[..., uv_mask][early_mask] if np.any(uv_mask) else np.array([])
                        early_uv_preds = temporal_preds[..., uv_mask][early_mask] if np.any(uv_mask) else np.array([])
                        if early_uv_targets.size:
                            metrics['Early_UV_R2'] = safe_r2_score(early_uv_targets, early_uv_preds)
                grad_losses = []
                for true_i, pred_i, z_i in zip(temporal_targets, temporal_preds, z_cm):
                    step_mask = (z_i[1:] <= float(early_z_max_cm)) & (z_i[:-1] <= float(early_z_max_cm))
                    if not np.any(step_mask):
                        continue
                    dz = np.maximum((z_i[1:] - z_i[:-1]) / 100.0, 1e-12)
                    true_grad = (true_i[1:] - true_i[:-1]) / dz[:, None]
                    pred_grad = (pred_i[1:] - pred_i[:-1]) / dz[:, None]
                    grad_losses.append(np.mean((true_grad[step_mask] - pred_grad[step_mask]) ** 2))
                if grad_losses:
                    metrics['Early_ZGradient_MSE'] = float(np.mean(grad_losses))
    
    return metrics

# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(history, output_path):
    """Plot training, validation, and optional test diagnostic curves."""
    epochs = [e['epoch'] for e in history['epochs']]
    train_loss = [e['train']['total'] for e in history['epochs']]
    val_loss = [e['val']['total'] for e in history['epochs']]
    val_r2 = [e['val_r2'] for e in history['epochs']]
    val_uv_r2 = [e.get('val_uv_r2', np.nan) for e in history['epochs']]
    val_temporal_r2 = [e.get('val_temporal_r2', np.nan) for e in history['epochs']]
    val_temporal_uv_r2 = [e.get('val_temporal_uv_r2', np.nan) for e in history['epochs']]
    val_early_z_r2 = [e.get('val_early_z_r2', np.nan) for e in history['epochs']]
    has_test = all('test' in e for e in history['epochs'])
    test_loss = [e['test']['total'] for e in history['epochs']] if has_test else None
    test_r2 = [e.get('test_r2', np.nan) for e in history['epochs']] if has_test else None
    test_uv_r2 = [e.get('test_uv_r2', np.nan) for e in history['epochs']] if has_test else None
    test_temporal_r2 = [e.get('test_temporal_r2', np.nan) for e in history['epochs']] if has_test else None
    test_temporal_uv_r2 = [e.get('test_temporal_uv_r2', np.nan) for e in history['epochs']] if has_test else None
    test_early_z_r2 = [e.get('test_early_z_r2', np.nan) for e in history['epochs']] if has_test else None
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))
    
    ax1.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    if has_test:
        ax1.plot(epochs, test_loss, label='Test Loss', linewidth=2, linestyle='--')
    if history.get('best_val_loss_epoch', 0):
        ax1.axvline(history['best_val_loss_epoch'], color='gray', linestyle=':', linewidth=1.2,
                    label='Best Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training / Validation / Test Loss' if has_test else 'Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_temporal_r2, label='Val Temporal R2', color='green', linewidth=2)
    ax2.plot(epochs, val_temporal_uv_r2, label='Val Temporal UV R2', color='tab:blue', linewidth=1.5)
    ax2.plot(epochs, val_early_z_r2, label='Val Early-Z R2', color='tab:purple', linewidth=1.5)
    if has_test:
        ax2.plot(epochs, test_temporal_r2, label='Test Temporal R2', color='orange', linewidth=2, linestyle='--')
        ax2.plot(epochs, test_temporal_uv_r2, label='Test Temporal UV R2', color='tab:red', linewidth=1.5, linestyle='--')
        ax2.plot(epochs, test_early_z_r2, label='Test Early-Z R2', color='tab:brown', linewidth=1.5, linestyle='--')
    if history.get('best_val_temporal_r2_epoch', 0):
        ax2.axvline(history['best_val_temporal_r2_epoch'], color='green', linestyle=':', linewidth=1.2,
                    label='Best Val Temporal R2')
    if history.get('best_val_temporal_uv_r2_epoch', 0):
        ax2.axvline(history['best_val_temporal_uv_r2_epoch'], color='tab:blue', linestyle=':', linewidth=1.2,
                    label='Best Val Temporal UV R2')
    if history.get('best_val_early_z_r2_epoch', 0):
        ax2.axvline(history['best_val_early_z_r2_epoch'], color='tab:purple', linestyle=':', linewidth=1.2,
                    label='Best Val Early-Z R2')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R2 Score')
    ax2.set_title('Temporal Evolution R2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(epochs, val_r2, label='Val Final R2', color='green', linewidth=2)
    ax3.plot(epochs, val_uv_r2, label='Val Final UV R2', color='tab:blue', linewidth=1.5)
    if has_test:
        ax3.plot(epochs, test_r2, label='Test Final R2', color='orange', linewidth=2, linestyle='--')
        ax3.plot(epochs, test_uv_r2, label='Test Final UV R2', color='tab:red', linewidth=1.5, linestyle='--')
    if history.get('best_val_r2_epoch', 0):
        ax3.axvline(history['best_val_r2_epoch'], color='green', linestyle=':', linewidth=1.2,
                    label='Best Val Final R2')
    if history.get('best_val_uv_r2_epoch', 0):
        ax3.axvline(history['best_val_uv_r2_epoch'], color='tab:blue', linestyle=':', linewidth=1.2,
                    label='Best Val Final UV R2')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('R2 Score')
    ax3.set_title('Final Spectrum R2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to: {output_path}")

def plot_prediction_comparison(y_true, y_pred, output_path, n_samples=5,
                               wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                               indices=None, source_files=None, source_indices=None,
                               y_scale='normalized', output_normalization=None):
    """Plot prediction vs ground truth spectra with actual wavelength axis.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground truth and predicted spectra, shape (N, n_wavelengths)
    output_path : str
        Output image path
    n_samples : int
        Number of samples to plot
    wavelength_range : tuple
        (min_wl, max_wl) in nanometers
    y_scale : str
        "normalized" for normalized log power, "db" for Luna-like dB display
    """
    if y_scale not in ('normalized', 'db', 'log_power', 'physical_db'):
        raise ValueError(f"Unsupported y_scale: {y_scale}")

    if indices is None or len(indices) == 0:
        n_samples = min(n_samples, len(y_true))
        indices = np.random.choice(len(y_true), n_samples, replace=False).tolist()
    else:
        indices = [idx for idx in indices if 0 <= idx < len(y_true)]
        n_samples = min(len(indices), len(y_true))
        if n_samples == 0:
            logging.warning("No valid --visualize-indices entries; falling back to random samples")
            n_samples = min(5, len(y_true))
            indices = np.random.choice(len(y_true), n_samples, replace=False).tolist()
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3*n_samples))
    if n_samples == 1:
        axes = [axes]
    
    n_points = y_true.shape[1]
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)

    if y_scale == 'db':
        ylabel = 'Luna-like SED (dB)'
        suptitle = 'Prediction vs Ground Truth Spectra (dB display, normalized scale)'
    elif y_scale == 'log_power':
        ylabel = 'Log10 Power'
        suptitle = 'Prediction vs Ground Truth Spectra (inverse-normalized log power)'
    elif y_scale == 'physical_db':
        ylabel = 'Relative SED (dB)'
        suptitle = 'Prediction vs Ground Truth Spectra (relative dB from log power)'
    else:
        ylabel = 'Normalized Log Power'
        suptitle = 'Prediction vs Ground Truth Spectra'
    
    for i, idx in enumerate(indices):
        if y_scale == 'db':
            true_plot = normalized_to_luna_db(y_true[idx])
            pred_plot = normalized_to_luna_db(y_pred[idx])
        elif y_scale == 'log_power':
            true_log = inverse_spectrum_normalization(y_true[idx], output_normalization)
            pred_log = inverse_spectrum_normalization(y_pred[idx], output_normalization)
            if true_log is None or pred_log is None:
                logging.warning("Skipping %s because output normalization is not invertible", output_path)
                plt.close()
                return list(indices)
            true_plot = true_log
            pred_plot = pred_log
        elif y_scale == 'physical_db':
            true_log = inverse_spectrum_normalization(y_true[idx], output_normalization)
            pred_log = inverse_spectrum_normalization(y_pred[idx], output_normalization)
            if true_log is None or pred_log is None:
                logging.warning("Skipping %s because output normalization is not invertible", output_path)
                plt.close()
                return list(indices)
            true_plot = relative_db_from_log_power(true_log)
            pred_plot = relative_db_from_log_power(pred_log)
        else:
            true_plot = y_true[idx]
            pred_plot = y_pred[idx]

        axes[i].plot(wavelengths, true_plot, label='Ground Truth', linewidth=2, color='#1f77b4')
        axes[i].plot(wavelengths, pred_plot, label='Prediction', linewidth=2, alpha=0.8, color='#ff7f0e')
        axes[i].set_ylabel(ylabel, fontsize=10)
        if source_files is not None and source_indices is not None and idx < len(source_files):
            title = f'test[{idx}] | {source_files[idx]} | source_index={source_indices[idx]}'
        else:
            title = f'test[{idx}] (source unknown)'
        if y_scale == 'db':
            title += ' | dB display'
        elif y_scale == 'log_power':
            title += ' | log10 power'
        elif y_scale == 'physical_db':
            title += ' | relative dB from log power'
        axes[i].set_title(title, fontsize=11)
        axes[i].legend(fontsize=9)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(TRAIN_WAVELENGTH_RANGE_NM[0], TRAIN_WAVELENGTH_RANGE_NM[1])
        if y_scale == 'db':
            axes[i].set_ylim(LUNA_STYLE_DB_MIN, 0)
        elif y_scale == 'physical_db':
            axes[i].set_ylim(LUNA_STYLE_DB_MIN, 0)
    
    axes[-1].set_xlabel('Wavelength (nm)', fontsize=11)
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logging.info(f"Prediction comparison saved to: {output_path}")
    return list(indices)


def plot_temporal_evolution_heatmap(temporal_pred, temporal_true, z_positions,
                                    output_path, wavelength_range=TRAIN_WAVELENGTH_RANGE_NM,
                                    n_samples=3, indices=None, z_range_cm=None,
                                    title_suffix=""):
    """Plot temporal evolution heatmaps comparing prediction vs ground truth.
    
    Creates a figure with 3 rows per sample:
    - Row 1: Ground truth temporal evolution
    - Row 2: Predicted temporal evolution
    - Row 3: Absolute difference heatmap
    
    Parameters
    ----------
    temporal_pred : np.ndarray, shape (N, n_z, n_wavelengths)
        Predicted temporal spectra
    temporal_true : np.ndarray, shape (N, n_z, n_wavelengths)
        Ground truth temporal spectra
    z_positions : np.ndarray or list
        Z positions for each sample (can be ragged)
    output_path : str
        Output image path
    wavelength_range : tuple
        (min_wl, max_wl) in nanometers
    n_samples : int
        Number of samples to visualize
    indices : list or None
        Test split indices to visualize. Random samples are selected when None.
    z_range_cm : tuple or None
        Optional propagation-distance zoom range in cm.
    """
    n_z, n_wl = temporal_pred.shape[1], temporal_pred.shape[2]
    wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wl)
    
    n_available = min(len(temporal_pred), len(temporal_true))
    if indices is None or len(indices) == 0:
        n_samples = min(n_samples, n_available)
        indices = np.random.choice(n_available, n_samples, replace=False).tolist()
    else:
        indices = [idx for idx in indices if 0 <= idx < n_available]
        n_samples = min(len(indices), n_available)
        if n_samples == 0:
            logging.warning("No valid temporal visualization indices; falling back to random samples")
            n_samples = min(3, n_available)
            indices = np.random.choice(n_available, n_samples, replace=False).tolist()
    
    fig, axes = plt.subplots(n_samples, 3, figsize=(18, 4*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        pred_sample = temporal_pred[idx]
        true_sample = temporal_true[idx]
        
        # Get z positions for this sample
        if isinstance(z_positions, np.ndarray) and z_positions.dtype == object:
            z_pos = np.array(z_positions[idx]).flatten()
        else:
            z_pos = np.linspace(0, 100, n_z) if z_positions is None else np.array(z_positions).flatten()[:n_z]
        
        # Ensure consistent z length
        z_len = min(len(z_pos), n_z)
        pred_sample = pred_sample[:z_len]
        true_sample = true_sample[:z_len]
        z_pos = z_pos[:z_len]

        # HDF5 z is usually in meters; convert to cm for plotting when needed.
        z_pos_cm = z_pos * 100.0 if np.nanmax(z_pos) <= 5.0 else z_pos
        if z_range_cm is not None:
            z_mask = (z_pos_cm >= z_range_cm[0]) & (z_pos_cm <= z_range_cm[1])
            if np.count_nonzero(z_mask) < 2:
                logging.warning(
                    "Sample %s has fewer than 2 z points in %.3f-%.3f cm; using first available points",
                    idx, z_range_cm[0], z_range_cm[1]
                )
                keep_n = min(2, len(z_pos_cm))
                z_mask = np.zeros_like(z_pos_cm, dtype=bool)
                z_mask[:keep_n] = True
            pred_sample = pred_sample[z_mask]
            true_sample = true_sample[z_mask]
            z_pos_cm = z_pos_cm[z_mask]
        
        true_db = normalized_to_luna_db(true_sample)
        pred_db = normalized_to_luna_db(pred_sample)
        diff_db = np.abs(pred_db - true_db)

        # Ground truth heatmap
        ax_true = axes[row, 0]
        im_true = ax_true.pcolormesh(wavelengths, z_pos_cm, true_db,
                                      shading='auto', cmap=LUNA_STYLE_CMAP,
                                      vmin=LUNA_STYLE_DB_MIN, vmax=0)
        ax_true.set_ylabel('Distance (cm)', fontsize=10)
        if row == 0:
            ax_true.set_title('Ground Truth', fontsize=12, fontweight='bold')
        cbar_true = plt.colorbar(im_true, ax=ax_true, fraction=0.046, pad=0.04)
        cbar_true.set_label('SED (dB)')
        
        # Prediction heatmap
        ax_pred = axes[row, 1]
        im_pred = ax_pred.pcolormesh(wavelengths, z_pos_cm, pred_db,
                                      shading='auto', cmap=LUNA_STYLE_CMAP,
                                      vmin=LUNA_STYLE_DB_MIN, vmax=0)
        if row == 0:
            ax_pred.set_title('Prediction', fontsize=12, fontweight='bold')
        cbar_pred = plt.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)
        cbar_pred.set_label('SED (dB)')
        
        # Difference heatmap
        ax_diff = axes[row, 2]
        im_diff = ax_diff.pcolormesh(wavelengths, z_pos_cm, diff_db,
                                      shading='auto', cmap='Reds',
                                      vmin=0, vmax=max(float(np.nanmax(diff_db)), 1.0))
        if row == 0:
            ax_diff.set_title('|Prediction - Truth|', fontsize=12, fontweight='bold')
        cbar_diff = plt.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
        cbar_diff.set_label('|Delta SED| (dB)')

        for ax in (ax_true, ax_pred, ax_diff):
            ax.set_xlim(TRAIN_WAVELENGTH_RANGE_NM[0], TRAIN_WAVELENGTH_RANGE_NM[1])
            if z_range_cm is not None:
                ax.set_ylim(z_range_cm[0], z_range_cm[1])
        
        # Set x labels on bottom row
        if row == n_samples - 1:
            ax_true.set_xlabel('Wavelength (nm)', fontsize=10)
            ax_pred.set_xlabel('Wavelength (nm)', fontsize=10)
            ax_diff.set_xlabel('Wavelength (nm)', fontsize=10)
    
    title = 'Temporal Evolution: Prediction vs Ground Truth'
    if title_suffix:
        title += f' ({title_suffix})'
    fig.suptitle(title,
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    logging.info(f"Temporal evolution heatmap saved to: {output_path}")
    return list(indices)

# ============================================================================
# Main
# ============================================================================

def run_experiment(args, model_type, output_subdir):
    """Run a single experiment with given model type."""
    os.makedirs(output_subdir, exist_ok=True)
    processing_params = load_processing_params(args.input_dir)
    wavelength_range = get_wavelength_range_nm(processing_params)
    if not processing_params:
        logging.warning(
            "processing_params.json not found; assuming training wavelength range %.0f-%.0f nm",
            wavelength_range[0], wavelength_range[1]
        )
    elif not is_expected_training_range(wavelength_range):
        logging.warning(
            "Processed spectra use %.1f-%.1f nm, but the intended ML training range is %.1f-%.1f nm. "
            "Re-run data_preprocessing.py with --wavelength-min-nm 200 --wavelength-max-nm 2500.",
            wavelength_range[0], wavelength_range[1],
            TRAIN_WAVELENGTH_RANGE_NM[0], TRAIN_WAVELENGTH_RANGE_NM[1]
        )
    else:
        logging.info("Training wavelength range: %.1f-%.1f nm", wavelength_range[0], wavelength_range[1])
    spectrum_normalization = processing_params.get('spectrum_normalization', 'per_sample_minmax')
    output_normalization = processing_params.get('output_normalization', {'mode': spectrum_normalization})
    output_activation = resolve_output_activation(args.output_activation, spectrum_normalization)
    sample_filter = processing_params.get('sample_filter', 'unknown')
    processed_n_z = processing_params.get('n_z')
    early_dense_detected = bool(processing_params.get('early_dense_detected', False))
    if processed_n_z is not None:
        logging.info(
            "Processed temporal sampling: n_z=%s, sample_filter=%s, early_dense_detected=%s",
            processed_n_z, sample_filter, early_dense_detected
        )
        if early_dense_detected:
            logging.info(
                "Early-dense temporal data detected. Use smaller batch sizes if GPU memory is tight."
            )
    logging.info("Output activation: %s", output_activation)
    
    logging.info("="*70)
    logging.info(f"Running experiment: {model_type}")
    logging.info("="*70)
    
    # Load data
    X_train = np.load(os.path.join(args.input_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(args.input_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(args.input_dir, 'X_test.npy'))
    
    use_temporal = False
    y_temporal_train = y_temporal_val = y_temporal_test = None
    z_train = z_val = z_test = None
    
    if model_type in ('lstm', 'temporal', 'temporal_cnn') or (
        model_type == 'transformer' and args.transformer_mode == 'temporal'
    ):
        try:
            y_temporal_train = np.load(os.path.join(args.input_dir, 'y_temporal_train.npy'))
            y_temporal_val = np.load(os.path.join(args.input_dir, 'y_temporal_val.npy'))
            y_temporal_test = np.load(os.path.join(args.input_dir, 'y_temporal_test.npy'))
            z_train = np.load(os.path.join(args.input_dir, 'z_train.npy'), allow_pickle=True)
            z_val = np.load(os.path.join(args.input_dir, 'z_val.npy'), allow_pickle=True)
            z_test = np.load(os.path.join(args.input_dir, 'z_test.npy'), allow_pickle=True)
            use_temporal = True
        except FileNotFoundError:
            logging.info("Temporal data not found, using final spectrum only")
    
    y_train = np.load(os.path.join(args.input_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(args.input_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(args.input_dir, 'y_test.npy'))
    source_files_test = None
    source_indices_test = None
    source_files_path = os.path.join(args.input_dir, 'source_files_test.npy')
    source_indices_path = os.path.join(args.input_dir, 'source_indices_test.npy')
    if os.path.exists(source_files_path) and os.path.exists(source_indices_path):
        source_files_test = np.load(source_files_path, allow_pickle=True)
        source_indices_test = np.load(source_indices_path)
        logging.info("Loaded test source mapping for visualization titles")

    split_report = write_split_distribution_report(
        output_subdir, X_train, X_val, X_test, y_train, y_val, y_test,
        wavelength_range, args.uv_min_nm, args.uv_max_nm
    )
    logging.info(
        "Split diagnostics: duplicate X train/test=%d, nearest test-X median distance=%.4g",
        split_report['duplicate_X_hash_counts']['train_test'],
        split_report['nearest_X_distance_to_train']['test'].get('median', float('nan'))
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    
    # Create datasets
    if use_temporal:
        y_temp_train_t = torch.tensor(y_temporal_train, dtype=torch.float32)
        y_temp_val_t = torch.tensor(y_temporal_val, dtype=torch.float32)
        y_temp_test_t = torch.tensor(y_temporal_test, dtype=torch.float32)
        
        # Pad z positions
        max_z = max(len(z) for z in z_train)
        def pad_z(z_list, max_len):
            padded = []
            for z in z_list:
                z_arr = np.array(z, dtype=np.float32)
                if len(z_arr) < max_len:
                    z_arr = np.pad(z_arr, (0, max_len - len(z_arr)), mode='edge')
                else:
                    z_arr = z_arr[:max_len]
                padded.append(z_arr)
            return np.array(padded, dtype=np.float32)
        
        z_train_p = pad_z(z_train, max_z)
        z_val_p = pad_z(z_val, max_z)
        z_test_p = pad_z(z_test, max_z)
        
        z_train_t = torch.tensor(z_train_p, dtype=torch.float32)
        z_val_t = torch.tensor(z_val_p, dtype=torch.float32)
        z_test_t = torch.tensor(z_test_p, dtype=torch.float32)
        
        train_ds = TensorDataset(X_train_t, y_temp_train_t, y_train_t, z_train_t)
        val_ds = TensorDataset(X_val_t, y_temp_val_t, y_val_t, z_val_t)
        test_ds = TensorDataset(X_test_t, y_temp_test_t, y_test_t, z_test_t)
    else:
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        test_ds = TensorDataset(X_test_t, y_test_t)
    
    pin_memory = args.pin_memory and torch.cuda.is_available() and not args.no_cuda
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory,
                              worker_init_fn=seed_worker, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers, pin_memory=pin_memory,
                            worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers, pin_memory=pin_memory,
                             worker_init_fn=seed_worker)
    
    # Initialize model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    logging.info(f"Input dimension: {input_dim}, output wavelength points: {output_dim}")
    band_indices = None
    
    if model_type == 'lstm':
        n_z = y_temporal_train.shape[1] if y_temporal_train is not None else 20
        model = TemporalLSTM(input_dim, output_dim, n_z_steps=n_z, output_activation=output_activation)
    elif model_type == 'temporal':
        n_z = y_temporal_train.shape[1] if y_temporal_train is not None else 20
        model = TemporalMLP(input_dim, output_dim, n_z_steps=n_z, output_activation=output_activation)
    elif model_type == 'temporal_cnn':
        if y_temporal_train is None:
            raise ValueError("temporal_cnn requires y_temporal_*.npy and z_*.npy in the input directory")
        n_z = y_temporal_train.shape[1]
        if args.temporal_cnn_architecture == 'multiscale':
            model = TemporalConv2DMultiScale(
                input_dim, output_dim, n_z_steps=n_z,
                base_channels=args.temporal_cnn_channels,
                z_init=min(args.temporal_cnn_z_init, n_z),
                lambda_init=args.temporal_cnn_lambda_init,
                wavelength_range=wavelength_range,
                band_boundaries_nm=args.temporal_cnn_band_boundaries_nm,
                coordinate_channels=args.temporal_cnn_coordinate_channels,
                output_activation=output_activation
            )
        else:
            model = TemporalConv2D(
                input_dim, output_dim, n_z_steps=n_z,
                base_channels=args.temporal_cnn_channels,
                z_init=min(args.temporal_cnn_z_init, n_z),
                lambda_init=args.temporal_cnn_lambda_init,
                output_activation=output_activation
            )
    elif model_type == 'transformer':
        if args.transformer_mode == 'temporal':
            if y_temporal_train is None:
                raise ValueError("transformer temporal mode requires y_temporal_*.npy and z_*.npy")
            n_z = y_temporal_train.shape[1]
            model = TemporalSpectrumTransformer(
                input_dim, output_dim, n_z_steps=n_z,
                d_model=args.transformer_d_model,
                nhead=args.transformer_heads,
                num_layers=args.transformer_layers,
                use_z_embedding=args.transformer_use_z_embedding,
                output_activation=output_activation,
                wavelength_range=wavelength_range,
                band_boundaries_nm=args.transformer_band_boundaries_nm
            )
            use_temporal = True
        else:
            model = SpectrumTransformer(
                input_dim, output_dim,
                d_model=args.transformer_d_model,
                nhead=args.transformer_heads,
                num_layers=args.transformer_layers,
                output_activation=output_activation
            )
    elif model_type == 'linear':
        model = LinearBaseline(input_dim, output_dim, output_activation=output_activation)
    elif model_type == 'shallow':
        model = ShallowMLP(input_dim, output_dim, output_activation=output_activation)
    elif model_type == 'banded_mlp':
        band_indices = make_band_indices(
            output_dim, wavelength_range=wavelength_range,
            num_bands=args.num_bands, band_boundaries_nm=args.band_boundaries_nm
        )
        model = BandedMLP(
            input_dim, output_dim, num_bands=args.num_bands,
            wavelength_range=wavelength_range,
            band_boundaries_nm=args.band_boundaries_nm,
            band_head_hidden=args.band_head_hidden,
            uv_band_head_hidden=args.uv_band_head_hidden,
            output_activation=output_activation
        )
    else:
        model = StandardMLP(input_dim, output_dim, output_activation=output_activation)
        use_temporal = False
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = model.to(device)
    finetune_load_report = None
    if args.finetune_from is not None:
        strict = args.finetune_strict.lower() == 'true'
        finetune_load_report = load_finetune_weights(model, args.finetune_from, device=device, strict=strict)
        logging.info(
            "Loaded fine-tune weights from %s (strict=%s, loaded=%d, skipped=%d)",
            args.finetune_from, strict,
            finetune_load_report['loaded'], len(finetune_load_report['skipped'])
        )

    if model_type == 'banded_mlp' and args.freeze_shared:
        for param in model.shared.parameters():
            param.requires_grad = False
        logging.info("Frozen BandedMLP.shared for fine-tuning")

    if model_type == 'banded_mlp' and args.freeze_non_uv_heads:
        if args.band_boundaries_nm is None:
            raise ValueError("--freeze-non-uv-heads requires --band-boundaries-nm so the first head is a physical UV band")
        for head_idx, head in enumerate(model.band_heads):
            if head_idx == 0:
                continue
            for param in head.parameters():
                param.requires_grad = False
        logging.info("Frozen non-UV BandedMLP heads; only first band head remains trainable")

    logging.info(f"Model: {type(model).__name__}, Device: {device}")
    parameter_count = count_parameters(model)
    logging.info(f"Trainable parameters: {parameter_count}")
    
    # Training
    criterion = TemporalLoss(
        alpha=args.alpha,
        gamma=args.gamma,
        spectral_gradient_weight=args.spectral_gradient_weight,
        short_wavelength_weight=args.short_wavelength_weight,
        short_wavelength_cutoff_nm=args.short_wavelength_cutoff_nm,
        wavelength_range=wavelength_range,
        output_dim=output_dim,
        uv_min_nm=args.uv_min_nm,
        uv_max_nm=args.uv_max_nm,
        uv_loss_weight=args.uv_loss_weight,
        uv_gradient_weight=args.uv_gradient_weight,
        uv_r2_weight=args.uv_r2_weight,
        num_bands=args.num_bands,
        band_boundary_weight=args.band_boundary_weight,
        band_boundary_indices=[int(idx[0]) for idx in band_indices[1:]] if band_indices is not None else None,
        uv_second_derivative_weight=args.uv_second_derivative_weight,
        long_wavelength_weight=args.long_wavelength_weight,
        long_wavelength_cutoff_nm=args.long_wavelength_cutoff_nm,
        z_gradient_weight=args.z_gradient_weight,
        lambda_gradient_weight=args.lambda_gradient_weight,
        temporal_late_weight=args.temporal_late_weight,
        early_z_loss_weight=args.early_z_loss_weight,
        early_z_max_cm=args.early_z_max_cm,
        early_z_gradient_weight=args.early_z_gradient_weight,
        z_gradient_coordinate=args.z_gradient_coordinate,
        output_normalization=output_normalization,
        log_integral_weight=args.log_integral_weight,
        log_peak_weight=args.log_peak_weight
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters remain after applying freeze options")
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, verbose=True)
    
    train_start = time.time()
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=args.epochs, patience=args.patience, use_temporal=use_temporal,
        device=device, test_loader=test_loader,
        selection_metric=args.selection_metric,
        checkpoint_dir=output_subdir
    )
    training_time_sec = time.time() - train_start
    history['config']['seed'] = args.seed
    history['config']['input_dir'] = args.input_dir
    history['config']['output_dir'] = output_subdir
    history['config']['wavelength_range_nm'] = list(wavelength_range)
    history['config']['spectral_gradient_weight'] = args.spectral_gradient_weight
    history['config']['short_wavelength_weight'] = args.short_wavelength_weight
    history['config']['short_wavelength_cutoff_nm'] = args.short_wavelength_cutoff_nm
    history['config']['uv_min_nm'] = args.uv_min_nm
    history['config']['uv_max_nm'] = args.uv_max_nm
    history['config']['uv_loss_weight'] = args.uv_loss_weight
    history['config']['uv_gradient_weight'] = args.uv_gradient_weight
    history['config']['uv_second_derivative_weight'] = args.uv_second_derivative_weight
    history['config']['uv_r2_weight'] = args.uv_r2_weight
    history['config']['long_wavelength_weight'] = args.long_wavelength_weight
    history['config']['long_wavelength_cutoff_nm'] = args.long_wavelength_cutoff_nm
    history['config']['num_bands'] = args.num_bands
    history['config']['band_boundaries_nm'] = args.band_boundaries_nm
    history['config']['band_head_hidden'] = args.band_head_hidden
    history['config']['uv_band_head_hidden'] = args.uv_band_head_hidden
    history['config']['band_boundary_weight'] = args.band_boundary_weight
    history['config']['z_gradient_weight'] = args.z_gradient_weight
    history['config']['z_gradient_coordinate'] = args.z_gradient_coordinate
    history['config']['lambda_gradient_weight'] = args.lambda_gradient_weight
    history['config']['temporal_late_weight'] = args.temporal_late_weight
    history['config']['early_z_loss_weight'] = args.early_z_loss_weight
    history['config']['early_z_max_cm'] = args.early_z_max_cm
    history['config']['early_z_gradient_weight'] = args.early_z_gradient_weight
    history['config']['temporal_cnn_channels'] = args.temporal_cnn_channels
    history['config']['temporal_cnn_lambda_init'] = args.temporal_cnn_lambda_init
    history['config']['temporal_cnn_z_init'] = args.temporal_cnn_z_init
    history['config']['temporal_cnn_architecture'] = args.temporal_cnn_architecture
    history['config']['temporal_cnn_band_boundaries_nm'] = args.temporal_cnn_band_boundaries_nm
    history['config']['temporal_cnn_coordinate_channels'] = bool(args.temporal_cnn_coordinate_channels)
    history['config']['transformer_mode'] = args.transformer_mode
    history['config']['transformer_d_model'] = args.transformer_d_model
    history['config']['transformer_heads'] = args.transformer_heads
    history['config']['transformer_layers'] = args.transformer_layers
    history['config']['transformer_use_z_embedding'] = bool(args.transformer_use_z_embedding)
    history['config']['transformer_band_boundaries_nm'] = args.transformer_band_boundaries_nm
    history['config']['finetune_from'] = args.finetune_from
    history['config']['finetune_load_report'] = finetune_load_report
    history['config']['freeze_shared'] = bool(args.freeze_shared)
    history['config']['freeze_non_uv_heads'] = bool(args.freeze_non_uv_heads)
    history['config']['finetune_strict'] = args.finetune_strict
    history['config']['output_activation'] = output_activation
    history['config']['output_normalization'] = output_normalization
    history['config']['log_integral_weight'] = args.log_integral_weight
    history['config']['log_peak_weight'] = args.log_peak_weight
    history['config']['selection_metric'] = args.selection_metric
    history['config']['parameter_count'] = parameter_count
    history['config']['training_time_sec'] = training_time_sec
    
    # Evaluation
    metrics = evaluate_model(
        model, test_loader, use_temporal=use_temporal, device=device,
        wavelength_range=wavelength_range, uv_min_nm=args.uv_min_nm, uv_max_nm=args.uv_max_nm,
        num_bands=len(band_indices) if band_indices is not None else (args.num_bands if model_type == 'banded_mlp' else None),
        band_indices=band_indices,
        early_z_max_cm=args.early_z_max_cm,
        output_normalization=output_normalization
    )
    metrics['Selection_Metric'] = args.selection_metric
    metrics['Restored_Metric'] = history.get('restored_metric')
    metrics['Restored_Epoch'] = history.get('restored_epoch', 0)
    metrics['Parameter_Count'] = parameter_count
    metrics['Training_Time_Sec'] = float(training_time_sec)
    metrics['Uses_Temporal_Labels'] = bool(use_temporal)
    metrics['Output_Activation'] = output_activation
    metrics['Spectrum_Normalization'] = spectrum_normalization
    metrics['Temporal_CNN_Architecture'] = args.temporal_cnn_architecture if model_type == 'temporal_cnn' else ''
    metrics['Temporal_CNN_Band_Boundaries_Nm'] = args.temporal_cnn_band_boundaries_nm if model_type == 'temporal_cnn' else []
    metrics['Temporal_CNN_Coordinate_Channels'] = bool(args.temporal_cnn_coordinate_channels) if model_type == 'temporal_cnn' else False
    metrics['Transformer_Mode'] = args.transformer_mode if model_type == 'transformer' else ''
    metrics['Transformer_D_Model'] = args.transformer_d_model if model_type == 'transformer' else 0
    metrics['Transformer_Heads'] = args.transformer_heads if model_type == 'transformer' else 0
    metrics['Transformer_Layers'] = args.transformer_layers if model_type == 'transformer' else 0
    metrics['Transformer_Use_Z_Embedding'] = bool(args.transformer_use_z_embedding) if model_type == 'transformer' else False
    metrics['Transformer_Band_Boundaries_Nm'] = args.transformer_band_boundaries_nm if model_type == 'transformer' else []
    metrics['Z_Gradient_Coordinate'] = args.z_gradient_coordinate

    selected_state_dict = copy.deepcopy(model.state_dict())
    checkpoint_files = {
        'Best_Val_Loss': 'best_val_loss_model.pth',
        'Best_Val_R2': 'best_val_r2_model.pth',
        'Best_Val_UV_R2': 'best_val_uv_r2_model.pth',
        'Best_Val_Temporal_R2': 'best_val_temporal_r2_model.pth',
        'Best_Val_Temporal_UV_R2': 'best_val_temporal_uv_r2_model.pth',
        'Best_Val_Early_Z_R2': 'best_val_early_z_r2_model.pth',
        'Best_Val_Early_UV_R2': 'best_val_early_uv_r2_model.pth',
    }
    checkpoint_metrics = {}
    for prefix, filename in checkpoint_files.items():
        checkpoint_path = os.path.join(output_subdir, filename)
        if not os.path.exists(checkpoint_path):
            continue
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            ckpt_metrics = evaluate_model(
                model, test_loader, use_temporal=use_temporal, device=device,
                wavelength_range=wavelength_range, uv_min_nm=args.uv_min_nm, uv_max_nm=args.uv_max_nm,
                num_bands=len(band_indices) if band_indices is not None else (args.num_bands if model_type == 'banded_mlp' else None),
                band_indices=band_indices,
                early_z_max_cm=args.early_z_max_cm,
                output_normalization=output_normalization
            )
            checkpoint_metrics[prefix] = ckpt_metrics
            metrics[f'{prefix}_Test_R2'] = ckpt_metrics.get('R2', float('nan'))
            metrics[f'{prefix}_Test_UV_R2'] = ckpt_metrics.get('UV_R2', float('nan'))
            metrics[f'{prefix}_Test_RMSE'] = ckpt_metrics.get('RMSE', float('nan'))
            metrics[f'{prefix}_Test_Temporal_R2'] = ckpt_metrics.get('Temporal_R2', float('nan'))
            metrics[f'{prefix}_Test_Temporal_UV_R2'] = ckpt_metrics.get('Temporal_UV_R2', float('nan'))
            metrics[f'{prefix}_Test_Early_Z_R2'] = ckpt_metrics.get('Early_Z_R2', float('nan'))
            metrics[f'{prefix}_Test_Early_UV_R2'] = ckpt_metrics.get('Early_UV_R2', float('nan'))
        except Exception as exc:
            logging.warning("Failed to evaluate %s checkpoint: %s", prefix, exc)
    model.load_state_dict(selected_state_dict)
    history['checkpoint_metrics'] = checkpoint_metrics
    
    logging.info("Test Metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            logging.info(f"  {k}: {float(v):.4f}")
        else:
            logging.info(f"  {k}: {v}")
    
    # Save
    torch.save(model.state_dict(), os.path.join(output_subdir, 'best_model.pth'))
    with open(os.path.join(output_subdir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_subdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    config = {
        'model': model_type,
        'input_dir': args.input_dir,
        'output_dir': output_subdir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'selection_metric': args.selection_metric,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'spectral_gradient_weight': args.spectral_gradient_weight,
        'short_wavelength_weight': args.short_wavelength_weight,
        'short_wavelength_cutoff_nm': args.short_wavelength_cutoff_nm,
        'uv_min_nm': args.uv_min_nm,
        'uv_max_nm': args.uv_max_nm,
        'uv_loss_weight': args.uv_loss_weight,
        'uv_gradient_weight': args.uv_gradient_weight,
        'uv_second_derivative_weight': args.uv_second_derivative_weight,
        'uv_r2_weight': args.uv_r2_weight,
        'long_wavelength_weight': args.long_wavelength_weight,
        'long_wavelength_cutoff_nm': args.long_wavelength_cutoff_nm,
        'num_bands': args.num_bands,
        'band_boundaries_nm': args.band_boundaries_nm,
        'band_head_hidden': args.band_head_hidden,
        'uv_band_head_hidden': args.uv_band_head_hidden,
        'band_boundary_weight': args.band_boundary_weight,
        'z_gradient_weight': args.z_gradient_weight,
        'z_gradient_coordinate': args.z_gradient_coordinate,
        'lambda_gradient_weight': args.lambda_gradient_weight,
        'temporal_late_weight': args.temporal_late_weight,
        'early_z_loss_weight': args.early_z_loss_weight,
        'early_z_max_cm': args.early_z_max_cm,
        'early_z_gradient_weight': args.early_z_gradient_weight,
        'temporal_cnn_channels': args.temporal_cnn_channels,
        'temporal_cnn_lambda_init': args.temporal_cnn_lambda_init,
        'temporal_cnn_z_init': args.temporal_cnn_z_init,
        'temporal_cnn_architecture': args.temporal_cnn_architecture,
        'temporal_cnn_band_boundaries_nm': args.temporal_cnn_band_boundaries_nm,
        'temporal_cnn_coordinate_channels': bool(args.temporal_cnn_coordinate_channels),
        'transformer_mode': args.transformer_mode,
        'transformer_d_model': args.transformer_d_model,
        'transformer_heads': args.transformer_heads,
        'transformer_layers': args.transformer_layers,
        'transformer_use_z_embedding': bool(args.transformer_use_z_embedding),
        'transformer_band_boundaries_nm': args.transformer_band_boundaries_nm,
        'finetune_from': args.finetune_from,
        'finetune_load_report': finetune_load_report,
        'freeze_shared': bool(args.freeze_shared),
        'freeze_non_uv_heads': bool(args.freeze_non_uv_heads),
        'finetune_strict': args.finetune_strict,
        'parameter_count': parameter_count,
        'training_time_sec': training_time_sec,
        'seed': args.seed,
        'num_workers': args.num_workers,
        'pin_memory': bool(pin_memory),
        'visualize_indices': args.visualize_indices,
        'wavelength_range_nm': list(wavelength_range),
        'spectrum_normalization': spectrum_normalization,
        'output_activation': output_activation,
        'output_normalization': output_normalization,
        'log_integral_weight': args.log_integral_weight,
        'log_peak_weight': args.log_peak_weight,
        'sample_filter': sample_filter,
        'processed_n_z': processed_n_z,
        'early_dense_detected': early_dense_detected,
        'processing_params': processing_params,
    }
    with open(os.path.join(output_subdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Visualization
    if args.visualize:
        plot_training_curves(history, os.path.join(output_subdir, 'training_curves.png'))
        
        # Get predictions for visualization
        model.eval()
        test_preds = []
        test_temporal_preds = []
        test_temporal_true = []
        test_z_positions = []
        
        with torch.no_grad():
            for batch_data in test_loader:
                if use_temporal and len(batch_data) == 4:
                    data, y_temp, _, z_pos = batch_data
                    data = data.to(device)
                    z_pos = z_pos.to(device)
                    temp_pred, pred = model(data, z_pos)
                    test_preds.extend(pred.cpu().numpy())
                    if temp_pred is not None:
                        test_temporal_preds.extend(temp_pred.cpu().numpy())
                    test_temporal_true.extend(y_temp.numpy())
                    test_z_positions.extend(z_pos.cpu().numpy())
                else:
                    data, _ = batch_data
                    data = data.to(device)
                    _, pred = model(data)
                    test_preds.extend(pred.cpu().numpy())
        
        test_preds = np.array(test_preds)
        prediction_indices = plot_prediction_comparison(
            y_test, test_preds,
            os.path.join(output_subdir, 'predictions.png'),
            wavelength_range=wavelength_range,
            indices=args.visualize_indices,
            source_files=source_files_test,
            source_indices=source_indices_test
        )
        plot_prediction_comparison(
            y_test, test_preds,
            os.path.join(output_subdir, 'predictions_db.png'),
            wavelength_range=wavelength_range,
            indices=prediction_indices,
            source_files=source_files_test,
            source_indices=source_indices_test,
            y_scale='db'
        )
        if inverse_spectrum_normalization(y_test[:1], output_normalization) is not None:
            plot_prediction_comparison(
                y_test, test_preds,
                os.path.join(output_subdir, 'predictions_log_power.png'),
                wavelength_range=wavelength_range,
                indices=prediction_indices,
                source_files=source_files_test,
                source_indices=source_indices_test,
                y_scale='log_power',
                output_normalization=output_normalization
            )
            plot_prediction_comparison(
                y_test, test_preds,
                os.path.join(output_subdir, 'predictions_physical_db.png'),
                wavelength_range=wavelength_range,
                indices=prediction_indices,
                source_files=source_files_test,
                source_indices=source_indices_test,
                y_scale='physical_db',
                output_normalization=output_normalization
            )
        
        # Temporal evolution heatmap (for temporal models)
        if use_temporal and len(test_temporal_preds) > 0:
            test_temporal_preds = np.array(test_temporal_preds)
            test_temporal_true = np.array(test_temporal_true)
            temporal_indices = plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=args.visualize_indices
            )
            temporal_index_payload = {'indices': [int(idx) for idx in temporal_indices]}
            if source_indices_test is not None:
                temporal_index_payload['source_indices'] = [
                    int(source_indices_test[idx]) for idx in temporal_indices
                ]
            if source_files_test is not None:
                temporal_index_payload['source_files'] = [
                    str(source_files_test[idx]) for idx in temporal_indices
                ]
            with open(os.path.join(output_subdir, 'temporal_visualization_indices.json'), 'w') as f:
                json.dump(temporal_index_payload, f, indent=2)
            plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution_z0_2cm.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=temporal_indices,
                z_range_cm=(0.0, 2.0),
                title_suffix='z = 0-2 cm zoom'
            )
            plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution_z0_10cm.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=temporal_indices,
                z_range_cm=(0.0, 10.0),
                title_suffix='z = 0-10 cm zoom'
            )
    
    return metrics, history

def build_model_for_type(model_type, input_dim, output_dim, args, wavelength_range,
                         y_temporal=None, output_activation='sigmoid'):
    """Construct a model without touching optimizer/training state."""
    band_indices = None
    use_temporal = model_type in ('lstm', 'temporal', 'temporal_cnn') or (
        model_type == 'transformer' and args.transformer_mode == 'temporal'
    )

    if model_type == 'lstm':
        n_z = y_temporal.shape[1] if y_temporal is not None else 20
        model = TemporalLSTM(input_dim, output_dim, n_z_steps=n_z, output_activation=output_activation)
    elif model_type == 'temporal':
        n_z = y_temporal.shape[1] if y_temporal is not None else 20
        model = TemporalMLP(input_dim, output_dim, n_z_steps=n_z, output_activation=output_activation)
    elif model_type == 'temporal_cnn':
        if y_temporal is None:
            raise ValueError("temporal_cnn eval requires y_temporal_test.npy and z_test.npy")
        n_z = y_temporal.shape[1]
        if args.temporal_cnn_architecture == 'multiscale':
            model = TemporalConv2DMultiScale(
                input_dim, output_dim, n_z_steps=n_z,
                base_channels=args.temporal_cnn_channels,
                z_init=min(args.temporal_cnn_z_init, n_z),
                lambda_init=args.temporal_cnn_lambda_init,
                wavelength_range=wavelength_range,
                band_boundaries_nm=args.temporal_cnn_band_boundaries_nm,
                coordinate_channels=args.temporal_cnn_coordinate_channels,
                output_activation=output_activation
            )
        else:
            model = TemporalConv2D(
                input_dim, output_dim, n_z_steps=n_z,
                base_channels=args.temporal_cnn_channels,
                z_init=min(args.temporal_cnn_z_init, n_z),
                lambda_init=args.temporal_cnn_lambda_init,
                output_activation=output_activation
            )
    elif model_type == 'transformer':
        if args.transformer_mode == 'temporal':
            if y_temporal is None:
                raise ValueError("transformer temporal eval requires y_temporal_test.npy and z_test.npy")
            n_z = y_temporal.shape[1]
            model = TemporalSpectrumTransformer(
                input_dim, output_dim, n_z_steps=n_z,
                d_model=args.transformer_d_model,
                nhead=args.transformer_heads,
                num_layers=args.transformer_layers,
                use_z_embedding=args.transformer_use_z_embedding,
                output_activation=output_activation,
                wavelength_range=wavelength_range,
                band_boundaries_nm=args.transformer_band_boundaries_nm
            )
            use_temporal = True
        else:
            model = SpectrumTransformer(
                input_dim, output_dim,
                d_model=args.transformer_d_model,
                nhead=args.transformer_heads,
                num_layers=args.transformer_layers,
                output_activation=output_activation
            )
            use_temporal = False
    elif model_type == 'linear':
        model = LinearBaseline(input_dim, output_dim, output_activation=output_activation)
        use_temporal = False
    elif model_type == 'shallow':
        model = ShallowMLP(input_dim, output_dim, output_activation=output_activation)
        use_temporal = False
    elif model_type == 'banded_mlp':
        band_indices = make_band_indices(
            output_dim, wavelength_range=wavelength_range,
            num_bands=args.num_bands, band_boundaries_nm=args.band_boundaries_nm
        )
        model = BandedMLP(
            input_dim, output_dim, num_bands=args.num_bands,
            wavelength_range=wavelength_range,
            band_boundaries_nm=args.band_boundaries_nm,
            band_head_hidden=args.band_head_hidden,
            uv_band_head_hidden=args.uv_band_head_hidden,
            output_activation=output_activation
        )
        use_temporal = False
    else:
        model = StandardMLP(input_dim, output_dim, output_activation=output_activation)
        use_temporal = False

    return model, use_temporal, band_indices

def pad_z_for_eval(z_list, max_len):
    padded = []
    for z in z_list:
        z_arr = np.array(z, dtype=np.float32).flatten()
        if len(z_arr) == 0:
            z_arr = np.zeros(max_len, dtype=np.float32)
        elif len(z_arr) < max_len:
            z_arr = np.pad(z_arr, (0, max_len - len(z_arr)), mode='edge')
        else:
            z_arr = z_arr[:max_len]
        padded.append(z_arr)
    return np.array(padded, dtype=np.float32)

def collect_model_predictions(model, test_loader, use_temporal, device):
    model.eval()
    test_preds = []
    test_temporal_preds = []
    test_temporal_true = []
    test_z_positions = []

    with torch.no_grad():
        for batch_data in test_loader:
            if use_temporal and len(batch_data) == 4:
                data, y_temp, _, z_pos = batch_data
                data = data.to(device)
                z_pos = z_pos.to(device)
                temp_pred, pred = model(data, z_pos)
                test_preds.extend(pred.cpu().numpy())
                if temp_pred is not None:
                    test_temporal_preds.extend(temp_pred.cpu().numpy())
                test_temporal_true.extend(y_temp.numpy())
                test_z_positions.extend(z_pos.cpu().numpy())
            else:
                data, _ = batch_data
                data = data.to(device)
                _, pred = model(data)
                test_preds.extend(pred.cpu().numpy())

    return (
        np.array(test_preds),
        np.array(test_temporal_preds) if len(test_temporal_preds) else None,
        np.array(test_temporal_true) if len(test_temporal_true) else None,
        test_z_positions
    )

def run_eval_only(args, model_type, output_subdir):
    """Load a checkpoint and evaluate only the input directory test split."""
    if args.checkpoint is None:
        raise ValueError("--eval-only requires --checkpoint PATH")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(output_subdir, exist_ok=True)
    processing_params = load_processing_params(args.input_dir)
    wavelength_range = get_wavelength_range_nm(processing_params)
    spectrum_normalization = processing_params.get('spectrum_normalization', 'per_sample_minmax')
    output_normalization = processing_params.get('output_normalization', {'mode': spectrum_normalization})
    output_activation = resolve_output_activation(args.output_activation, spectrum_normalization)
    sample_filter = processing_params.get('sample_filter', 'unknown')
    processed_n_z = processing_params.get('n_z')
    early_dense_detected = bool(processing_params.get('early_dense_detected', False))
    if processed_n_z is not None:
        logging.info(
            "Eval temporal sampling: n_z=%s, sample_filter=%s, early_dense_detected=%s",
            processed_n_z, sample_filter, early_dense_detected
        )
    logging.info("Eval output activation: %s", output_activation)

    X_test = np.load(os.path.join(args.input_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(args.input_dir, 'y_test.npy'))
    if len(X_test) == 0:
        raise ValueError("X_test.npy is empty; eval-only requires at least one test sample")

    y_temporal_test = None
    z_test = None
    use_temporal_requested = model_type in ('lstm', 'temporal', 'temporal_cnn') or (
        model_type == 'transformer' and args.transformer_mode == 'temporal'
    )
    if use_temporal_requested:
        y_temporal_test = np.load(os.path.join(args.input_dir, 'y_temporal_test.npy'))
        z_test = np.load(os.path.join(args.input_dir, 'z_test.npy'), allow_pickle=True)

    source_files_test = None
    source_indices_test = None
    source_files_path = os.path.join(args.input_dir, 'source_files_test.npy')
    source_indices_path = os.path.join(args.input_dir, 'source_indices_test.npy')
    if os.path.exists(source_files_path) and os.path.exists(source_indices_path):
        source_files_test = np.load(source_files_path, allow_pickle=True)
        source_indices_test = np.load(source_indices_path)

    input_dim = X_test.shape[1]
    output_dim = y_test.shape[1]
    model, use_temporal, band_indices = build_model_for_type(
        model_type, input_dim, output_dim, args, wavelength_range,
        y_temporal=y_temporal_test,
        output_activation=output_activation
    )

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    if use_temporal:
        y_temp_test_t = torch.tensor(y_temporal_test, dtype=torch.float32)
        z_test_p = pad_z_for_eval(z_test, y_temporal_test.shape[1])
        z_test_t = torch.tensor(z_test_p, dtype=torch.float32)
        test_ds = TensorDataset(X_test_t, y_temp_test_t, y_test_t, z_test_t)
    else:
        test_ds = TensorDataset(X_test_t, y_test_t)

    pin_memory = args.pin_memory and torch.cuda.is_available() and not args.no_cuda
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker
    )

    metrics = evaluate_model(
        model, test_loader, use_temporal=use_temporal, device=device,
        wavelength_range=wavelength_range, uv_min_nm=args.uv_min_nm, uv_max_nm=args.uv_max_nm,
        num_bands=len(band_indices) if band_indices is not None else (args.num_bands if model_type == 'banded_mlp' else None),
        band_indices=band_indices,
        early_z_max_cm=args.early_z_max_cm,
        output_normalization=output_normalization
    )
    metrics.update({
        'Eval_Only': True,
        'Checkpoint': args.checkpoint,
        'Model': model_type,
        'Parameter_Count': count_parameters(model),
        'Uses_Temporal_Labels': bool(use_temporal),
        'Output_Activation': output_activation,
        'Spectrum_Normalization': spectrum_normalization,
        'Temporal_CNN_Architecture': args.temporal_cnn_architecture if model_type == 'temporal_cnn' else '',
        'Temporal_CNN_Band_Boundaries_Nm': args.temporal_cnn_band_boundaries_nm if model_type == 'temporal_cnn' else [],
        'Temporal_CNN_Coordinate_Channels': bool(args.temporal_cnn_coordinate_channels) if model_type == 'temporal_cnn' else False,
        'Transformer_Mode': args.transformer_mode if model_type == 'transformer' else '',
        'Transformer_D_Model': args.transformer_d_model if model_type == 'transformer' else 0,
        'Transformer_Heads': args.transformer_heads if model_type == 'transformer' else 0,
        'Transformer_Layers': args.transformer_layers if model_type == 'transformer' else 0,
        'Transformer_Use_Z_Embedding': bool(args.transformer_use_z_embedding) if model_type == 'transformer' else False,
        'Transformer_Band_Boundaries_Nm': args.transformer_band_boundaries_nm if model_type == 'transformer' else [],
        'Z_Gradient_Coordinate': args.z_gradient_coordinate,
    })

    logging.info("Eval-only metrics:")
    for k, v in metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating, bool)):
            logging.info("  %s: %.4f", k, float(v))
        else:
            logging.info("  %s: %s", k, v)

    with open(os.path.join(output_subdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    config = {
        'eval_only': True,
        'model': model_type,
        'checkpoint': args.checkpoint,
        'input_dir': args.input_dir,
        'output_dir': output_subdir,
        'batch_size': args.batch_size,
        'wavelength_range_nm': list(wavelength_range),
        'spectrum_normalization': spectrum_normalization,
        'output_activation': output_activation,
        'output_normalization': output_normalization,
        'log_integral_weight': args.log_integral_weight,
        'log_peak_weight': args.log_peak_weight,
        'sample_filter': sample_filter,
        'processed_n_z': processed_n_z,
        'early_dense_detected': early_dense_detected,
        'processing_params': processing_params,
        'band_boundaries_nm': args.band_boundaries_nm,
        'band_head_hidden': args.band_head_hidden,
        'uv_band_head_hidden': args.uv_band_head_hidden,
        'temporal_cnn_channels': args.temporal_cnn_channels,
        'temporal_cnn_lambda_init': args.temporal_cnn_lambda_init,
        'temporal_cnn_z_init': args.temporal_cnn_z_init,
        'temporal_cnn_architecture': args.temporal_cnn_architecture,
        'temporal_cnn_band_boundaries_nm': args.temporal_cnn_band_boundaries_nm,
        'temporal_cnn_coordinate_channels': bool(args.temporal_cnn_coordinate_channels),
        'transformer_mode': args.transformer_mode,
        'transformer_d_model': args.transformer_d_model,
        'transformer_heads': args.transformer_heads,
        'transformer_layers': args.transformer_layers,
        'transformer_use_z_embedding': bool(args.transformer_use_z_embedding),
        'transformer_band_boundaries_nm': args.transformer_band_boundaries_nm,
        'z_gradient_coordinate': args.z_gradient_coordinate,
        'uv_min_nm': args.uv_min_nm,
        'uv_max_nm': args.uv_max_nm,
    }
    with open(os.path.join(output_subdir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if args.visualize:
        test_preds, test_temporal_preds, test_temporal_true, test_z_positions = collect_model_predictions(
            model, test_loader, use_temporal, device
        )
        prediction_indices = plot_prediction_comparison(
            y_test, test_preds,
            os.path.join(output_subdir, 'predictions.png'),
            wavelength_range=wavelength_range,
            indices=args.visualize_indices,
            source_files=source_files_test,
            source_indices=source_indices_test
        )
        plot_prediction_comparison(
            y_test, test_preds,
            os.path.join(output_subdir, 'predictions_db.png'),
            wavelength_range=wavelength_range,
            indices=prediction_indices,
            source_files=source_files_test,
            source_indices=source_indices_test,
            y_scale='db'
        )
        if inverse_spectrum_normalization(y_test[:1], output_normalization) is not None:
            plot_prediction_comparison(
                y_test, test_preds,
                os.path.join(output_subdir, 'predictions_log_power.png'),
                wavelength_range=wavelength_range,
                indices=prediction_indices,
                source_files=source_files_test,
                source_indices=source_indices_test,
                y_scale='log_power',
                output_normalization=output_normalization
            )
            plot_prediction_comparison(
                y_test, test_preds,
                os.path.join(output_subdir, 'predictions_physical_db.png'),
                wavelength_range=wavelength_range,
                indices=prediction_indices,
                source_files=source_files_test,
                source_indices=source_indices_test,
                y_scale='physical_db',
                output_normalization=output_normalization
            )
        if use_temporal and test_temporal_preds is not None:
            temporal_indices = plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=args.visualize_indices
            )
            temporal_index_payload = {'indices': [int(idx) for idx in temporal_indices]}
            if source_indices_test is not None:
                temporal_index_payload['source_indices'] = [
                    int(source_indices_test[idx]) for idx in temporal_indices
                ]
            if source_files_test is not None:
                temporal_index_payload['source_files'] = [
                    str(source_files_test[idx]) for idx in temporal_indices
                ]
            with open(os.path.join(output_subdir, 'temporal_visualization_indices.json'), 'w') as f:
                json.dump(temporal_index_payload, f, indent=2)
            plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution_z0_2cm.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=temporal_indices,
                z_range_cm=(0.0, 2.0),
                title_suffix='z = 0-2 cm zoom'
            )
            plot_temporal_evolution_heatmap(
                test_temporal_preds, test_temporal_true, test_z_positions,
                os.path.join(output_subdir, 'temporal_evolution_z0_10cm.png'),
                wavelength_range=wavelength_range,
                n_samples=3,
                indices=temporal_indices,
                z_range_cm=(0.0, 10.0),
                title_suffix='z = 0-10 cm zoom'
            )

    return metrics

def main():
    args = parse_arguments()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("="*70)
    logging.info("Model Training v4.0 - Multi-thickness Anti-resonant Fiber")
    logging.info("="*70)
    
    if args.eval_only:
        if args.compare_all:
            raise ValueError("--eval-only does not support --compare-all; run one checkpoint/model at a time")
        run_eval_only(args, args.model, args.output_dir)
    elif args.compare_all:
        models = ['linear', 'shallow', 'mlp', 'temporal', 'lstm', 'transformer']
        results = {}
        
        for model_type in models:
            subdir = os.path.join(args.output_dir, f'experiment_{model_type}')
            metrics, history = run_experiment(args, model_type, subdir)
            results[model_type] = {'metrics': metrics, 'history': history}
        
        # Generate comparison report
        logging.info("\n" + "="*70)
        logging.info("Comparison Report")
        logging.info("="*70)
        for model_type, result in results.items():
            logging.info(f"\n{model_type.upper()}:")
            for k, v in result['metrics'].items():
                if isinstance(v, (int, float, np.integer, np.floating, bool)):
                    logging.info(f"  {k}: {float(v):.4f}")
                else:
                    logging.info(f"  {k}: {v}")
        
        # Save comparison
        with open(os.path.join(args.output_dir, 'comparison_report.json'), 'w') as f:
            json.dump({k: v['metrics'] for k, v in results.items()}, f, indent=2)
    else:
        run_experiment(args, args.model, args.output_dir)

if __name__ == "__main__":
    main()
