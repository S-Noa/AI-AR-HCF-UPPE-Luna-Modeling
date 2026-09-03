#!/usr/bin/env python3
"""
Data Preprocessing Script v4.0 - Anti-resonant Fiber Multi-thickness Support

Features:
  - Stratified sampling by wall thickness (ensuring consistent thickness ratios)
  - Data cleaning (outlier detection, missing value handling)
  - Feature engineering (polynomial features, thickness-as-category encoding)
  - Min-Max normalization with scaler persistence
  - Support for 4 thicknesses: 650nm, 360nm, 200nm, 80nm

Usage:
  python data_preprocessing.py --input-dir /path/to/data --output-dir processed_data
  python data_preprocessing.py --thickness 0.65 --output-dir processed_data_t650
"""

import os
import sys
import argparse
import h5py
import numpy as np
import re
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ============================================================================
# Configuration
# ============================================================================

WAVELENGTH_RANGE = (200e-9, 2500e-9)
TARGET_POINTS = 500
SPECTRUM_NORMALIZATION = 'per_sample_minmax'
EPSILON = 1e-15
C = 299792458

# Thickness categories (in meters)
THICKNESS_CATEGORIES = {
    0.65e-6: 'T650',
    0.36e-6: 'T360',
    0.20e-6: 'T200',
    0.08e-6: 'T080'
}

# ============================================================================
# Command Line Arguments
# ============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Preprocess anti-resonant fiber HDF5 data for ML training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_preprocessing.py --input-dir ./training_data --output-dir processed_data
  python data_preprocessing.py --thickness 0.65 --single-thickness-mode
  python data_preprocessing.py --test-size 0.15 --val-size 0.15
        """
    )
    parser.add_argument('--input-dir', '-i', type=str, default='training_data',
                        help='Input directory containing HDF5 files')
    parser.add_argument('--output-dir', '-o', type=str, default='processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--thickness', '-t', type=float, default=None,
                        help='Filter by specific wall thickness (in um, e.g., 0.65)')
    parser.add_argument('--single-thickness-mode', action='store_true',
                        help='Process only one thickness category')
    parser.add_argument('--test-size', type=float, default=0.15,
                        help='Test set proportion (default: 0.15)')
    parser.add_argument('--val-size', type=float, default=0.15,
                        help='Validation set proportion (default: 0.15)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='HDF5 reading batch size')
    parser.add_argument('--wavelength-min-nm', type=float, default=200.0,
                        help='Minimum wavelength for ML spectra in nm (default: 200)')
    parser.add_argument('--wavelength-max-nm', type=float, default=2500.0,
                        help='Maximum wavelength for ML spectra in nm (default: 2500)')
    parser.add_argument('--target-points', type=int, default=500,
                        help='Number of wavelength samples per spectrum (default: 500)')
    parser.add_argument('--spectrum-normalization', type=str, default='per_sample_minmax',
                        choices=['per_sample_minmax', 'global_log_standard', 'none_log', 'rnn_paper_db'],
                        help='Spectrum normalization mode')
    parser.add_argument('--sample-filter', type=str, default='normal',
                        choices=['normal', 'earlydense', 'all'],
                        help='Which HDF5 samples to process: normal excludes *_earlydense.h5, earlydense only includes them, all includes both')
    parser.add_argument('--feature-scaler', type=str, default=None,
                        help='Optional existing scaler_X.joblib to transform input features, required for eval-only use of trained models')
    parser.add_argument('--split-strategy', type=str, default='random',
                        choices=['random', 'holdout_feature', 'all_test'],
                        help='Train/val/test split strategy (default: random)')
    parser.add_argument('--holdout-feature', type=str, default='pressure',
                        choices=['energy', 'tau', 'pressure', 'diameter', 'wallthickness'],
                        help='Physical input feature used for holdout_feature split')
    parser.add_argument('--holdout-quantile-low', type=float, default=0.85,
                        help='Lower quantile of holdout feature for test set (default: 0.85)')
    parser.add_argument('--holdout-quantile-high', type=float, default=1.0,
                        help='Upper quantile of holdout feature for test set (default: 1.0)')
    return parser.parse_args()

# ============================================================================
# File Operations
# ============================================================================

def is_file_complete(filepath, min_size=100000):
    """Check if HDF5 file is complete and valid."""
    try:
        if not os.path.exists(filepath):
            return False, "File not found"
        file_size = os.path.getsize(filepath)
        if file_size < min_size:
            return False, f"File too small ({file_size} bytes)"
        with h5py.File(filepath, 'r') as f:
            required = ['Eω', 'z', 'stats', 'physics_features']
            missing = [d for d in required if d not in f]
            if missing:
                return False, f"Missing datasets: {', '.join(missing)}"
            if 'Eω' in f:
                shape = f['Eω'].shape
                if len(shape) < 2 or shape[0] == 0 or shape[1] == 0:
                    return False, f"Invalid Eω shape: {shape}"
        return True, "File complete"
    except Exception as e:
        return False, f"Error: {str(e)}"

# ============================================================================
# HDF5 Reading
# ============================================================================

def determine_eω_layout(Eω, z):
    """Auto-detect Eω dimension layout."""
    z_len = len(z)
    if z_len == Eω.shape[1]:
        return 'freq_z'
    elif z_len == Eω.shape[0]:
        return 'z_freq'
    logging.warning(f"Cannot determine Eω layout: z_len={z_len}, Eω_shape={Eω.shape}")
    return 'freq_z'

def get_freq_count(Eω, z):
    """Get frequency point count."""
    layout = determine_eω_layout(Eω, z)
    return Eω.shape[0] if layout == 'freq_z' else Eω.shape[1]

def parse_byte_str(val):
    """Parse Julia/HDF5 byte strings to Python values."""
    if isinstance(val, bytes):
        s = val.decode('utf-8').strip()
        if s in ('nothing', 'NaN', 'Inf', '-Inf'):
            return None
        if s in ('true', 'True'):
            return True
        if s in ('false', 'False'):
            return False
        if s.endswith('[]') and '[' not in s[:-2]:
            return None
        if s.startswith('[') and s.endswith(']'):
            inner = s[1:-1].strip()
            if not inner:
                return None
            try:
                return [float(x.strip()) for x in inner.split(',')]
            except ValueError:
                return s
        try:
            return float(s)
        except ValueError:
            return s
    return val

def read_hdf5_file(filepath):
    """Read HDF5 and extract all relevant data."""
    try:
        with h5py.File(filepath, 'r') as f:
            Eω = f['Eω'][:]
            z = f['z'][:]
            
            stats = {key: f['stats'][key][:] for key in f['stats']} if 'stats' in f else {}
            
            physics_features = {}
            if 'physics_features' in f:
                for key in f['physics_features']:
                    physics_features[key] = f['physics_features'][key][()]
            
            sim_params = {}
            if 'prop_capillary_args' in f:
                pca = f['prop_capillary_args']
                param_map = {'energy': 'energy', 'τfwhm': 'tau', 'ϕ': 'phi',
                             'λ0': 'lambda0', 'pressure': 'pressure',
                             'flength': 'length', 'radius': 'radius'}
                for hdf5_key, param_name in param_map.items():
                    if hdf5_key in pca:
                        val = parse_byte_str(pca[hdf5_key][()])
                        if val is not None:
                            sim_params[param_name] = val

            # Backward compatibility for older generated files that saved
            # user-facing parameters under an "input" group instead of
            # prop_capillary_args.
            if 'input' in f:
                inp = f['input']
                input_map = {
                    'energy': ('energy', 1e-6),      # stored in microjoules
                    'tau': ('tau', 1e-15),           # stored in femtoseconds
                    'pressure': ('pressure', 1.0),
                    'diameter': ('diameter', 1.0),   # stored in micrometers
                    'wallthickness': ('wallthickness', 1e-6),
                    'chirp': ('phi', 1.0),
                    'flength': ('length', 1.0),
                }
                for hdf5_key, (param_name, scale) in input_map.items():
                    if param_name not in sim_params and hdf5_key in inp:
                        val = safe_float(parse_byte_str(inp[hdf5_key][()]), default=None)
                        if val is not None:
                            sim_params[param_name] = val * scale
                if 'radius' not in sim_params and 'diameter' in sim_params:
                    sim_params['radius'] = sim_params['diameter'] * 0.5e-6
            
            # Get frequency grid
            ω_effective = None
            freq_count = get_freq_count(Eω, z)
            if 'grid' in f:
                grid = f['grid']
                for candidate in ['ω', 'omega', 'w']:
                    if candidate in grid:
                        ds = grid[candidate]
                        if ds.shape != () and len(ds[:]) == freq_count:
                            ω_effective = ds[:]
                            break
                if ω_effective is None and 'sidx' in grid and 'ωo' in grid:
                    sidx = grid['sidx'][:]
                    ωo = grid['ωo'][:]
                    ω_cropped = ωo[sidx]
                    if len(ω_cropped) == freq_count:
                        ω_effective = ω_cropped
            
            return Eω, z, stats, physics_features, ω_effective, sim_params
    except Exception as e:
        logging.error(f"Error reading {filepath}: {e}")
        return None, None, None, None, None, None

# ============================================================================
# Spectrum Processing
# ============================================================================

def estimate_wavelength(n_points):
    """Estimate wavelength array as fallback."""
    λ_min, λ_max = WAVELENGTH_RANGE
    freq = np.linspace(C / λ_max, C / λ_min, n_points)
    return C / freq

def spectrum_scale_features(log_spectrum, linear_power):
    """Return compact absolute-scale descriptors for one interpolated spectrum."""
    return np.array([
        np.nanmin(log_spectrum),
        np.nanmax(log_spectrum),
        np.nansum(log_spectrum),
        np.nanmax(linear_power),
        np.nansum(linear_power),
    ], dtype=np.float64)

def default_spectrum_quality():
    """Quality diagnostics for a spectrum that could not be processed."""
    return {
        'extrapolation_ratio': 1.0,
        'peak_index': -1,
        'peak_wavelength_nm': float('nan'),
        'edge_peak_flag': True,
        'log_dynamic_range': 0.0,
        'linear_power_sum': 0.0,
    }

def spectrum_quality(log_spectrum, linear_power, target_wavelength, extrapolation_ratio):
    """Return diagnostics that help identify normalization-amplified edge artifacts."""
    if log_spectrum.size == 0 or linear_power.size == 0:
        return default_spectrum_quality()
    peak_index = int(np.nanargmax(log_spectrum))
    edge_width = max(1, int(np.ceil(0.01 * len(log_spectrum))))
    edge_peak = peak_index < edge_width or peak_index >= len(log_spectrum) - edge_width
    return {
        'extrapolation_ratio': float(extrapolation_ratio),
        'peak_index': peak_index,
        'peak_wavelength_nm': float(target_wavelength[peak_index] * 1e9),
        'edge_peak_flag': bool(edge_peak),
        'log_dynamic_range': float(np.nanmax(log_spectrum) - np.nanmin(log_spectrum)),
        'linear_power_sum': float(np.nansum(linear_power)),
    }

def parse_source_sample_index(filepath):
    """Parse sample_000123.h5 style indices; return -1 when unavailable."""
    match = re.search(r'sample_(\d+)(?:_earlydense)?\.h5$', os.path.basename(filepath))
    return int(match.group(1)) if match else -1


def is_earlydense_filename(filename):
    return filename.endswith('_earlydense.h5')


def sample_file_matches_filter(filename, sample_filter):
    if not filename.endswith('.h5'):
        return False
    is_earlydense = is_earlydense_filename(filename)
    if sample_filter == 'normal':
        return not is_earlydense
    if sample_filter == 'earlydense':
        return is_earlydense
    return True

def process_spectrum(Eω_output, omega=None):
    """Process one spectrum to fixed-grid log-power data plus scale features."""
    try:
        n_freq = len(Eω_output)
        
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
        
        power = np.abs(Eω_output)**2
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
            logging.warning("Insufficient data points, returning zero spectrum")
            return np.zeros(TARGET_POINTS), np.zeros(5), default_spectrum_quality()
        
        target_wavelength = np.linspace(WAVELENGTH_RANGE[0], WAVELENGTH_RANGE[1], TARGET_POINTS)
        extrapolation_ratio = np.mean(
            (target_wavelength < wavelength_cropped[0]) |
            (target_wavelength > wavelength_cropped[-1])
        )
        if extrapolation_ratio > 0.05:
            logging.warning(
                "Spectrum interpolation uses endpoint fill for %.1f%% of target wavelengths",
                extrapolation_ratio * 100
            )
        spectrum_interp = np.interp(target_wavelength, wavelength_cropped, spectrum_cropped)
        
        log_spectrum = np.log10(spectrum_interp + EPSILON)
        scale_features = spectrum_scale_features(log_spectrum, spectrum_interp)
        quality = spectrum_quality(log_spectrum, spectrum_interp, target_wavelength, extrapolation_ratio)

        if SPECTRUM_NORMALIZATION == 'per_sample_minmax':
            min_val, max_val = log_spectrum.min(), log_spectrum.max()
            if max_val > min_val:
                processed = (log_spectrum - min_val) / (max_val - min_val)
            else:
                processed = np.zeros_like(log_spectrum)
        elif SPECTRUM_NORMALIZATION in ('global_log_standard', 'none_log'):
            processed = log_spectrum
        elif SPECTRUM_NORMALIZATION == 'rnn_paper_db':
            processed = spectrum_interp
        else:
            raise ValueError(f"Unsupported spectrum normalization: {SPECTRUM_NORMALIZATION}")
        
        return processed, scale_features, quality
    except Exception as e:
        logging.error(f"Spectrum processing error: {e}")
        return np.zeros(TARGET_POINTS), np.zeros(5), default_spectrum_quality()

def process_temporal_spectra(Eω, z, omega=None):
    """Process full z-sequence spectra and return sample-level scale features."""
    try:
        layout = determine_eω_layout(Eω, z)
        n_z = Eω.shape[1] if layout == 'freq_z' else Eω.shape[0]
        
        temporal_spectra = []
        scale_features = []
        qualities = []
        for i in range(n_z):
            spectrum_i = Eω[:, i] if layout == 'freq_z' else Eω[i, :]
            processed, scale_i, quality_i = process_spectrum(spectrum_i, omega=omega)
            temporal_spectra.append(processed)
            scale_features.append(scale_i)
            qualities.append(quality_i)
        
        temporal_spectra = np.array(temporal_spectra)
        scale_features = np.array(scale_features)
        final_quality = qualities[-1] if qualities else default_spectrum_quality()
        sample_scale = np.array([
            np.nanmin(scale_features[:, 0]),
            np.nanmax(scale_features[:, 1]),
            np.nanmean(scale_features[:, 2]),
            np.nanmax(scale_features[:, 3]),
            np.nanmean(scale_features[:, 4]),
        ], dtype=np.float64)
        max_extrapolation = max(q['extrapolation_ratio'] for q in qualities) if qualities else 1.0
        mean_extrapolation = float(np.mean([q['extrapolation_ratio'] for q in qualities])) if qualities else 1.0
        sample_quality = {
            'final_peak_wavelength_nm': final_quality['peak_wavelength_nm'],
            'final_peak_index': final_quality['peak_index'],
            'final_edge_peak_flag': final_quality['edge_peak_flag'],
            'max_extrapolation_ratio': float(max_extrapolation),
            'mean_extrapolation_ratio': float(mean_extrapolation),
            'sample_log_dynamic_range': float(np.nanmax(scale_features[:, 1]) - np.nanmin(scale_features[:, 0])),
            'final_log_dynamic_range': final_quality['log_dynamic_range'],
            'linear_power_sum_mean': float(np.nanmean(scale_features[:, 4])),
        }
        return temporal_spectra, sample_scale, sample_quality
    except Exception as e:
        logging.error(f"Temporal spectra processing error: {e}")
        return None, None, None

# ============================================================================
# Feature Extraction
# ============================================================================

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        if isinstance(value, complex):
            return float(value.real)
        if isinstance(value, (list, tuple)):
            arr = np.asarray(value).flatten()
            if len(arr) > 0:
                v = arr[0]
                return float(v.real) if isinstance(v, complex) else float(v)
            return default
        if hasattr(value, '__len__') and not isinstance(value, str):
            arr = np.asarray(value).flatten()
            if len(arr) > 0:
                v = arr[0]
                return float(v.real) if isinstance(v, complex) else float(v)
            return default
        return float(value)
    except:
        return default

def get_thickness_category(wallthickness_m):
    """Map wall thickness to category label."""
    wt_um = wallthickness_m * 1e6
    # Find closest category
    categories = {0.65: 'T650', 0.36: 'T360', 0.20: 'T200', 0.08: 'T080'}
    closest = min(categories.keys(), key=lambda x: abs(x - wt_um))
    if abs(closest - wt_um) < 0.05:  # 50nm tolerance
        return categories[closest]
    return None

def extract_features(physics_features, sim_params=None):
    """
    Extract 13-dimensional input features.
    
    Feature order:
    [energy, tau, pressure, diameter, wallthickness,
     beta2, gamma, N, L0, gamma_K, P_ratio, Aeff, neff]
    """
    features = []
    
    if sim_params:
        features.append(safe_float(sim_params.get('energy', 0)))
        features.append(safe_float(sim_params.get('tau', 0)))
        features.append(safe_float(sim_params.get('pressure', 0)))
        radius = safe_float(sim_params.get('radius', 0))
        features.append(radius * 2 * 1e6 if radius > 0 else 0.0)
        wallthickness = safe_float(physics_features.get('wallthickness', 0.5e-6))
        features.append(wallthickness * 1e6 if wallthickness > 0 else 0.5)
    else:
        features.extend([0.0]*5)
    
    feature_keys = ['beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff']
    for key in feature_keys:
        value = physics_features.get(key, 0.0)
        features.append(safe_float(value))
    
    result = np.array(features, dtype=np.float64)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result

# ============================================================================
# Data Cleaning
# ============================================================================

def clean_data(X, y_temporal, y_final, z_positions, thickness_labels, scale_features=None,
               source_files=None, source_paths=None, source_indices=None, quality_rows=None):
    """
    Clean dataset by removing invalid samples.
    
    Cleaning rules:
    1. Remove samples with NaN/Inf in features
    2. Remove samples with NaN/Inf in spectra
    3. Remove outliers based on Z-score (>3) for energy and pressure
    4. Remove samples with all-zero spectra
    """
    n_original = len(X)
    valid_mask = np.ones(n_original, dtype=bool)
    
    # Rule 1: NaN/Inf in features
    valid_mask &= np.isfinite(X).all(axis=1)
    
    # Rule 2: NaN/Inf in temporal spectra
    for i in range(n_original):
        if not np.isfinite(y_temporal[i]).all():
            valid_mask[i] = False
    
    # Rule 3: Z-score outlier detection for energy (feature 0) and pressure (feature 2)
    from scipy import stats
    for feat_idx in [0, 2]:  # energy, pressure
        z_scores = np.abs(stats.zscore(X[:, feat_idx]))
        valid_mask &= (z_scores < 3.0)
    
    # Rule 4: All-zero spectra
    for i in range(n_original):
        if y_temporal[i].max() < 1e-10:
            valid_mask[i] = False
    
    n_removed = n_original - np.sum(valid_mask)
    logging.info(f"Data cleaning: removed {n_removed}/{n_original} samples ({n_removed/n_original*100:.1f}%)")
    
    cleaned = (X[valid_mask], y_temporal[valid_mask], y_final[valid_mask],
               z_positions[valid_mask], [thickness_labels[i] for i in range(n_original) if valid_mask[i]])
    extras = []
    if scale_features is not None:
        extras.append(scale_features[valid_mask])
    if source_files is not None:
        extras.append(np.asarray(source_files, dtype=object)[valid_mask])
    if source_paths is not None:
        extras.append(np.asarray(source_paths, dtype=object)[valid_mask])
    if source_indices is not None:
        extras.append(np.asarray(source_indices, dtype=np.int64)[valid_mask])
    if quality_rows is not None:
        extras.append(np.asarray(quality_rows, dtype=object)[valid_mask])
    if extras:
        return cleaned + tuple(extras)
    return cleaned

# ============================================================================
# Stratified Splitting
# ============================================================================

def stratified_split(X, y_temporal, y_final, z_positions, thickness_labels, test_size, val_size, random_state,
                     scale_features=None, source_files=None, source_paths=None, source_indices=None, quality_rows=None,
                     split_strategy='random', holdout_feature='pressure',
                     holdout_quantile_low=0.85, holdout_quantile_high=1.0):
    """
    Stratified train/val/test split ensuring consistent thickness ratios.
    
    Strategy:
    1. First split: train (1-test_size-val_size) / temp (test_size+val_size)
    2. Second split: val (val_size/(test_size+val_size)) / test (test_size/(test_size+val_size))
    """
    def can_stratify(labels, min_count=2):
        _, counts = np.unique(labels, return_counts=True)
        return len(counts) > 1 and np.all(counts >= min_count)

    # Create thickness category array for stratification. Upstream currently
    # passes labels like "T650"; keep compatibility with older callers that
    # may pass wall thickness values in meters.
    thickness_array = np.array([
        t if isinstance(t, str) else THICKNESS_CATEGORIES.get(t, 'UNKNOWN')
        for t in thickness_labels
    ])
    stratify_first = thickness_array if can_stratify(thickness_array) else None
    if stratify_first is None:
        logging.warning("Stratified split disabled: not enough samples per thickness category")
    
    indices = np.arange(len(X))
    feature_indices = {
        'energy': 0,
        'tau': 1,
        'pressure': 2,
        'diameter': 3,
        'wallthickness': 4,
    }
    split_metadata = {'split_strategy': split_strategy}

    if split_strategy == 'all_test':
        train_idx = indices[:0]
        val_idx = indices[:0]
        test_idx = indices
        split_metadata.update({'eval_only': True, 'test_size': int(len(test_idx))})
        logging.info("all_test split: using all %d samples as the test/evaluation set", len(test_idx))
    elif split_strategy == 'holdout_feature':
        if not (0.0 <= holdout_quantile_low < holdout_quantile_high <= 1.0):
            raise ValueError("holdout quantiles must satisfy 0 <= low < high <= 1")
        feat_idx = feature_indices[holdout_feature]
        feature_values = X[:, feat_idx]
        q_low = float(np.quantile(feature_values, holdout_quantile_low))
        q_high = float(np.quantile(feature_values, holdout_quantile_high))
        test_mask = (feature_values >= q_low) & (feature_values <= q_high)
        test_idx = indices[test_mask]
        remaining_idx = indices[~test_mask]
        if len(test_idx) == 0 or len(remaining_idx) < 2:
            raise ValueError(
                f"holdout_feature split produced test={len(test_idx)}, remaining={len(remaining_idx)} samples"
            )

        thick_remaining = thickness_array[remaining_idx]
        stratify_remaining = thick_remaining if can_stratify(thick_remaining) else None
        if stratify_remaining is None:
            logging.warning("Train/val stratification disabled for holdout split")
        train_rel_idx, val_rel_idx = train_test_split(
            np.arange(len(remaining_idx)), test_size=val_size,
            random_state=random_state, stratify=stratify_remaining
        )
        train_idx = remaining_idx[train_rel_idx]
        val_idx = remaining_idx[val_rel_idx]
        logging.info(
            "Holdout split: feature=%s, quantiles=[%.3f, %.3f], values=[%.6g, %.6g], test=%d",
            holdout_feature, holdout_quantile_low, holdout_quantile_high, q_low, q_high, len(test_idx)
        )
        split_metadata.update({
            'holdout_feature': holdout_feature,
            'holdout_quantile_low': float(holdout_quantile_low),
            'holdout_quantile_high': float(holdout_quantile_high),
            'holdout_value_low': q_low,
            'holdout_value_high': q_high,
        })
    else:
        # First split: train / temp
        temp_size = test_size + val_size
        train_idx, temp_idx = train_test_split(
            indices, test_size=temp_size, random_state=random_state, stratify=stratify_first
        )

        # Second split: val / test
        thick_temp = thickness_array[temp_idx]
        stratify_second = thick_temp if can_stratify(thick_temp) else None
        if stratify_second is None:
            logging.warning("Validation/test stratification disabled: not enough temp samples per thickness category")
        val_rel_idx, test_rel_idx = train_test_split(
            np.arange(len(temp_idx)), test_size=test_size / temp_size,
            random_state=random_state, stratify=stratify_second
        )
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
        split_metadata.update({'test_size': float(test_size), 'val_size': float(val_size)})

    thick_train = thickness_array[train_idx]
    thick_val = thickness_array[val_idx]
    thick_test = thickness_array[test_idx]
    
    logging.info(f"Stratified split:")
    logging.info(f"  Train: {len(train_idx)} samples, thickness dist: {dict(zip(*np.unique(thick_train, return_counts=True)))}")
    logging.info(f"  Val: {len(val_idx)} samples, thickness dist: {dict(zip(*np.unique(thick_val, return_counts=True)))}")
    logging.info(f"  Test: {len(test_idx)} samples, thickness dist: {dict(zip(*np.unique(thick_test, return_counts=True)))}")
    
    def take(arr, idx):
        if arr is None:
            return None
        return np.asarray(arr, dtype=object)[idx] if np.asarray(arr).dtype == object else np.asarray(arr)[idx]

    return {
        'X_train': X[train_idx], 'X_val': X[val_idx], 'X_test': X[test_idx],
        'y_temp_train': y_temporal[train_idx], 'y_temp_val': y_temporal[val_idx], 'y_temp_test': y_temporal[test_idx],
        'y_final_train': y_final[train_idx], 'y_final_val': y_final[val_idx], 'y_final_test': y_final[test_idx],
        'z_train': z_positions[train_idx], 'z_val': z_positions[val_idx], 'z_test': z_positions[test_idx],
        'scale_train': take(scale_features, train_idx), 'scale_val': take(scale_features, val_idx), 'scale_test': take(scale_features, test_idx),
        'source_files_train': take(source_files, train_idx), 'source_files_val': take(source_files, val_idx), 'source_files_test': take(source_files, test_idx),
        'source_paths_train': take(source_paths, train_idx), 'source_paths_val': take(source_paths, val_idx), 'source_paths_test': take(source_paths, test_idx),
        'source_indices_train': take(source_indices, train_idx), 'source_indices_val': take(source_indices, val_idx), 'source_indices_test': take(source_indices, test_idx),
        'quality_train': take(quality_rows, train_idx), 'quality_val': take(quality_rows, val_idx), 'quality_test': take(quality_rows, test_idx),
        'split_metadata': split_metadata,
    }

# ============================================================================
# Feature Engineering
# ============================================================================

def add_polynomial_features(X):
    """Add polynomial interaction features."""
    # energy * pressure interaction
    energy_pressure = X[:, 0:1] * X[:, 2:3]
    # diameter * pressure interaction
    diameter_pressure = X[:, 3:4] * X[:, 2:3]

    return np.hstack([X, energy_pressure, diameter_pressure])

def encode_thickness_onehot(X, thickness_labels):
    """One-hot encode thickness categories."""
    categories = ['T650', 'T360', 'T200', 'T080']
    onehot = np.zeros((len(X), len(categories)))
    for i, label in enumerate(thickness_labels):
        if label in categories:
            idx = categories.index(label)
            onehot[i, idx] = 1.0
    return np.hstack([X, onehot])

def write_quality_csv(filepath, source_files, source_indices, quality_rows):
    """Write per-sample spectrum quality diagnostics."""
    fieldnames = [
        'split_index', 'source_file', 'source_index',
        'final_peak_wavelength_nm', 'final_peak_index', 'final_edge_peak_flag',
        'max_extrapolation_ratio', 'mean_extrapolation_ratio',
        'sample_log_dynamic_range', 'final_log_dynamic_range',
        'linear_power_sum_mean', 'suspicious'
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, quality in enumerate(quality_rows):
            q = dict(quality)
            suspicious = (
                bool(q.get('final_edge_peak_flag', False)) or
                float(q.get('max_extrapolation_ratio', 0.0)) > 0.05 or
                (
                    float(q.get('sample_log_dynamic_range', 0.0)) < 1.0 and
                    bool(q.get('final_edge_peak_flag', False))
                )
            )
            row = {
                'split_index': i,
                'source_file': str(source_files[i]) if source_files is not None else '',
                'source_index': int(source_indices[i]) if source_indices is not None else -1,
                'suspicious': suspicious,
            }
            row.update(q)
            writer.writerow(row)

def write_suspicious_csv(filepath, source_files, source_indices, quality_rows):
    """Write only suspicious samples for quick manual review."""
    suspicious_rows = []
    for i, quality in enumerate(quality_rows):
        q = dict(quality)
        suspicious = (
            bool(q.get('final_edge_peak_flag', False)) or
            float(q.get('max_extrapolation_ratio', 0.0)) > 0.05 or
            (
                float(q.get('sample_log_dynamic_range', 0.0)) < 1.0 and
                bool(q.get('final_edge_peak_flag', False))
            )
        )
        if suspicious:
            row = {
                'split_index': i,
                'source_file': str(source_files[i]) if source_files is not None else '',
                'source_index': int(source_indices[i]) if source_indices is not None else -1,
            }
            row.update(q)
            suspicious_rows.append(row)

    fieldnames = [
        'split_index', 'source_file', 'source_index',
        'final_peak_wavelength_nm', 'final_peak_index', 'final_edge_peak_flag',
        'max_extrapolation_ratio', 'mean_extrapolation_ratio',
        'sample_log_dynamic_range', 'final_log_dynamic_range',
        'linear_power_sum_mean'
    ]
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in suspicious_rows:
            writer.writerow(row)

# ============================================================================
# Main Processing
# ============================================================================

def main():
    global WAVELENGTH_RANGE, TARGET_POINTS, SPECTRUM_NORMALIZATION
    args = parse_arguments()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.wavelength_min_nm >= args.wavelength_max_nm:
        raise ValueError("--wavelength-min-nm must be smaller than --wavelength-max-nm")
    if args.target_points < 10:
        raise ValueError("--target-points must be at least 10")
    WAVELENGTH_RANGE = (args.wavelength_min_nm * 1e-9, args.wavelength_max_nm * 1e-9)
    TARGET_POINTS = args.target_points
    SPECTRUM_NORMALIZATION = args.spectrum_normalization
    
    logging.info("="*70)
    logging.info("Data Preprocessing v4.0 - Multi-thickness Anti-resonant Fiber")
    logging.info("="*70)
    logging.info(f"Input dir: {input_dir}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"Test size: {args.test_size}, Val size: {args.val_size}")
    logging.info(f"Wavelength range: {args.wavelength_min_nm}-{args.wavelength_max_nm} nm")
    logging.info(f"Target points: {TARGET_POINTS}")
    logging.info(f"Spectrum normalization: {SPECTRUM_NORMALIZATION}")
    logging.info(f"Sample filter: {args.sample_filter}")
    
    # Find all HDF5 files
    hdf5_files = []
    skipped_files = []
    filter_skipped_count = 0
    for filename in os.listdir(input_dir):
        if not filename.endswith('.h5'):
            continue
        if not sample_file_matches_filter(filename, args.sample_filter):
            filter_skipped_count += 1
            continue
        filepath = os.path.join(input_dir, filename)
        is_complete, reason = is_file_complete(filepath)
        if is_complete:
            hdf5_files.append(filepath)
        else:
            skipped_files.append((filename, reason))
    
    if skipped_files:
        logging.info(f"Skipped {len(skipped_files)} files")
    if filter_skipped_count:
        logging.info(f"Filtered out {filter_skipped_count} HDF5 files by --sample-filter={args.sample_filter}")
    
    hdf5_files = sorted(hdf5_files)
    logging.info(f"Found {len(hdf5_files)} valid HDF5 files")
    
    # Process files
    all_features = []
    all_temporal_spectra = []
    all_final_spectra = []
    all_z_positions = []
    all_thickness_labels = []
    all_scale_features = []
    all_source_files = []
    all_source_paths = []
    all_source_indices = []
    all_quality_rows = []
    
    for i, filepath in enumerate(hdf5_files):
        if i % 100 == 0:
            logging.info(f"Processing {i+1}/{len(hdf5_files)}: {os.path.basename(filepath)}")
        
        Eω, z, stats, physics_features, omega, sim_params = read_hdf5_file(filepath)
        if Eω is None:
            continue
        
        # Extract features
        features = extract_features(physics_features, sim_params=sim_params)
        
        # Get thickness category
        wallthickness = safe_float(physics_features.get('wallthickness', 0.5e-6))
        thickness_cat = get_thickness_category(wallthickness)
        
        # Filter by thickness if specified
        if args.thickness is not None:
            target_wt = args.thickness * 1e-6
            if abs(wallthickness - target_wt) > 0.05e-6:
                continue
        
        # Process temporal spectra
        temporal_spectra, scale_features, quality_row = process_temporal_spectra(Eω, z, omega=omega)
        if temporal_spectra is None:
            continue
        
        final_spectrum = temporal_spectra[-1, :]
        
        all_features.append(features)
        all_temporal_spectra.append(temporal_spectra)
        all_final_spectra.append(final_spectrum)
        all_z_positions.append(z.copy())
        all_thickness_labels.append(thickness_cat if thickness_cat else 'UNKNOWN')
        all_scale_features.append(scale_features)
        all_source_files.append(os.path.basename(filepath))
        all_source_paths.append(filepath)
        all_source_indices.append(parse_source_sample_index(filepath))
        all_quality_rows.append(quality_row)
    
    if not all_features:
        logging.error("No valid samples processed!")
        return

    n_z_values = sorted({arr.shape[0] for arr in all_temporal_spectra})
    if len(n_z_values) != 1:
        raise ValueError(
            "Selected samples have inconsistent z point counts "
            f"{n_z_values}. Do not mix normal samples with early-dense samples, "
            "and do not mix different --early-dense-saveN settings. "
            "Use --sample-filter normal or --sample-filter earlydense."
        )
    n_z_points = n_z_values[0]
    early_dense_detected = any(is_earlydense_filename(name) for name in all_source_files)
    
    X = np.array(all_features)
    y_temporal = np.array(all_temporal_spectra)
    y_final = np.array(all_final_spectra)
    z_positions = np.array(all_z_positions, dtype=object)
    scale_features = np.array(all_scale_features)
    source_files = np.array(all_source_files, dtype=object)
    source_paths = np.array(all_source_paths, dtype=object)
    source_indices = np.array(all_source_indices, dtype=np.int64)
    quality_rows = np.array(all_quality_rows, dtype=object)
    
    logging.info(f"\nRaw data: X={X.shape}, y_temporal={y_temporal.shape}, y_final={y_final.shape}")
    logging.info(f"Temporal z points per sample: {n_z_points}")
    logging.info(f"Thickness distribution: {dict(zip(*np.unique(all_thickness_labels, return_counts=True)))}")
    
    # Data cleaning
    (X, y_temporal, y_final, z_positions, thickness_labels, scale_features,
     source_files, source_paths, source_indices, quality_rows) = clean_data(
        X, y_temporal, y_final, z_positions, all_thickness_labels,
        scale_features=scale_features,
        source_files=source_files,
        source_paths=source_paths,
        source_indices=source_indices,
        quality_rows=quality_rows
    )
    
    logging.info(f"After cleaning: X={X.shape}, y_temporal={y_temporal.shape}")
    
    # Feature engineering: polynomial interactions (without thickness interaction)
    X_poly = add_polynomial_features(X)
    logging.info(f"After polynomial features: X={X_poly.shape}")
    
    # Stratified split (without one-hot encoding - thickness is handled by separate models)
    splits = stratified_split(
        X_poly, y_temporal, y_final, z_positions, thickness_labels,
        args.test_size, args.val_size, args.random_seed,
        scale_features=scale_features,
        source_files=source_files,
        source_paths=source_paths,
        source_indices=source_indices,
        quality_rows=quality_rows,
        split_strategy=args.split_strategy,
        holdout_feature=args.holdout_feature,
        holdout_quantile_low=args.holdout_quantile_low,
        holdout_quantile_high=args.holdout_quantile_high
    )
    X_train, X_val, X_test = splits['X_train'], splits['X_val'], splits['X_test']
    y_temp_train, y_temp_val, y_temp_test = splits['y_temp_train'], splits['y_temp_val'], splits['y_temp_test']
    y_final_train, y_final_val, y_final_test = splits['y_final_train'], splits['y_final_val'], splits['y_final_test']
    z_train, z_val, z_test = splits['z_train'], splits['z_val'], splits['z_test']
    eval_only = args.split_strategy == 'all_test'
    
    # Min-Max normalization for input features. For eval-only datasets, pass
    # the training run's scaler_X.joblib so checkpoint inputs use the same scale.
    import joblib
    if args.feature_scaler is not None:
        logging.info("Loading input feature scaler from: %s", args.feature_scaler)
        scaler_X = joblib.load(args.feature_scaler)
    else:
        if eval_only:
            logging.warning(
                "all_test preprocessing without --feature-scaler fits a scaler on evaluation samples; "
                "use the training scaler for checkpoint evaluation."
            )
        scaler_X = MinMaxScaler()
        scaler_fit_X = X_test if eval_only else X_train
        if len(scaler_fit_X) == 0:
            raise ValueError("No samples available to fit input feature scaler")
        scaler_X.fit(scaler_fit_X)
    X_train_scaled = scaler_X.transform(X_train) if len(X_train) else np.empty((0, X.shape[1]), dtype=np.float64)
    X_val_scaled = scaler_X.transform(X_val) if len(X_val) else np.empty((0, X.shape[1]), dtype=np.float64)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Temporal output normalization.
    # per_sample_minmax preserves legacy behavior; global_log_standard keeps
    # between-sample intensity differences in log-power space. rnn_paper_db
    # follows the Salmela RNN baseline style: global maximum, dB clipping, [0,1].
    output_normalization = {'mode': SPECTRUM_NORMALIZATION}
    if SPECTRUM_NORMALIZATION == 'per_sample_minmax':
        y_temp_train_scaled = np.zeros_like(y_temp_train)
        for i in range(len(y_temp_train)):
            sample_min = y_temp_train[i].min()
            sample_max = y_temp_train[i].max()
            if sample_max > sample_min:
                y_temp_train_scaled[i] = (y_temp_train[i] - sample_min) / (sample_max - sample_min)

        y_temp_val_scaled = np.zeros_like(y_temp_val)
        for i in range(len(y_temp_val)):
            sample_min = y_temp_val[i].min()
            sample_max = y_temp_val[i].max()
            if sample_max > sample_min:
                y_temp_val_scaled[i] = (y_temp_val[i] - sample_min) / (sample_max - sample_min)

        y_temp_test_scaled = np.zeros_like(y_temp_test)
        for i in range(len(y_temp_test)):
            sample_min = y_temp_test[i].min()
            sample_max = y_temp_test[i].max()
            if sample_max > sample_min:
                y_temp_test_scaled[i] = (y_temp_test[i] - sample_min) / (sample_max - sample_min)
    elif SPECTRUM_NORMALIZATION == 'global_log_standard':
        reference_temporal = y_temp_test if eval_only else y_temp_train
        log_mean = float(np.mean(reference_temporal))
        log_std = float(np.std(reference_temporal))
        if log_std <= EPSILON:
            logging.warning("Training log spectrum std is near zero; using std=1.0")
            log_std = 1.0
        y_temp_train_scaled = (y_temp_train - log_mean) / log_std
        y_temp_val_scaled = (y_temp_val - log_mean) / log_std
        y_temp_test_scaled = (y_temp_test - log_mean) / log_std
        output_normalization.update({'log_mean': log_mean, 'log_std': log_std})
    elif SPECTRUM_NORMALIZATION == 'none_log':
        y_temp_train_scaled = y_temp_train.copy()
        y_temp_val_scaled = y_temp_val.copy()
        y_temp_test_scaled = y_temp_test.copy()
    elif SPECTRUM_NORMALIZATION == 'rnn_paper_db':
        reference_temporal = y_temp_test if eval_only else y_temp_train
        rnn_db_reference_max = float(np.max(np.abs(reference_temporal)))
        rnn_db_floor = -55.0
        rnn_db_eps = EPSILON
        if rnn_db_reference_max <= rnn_db_eps:
            logging.warning("RNN paper dB reference max is near zero; using 1.0")
            rnn_db_reference_max = 1.0

        def rnn_paper_db_scale(arr):
            normalized = np.maximum(np.asarray(arr, dtype=np.float64) / rnn_db_reference_max, rnn_db_eps)
            db = 10.0 * np.log10(normalized)
            db = np.clip(db, rnn_db_floor, 0.0)
            return (db / abs(rnn_db_floor) + 1.0).astype(np.float32)

        y_temp_train_scaled = rnn_paper_db_scale(y_temp_train)
        y_temp_val_scaled = rnn_paper_db_scale(y_temp_val)
        y_temp_test_scaled = rnn_paper_db_scale(y_temp_test)
        output_normalization.update({
            'rnn_db_reference_max': rnn_db_reference_max,
            'rnn_db_floor': rnn_db_floor,
            'rnn_db_eps': rnn_db_eps,
            'target_range': [0.0, 1.0],
        })
    else:
        raise ValueError(f"Unsupported spectrum normalization: {SPECTRUM_NORMALIZATION}")

    # Save scaler
    joblib.dump(scaler_X, os.path.join(output_dir, 'scaler_X.joblib'))

    # Save processed data
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_scaled)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val_scaled)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test_scaled)

    np.save(os.path.join(output_dir, 'y_temporal_train.npy'), y_temp_train_scaled)
    np.save(os.path.join(output_dir, 'y_temporal_val.npy'), y_temp_val_scaled)
    np.save(os.path.join(output_dir, 'y_temporal_test.npy'), y_temp_test_scaled)
    
    # y_final must be extracted from the normalized temporal tensor so the final
    # and temporal training targets stay exactly consistent for every mode.
    y_final_train_scaled = y_temp_train_scaled[:, -1, :]
    y_final_val_scaled = y_temp_val_scaled[:, -1, :]
    y_final_test_scaled = y_temp_test_scaled[:, -1, :]

    np.save(os.path.join(output_dir, 'y_train.npy'), y_final_train_scaled)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_final_val_scaled)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_final_test_scaled)
    
    np.save(os.path.join(output_dir, 'z_train.npy'), z_train, allow_pickle=True)
    np.save(os.path.join(output_dir, 'z_val.npy'), z_val, allow_pickle=True)
    np.save(os.path.join(output_dir, 'z_test.npy'), z_test, allow_pickle=True)
    np.save(os.path.join(output_dir, 'spectrum_scale_features.npy'), scale_features)
    np.save(os.path.join(output_dir, 'spectrum_scale_features_train.npy'), splits['scale_train'])
    np.save(os.path.join(output_dir, 'spectrum_scale_features_val.npy'), splits['scale_val'])
    np.save(os.path.join(output_dir, 'spectrum_scale_features_test.npy'), splits['scale_test'])
    np.save(os.path.join(output_dir, 'source_files_train.npy'), splits['source_files_train'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_files_val.npy'), splits['source_files_val'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_files_test.npy'), splits['source_files_test'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_paths_train.npy'), splits['source_paths_train'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_paths_val.npy'), splits['source_paths_val'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_paths_test.npy'), splits['source_paths_test'], allow_pickle=True)
    np.save(os.path.join(output_dir, 'source_indices_train.npy'), splits['source_indices_train'])
    np.save(os.path.join(output_dir, 'source_indices_val.npy'), splits['source_indices_val'])
    np.save(os.path.join(output_dir, 'source_indices_test.npy'), splits['source_indices_test'])
    write_quality_csv(
        os.path.join(output_dir, 'spectrum_quality_train.csv'),
        splits['source_files_train'], splits['source_indices_train'], splits['quality_train']
    )
    write_quality_csv(
        os.path.join(output_dir, 'spectrum_quality_val.csv'),
        splits['source_files_val'], splits['source_indices_val'], splits['quality_val']
    )
    write_quality_csv(
        os.path.join(output_dir, 'spectrum_quality_test.csv'),
        splits['source_files_test'], splits['source_indices_test'], splits['quality_test']
    )
    write_suspicious_csv(
        os.path.join(output_dir, 'suspicious_samples_test.csv'),
        splits['source_files_test'], splits['source_indices_test'], splits['quality_test']
    )
    
    # Save processing parameters
    processing_params = {
        'version': 'v4.4-source-tracking-quality-sample-filter',
        'input_features': ['energy', 'tau', 'pressure', 'diameter', 'wallthickness',
                          'beta2', 'gamma', 'N', 'L0', 'gamma_K', 'P_ratio', 'Aeff', 'neff',
                          'energy_pressure', 'diameter_pressure'],
        'feature_dim': X_poly.shape[1],
        'data_format': {
            'X': f'(N, {X_poly.shape[1]}) - input features without one-hot thickness',
            'y_temporal': f'(N, n_z, {TARGET_POINTS}) - temporal spectra',
            'y_final': f'(N, {TARGET_POINTS}) - final spectrum',
            'spectrum_scale_features': '(N, 5) - log/intensity scale descriptors',
            'source_files_*': '(N,) - source HDF5 basename for each split',
            'source_indices_*': '(N,) - parsed sample index from sample_XXXXXX.h5',
            'spectrum_quality_*': 'CSV diagnostics for interpolation, edge peaks, and dynamic range'
        },
        'wavelength_range_m': list(WAVELENGTH_RANGE),
        'wavelength_range_nm': [args.wavelength_min_nm, args.wavelength_max_nm],
        'target_points': TARGET_POINTS,
        'sample_filter': args.sample_filter,
        'early_dense_detected': bool(early_dense_detected),
        'n_z': int(n_z_points),
        'spectrum_normalization': SPECTRUM_NORMALIZATION,
        'feature_scaler_source': args.feature_scaler,
        'output_normalization': output_normalization,
        'spectrum_scale_feature_names': [
            'log_power_min',
            'log_power_max',
            'log_power_sum_mean',
            'linear_power_peak',
            'linear_power_sum_mean'
        ],
        'legacy_notes': {
            'spectrum_scale_features.npy': 'All cleaned samples before train/val/test split; do not index it with y_test.',
            'spectrum_scale_features_train_val_test': 'Use split-specific spectrum_scale_features_*.npy for aligned metadata.'
        },
        'thickness_categories': list(THICKNESS_CATEGORIES.values()),
        'split_ratio': (
            {'train': 0.0, 'val': 0.0, 'test': 1.0} if eval_only
            else {'train': 1-args.test_size-args.val_size, 'val': args.val_size, 'test': args.test_size}
        ),
        'split_sizes': {
            'train': int(len(X_train_scaled)),
            'val': int(len(X_val_scaled)),
            'test': int(len(X_test_scaled))
        },
        'split_strategy': args.split_strategy,
        'eval_only': bool(eval_only),
        'holdout_feature': args.holdout_feature if args.split_strategy == 'holdout_feature' else None,
        'holdout_quantile_low': args.holdout_quantile_low if args.split_strategy == 'holdout_feature' else None,
        'holdout_quantile_high': args.holdout_quantile_high if args.split_strategy == 'holdout_feature' else None,
        'split_metadata': splits.get('split_metadata', {}),
        'random_seed': args.random_seed,
        'wallthickness_range': [0.08, 0.65],
        'architecture': 'Independent model per thickness (no one-hot encoding)'
    }
    
    with open(os.path.join(output_dir, 'processing_params.json'), 'w') as f:
        json.dump(processing_params, f, indent=4)
    
    logging.info("\n" + "="*70)
    logging.info("Data preprocessing completed!")
    logging.info("="*70)
    logging.info(f"Train: X={X_train_scaled.shape}, y_temporal={y_temp_train_scaled.shape}")
    logging.info(f"Val: X={X_val_scaled.shape}, y_temporal={y_temp_val_scaled.shape}")
    logging.info(f"Test: X={X_test_scaled.shape}, y_temporal={y_temp_test_scaled.shape}")
    logging.info(f"Feature dimension: {X_poly.shape[1]} (13 original + 2 polynomial, no one-hot)")
    logging.info("Architecture: Independent model per thickness category")

if __name__ == "__main__":
    main()
