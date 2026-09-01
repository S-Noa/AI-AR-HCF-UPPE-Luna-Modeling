#!/usr/bin/env python3
"""Train a PyTorch RNNnonlinear-style LSTM on Luna temporal spectra."""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time

import h5py
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LunaRNN(nn.Module):
    """LSTM -> Dense -> Dense -> output, matching the RNNnonlinear baseline."""

    def __init__(self, n_grid, hidden=250, output_activation="sigmoid", conditioning_dim=0):
        super().__init__()
        self.n_grid = int(n_grid)
        self.conditioning_dim = int(conditioning_dim)
        self.output_activation = output_activation
        self.lstm = nn.LSTM(input_size=n_grid + self.conditioning_dim, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_grid),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])
        if self.output_activation == "sigmoid":
            y = torch.sigmoid(y)
        return y


def parse_args():
    parser = argparse.ArgumentParser(description="Train RNNnonlinear-style LSTM on Luna .mat data")
    parser.add_argument("--data", required=True, help=".mat file containing data=(N,n_grid,n_steps)")
    parser.add_argument("--output-dir", default="luna_rnn_results")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=250)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--train-evolutions", type=int, default=None,
                        help="Optional explicit number of training evolutions, matching RNNnonlinear train_evo")
    parser.add_argument("--test-evolutions", type=int, default=None,
                        help="Optional explicit number of test evolutions, matching RNNnonlinear test_evo")
    parser.add_argument("--output-activation", choices=["sigmoid", "identity"], default="sigmoid")
    parser.add_argument("--conditioning", choices=["auto", "none", "features", "z", "features_z"], default="auto",
                        help="Condition LSTM inputs on exported sample features and/or normalized z")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N epochs")
    parser.add_argument(
        "--training-mode",
        choices=[
            "one_step", "scheduled_sampling", "rollout",
            "open_source_legacy", "conditional_legacy", "rollout_robust",
        ],
        default="one_step",
        help=(
            "Training mode. one_step/scheduled_sampling/rollout are explicit modes; "
            "open_source_legacy, conditional_legacy, and rollout_robust are presets "
            "matching the staged RNNnonlinear-to-Luna migration."
        ),
    )
    parser.add_argument("--rollout-steps", type=int, default=20,
                        help="Number of consecutive predicted steps for recursive training")
    parser.add_argument("--rollout-loss-weight", type=float, default=1.0,
                        help="Extra rollout-loss weight used in --training-mode rollout")
    parser.add_argument("--scheduled-sampling-start", type=float, default=0.0,
                        help="Initial probability of feeding a detached model prediction back into the input window")
    parser.add_argument("--scheduled-sampling-end", type=float, default=0.5,
                        help="Final probability of feeding a detached model prediction back into the input window")
    parser.add_argument("--eval-autoregressive-every", type=int, default=5,
                        help="Run a bounded autoregressive validation rollout every N epochs; 0 disables it")
    parser.add_argument("--eval-autoregressive-samples", type=int, default=64,
                        help="Maximum test evolutions used for periodic autoregressive validation")
    parser.add_argument("--eval-stepwise-samples", type=int, default=256,
                        help="Maximum test evolutions used for periodic stepwise validation; final metrics use all tests")
    parser.add_argument("--autoregressive-eval-batch-size", type=int, default=16,
                        help="Batch size for autoregressive validation and final rollout")
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="Save latest training state and JSON progress every N epochs")
    return parser.parse_args()


def apply_training_preset(args):
    """Resolve high-level migration presets to concrete training/conditioning settings."""
    requested_mode = args.training_mode
    preset = "manual"
    if requested_mode == "open_source_legacy":
        preset = requested_mode
        args.training_mode = "one_step"
        args.conditioning = "none"
    elif requested_mode == "conditional_legacy":
        preset = requested_mode
        args.training_mode = "one_step"
        args.conditioning = "features_z"
    elif requested_mode == "rollout_robust":
        preset = requested_mode
        args.training_mode = "scheduled_sampling"
        args.conditioning = "features_z"
        args.rollout_steps = max(int(args.rollout_steps), 50)
        args.eval_autoregressive_every = min(int(args.eval_autoregressive_every), 2)
        args.eval_autoregressive_samples = max(int(args.eval_autoregressive_samples), 64)
    args.requested_training_mode = requested_mode
    args.training_preset = preset
    return args


def setup_logging(log_file=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_hdf5_strings(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def load_luna_mat_bundle(path):
    """Load data and optional conditional arrays from MATLAB v5 or HDF5/v7.3-style files."""
    scipy_error = None
    try:
        mat = sio.loadmat(path)
        if "data" not in mat:
            raise KeyError(".mat file must contain variable 'data'")
        data = mat["data"].astype(np.float32)
        features = mat.get("features")
        if features is not None:
            features = np.asarray(features, dtype=np.float32)
        z_norm = mat.get("z_norm")
        if z_norm is not None:
            z_norm = np.asarray(z_norm, dtype=np.float32).reshape(-1)
        return {
            "data": data,
            "features": features,
            "z_norm": z_norm,
            "feature_names": None,
        }
    except (NotImplementedError, ValueError, OSError, KeyError) as exc:
        scipy_error = exc

    try:
        with h5py.File(path, "r") as f:
            if "data" not in f:
                raise KeyError("HDF5 file must contain dataset '/data'")
            features = np.asarray(f["features"][:], dtype=np.float32) if "features" in f else None
            z_norm = np.asarray(f["z_norm"][:], dtype=np.float32).reshape(-1) if "z_norm" in f else None
            feature_names = decode_hdf5_strings(f["feature_names"][:]) if "feature_names" in f else None
            return {
                "data": np.asarray(f["data"][:], dtype=np.float32),
                "features": features,
                "z_norm": z_norm,
                "feature_names": feature_names,
            }
    except Exception as h5_error:
        raise RuntimeError(
            f"Failed to load {path} as MATLAB v5 or HDF5/v7.3. "
            f"scipy error: {scipy_error}; hdf5 error: {h5_error}"
        ) from h5_error


def resolve_conditioning(mode, features, z_norm):
    if mode == "auto":
        mode = "features_z" if features is not None else "none"
    use_features = mode in ("features", "features_z")
    use_z = mode in ("z", "features_z")
    if use_features and features is None:
        raise ValueError("--conditioning requires /features in the .mat file; rerun prepare_luna_data.py --include-features")
    if use_z and z_norm is None:
        raise ValueError("--conditioning requires /z_norm in the .mat file; rerun prepare_luna_data.py")
    return mode, use_features, use_z


def conditioning_dim_for(use_features, use_z, features):
    return (features.shape[1] if use_features else 0) + (1 if use_z else 0)


def concat_condition_np(spectra, feature=None, z_values=None):
    """Append per-sample features and per-step z to a spectral input window."""
    parts = [spectra.astype(np.float32, copy=False)]
    n_steps = spectra.shape[0]
    if feature is not None:
        parts.append(np.repeat(feature.reshape(1, -1), n_steps, axis=0).astype(np.float32, copy=False))
    if z_values is not None:
        parts.append(np.asarray(z_values, dtype=np.float32).reshape(n_steps, 1))
    return np.concatenate(parts, axis=1)


def concat_condition_torch(spectra, feature=None, z_values=None):
    """Torch version of concat_condition_np for rollout windows."""
    parts = [spectra]
    batch, n_steps, _ = spectra.shape
    if feature is not None:
        parts.append(feature.unsqueeze(1).expand(batch, n_steps, feature.size(1)))
    if z_values is not None:
        z = z_values.reshape(1, n_steps, 1).to(spectra.device).expand(batch, n_steps, 1)
        parts.append(z)
    return torch.cat(parts, dim=2)


def make_series(data, window_size):
    """Return X/Y one-step series and original evolution split metadata."""
    n_evo, n_grid, n_steps = data.shape
    evo_size = n_steps - 1
    x_all = np.zeros((n_evo * evo_size, window_size, n_grid), dtype=np.float32)
    y_all = np.zeros((n_evo * evo_size, n_grid), dtype=np.float32)
    for evo in range(n_evo):
        evo_data = data[evo].T
        prefix = np.tile(evo_data[0], (window_size - 1, 1))
        padded = np.vstack([prefix, evo_data])
        for step in range(evo_size):
            idx = evo * evo_size + step
            x_all[idx] = padded[step:step + window_size]
            y_all[idx] = padded[step + window_size]
    return x_all, y_all, evo_size


class OneStepEvolutionDataset(Dataset):
    """Lazy one-step windows without materializing every overlapping spectrum."""

    def __init__(self, data, window_size, features=None, z_norm=None, use_features=False, use_z=False):
        self.data = data
        self.window_size = int(window_size)
        self.n_evo, self.n_grid, self.n_steps = data.shape
        self.steps_per_evolution = self.n_steps - 1
        self.features = features
        self.z_norm = z_norm
        self.use_features = bool(use_features)
        self.use_z = bool(use_z)

    def __len__(self):
        return self.n_evo * self.steps_per_evolution

    def __getitem__(self, index):
        evo_idx = int(index // self.steps_per_evolution)
        step_idx = int(index % self.steps_per_evolution)
        sequence = self.data[evo_idx].T
        start = max(0, step_idx - self.window_size + 1)
        history = sequence[start:step_idx + 1]
        if history.shape[0] < self.window_size:
            prefix = np.repeat(sequence[:1], self.window_size - history.shape[0], axis=0)
            history = np.vstack([prefix, history])
        feature = self.features[evo_idx] if self.use_features else None
        z_window = None
        if self.use_z:
            z_seq = self.z_norm
            z_history = z_seq[start:step_idx + 1]
            if z_history.shape[0] < self.window_size:
                z_prefix = np.repeat(z_seq[:1], self.window_size - z_history.shape[0], axis=0)
                z_history = np.concatenate([z_prefix, z_history])
            z_window = z_history
        history = concat_condition_np(history, feature=feature, z_values=z_window)
        target = sequence[step_idx + 1]
        return torch.from_numpy(history.copy()), torch.from_numpy(target.copy())


class EvolutionDataset(Dataset):
    """Return complete z-evolution sequences for short recursive rollout training."""

    def __init__(self, data, features=None, use_features=False):
        self.data = data
        self.features = features
        self.use_features = bool(use_features)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        sequence = torch.from_numpy(self.data[index].T.copy())
        if self.use_features:
            return sequence, torch.from_numpy(self.features[index].copy())
        return sequence


def regression_from_sums(count, true_sum, true_sq_sum, squared_error_sum):
    """Return MSE/R2 from streamed regression sufficient statistics."""
    if count <= 0:
        return {"mse": float("nan"), "r2": float("nan")}
    mse = squared_error_sum / count
    ss_tot = true_sq_sum - (true_sum ** 2) / count
    return {
        "mse": float(mse),
        "r2": float(1.0 - squared_error_sum / max(ss_tot, 1e-12)),
    }


def write_json_atomic(path, payload):
    """Write inspectable progress JSON without leaving a partially written file."""
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(temp_path, path)


def append_jsonl(path, payload):
    """Append a completed epoch record for monitoring interrupted runs."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def global_r2(y_true, y_pred, eps=1e-12):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))


def evaluate_stepwise(model, loader, device, return_predictions=False):
    model.eval()
    preds = [] if return_predictions else None
    targets = [] if return_predictions else None
    count = 0
    true_sum = 0.0
    true_sq_sum = 0.0
    squared_error_sum = 0.0
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            error = pred - y
            count += y.numel()
            true_sum += float(torch.sum(y).cpu())
            true_sq_sum += float(torch.sum(y ** 2).cpu())
            squared_error_sum += float(torch.sum(error ** 2).cpu())
            if return_predictions:
                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
    summary = regression_from_sums(count, true_sum, true_sq_sum, squared_error_sum)
    if return_predictions:
        summary["pred"] = np.concatenate(preds, axis=0)
        summary["true"] = np.concatenate(targets, axis=0)
    return summary


def predict_autoregressive(model, data_test, window_size, device,
                           batch_size=16, max_evolutions=None, return_predictions=False,
                           features_test=None, z_norm=None, use_features=False, use_z=False):
    """Roll out complete propagation trajectories from each initial spectrum."""
    model.eval()
    if max_evolutions is not None:
        data_test = data_test[:min(int(max_evolutions), data_test.shape[0])]
    n_evo, n_grid, n_steps = data_test.shape
    evo_size = n_steps - 1
    y_submit = np.empty((n_evo, evo_size, n_grid), dtype=np.float32) if return_predictions else None
    count = 0
    true_sum = 0.0
    true_sq_sum = 0.0
    squared_error_sum = 0.0
    final_count = 0
    final_true_sum = 0.0
    final_true_sq_sum = 0.0
    final_squared_error_sum = 0.0

    with torch.no_grad():
        for start in range(0, n_evo, batch_size):
            stop = min(start + batch_size, n_evo)
            batch = torch.tensor(data_test[start:stop], dtype=torch.float32, device=device)
            feature_batch = None
            if use_features:
                feature_batch = torch.tensor(features_test[start:stop], dtype=torch.float32, device=device)
            z_tensor = torch.tensor(z_norm, dtype=torch.float32, device=device) if use_z else None
            initial_spectrum = batch[:, :, 0].unsqueeze(1).repeat(1, window_size, 1)
            z_window = z_tensor[:1].repeat(window_size) if use_z else None
            current = concat_condition_torch(initial_spectrum, feature=feature_batch, z_values=z_window)
            for step in range(evo_size):
                pred = model(current)
                target = batch[:, :, step + 1]
                error = pred - target
                count += target.numel()
                true_sum += float(torch.sum(target).cpu())
                true_sq_sum += float(torch.sum(target ** 2).cpu())
                squared_error_sum += float(torch.sum(error ** 2).cpu())
                if step == evo_size - 1:
                    final_count += target.numel()
                    final_true_sum += float(torch.sum(target).cpu())
                    final_true_sq_sum += float(torch.sum(target ** 2).cpu())
                    final_squared_error_sum += float(torch.sum(error ** 2).cpu())
                if return_predictions:
                    y_submit[start:stop, step, :] = pred.cpu().numpy()
                next_z = z_tensor[step + 1:step + 2] if use_z else None
                next_input = concat_condition_torch(pred.unsqueeze(1), feature=feature_batch, z_values=next_z)
                current = torch.cat([current[:, 1:, :], next_input], dim=1)

    summary = regression_from_sums(count, true_sum, true_sq_sum, squared_error_sum)
    final_summary = regression_from_sums(
        final_count, final_true_sum, final_true_sq_sum, final_squared_error_sum
    )
    summary["final_mse"] = final_summary["mse"]
    summary["final_r2"] = final_summary["r2"]
    if return_predictions:
        y_true = np.transpose(data_test[:, :, 1:], (0, 2, 1)).reshape(n_evo * evo_size, n_grid)
        summary["pred"] = y_submit.reshape(n_evo * evo_size, n_grid)
        summary["true"] = y_true
    return summary


def build_rollout_window(sequence, start_step, window_size, feature=None, z_norm=None, use_features=False, use_z=False):
    """Build an initial window ending at a shared propagation index."""
    prefix = sequence[:, :1, :].expand(-1, window_size - 1, -1)
    padded = torch.cat([prefix, sequence], dim=1)
    spectra = padded[:, start_step:start_step + window_size]
    z_window = None
    if use_z:
        z_prefix = z_norm[:1].expand(window_size - 1)
        z_padded = torch.cat([z_prefix, z_norm], dim=0)
        z_window = z_padded[start_step:start_step + window_size]
    return concat_condition_torch(
        spectra,
        feature=feature if use_features else None,
        z_values=z_window
    )


def train_recursive_epoch(model, loader, optimizer, loss_fn, device, mode,
                          rollout_steps, feedback_probability, rollout_loss_weight,
                          z_norm=None, use_features=False, use_z=False):
    """Train on short recursive trajectories with scheduled feedback."""
    model.train()
    total_loss_sum = 0.0
    one_step_loss_sum = 0.0
    rollout_loss_sum = 0.0
    batches = 0

    z_tensor = torch.tensor(z_norm, dtype=torch.float32, device=device) if use_z else None

    for batch in loader:
        if use_features:
            sequence, feature = batch
            feature = feature.to(device)
        else:
            sequence = batch
            feature = None
        sequence = sequence.to(device)
        n_steps = sequence.size(1)
        horizon = min(int(rollout_steps), n_steps - 1)
        max_start = n_steps - horizon - 1
        start_step = random.randint(0, max(0, max_start))
        current = build_rollout_window(
            sequence, start_step, model_window_size(model),
            feature=feature, z_norm=z_tensor,
            use_features=use_features, use_z=use_z
        )

        predictions = []
        targets = []
        for step in range(horizon):
            pred = model(current)
            target = sequence[:, start_step + step + 1, :]
            predictions.append(pred)
            targets.append(target)

            if mode == "scheduled_sampling":
                use_prediction = torch.rand((sequence.size(0), 1), device=device) < feedback_probability
                feedback = torch.where(use_prediction, pred.detach(), target)
            else:
                feedback = pred.detach()
            next_z = z_tensor[start_step + step + 1:start_step + step + 2] if use_z else None
            feedback_input = concat_condition_torch(
                feedback.unsqueeze(1),
                feature=feature if use_features else None,
                z_values=next_z
            )
            current = torch.cat([current[:, 1:, :], feedback_input], dim=1)

        prediction_stack = torch.stack(predictions, dim=1)
        target_stack = torch.stack(targets, dim=1)
        one_step_loss = loss_fn(prediction_stack[:, 0, :], target_stack[:, 0, :])
        rollout_loss = loss_fn(prediction_stack, target_stack)
        if mode == "rollout":
            total_loss = one_step_loss + rollout_loss_weight * rollout_loss
        else:
            total_loss = rollout_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_loss_sum += float(total_loss.detach().cpu())
        one_step_loss_sum += float(one_step_loss.detach().cpu())
        rollout_loss_sum += float(rollout_loss.detach().cpu())
        batches += 1

    return {
        "total_mse": total_loss_sum / max(batches, 1),
        "one_step_mse": one_step_loss_sum / max(batches, 1),
        "rollout_mse": rollout_loss_sum / max(batches, 1),
        "feedback_probability": float(feedback_probability),
    }


def model_window_size(model):
    """Store the configured input-window size on the model at construction time."""
    return int(model.window_size)


def main():
    args = parse_args()
    args = apply_training_preset(args)
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = args.log_file or os.path.join(args.output_dir, "train.log")
    setup_logging(log_file)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    history_path = os.path.join(args.output_dir, "history.json")
    history_jsonl_path = os.path.join(args.output_dir, "history.jsonl")
    latest_metrics_path = os.path.join(args.output_dir, "latest_metrics.json")
    latest_state_path = os.path.join(args.output_dir, "training_state_latest.pth")

    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1")
    if not 0.0 <= args.scheduled_sampling_start <= 1.0:
        raise ValueError("--scheduled-sampling-start must be within [0, 1]")
    if not 0.0 <= args.scheduled_sampling_end <= 1.0:
        raise ValueError("--scheduled-sampling-end must be within [0, 1]")
    logging.info(
        "Requested training mode: %s | preset: %s | resolved mode: %s | conditioning: %s",
        args.requested_training_mode, args.training_preset, args.training_mode, args.conditioning
    )

    logging.info("Loading data: %s", args.data)
    bundle = load_luna_mat_bundle(args.data)
    data = bundle["data"]
    features = bundle["features"]
    z_norm = bundle["z_norm"]
    if data.ndim != 3:
        raise ValueError(f"data must be (N,n_grid,n_steps), got {data.shape}")
    if features is not None and features.shape[0] != data.shape[0]:
        raise ValueError(f"features sample count {features.shape[0]} does not match data {data.shape[0]}")
    if z_norm is not None and z_norm.shape[0] != data.shape[2]:
        raise ValueError(f"z_norm length {z_norm.shape[0]} does not match n_steps {data.shape[2]}")
    conditioning_mode, use_features, use_z = resolve_conditioning(args.conditioning, features, z_norm)
    conditioning_dim = conditioning_dim_for(use_features, use_z, features)
    logging.info("Loaded data shape: %s", data.shape)
    logging.info(
        "Conditioning mode: %s | features=%s | z=%s | conditioning_dim=%d",
        conditioning_mode,
        None if features is None else features.shape,
        None if z_norm is None else z_norm.shape,
        conditioning_dim,
    )
    if bundle.get("feature_names"):
        logging.info("Feature names: %s", ", ".join(bundle["feature_names"]))
    logging.info("Device: %s", device)

    n_evo = data.shape[0]
    explicit_split = args.train_evolutions is not None or args.test_evolutions is not None
    if explicit_split:
        if args.train_evolutions is None or args.test_evolutions is None:
            raise ValueError("--train-evolutions and --test-evolutions must be provided together")
        n_train = int(args.train_evolutions)
        n_test = int(args.test_evolutions)
        if n_train < 1 or n_test < 1:
            raise ValueError("--train-evolutions and --test-evolutions must be positive")
        if n_train + n_test > n_evo:
            raise ValueError(
                f"Requested train+test evolutions {n_train + n_test} exceeds available {n_evo}"
            )
        data = data[:n_train + n_test]
        if features is not None:
            features = features[:n_train + n_test]
    else:
        n_test = max(1, int(round(n_evo * args.test_fraction)))
        n_train = n_evo - n_test
    if n_train <= 0:
        raise ValueError("Need at least one training evolution")
    data_train = data[:n_train]
    data_test = data[n_train:]
    features_train = features[:n_train] if features is not None else None
    features_test = features[n_train:] if features is not None else None
    logging.info(
        "Split evolutions: train=%d test=%d split=%s window_size=%d output_activation=%s training_mode=%s preset=%s",
        n_train, n_test, "explicit" if explicit_split else "test_fraction",
        args.window_size, args.output_activation, args.training_mode, args.training_preset
    )

    train_one_step_ds = OneStepEvolutionDataset(
        data_train, args.window_size,
        features=features_train, z_norm=z_norm,
        use_features=use_features, use_z=use_z
    )
    test_one_step_ds = OneStepEvolutionDataset(
        data_test, args.window_size,
        features=features_test, z_norm=z_norm,
        use_features=use_features, use_z=use_z
    )
    periodic_stepwise_n = min(int(args.eval_stepwise_samples), n_test)
    periodic_test_ds = OneStepEvolutionDataset(
        data_test[:periodic_stepwise_n], args.window_size,
        features=features_test[:periodic_stepwise_n] if features_test is not None else None,
        z_norm=z_norm, use_features=use_features, use_z=use_z
    )
    test_loader = DataLoader(periodic_test_ds, batch_size=args.batch_size, shuffle=False)
    final_test_loader = DataLoader(test_one_step_ds, batch_size=args.batch_size, shuffle=False)
    if args.training_mode == "one_step":
        train_loader = DataLoader(train_one_step_ds, batch_size=args.batch_size, shuffle=True)
        rollout_loader = None
        logging.info("Using lazy one-step dataset: train=%d windows, periodic test=%d/%d evolutions",
                     len(train_one_step_ds), periodic_stepwise_n, n_test)
    else:
        train_loader = None
        rollout_loader = DataLoader(
            EvolutionDataset(data_train, features=features_train, use_features=use_features),
            batch_size=args.batch_size, shuffle=True
        )
        logging.info("Using recursive evolution dataset: train=%d trajectories, rollout_steps=%d",
                     len(rollout_loader.dataset), args.rollout_steps)

    model = LunaRNN(
        data.shape[1],
        hidden=args.hidden,
        output_activation=args.output_activation,
        conditioning_dim=conditioning_dim
    ).to(device)
    model.window_size = int(args.window_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9)
    loss_fn = nn.MSELoss()

    history = []
    start = time.time()
    best_stepwise_r2 = -float("inf")
    best_autoregressive_r2 = -float("inf")
    best_stepwise_state = copy.deepcopy(model.state_dict())
    best_autoregressive_state = None
    best_stepwise_epoch = 0
    best_autoregressive_epoch = 0
    config = vars(args).copy()
    config.update({
        "data_shape": list(data.shape),
        "train_evolutions": int(n_train),
        "test_evolutions": int(n_test),
        "split_mode": "explicit" if explicit_split else "test_fraction",
        "device": str(device),
        "log_file": log_file,
        "conditioning_mode": conditioning_mode,
        "conditioning_dim": int(conditioning_dim),
        "feature_names": bundle.get("feature_names"),
    })
    write_json_atomic(os.path.join(args.output_dir, "config.json"), config)
    logging.info("Starting training for %d epochs; progress log: %s", args.epochs, log_file)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        feedback_probability = 0.0
        if args.training_mode == "one_step":
            train_loss = 0.0
            batches = 0
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.detach().cpu())
                batches += 1
            train_summary = {
                "total_mse": train_loss / max(batches, 1),
                "one_step_mse": train_loss / max(batches, 1),
                "rollout_mse": float("nan"),
                "feedback_probability": 0.0,
            }
        else:
            if args.training_mode == "scheduled_sampling":
                progress = epoch / max(args.epochs - 1, 1)
                feedback_probability = (
                    args.scheduled_sampling_start
                    + progress * (args.scheduled_sampling_end - args.scheduled_sampling_start)
                )
            else:
                feedback_probability = 1.0
            train_summary = train_recursive_epoch(
                model, rollout_loader, optimizer, loss_fn, device,
                args.training_mode, args.rollout_steps,
                feedback_probability, args.rollout_loss_weight,
                z_norm=z_norm,
                use_features=use_features,
                use_z=use_z
            )

        stepwise = evaluate_stepwise(model, test_loader, device, return_predictions=False)
        run_autoregressive_eval = (
            args.eval_autoregressive_every > 0
            and ((epoch + 1) % args.eval_autoregressive_every == 0 or epoch == 0 or epoch == args.epochs - 1)
        )
        autoregressive = None
        if run_autoregressive_eval:
            autoregressive = predict_autoregressive(
                model, data_test, args.window_size, device,
                batch_size=args.autoregressive_eval_batch_size,
                max_evolutions=args.eval_autoregressive_samples,
                return_predictions=False,
                features_test=features_test,
                z_norm=z_norm,
                use_features=use_features,
                use_z=use_z
            )

        elapsed = time.time() - start
        epoch_duration = time.time() - epoch_start
        eta_sec = epoch_duration * max(args.epochs - epoch - 1, 0)
        row = {
            "epoch": epoch + 1,
            "training_mode": args.training_mode,
            "train_mse": train_summary["total_mse"],
            "train_one_step_mse": train_summary["one_step_mse"],
            "train_rollout_mse": train_summary["rollout_mse"],
            "feedback_probability": train_summary["feedback_probability"],
            "test_stepwise_mse": stepwise["mse"],
            "test_stepwise_r2": stepwise["r2"],
            "test_autoregressive_mse": autoregressive["mse"] if autoregressive else None,
            "test_autoregressive_r2": autoregressive["r2"] if autoregressive else None,
            "test_autoregressive_final_r2": autoregressive["final_r2"] if autoregressive else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_sec": epoch_duration,
            "elapsed_sec": elapsed,
            "eta_sec": eta_sec,
        }
        history.append(row)

        if np.isfinite(stepwise["r2"]) and stepwise["r2"] > best_stepwise_r2:
            best_stepwise_r2 = stepwise["r2"]
            best_stepwise_state = copy.deepcopy(model.state_dict())
            best_stepwise_epoch = epoch + 1
            torch.save(best_stepwise_state, os.path.join(args.output_dir, "best_stepwise_model.pth"))
        if autoregressive and np.isfinite(autoregressive["r2"]) and autoregressive["r2"] > best_autoregressive_r2:
            best_autoregressive_r2 = autoregressive["r2"]
            best_autoregressive_state = copy.deepcopy(model.state_dict())
            best_autoregressive_epoch = epoch + 1
            torch.save(best_autoregressive_state, os.path.join(args.output_dir, "best_autoregressive_model.pth"))

        append_jsonl(history_jsonl_path, row)
        write_json_atomic(history_path, history)
        latest_metrics = {
            "status": "training",
            "latest_epoch": epoch + 1,
            "best_stepwise_r2": best_stepwise_r2,
            "best_stepwise_epoch": best_stepwise_epoch,
            "best_autoregressive_r2": best_autoregressive_r2 if best_autoregressive_state is not None else None,
            "best_autoregressive_epoch": best_autoregressive_epoch,
            "latest": row,
        }
        write_json_atomic(latest_metrics_path, latest_metrics)

        should_checkpoint = (
            args.checkpoint_every > 0
            and ((epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1)
        )
        if should_checkpoint:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_stepwise_r2": best_stepwise_r2,
                "best_stepwise_epoch": best_stepwise_epoch,
                "best_autoregressive_r2": best_autoregressive_r2,
                "best_autoregressive_epoch": best_autoregressive_epoch,
                "args": config,
            }, latest_state_path)

        if (epoch + 1) % max(1, args.log_every) == 0 or epoch == 0 or epoch == args.epochs - 1:
            autoreg_msg = ""
            if autoregressive:
                autoreg_msg = (
                    f" | autoreg_r2={autoregressive['r2']:.6f}"
                    f" | autoreg_final_r2={autoregressive['final_r2']:.6f}"
                )
            logging.info(
                "Epoch %d/%d | mode=%s | train_mse=%.6g | one_step=%.6g | rollout=%.6g | "
                "feedback=%.3f | stepwise_r2=%.6f%s | lr=%.3g | elapsed=%.1fs | eta=%.1fs",
                epoch + 1, args.epochs, args.training_mode,
                row["train_mse"], row["train_one_step_mse"], row["train_rollout_mse"],
                row["feedback_probability"], row["test_stepwise_r2"], autoreg_msg,
                row["learning_rate"], row["elapsed_sec"], row["eta_sec"]
            )

    if best_autoregressive_state is not None:
        model.load_state_dict(best_autoregressive_state)
        selected_model = "best_autoregressive"
        selected_epoch = best_autoregressive_epoch
    else:
        model.load_state_dict(best_stepwise_state)
        selected_model = "best_stepwise"
        selected_epoch = best_stepwise_epoch
    logging.info("Restored %s checkpoint from epoch %d for final evaluation", selected_model, selected_epoch)

    logging.info("Running final full-test stepwise evaluation")
    stepwise = evaluate_stepwise(model, final_test_loader, device, return_predictions=True)
    logging.info("Running autoregressive evaluation")
    autoreg = predict_autoregressive(
        model, data_test, args.window_size, device,
        batch_size=args.autoregressive_eval_batch_size,
        return_predictions=True,
        features_test=features_test,
        z_norm=z_norm,
        use_features=use_features,
        use_z=use_z
    )
    metrics = {
        "data": args.data,
        "data_shape": list(data.shape),
        "train_evolutions": int(n_train),
        "test_evolutions": int(n_test),
        "split_mode": "explicit" if explicit_split else "test_fraction",
        "window_size": args.window_size,
        "hidden": args.hidden,
        "output_activation": args.output_activation,
        "conditioning_mode": conditioning_mode,
        "conditioning_dim": int(conditioning_dim),
        "feature_names": bundle.get("feature_names"),
        "requested_training_mode": args.requested_training_mode,
        "training_preset": args.training_preset,
        "training_mode": args.training_mode,
        "rollout_steps": args.rollout_steps,
        "rollout_loss_weight": args.rollout_loss_weight,
        "scheduled_sampling_start": args.scheduled_sampling_start,
        "scheduled_sampling_end": args.scheduled_sampling_end,
        "training_time_sec": time.time() - start,
        "selected_model": selected_model,
        "selected_epoch": selected_epoch,
        "best_stepwise_r2": best_stepwise_r2,
        "best_stepwise_epoch": best_stepwise_epoch,
        "best_autoregressive_r2_subset": best_autoregressive_r2 if best_autoregressive_state is not None else None,
        "best_autoregressive_epoch": best_autoregressive_epoch,
        "periodic_stepwise_evolutions": periodic_stepwise_n,
        "periodic_autoregressive_evolutions": min(int(args.eval_autoregressive_samples), n_test),
        "stepwise_mse": stepwise["mse"],
        "stepwise_r2": stepwise["r2"],
        "autoregressive_mse": autoreg["mse"],
        "autoregressive_r2": autoreg["r2"],
        "autoregressive_final_mse": autoreg["final_mse"],
        "autoregressive_final_r2": autoreg["final_r2"],
    }

    logging.info("Saving outputs to %s", args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "luna_rnn_model.pth"))
    torch.save(model.state_dict(), os.path.join(args.output_dir, "final_selected_model.pth"))
    write_json_atomic(os.path.join(args.output_dir, "metrics.json"), metrics)
    write_json_atomic(history_path, history)
    write_json_atomic(latest_metrics_path, {"status": "complete", **metrics})
    sio.savemat(os.path.join(args.output_dir, "stepwise_predictions.mat"),
                {"Y_submit": stepwise["pred"], "Y_test": stepwise["true"]})
    sio.savemat(os.path.join(args.output_dir, "autoregressive_predictions.mat"),
                {"Y_submit": autoreg["pred"], "Y_test": autoreg["true"]})
    logging.info("Final metrics:\n%s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
