#!/usr/bin/env python3
"""Gradient-based inverse design using a trained train_mlp.py forward surrogate.

This first-stage inverse design script optimizes the processed model input
feature vector directly. It is intended for candidate discovery followed by
Luna verification, not as a replacement for physics simulation.
"""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import joblib
import numpy as np
import torch

import train_mlp


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize input features through a trained forward surrogate")
    parser.add_argument("--input-dir", required=True, help="Processed data directory used by the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Forward surrogate .pth checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory for inverse-design candidates")
    parser.add_argument("--model", default="transformer",
                        choices=["mlp", "temporal", "lstm", "transformer", "linear", "shallow", "banded_mlp", "temporal_cnn"])
    parser.add_argument("--objective", default="uv_fraction",
                        choices=["uv_fraction", "uv_peak", "target_spectrum"],
                        help="Objective optimized on the predicted final spectrum")
    parser.add_argument("--target-spectrum", default=None,
                        help="Optional .npy target spectrum for --objective target_spectrum")
    parser.add_argument("--n-restarts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")

    # Architecture compatibility knobs; keep in sync with train_mlp.py defaults.
    parser.add_argument("--transformer-mode", default="temporal", choices=["final", "temporal"])
    parser.add_argument("--transformer-d-model", type=int, default=192)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-use-z-embedding", action="store_true")
    parser.add_argument("--transformer-band-boundaries-nm", type=float, nargs="+",
                        default=[200.0, 700.0, 1200.0, 1800.0, 2500.0])
    parser.add_argument("--temporal-cnn-architecture", default="basic", choices=["basic", "multiscale"])
    parser.add_argument("--temporal-cnn-channels", type=int, default=128)
    parser.add_argument("--temporal-cnn-z-init", type=int, default=16)
    parser.add_argument("--temporal-cnn-lambda-init", type=int, default=64)
    parser.add_argument("--temporal-cnn-band-boundaries-nm", type=float, nargs="+",
                        default=[200.0, 700.0, 1200.0, 1800.0, 2500.0])
    parser.add_argument("--temporal-cnn-coordinate-channels", action="store_true")
    parser.add_argument("--num-bands", type=int, default=4)
    parser.add_argument("--band-boundaries-nm", type=float, nargs="+", default=None)
    parser.add_argument("--band-head-hidden", type=int, default=128)
    parser.add_argument("--uv-band-head-hidden", type=int, default=256)
    parser.add_argument("--output-activation", default="auto", choices=["auto", "sigmoid", "identity"])
    parser.add_argument("--uv-min-nm", type=float, default=200.0)
    parser.add_argument("--uv-max-nm", type=float, default=700.0)
    return parser.parse_args()


def load_processed_arrays(input_dir, model, transformer_mode):
    x_parts = []
    for split in ("train", "val", "test"):
        path = os.path.join(input_dir, f"X_{split}.npy")
        if os.path.exists(path):
            arr = np.load(path)
            if arr.size:
                x_parts.append(arr.astype(np.float32))
    if not x_parts:
        raise ValueError(f"No non-empty X split found in {input_dir}")
    x_all = np.vstack(x_parts)

    y_test = np.load(os.path.join(input_dir, "y_test.npy")).astype(np.float32)
    y_temporal = None
    z_reference = None
    temporal_requested = model in ("temporal", "lstm", "temporal_cnn") or (
        model == "transformer" and transformer_mode == "temporal"
    )
    if temporal_requested:
        y_temporal = np.load(os.path.join(input_dir, "y_temporal_test.npy")).astype(np.float32)
        z_raw = np.load(os.path.join(input_dir, "z_test.npy"), allow_pickle=True)
        z_reference = train_mlp.pad_z_for_eval(z_raw[:1], y_temporal.shape[1])[0]
    return x_all, y_test, y_temporal, z_reference


def model_args_from_cli(args):
    """Build a train_mlp-compatible namespace for model construction."""
    return SimpleNamespace(
        transformer_mode=args.transformer_mode,
        transformer_d_model=args.transformer_d_model,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_use_z_embedding=args.transformer_use_z_embedding,
        transformer_band_boundaries_nm=args.transformer_band_boundaries_nm,
        temporal_cnn_architecture=args.temporal_cnn_architecture,
        temporal_cnn_channels=args.temporal_cnn_channels,
        temporal_cnn_z_init=args.temporal_cnn_z_init,
        temporal_cnn_lambda_init=args.temporal_cnn_lambda_init,
        temporal_cnn_band_boundaries_nm=args.temporal_cnn_band_boundaries_nm,
        temporal_cnn_coordinate_channels=args.temporal_cnn_coordinate_channels,
        num_bands=args.num_bands,
        band_boundaries_nm=args.band_boundaries_nm,
        band_head_hidden=args.band_head_hidden,
        uv_band_head_hidden=args.uv_band_head_hidden,
    )


def predict_final(model, x, use_temporal, z_reference):
    if use_temporal:
        z = torch.tensor(z_reference, dtype=torch.float32, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        _, pred = model(x, z)
    else:
        _, pred = model(x)
    return pred


def objective_loss(pred, objective, uv_mask, target=None):
    if objective == "target_spectrum":
        return torch.mean((pred - target) ** 2), -torch.mean((pred - target) ** 2)

    positive = torch.relu(pred)
    uv = positive[:, uv_mask]
    if objective == "uv_peak":
        score = torch.mean(torch.max(uv, dim=1).values)
    else:
        score = torch.mean(torch.sum(uv, dim=1) / (torch.sum(positive, dim=1) + 1e-8))
    return -score, score


def luna_command_from_features(row):
    required = ["energy", "tau", "pressure", "diameter", "wallthickness"]
    if not all(name in row for name in required):
        return ""
    return (
        "julia --project=../.. anti_resonant_simulation.jl "
        f"-e {row['energy']:.6g} --tau {row['tau']:.6g} "
        f"-p {row['pressure']:.6g} -d {row['diameter']:.6g} "
        f"-t {row['wallthickness']:.6g}"
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    processing_params = train_mlp.load_processing_params(args.input_dir)
    wavelength_range = train_mlp.get_wavelength_range_nm(processing_params)
    spectrum_normalization = processing_params.get("spectrum_normalization", "per_sample_minmax")
    output_activation = train_mlp.resolve_output_activation(args.output_activation, spectrum_normalization)
    feature_names = processing_params.get("input_features") or []

    x_all, y_test, y_temporal, z_reference = load_processed_arrays(
        args.input_dir, args.model, args.transformer_mode
    )
    input_dim = x_all.shape[1]
    output_dim = y_test.shape[1]
    wavelength_nm = np.linspace(wavelength_range[0], wavelength_range[1], output_dim)
    uv_mask_np = (wavelength_nm >= args.uv_min_nm) & (wavelength_nm <= args.uv_max_nm)
    uv_mask = torch.tensor(uv_mask_np, dtype=torch.bool, device=device)

    model, use_temporal, _ = train_mlp.build_model_for_type(
        args.model, input_dim, output_dim, model_args_from_cli(args), wavelength_range,
        y_temporal=y_temporal, output_activation=output_activation
    )
    model = model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    scaler_path = os.path.join(args.input_dir, "scaler_X.joblib")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    lower = np.min(x_all, axis=0)
    upper = np.max(x_all, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    init_indices = rng.integers(0, x_all.shape[0], size=args.n_restarts)
    init = x_all[init_indices] + rng.normal(0.0, 0.03, size=(args.n_restarts, input_dim)).astype(np.float32) * span
    init = np.clip(init, lower, upper)
    x = torch.tensor(init, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=args.learning_rate)

    target = None
    if args.objective == "target_spectrum":
        if args.target_spectrum is None:
            raise ValueError("--objective target_spectrum requires --target-spectrum")
        target_np = np.load(args.target_spectrum).astype(np.float32)
        if target_np.ndim == 1:
            target_np = target_np.reshape(1, -1)
        if target_np.shape[1] != output_dim:
            raise ValueError(f"target spectrum length {target_np.shape[1]} != model output {output_dim}")
        target = torch.tensor(target_np[:1], dtype=torch.float32, device=device).expand(args.n_restarts, -1)

    lower_t = torch.tensor(lower, dtype=torch.float32, device=device)
    upper_t = torch.tensor(upper, dtype=torch.float32, device=device)
    history = []
    for step in range(args.steps):
        optimizer.zero_grad()
        pred = predict_final(model, x, use_temporal, z_reference)
        loss, score = objective_loss(pred, args.objective, uv_mask, target=target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(lower_t, upper_t)
        if step == 0 or (step + 1) % 25 == 0 or step == args.steps - 1:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu()), "score": float(score.detach().cpu())})

    with torch.no_grad():
        pred = predict_final(model, x, use_temporal, z_reference)
        _, scores = objective_loss(pred, args.objective, uv_mask, target=target)
        if args.objective == "target_spectrum":
            per_score = -torch.mean((pred - target) ** 2, dim=1)
        elif args.objective == "uv_peak":
            per_score = torch.max(torch.relu(pred)[:, uv_mask], dim=1).values
        else:
            positive = torch.relu(pred)
            per_score = torch.sum(positive[:, uv_mask], dim=1) / (torch.sum(positive, dim=1) + 1e-8)
        order = torch.argsort(per_score, descending=True)[:args.top_k].cpu().numpy()
        x_scaled = x.detach().cpu().numpy()
        x_raw = scaler.inverse_transform(x_scaled) if scaler is not None else x_scaled.copy()

    rows = []
    for rank, idx in enumerate(order, start=1):
        row = {
            "rank": rank,
            "restart_index": int(idx),
            "score": float(per_score[idx].detach().cpu()),
        }
        for j in range(input_dim):
            name = feature_names[j] if j < len(feature_names) else f"feature_{j}"
            row[f"{name}_scaled"] = float(x_scaled[idx, j])
            row[name] = float(x_raw[idx, j])
        row["luna_command"] = luna_command_from_features(row)
        rows.append(row)

    csv_path = os.path.join(args.output_dir, "inverse_candidates.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    commands_path = os.path.join(args.output_dir, "luna_validation_commands.sh")
    with open(commands_path, "w", encoding="utf-8") as f:
        for row in rows:
            if row["luna_command"]:
                f.write(row["luna_command"] + "\n")

    summary = {
        "input_dir": args.input_dir,
        "checkpoint": args.checkpoint,
        "model": args.model,
        "objective": args.objective,
        "spectrum_normalization": spectrum_normalization,
        "output_activation": output_activation,
        "use_temporal": use_temporal,
        "history": history,
        "note": (
            "Candidates are optimized in processed feature space. Verify all top candidates with Luna; "
            "for strict physical consistency, retrain with derived features recomputed from candidate parameters."
        ),
    }
    with open(os.path.join(args.output_dir, "inverse_design_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved {csv_path}")
    print(f"Saved {commands_path}")


if __name__ == "__main__":
    main()
