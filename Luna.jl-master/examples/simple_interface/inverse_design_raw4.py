#!/usr/bin/env python3
"""Gradient-based UV-fraction inverse design constrained to valid raw4 inputs."""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import joblib
import numpy as np
import torch

import train_mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--n-restarts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.input_dir, "processing_params.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    if params.get("input_features") != ["energy", "tau", "pressure", "diameter"]:
        raise ValueError("Use a raw4 feature view; derived features must not be optimized independently")
    if params.get("spectrum_normalization") != "global_log_standard":
        raise ValueError("UV fraction requires global-log targets")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = np.load(os.path.join(args.input_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(args.input_dir, "y_train.npy")).astype(np.float32)
    scaler = joblib.load(os.path.join(args.input_dir, "scaler_X.joblib"))
    raw_train = scaler.inverse_transform(x_train)
    lower, upper = raw_train.min(axis=0), raw_train.max(axis=0)
    model_args = SimpleNamespace(num_bands=4, band_boundaries_nm=None, band_head_hidden=128, uv_band_head_hidden=256)
    model, _, _ = train_mlp.build_model_for_type("banded_mlp", 4, y_train.shape[1], model_args,
        train_mlp.get_wavelength_range_nm(params), output_activation="identity")
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    wavelength = np.linspace(*train_mlp.get_wavelength_range_nm(params), y_train.shape[1])
    uv_mask = torch.tensor((wavelength >= 200.0) & (wavelength <= 700.0), device=device)
    mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    raw_lower = torch.tensor(lower, dtype=torch.float32, device=device)
    raw_span = torch.tensor(upper - lower, dtype=torch.float32, device=device)
    norm = params["output_normalization"]
    unit = torch.tensor(rng.uniform(0, 1, size=(args.n_restarts, 4)), dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([unit], lr=args.learning_rate)
    for _ in range(args.steps):
        optimizer.zero_grad()
        scaled = (raw_lower + unit * raw_span - mean) / scale
        log_power = model(scaled) * float(norm["log_std"]) + float(norm["log_mean"])
        power = torch.pow(10.0, torch.clamp(log_power, -30.0, 30.0))
        scores = torch.sum(power[:, uv_mask], dim=1) / (torch.sum(power, dim=1) + 1e-20)
        (-scores.mean()).backward(); optimizer.step(); unit.data.clamp_(0.0, 1.0)
    scores = scores.detach().cpu().numpy(); raw = (raw_lower + unit * raw_span).detach().cpu().numpy()
    order = np.argsort(scores)[::-1][:args.top_k]
    rows = [{"rank": int(rank + 1), "uv_fraction_surrogate": float(scores[i]),
             "energy_uj": float(raw[i, 0]), "tau_fs": float(raw[i, 1]),
             "pressure_bar": float(raw[i, 2]), "diameter_um": float(raw[i, 3])}
            for rank, i in enumerate(order)]
    with open(os.path.join(args.output_dir, "inverse_candidates.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
