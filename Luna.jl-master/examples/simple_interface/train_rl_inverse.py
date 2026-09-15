#!/usr/bin/env python3
"""Train and compare raw4 SAC inverse design against surrogate baselines."""

import argparse
import csv
import json
import os
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from scipy.optimize import differential_evolution

import train_mlp
from rl_inverse_env import SurrogateUVEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--episode-steps", type=int, default=20)
    parser.add_argument("--action-scale", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--baseline-budget", type=int, default=200000)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--band-head-hidden", type=int, default=128)
    parser.add_argument("--uv-band-head-hidden", type=int, default=256)
    return parser.parse_args()


def build_model(input_dim, output_dim, args, wavelength_range, device):
    model_args = SimpleNamespace(
        num_bands=4, band_boundaries_nm=[200.0, 700.0, 1200.0, 1800.0, 2500.0],
        band_head_hidden=args.band_head_hidden, uv_band_head_hidden=args.uv_band_head_hidden,
    )
    model, _, _ = train_mlp.build_model_for_type(
        "banded_mlp", input_dim, output_dim, model_args, wavelength_range,
        output_activation="identity"
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    with open(os.path.join(args.input_dir, "processing_params.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    if params.get("input_features") != ["energy", "tau", "pressure", "diameter"]:
        raise ValueError("RL requires a raw4 feature view with exactly energy, tau, pressure, diameter")
    if params.get("spectrum_normalization") != "global_log_standard":
        raise ValueError("RL UV-fraction reward requires global_log_standard targets")
    wavelength_range = train_mlp.get_wavelength_range_nm(params)
    x_train = np.load(os.path.join(args.input_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(args.input_dir, "y_train.npy")).astype(np.float32)
    scaler = joblib.load(os.path.join(args.input_dir, "scaler_X.joblib"))
    raw_train = scaler.inverse_transform(x_train)
    lower, upper = raw_train.min(axis=0), raw_train.max(axis=0)
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)
    wavelength_nm = np.linspace(wavelength_range[0], wavelength_range[1], y_train.shape[1])
    uv_mask = torch.tensor((wavelength_nm >= 200.0) & (wavelength_nm <= 700.0), device=device)
    output_norm = params["output_normalization"]
    log_mean = float(output_norm["log_mean"])
    log_std = float(output_norm["log_std"])
    model = build_model(4, y_train.shape[1], args, wavelength_range, device)

    def scores_from_unit(unit_params, require_grad=False):
        unit = np.asarray(unit_params, dtype=np.float32)
        raw = lower + unit * (upper - lower)
        scaled = scaler.transform(raw).astype(np.float32)
        x = torch.tensor(scaled, device=device, requires_grad=require_grad)
        _, pred = model(x)
        log_power = pred * log_std + log_mean
        linear_power = torch.pow(10.0, torch.clamp(log_power, -30.0, 30.0))
        score = torch.sum(linear_power[:, uv_mask], dim=1) / (torch.sum(linear_power, dim=1) + 1e-20)
        # The ratio is physically bounded. Clamp tiny floating-point overshoots
        # before it is used by the RL environment or written to candidate files.
        score = torch.clamp(score, 0.0, 1.0)
        return score, x

    def predictor(unit_params):
        with torch.no_grad():
            score, _ = scores_from_unit(unit_params)
        return score.detach().cpu().numpy()

    env = SurrogateUVEnv(predictor, lower, upper, max_steps=args.episode_steps,
                          action_scale=args.action_scale, seed=args.seed)
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise ImportError("Install requirements-rl.txt in the RL environment before training") from exc
    agent = SAC("MlpPolicy", env, seed=args.seed, verbose=1, device=str(device),
                learning_rate=3e-4, buffer_size=100000, learning_starts=2000,
                batch_size=256, gamma=0.99)
    agent.learn(total_timesteps=args.timesteps, progress_bar=False)
    agent.save(os.path.join(args.output_dir, "sac_agent"))

    candidates = []
    obs, _ = env.reset(seed=args.seed)
    for _ in range(args.top_k * 20):
        action, _ = agent.predict(obs, deterministic=False)
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            candidates.append((float(info["uv_fraction"]), env.params.copy(), "sac"))
            obs, _ = env.reset()

    random_unit = np.random.default_rng(args.seed).uniform(0.0, 1.0, size=(args.baseline_budget, 4))
    random_scores = predictor(random_unit)
    best_random = np.argsort(random_scores)[-args.top_k:]
    candidates.extend((float(random_scores[i]), random_unit[i], "random") for i in best_random)

    result = differential_evolution(lambda x: -float(predictor(np.asarray(x)[None, :])[0]),
                                    bounds=[(0.0, 1.0)] * 4, seed=args.seed,
                                    maxiter=max(1, args.baseline_budget // 60), popsize=15,
                                    polish=False, workers=1)
    candidates.append((-float(result.fun), np.asarray(result.x, dtype=np.float32), "differential_evolution"))

    unit = torch.rand((args.top_k, 4), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([unit], lr=0.03)
    raw_lower = torch.tensor(lower, device=device)
    raw_span = torch.tensor(upper - lower, device=device)
    for _ in range(300):
        optimizer.zero_grad()
        raw = raw_lower + torch.clamp(unit, 0.0, 1.0) * raw_span
        scaled = (raw - scaler_mean) / scaler_scale
        _, pred = model(scaled)
        log_power = pred * log_std + log_mean
        power = torch.pow(10.0, torch.clamp(log_power, -30.0, 30.0))
        score = torch.sum(power[:, uv_mask], dim=1) / (torch.sum(power, dim=1) + 1e-20)
        score = torch.clamp(score, 0.0, 1.0)
        (-score.mean()).backward()
        optimizer.step()
        unit.data.clamp_(0.0, 1.0)
    gradient_scores = predictor(unit.detach().cpu().numpy())
    candidates.extend((float(score), params_unit, "gradient") for score, params_unit in zip(gradient_scores, unit.detach().cpu().numpy()))

    candidates.sort(key=lambda row: row[0], reverse=True)
    rows = []
    for rank, (score, unit_params, method) in enumerate(candidates[:args.top_k], start=1):
        raw = lower + unit_params * (upper - lower)
        # The established preprocessing schema is mixed-unit: energy is J,
        # tau is s, pressure is bar, and diameter is already in um.
        rows.append({"rank": rank, "method": method,
                     "uv_fraction_surrogate": float(np.clip(score, 0.0, 1.0)),
                     "energy_j": float(raw[0]), "tau_s": float(raw[1]),
                     "pressure_bar": float(raw[2]), "diameter_m": float(raw[3] * 1e-6),
                     "energy_uj": float(raw[0] * 1e6), "tau_fs": float(raw[1] * 1e15),
                     "diameter_um": float(raw[3])})
    with open(os.path.join(args.output_dir, "inverse_candidates.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(os.path.join(args.output_dir, "run_luna_validation.sh"), "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write("julia --project=../.. anti_resonant_simulation.jl "
                         f"-e {row['energy_uj']:.6g} --tau {row['tau_fs']:.6g} "
                         f"-p {row['pressure_bar']:.6g} -d {row['diameter_um']:.6g} -t 0.65\n")
    with open(os.path.join(args.output_dir, "rl_summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"seed": args.seed, "timesteps": args.timesteps, "device": str(device),
                   "parameter_lower": lower.tolist(), "parameter_upper": upper.tolist(), "top_k": rows}, handle, indent=2)


if __name__ == "__main__":
    main()
