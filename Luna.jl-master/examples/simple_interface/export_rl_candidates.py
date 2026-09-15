#!/usr/bin/env python3
"""Re-export physical RL candidates from saved SAC agents without retraining.

The legacy preprocessing schema uses a mixed physical-unit convention: energy
is stored in joules, duration in seconds, pressure in bar, and diameter in
micrometres. This script writes both canonical SI fields and the uJ/fs/um
values required by ``anti_resonant_simulation.jl``. It also re-scores candidates
with a frozen forward surrogate and clamps UV fraction to [0, 1].
"""

import argparse
import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import torch

import train_mlp
from rl_inverse_env import SurrogateUVEnv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rl-root", required=True,
                        help="Directory containing rl_sac_seed*_v1/sac_agent.zip")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 456, 789])
    parser.add_argument("--top-k-per-seed", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def build_model(input_dim, output_dim, wavelength_range, checkpoint, device):
    args = SimpleNamespace(
        num_bands=4,
        band_boundaries_nm=[200.0, 700.0, 1200.0, 1800.0, 2500.0],
        band_head_hidden=128,
        uv_band_head_hidden=256,
    )
    model, _, _ = train_mlp.build_model_for_type(
        "banded_mlp", input_dim, output_dim, args, wavelength_range, output_activation="identity"
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    with open(os.path.join(args.input_dir, "processing_params.json"), encoding="utf-8") as handle:
        processing = json.load(handle)
    if processing.get("input_features") != ["energy", "tau", "pressure", "diameter"]:
        raise ValueError("Expected the raw4 feature view")
    if processing.get("spectrum_normalization") != "global_log_standard":
        raise ValueError("Expected global_log_standard targets")

    x_train = np.load(os.path.join(args.input_dir, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(args.input_dir, "y_train.npy")).astype(np.float32)
    scaler = joblib.load(os.path.join(args.input_dir, "scaler_X.joblib"))
    raw_train = scaler.inverse_transform(x_train)
    lower, upper = raw_train.min(axis=0), raw_train.max(axis=0)
    wavelength_range = train_mlp.get_wavelength_range_nm(processing)
    wavelength_nm = np.linspace(wavelength_range[0], wavelength_range[1], y_train.shape[1])
    uv_mask = torch.tensor((wavelength_nm >= 200.0) & (wavelength_nm <= 700.0), device=device)
    output_norm = processing["output_normalization"]
    log_mean, log_std = float(output_norm["log_mean"]), float(output_norm["log_std"])
    model = build_model(4, y_train.shape[1], wavelength_range, args.checkpoint, device)

    def predictor(unit_params):
        raw = lower + np.asarray(unit_params, dtype=np.float32) * (upper - lower)
        scaled = scaler.transform(raw).astype(np.float32)
        with torch.no_grad():
            _, prediction = model(torch.as_tensor(scaled, device=device))
            log_power = prediction * log_std + log_mean
            power = torch.pow(10.0, torch.clamp(log_power, -30.0, 30.0))
            score = torch.sum(power[:, uv_mask], dim=1) / (torch.sum(power, dim=1) + 1e-20)
        return torch.clamp(score, 0.0, 1.0).cpu().numpy()

    rows = []
    try:
        from stable_baselines3 import SAC
    except ImportError as exc:
        raise ImportError("stable-baselines3 is required to load saved SAC agents") from exc

    for seed in args.seeds:
        agent_path = Path(args.rl_root) / f"rl_sac_seed{seed}_v1" / "sac_agent.zip"
        if not agent_path.is_file():
            raise FileNotFoundError(agent_path)
        env = SurrogateUVEnv(predictor, lower, upper, max_steps=20, action_scale=0.10, seed=seed)
        agent = SAC.load(str(agent_path), env=env, device=str(device))
        seed_rows = []
        for episode in range(args.episodes):
            observation, _ = env.reset(seed=seed * 10_000 + episode)
            terminated = False
            while not terminated:
                action, _ = agent.predict(observation, deterministic=True)
                observation, _, terminated, _, _ = env.step(action)
            seed_rows.append((float(env.score), env.params.copy(), seed))
        seed_rows.sort(key=lambda item: item[0], reverse=True)
        rows.extend(seed_rows[:args.top_k_per_seed])

    rows.sort(key=lambda item: item[0], reverse=True)
    fieldnames = ["rank", "seed", "uv_fraction_surrogate", "energy_j", "tau_s", "pressure_bar",
                  "diameter_m", "energy_uj", "tau_fs", "diameter_um"]
    output_rows = []
    for rank, (score, unit, seed) in enumerate(rows, start=1):
        raw = lower + unit * (upper - lower)
        output_rows.append({
            "rank": rank, "seed": seed, "uv_fraction_surrogate": float(np.clip(score, 0.0, 1.0)),
            "energy_j": float(raw[0]), "tau_s": float(raw[1]), "pressure_bar": float(raw[2]),
            "diameter_m": float(raw[3] * 1e-6), "energy_uj": float(raw[0] * 1e6),
            "tau_fs": float(raw[1] * 1e15), "diameter_um": float(raw[3]),
        })
    with open(os.path.join(args.output_dir, "rl_sac_candidates_reexported.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    with open(os.path.join(args.output_dir, "run_luna_validation.sh"), "w", encoding="utf-8") as handle:
        handle.write("#!/usr/bin/env bash\nset -euo pipefail\n")
        for row in output_rows:
            filename = f"candidate_rank{row['rank']:03d}_seed{row['seed']}.h5"
            handle.write(
                "julia --project=../.. anti_resonant_simulation.jl "
                f"-e {row['energy_uj']:.9g} --tau {row['tau_fs']:.9g} "
                f"-p {row['pressure_bar']:.9g} -d {row['diameter_um']:.9g} -t 0.65 "
                f"-o validation_h5/{filename}\n"
            )
    os.chmod(os.path.join(args.output_dir, "run_luna_validation.sh"), 0o755)
    with open(os.path.join(args.output_dir, "rl_sac_candidates_reexported.json"), "w", encoding="utf-8") as handle:
        json.dump({"n_candidates": len(output_rows), "source": "saved SAC agents", "rows": output_rows}, handle, indent=2)
    print(f"Wrote {len(output_rows)} candidates to {args.output_dir}")


if __name__ == "__main__":
    main()
