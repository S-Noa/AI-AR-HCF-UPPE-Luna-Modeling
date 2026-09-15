#!/usr/bin/env python3
"""Evaluate raw4 global-log surrogate quality for UV-fraction inverse design."""

import argparse
import json
import os
from types import SimpleNamespace

import joblib
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

import train_mlp


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--band-head-hidden", type=int, default=128)
    parser.add_argument("--uv-band-head-hidden", type=int, default=256)
    return parser.parse_args()


def load_model(params, checkpoint, args, device, output_dim):
    model_args = SimpleNamespace(
        num_bands=4,
        band_boundaries_nm=[200.0, 700.0, 1200.0, 1800.0, 2500.0],
        band_head_hidden=args.band_head_hidden,
        uv_band_head_hidden=args.uv_band_head_hidden,
    )
    wavelength_range = train_mlp.get_wavelength_range_nm(params)
    model, _, _ = train_mlp.build_model_for_type(
        "banded_mlp", 4, output_dim, model_args, wavelength_range,
        output_activation="identity",
    )
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), wavelength_range


def uv_fraction(standardized, log_mean, log_std, uv_mask):
    log_power = standardized * log_std + log_mean
    linear_power = np.power(10.0, np.clip(log_power, -30.0, 30.0))
    return linear_power[:, uv_mask].sum(axis=1) / (linear_power.sum(axis=1) + 1e-20)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.input_dir, "processing_params.json"), encoding="utf-8") as handle:
        params = json.load(handle)
    if params.get("input_features") != ["energy", "tau", "pressure", "diameter"]:
        raise ValueError("Expected the four-variable raw4 feature view")
    if params.get("spectrum_normalization") != "global_log_standard":
        raise ValueError("Expected global_log_standard targets")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    x_test = np.load(os.path.join(args.input_dir, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(args.input_dir, "y_test.npy")).astype(np.float32)
    _ = joblib.load(os.path.join(args.input_dir, "scaler_X.joblib"))
    model, wavelength_range = load_model(params, args.checkpoint, args, device, y_test.shape[1])

    predictions = []
    with torch.no_grad():
        for start in range(0, len(x_test), args.batch_size):
            batch = torch.from_numpy(x_test[start:start + args.batch_size]).to(device)
            predictions.append(model(batch).cpu().numpy())
    prediction = np.concatenate(predictions, axis=0)

    wavelength_nm = np.linspace(wavelength_range[0], wavelength_range[1], y_test.shape[1])
    uv_mask = (wavelength_nm >= 200.0) & (wavelength_nm <= 700.0)
    norm = params["output_normalization"]
    target_uv = uv_fraction(y_test, float(norm["log_mean"]), float(norm["log_std"]), uv_mask)
    pred_uv = uv_fraction(prediction, float(norm["log_mean"]), float(norm["log_std"]), uv_mask)
    top_k = min(args.top_k, len(target_uv))
    target_top = set(np.argsort(target_uv)[-top_k:])
    pred_top = set(np.argsort(pred_uv)[-top_k:])
    summary = {
        "n_test": int(len(y_test)),
        "final_logpower_r2": float(r2_score(y_test.ravel(), prediction.ravel())),
        "uv_logpower_r2": float(r2_score(y_test[:, uv_mask].ravel(), prediction[:, uv_mask].ravel())),
        "uv_fraction_spearman": float(spearmanr(target_uv, pred_uv).statistic),
        "uv_fraction_pearson": float(np.corrcoef(target_uv, pred_uv)[0, 1]),
        "top_k": int(top_k),
        "top_k_overlap": int(len(target_top & pred_top)),
        "top_k_recall": float(len(target_top & pred_top) / top_k),
    }
    with open(os.path.join(args.output_dir, "uv_ranking_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    np.savez_compressed(
        os.path.join(args.output_dir, "uv_ranking_predictions.npz"),
        target_uv_fraction=target_uv,
        predicted_uv_fraction=pred_uv,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
