#!/usr/bin/env python3
"""Run the unmodified RNNnonlinear Keras components on a benchmark .mat file."""

import argparse
import json
import os

import numpy as np
import scipy.io as sio
from sklearn.metrics import r2_score

from load_data import load_data
from make_RNN_model import make_RNN_model, update_RNN_model
from pred_evo import pred_evo


def fixed_horizon_metrics(model, truth, window_size, horizons):
    """Exact recursive k-step evaluation from true local histories for Keras."""
    output = {}
    n_evo, n_steps, n_grid = truth.shape
    for origin in ("all", "zero"):
        output[origin] = {}
        for horizon in horizons:
            starts = [0] if origin == "zero" else range(n_steps - horizon)
            predictions, targets = [], []
            for start in starts:
                prefix = np.repeat(truth[:, :1, :], window_size - 1, axis=1)
                padded = np.concatenate([prefix, truth], axis=1)
                current = padded[:, start:start + window_size, :]
                for _ in range(horizon):
                    prediction = model.predict_proba(current)
                    current = np.concatenate([current[:, 1:, :], prediction[:, None, :]], axis=1)
                predictions.append(prediction)
                targets.append(truth[:, start + horizon, :])
            prediction = np.concatenate(predictions, axis=0)
            target = np.concatenate(targets, axis=0)
            output[origin][str(horizon)] = {
                "horizon": horizon,
                "origin": origin,
                "n_start_positions": len(starts),
                "mse": float(np.mean((prediction - target) ** 2)),
                "r2": float(r2_score(target.reshape(-1), prediction.reshape(-1))),
            }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization", choices=["none", "dBm"], required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train-evolutions", type=int, default=1250)
    parser.add_argument("--test-evolutions", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--epochs-stage1", type=int, default=50)
    parser.add_argument("--epochs-stage2", type=int, default=30)
    parser.add_argument(
        "--eval-fixed-horizons", type=int, nargs="+", default=list(range(1, 11)),
        help="Exact recursive horizons to evaluate from true histories (default: 1 through 10).",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    i_x, x_train, x_test, y_train, y_test = load_data(
        args.data, args.train_evolutions, args.test_evolutions, args.steps,
        args.window_size, args.normalization
    )
    model = make_RNN_model(args.window_size, i_x)
    history1 = model.fit(x_train, y_train, epochs=args.epochs_stage1, validation_split=0.1, verbose=2)
    model = update_RNN_model(model)
    history2 = model.fit(x_train, y_train, epochs=args.epochs_stage2, validation_split=0.1, verbose=2)
    stepwise = model.predict_proba(x_test)
    autoreg = pred_evo(model, x_test, args.test_evolutions, args.steps, args.window_size, i_x)
    evo_size = args.steps - 1
    autoreg_true_map = y_test.reshape(args.test_evolutions, evo_size, i_x)
    autoreg_pred_map = autoreg.reshape(args.test_evolutions, evo_size, i_x)
    input_profiles = x_test[::evo_size, 0, :]
    truth = np.concatenate([input_profiles[:, None, :], autoreg_true_map], axis=1)
    fixed_horizon = fixed_horizon_metrics(
        model, truth, args.window_size, args.eval_fixed_horizons
    )
    metrics = {
        "stepwise_r2": float(r2_score(y_test.reshape(-1), stepwise.reshape(-1))),
        "autoregressive_r2": float(r2_score(y_test.reshape(-1), autoreg.reshape(-1))),
        "autoregressive_final_r2": float(r2_score(autoreg_true_map[:, -1].reshape(-1), autoreg_pred_map[:, -1].reshape(-1))),
        "normalization": args.normalization,
        "seed": args.seed,
        "steps": args.steps,
        "window_size": args.window_size,
        "fixed_horizon_metrics": fixed_horizon,
    }
    model.save(os.path.join(args.output_dir, "keras_model.h5"))
    sio.savemat(os.path.join(args.output_dir, "stepwise_predictions.mat"), {"Y_submit": stepwise, "Y_test": y_test})
    sio.savemat(os.path.join(args.output_dir, "autoregressive_predictions.mat"), {"Y_submit": autoreg, "Y_test": y_test})
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with open(os.path.join(args.output_dir, "fixed_horizon_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(fixed_horizon, handle, indent=2)
    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage1": history1.history, "stage2": history2.history}, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
