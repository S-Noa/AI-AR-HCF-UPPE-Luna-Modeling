# Luna Data Adaptation Notes

## RNNnonlinear principle

The original project trains an LSTM as a one-step nonlinear propagation map:

```text
[spectrum(z-k+1), ..., spectrum(z)] -> spectrum(z+1)
```

At inference time, the predicted next spectrum is appended back into the input
window, so the full propagation evolution is produced autoregressively.

The historical Keras model is:

```text
LSTM -> Dense(ReLU) -> Dense(ReLU) -> Dense(Sigmoid)
```

with RMSprop and MSE loss.

## Luna conversion

`prepare_luna_data.py` converts Luna processed temporal labels:

```text
y_temporal_<split>.npy: (N, n_z, n_lambda)
```

to the RNNnonlinear MATLAB layout:

```text
data: (N, n_lambda, n_z)
```

Example:

```bash
python prepare_luna_data.py \
  --input-dir ../../Luna.jl-master/examples/simple_interface/processed_data_t650_nochirp_v1 \
  --output simulations/luna_t650_temporal.mat
```

## PyTorch baseline

`train_luna_rnn.py` is a PyTorch equivalent of the old Keras baseline. It avoids
pinning the experiment to TensorFlow 1.x / Keras 2.2.

Example:

```bash
python train_luna_rnn.py \
  --data simulations/luna_t650_temporal.mat \
  --output-dir results_luna_rnn_t650 \
  --window-size 10 \
  --epochs 50 \
  --batch-size 128 \
  --output-activation sigmoid
```

It reports both:

- step-wise teacher-forced prediction
- autoregressive full-evolution prediction

## Why this can preserve real intensity

The original RNNnonlinear preprocessing uses global normalization choices, such
as global max normalization, fixed dB dynamic range, or manual physical scale.
That preserves relative amplitude differences between samples.

The default Luna ML preprocessing uses per-sample min-max normalization of log
spectra, then another per-sample temporal min-max normalization. This removes
absolute power, peak intensity, spectral integral, and between-sample scale
information. To train absolute/log intensity models, regenerate processed data
with `--spectrum-normalization global_log_standard` or `none_log`, and use model
heads without mandatory sigmoid output.
