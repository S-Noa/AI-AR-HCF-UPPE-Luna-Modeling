# Next-stage Luna Surrogate Commands

Run from the corresponding cloud working directories.

## 1. Export conditional Luna data for RNN

```bash
cd /mnt/Luna.jl-master/rnnnonlinear-master/rnnnonlinear-master

nohup python3 prepare_luna_data.py \
  --input-dir ../../processed_data_t650_v7 \
  --output simulations/luna_t650_temporal_conditional.mat \
  --output-format hdf5 \
  --compression gzip \
  --include-features \
  > prepare_luna_t650_conditional.log 2>&1 &
```

## 2. Train conditional autoregressive RNN

```bash
cd /mnt/Luna.jl-master/rnnnonlinear-master/rnnnonlinear-master

nohup python3 train_luna_rnn.py \
  --data simulations/luna_t650_temporal_conditional.mat \
  --output-dir results_luna_rnn_t650_conditional_rollout_v2 \
  --conditioning features_z \
  --training-mode scheduled_sampling \
  --rollout-steps 50 \
  --scheduled-sampling-start 0.0 \
  --scheduled-sampling-end 0.5 \
  --eval-autoregressive-every 2 \
  --eval-autoregressive-samples 64 \
  --epochs 100 \
  --batch-size 16 \
  --log-file results_luna_rnn_t650_conditional_rollout_v2/train.log \
  > train_luna_rnn_conditional_rollout_v2.nohup.log 2>&1 &
```

## 3. Fine-tune temporal Transformer v3 for evolution-map quality

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

nohup python3 train_mlp.py \
  --input-dir processed_data_t650_v7 \
  --output-dir models_t650_transformer_temporal_finetune_v4 \
  --model transformer \
  --transformer-mode temporal \
  --transformer-d-model 192 \
  --transformer-heads 4 \
  --transformer-layers 4 \
  --transformer-use-z-embedding \
  --transformer-band-boundaries-nm 200 700 1200 1800 2500 \
  --finetune-from models_t650_transformer_temporal_v3/best_val_r2_model.pth \
  --learning-rate 5e-5 \
  --epochs 100 \
  --batch-size 4 \
  --selection-metric val_temporal_r2 \
  --early-z-max-cm 10 \
  --early-z-loss-weight 1.2 \
  --lambda-gradient-weight 0.03 \
  --uv-loss-weight 1.2 \
  --uv-second-derivative-weight 0.003 \
  --long-wavelength-weight 1.3 \
  --long-wavelength-cutoff-nm 1200 \
  --visualize \
  > train_transformer_temporal_finetune_v4.log 2>&1 &
```

## 4. Train global-log real-intensity baselines

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

nohup python3 train_mlp.py \
  --input-dir processed_data_t650_v7_global_log \
  --output-dir models_t650_banded_global_log_v1 \
  --model banded_mlp \
  --band-boundaries-nm 200 700 1200 1800 2500 \
  --output-activation identity \
  --epochs 300 \
  --batch-size 128 \
  --selection-metric val_r2 \
  --long-wavelength-weight 1.2 \
  --visualize \
  > train_banded_global_log_v1.log 2>&1 &
```

```bash
nohup python3 train_mlp.py \
  --input-dir processed_data_t650_v7_global_log \
  --output-dir models_t650_temporal_cnn_global_log_v1 \
  --model temporal_cnn \
  --temporal-cnn-architecture multiscale \
  --temporal-cnn-coordinate-channels \
  --output-activation identity \
  --epochs 500 \
  --batch-size 4 \
  --learning-rate 1e-3 \
  --selection-metric val_temporal_r2 \
  --early-z-max-cm 10 \
  --early-z-loss-weight 1.5 \
  --lambda-gradient-weight 0.03 \
  --long-wavelength-weight 1.3 \
  --visualize \
  > train_temporal_cnn_global_log_v1.log 2>&1 &
```

## 5. First-stage inverse design with a forward surrogate

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

python3 inverse_design.py \
  --input-dir processed_data_t650_v7 \
  --checkpoint models_t650_transformer_temporal_v3/best_val_r2_model.pth \
  --output-dir inverse_uv_fraction_transformer_v3 \
  --model transformer \
  --transformer-mode temporal \
  --transformer-d-model 192 \
  --transformer-heads 4 \
  --transformer-layers 4 \
  --transformer-use-z-embedding \
  --transformer-band-boundaries-nm 200 700 1200 1800 2500 \
  --objective uv_fraction \
  --n-restarts 64 \
  --steps 400 \
  --top-k 12
```

Then verify the generated candidates with:

```bash
bash inverse_uv_fraction_transformer_v3/luna_validation_commands.sh
```
