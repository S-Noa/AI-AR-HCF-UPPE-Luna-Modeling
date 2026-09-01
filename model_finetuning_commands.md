# CNN 与 Transformer 微调命令建议

## 原则

- `--finetune-from` 是 fine-tuning，不是完整断点续训；它只加载模型权重，不恢复 optimizer/scheduler。
- 微调时必须保持模型结构参数与 checkpoint 完全一致，否则会出现权重 shape mismatch。
- 对传播演化任务，优先使用 `--selection-metric val_temporal_r2` 或 `val_temporal_uv_r2`，不要只看最终光谱 `val_r2`。

## Transformer temporal v3 微调

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
  --early-z-loss-weight 1.5 \
  --lambda-gradient-weight 0.02 \
  --uv-loss-weight 1.0 \
  --long-wavelength-weight 1.2 \
  --long-wavelength-cutoff-nm 1200 \
  --visualize \
  > train_transformer_temporal_finetune_v4.log 2>&1 &
```

如果更关注 UV 演化，把选择指标改为：

```bash
--selection-metric val_temporal_uv_r2
```

如果更关注前 10 cm 早期传播，把选择指标改为：

```bash
--selection-metric val_early_z_r2
```

## Temporal CNN 微调模板

先确认原 CNN 训练时使用的是 `basic` 还是 `multiscale` 架构，并保持一致。以下是 multiscale 版本模板：

```bash
cd /mnt/Luna.jl-master/Luna.jl-master/examples/simple_interface

nohup python3 train_mlp.py \
  --input-dir processed_data_t650_v7 \
  --output-dir models_t650_temporal_cnn_finetune_v4 \
  --model temporal_cnn \
  --temporal-cnn-architecture multiscale \
  --temporal-cnn-coordinate-channels \
  --temporal-cnn-band-boundaries-nm 200 700 1200 1800 2500 \
  --temporal-cnn-channels 128 \
  --temporal-cnn-lambda-init 128 \
  --temporal-cnn-z-init 16 \
  --finetune-from PATH_TO_CNN_CHECKPOINT.pth \
  --learning-rate 5e-5 \
  --epochs 100 \
  --batch-size 8 \
  --selection-metric val_temporal_r2 \
  --early-z-max-cm 10 \
  --early-z-loss-weight 1.5 \
  --lambda-gradient-weight 0.04 \
  --uv-loss-weight 1.0 \
  --long-wavelength-weight 1.4 \
  --long-wavelength-cutoff-nm 1200 \
  --visualize \
  > train_temporal_cnn_finetune_v4.log 2>&1 &
```

如果原 CNN 是 `basic` 架构，应删去：

```bash
--temporal-cnn-architecture multiscale
--temporal-cnn-coordinate-channels
--temporal-cnn-band-boundaries-nm 200 700 1200 1800 2500
```

并把 `--temporal-cnn-lambda-init` 改回原训练值。

## 2 epoch smoke test

正式跑长训练前，先测试 checkpoint 是否能加载：

```bash
python3 train_mlp.py \
  --input-dir processed_data_t650_v7 \
  --output-dir smoke_finetune_load \
  --model transformer \
  --transformer-mode temporal \
  --transformer-d-model 192 \
  --transformer-heads 4 \
  --transformer-layers 4 \
  --transformer-use-z-embedding \
  --transformer-band-boundaries-nm 200 700 1200 1800 2500 \
  --finetune-from models_t650_transformer_temporal_v3/best_val_r2_model.pth \
  --learning-rate 5e-5 \
  --epochs 2 \
  --batch-size 2 \
  --selection-metric val_temporal_r2 \
  --visualize
```
