# Efficient3D:AUnifiedFrameworkforAdaptiveandDebiasedTokenReduction in3DMLLMs

本仓库用于 3D MLLM 的高效 Token 裁剪，当前包含两部分：

- `Efficient3D`：Chat-Scene 注意力图提取 + 基于预测图的加速推理（固定比例/自适应比例）。
- `DVTIE`：用于学习并预测全局注意力图。

## 目录结构

```text
Efficient3D/
├─ DVTIE/
├─ scripts/
├─ tasks/
├─ models/
├─ dataset/
├─ transformers/
└─ ...
```

## 0. 环境准备

在 `Efficient3D` 根目录执行：

```bash
pip install -r requirements.txt
```

## 1. 提取 Chat-Scene 注意力图

先修改脚本中的路径与标签参数：

- `scripts/extract_efficient3d_attn_maps.sh`
  - `llama_model_path`
  - `pretrained_path`
  - `train_tag` / `val_tag`

执行：

```bash
bash scripts/extract_efficient3d_attn_maps.sh
```

输出目录示例：

- `outputs/<timestamp>_...__extract_efficient3d_attn_maps/`
- 其中会保存 `infer_attn_maps_*.pt`，可作为 DVTIE 的训练目标。

## 2. 训练/验证 DVTIE 并保存结果

### 2.1 配置

按需修改：

- `DVTIE/config/train.yaml`
- `DVTIE/config/test.yaml`

关键字段：

- `annotation_root`
- `attn_maps_root`
- `output_dir`
- `pretrained_model_path`（仅推理时必须）

### 2.2 训练并验证（同时可导出注意力预测图）

```bash
python DVTIE/training_dvtie.py --config DVTIE/config/train.yaml --mode train_infer
```

### 2.3 仅验证/推理并保存 DVTIE 预测图

```bash
python DVTIE/training_dvtie.py \
  --config DVTIE/config/test.yaml \
  --mode infer \
  --pretrained_model_path <path_to_checkpoint_best.pth>
```

输出目录示例：

- `DVTIE/outputs/train_dvtie-.../checkpoint_best.pth`
- `DVTIE/outputs/eval_dvtie-.../infer_attn_maps_val_*.pt`

其中 `infer_attn_maps_val_*.pt` 即后续 Efficient3D 推理使用的预测图。

## 3. 使用 DVTIE 结果做固定裁剪比例推理

先修改：

- `scripts/batch_eval_efficient3d_pred_attn.sh`
  - `val_attn_maps_path` 指向包含 `infer_attn_maps_val_*.pt` 的目录
  - `rank_list`（固定裁剪保留 token 数）
  - `Ks`（聚合层）
  - 权重路径参数（`llama_model_path`、`pretrained_path`）

执行：

```bash
bash scripts/batch_eval_efficient3d_pred_attn.sh
```

## 4. 使用 DVTIE 结果做自适应裁剪比例推理

本仓库提供了“注释版 Adaptive 逻辑”开关脚本：

- `scripts/batch_eval_efficient3d_pred_attn_commented_adaptive.sh`

推荐流程：

1. 修改 `scripts/batch_eval_efficient3d_pred_attn_adaptive.sh`：
   - `val_attn_maps_path` 指向 DVTIE 预测图目录
   - `alpha_list`、`Ks`、模型路径参数
2. 执行：

```bash
bash scripts/batch_eval_efficient3d_pred_attn_commented_adaptive.sh \
  bash scripts/batch_eval_efficient3d_pred_attn_adaptive.sh
```

该命令会临时启用 `transformers/src/transformers/models/llama/modeling_llama.py` 中注释的 Adaptive 分支，运行结束后自动恢复原文件。

## 致谢

本项目参考并感谢以下开源项目：

- Fast3D: https://github.com/wencan25/Fast3D
- Chat-Scene: https://github.com/ZzZZCHS/Chat-Scene
