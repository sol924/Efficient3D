#!/usr/bin/env bash
set -euo pipefail

# nohup bash scripts/batch_eval_efficient3d_pred_attn.sh > "outputs/batch_eval_efficient3d_pred_attn_run1.log" 2>&1 &

which_python=$(which python)
export PYTHONPATH="${PYTHONPATH:-}:${which_python}:."
echo "[INFO] PYTHONPATH=${PYTHONPATH}"

if command -v ip >/dev/null 2>&1; then
  export MASTER_ADDR=$(ip route get 1.1.1.1 2>/dev/null | awk '/src/ {for(i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}')
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  export MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
if [ -z "${MASTER_ADDR:-}" ]; then
  export MASTER_ADDR=127.0.0.1
fi

export TORCHELASTIC_PORT_RANGE=20000:39999
export MASTER_PORT="$(
python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"

IFACE=$(ip route get 1.1.1.1 2>/dev/null | awk '/dev/ {print $5; exit}') || true
if [ -z "${IFACE:-}" ]; then
  IFACE=$(ls /sys/class/net | grep -E '^(ib|enp|eno|eth|em)' | head -n1 || true)
fi
export NCCL_SOCKET_IFNAME="${IFACE:-}"
export GLOO_SOCKET_IFNAME="${IFACE:-}"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

echo "[DDP] IFACE=${IFACE:-<none>}"
echo "[DDP] MASTER_ADDR=$MASTER_ADDR"
echo "[DDP] MASTER_PORT=$MASTER_PORT"

epoch=3
batch_size=32
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
config=""
max_grad_norm=0.01
seed=42
use_location_token=False

llama_model_path="/gpfs/work/cpt/yuhuilin21/PointLLM/llm_ckpt"

train_tag="scanrefer#scan2cap#multi3dref#scanqa#sqa3d"
val_tag="scanrefer"
evaluate=True
debug=False
enable_wandb=False
gpu_num=4
do_save=False
pretrained_path="/gpfs/work/cpt/yuhuilin21/PointLLM/chat_scene/ckpt_01_3446.pth"
num_workers=4

use_fast_v=False
use_fast_v_oracle=True
token_pruning=False
use_external_attn_maps=True
use_a_map_ori=False
val_attn_maps_path="/path/to/infer_attn_maps_dir"

# Fixed pruning ratios (keep from 300 visual tokens)
rank_list=(15 60 90)
Ks=(2 6 16)

if [ -z "${gpu_num:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    gpu_num=$(
      python - <<'PY'
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
print(len([x for x in cvd.split(",") if x.strip()])) if cvd else print(1)
PY
    )
  fi
fi
if [ -z "${gpu_num:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_num=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}')
  else
    gpu_num=1
  fi
fi
if ! [[ "$gpu_num" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] Invalid gpu_num='$gpu_num'. Please set gpu_num to a positive integer." >&2
  exit 1
fi
NPROC="$gpu_num"
echo "[DDP] Using NPROC (nproc_per_node) = $NPROC"

export CUDA_VISIBLE_DEVICES=0,1,2,3

len=${#Ks[@]}
for ((i=0; i<len; i++)); do
  k=${Ks[i]}
  rank=${rank_list[i]}

  other_info="efficient3d_pred_attn_${use_external_attn_maps}_layer${k}_rank${rank}_oracle${use_fast_v_oracle}_a_map_ori${use_a_map_ori}"
  tag="${val_tag}__${other_info}"

  OUTPUT_DIR="outputs_mask_llm/$(date +'%Y%m%d_%H%M%S')_lr${lr}_ep${epoch}_${tag}"
  mkdir -p "${OUTPUT_DIR}"

  echo "[DDP] Launch torchrun with --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
  echo "[OUT] OUTPUT_DIR=${OUTPUT_DIR}"

  torchrun \
    --nproc_per_node="${NPROC}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    tasks/inference_efficient3d_pred_attn.py \
      "$(dirname "$0")/${config}config.py" \
      output_dir "$OUTPUT_DIR" \
      scheduler.epochs "$epoch" \
      optimizer.lr "$lr" \
      model.add_scene_token "$add_scene_token" \
      model.add_img_token "$add_img_token" \
      pretrained_path "$pretrained_path" \
      evaluate "$evaluate" \
      wandb.enable "$enable_wandb" \
      gpu_num "$gpu_num" \
      do_save "$do_save" \
      batch_size "$batch_size" \
      num_workers "$num_workers" \
      model.train_emb "$train_emb" \
      model.train_img_proj "$train_img_proj" \
      train_tag "$train_tag" \
      val_tag "$val_tag" \
      use_fast_v "$use_fast_v" \
      use_fast_v_oracle "$use_fast_v_oracle" \
      token_pruning "$token_pruning" \
      use_external_attn_maps "$use_external_attn_maps" \
      use_a_map_ori "$use_a_map_ori" \
      val_attn_maps_path "$val_attn_maps_path" \
      fast_v_agg_layer "$k" \
      fast_v_attention_rank "$rank" \
      model.no_obj "$no_obj" \
      model.input_dim "$input_dim" \
      model.bidirection "$bidirection" \
      optimizer.different_lr.enable "$different_lr" \
      model.max_obj_num "$max_obj_num" \
      lora.lora_r "$lora_r" \
      lora.lora_alpha "$lora_alpha" \
      model.add_pos_emb "$add_pos_emb" \
      model.feat_fusion "$feat_fusion" \
      optimizer.max_grad_norm "$max_grad_norm" \
      seed "$seed" \
      model.fuse_with_id "$fuse_with_id" \
      model.llama_model_path "$llama_model_path" \
      model.use_location_token "$use_location_token"
done

