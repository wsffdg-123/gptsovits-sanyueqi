#!/usr/bin/env bash
# v5 微调（LoRA）: 复用 t120 预处理产物，SoVITS v5turbo + v5dev 两变体串行
# GPT 不重训（t2s 各版本通用，推理时配 mar7th_t120_v2pp-e15.ckpt）
# 路线: s2_train_v3_lora.py 纯 CFM LoRA（无判别器，不需要 s2D），底模 = gsv-v5-pretrained/s2G<variant>.pth
# 用法: nohup bash ~/tts-sovits/run_train_t120_v5.sh > ~/tts-sovits/logs/train_t120_v5.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"
export _CUDA_VISIBLE_DEVICES=0
export TORCH_CUDNN_V8_API_DISABLED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # DiT 注意力显存碎片

PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
EXP_SRC=mar7th_t120_v2pp   # 复用其预处理产物 (2-name2text / 4-cnhubert / 5-wav32k / 6-name2semantic)
EXP_DIR="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP_SRC"

for VARIANT in v5turbo v5dev; do
  EXP="mar7th_t120_${VARIANT}"
  echo "== [$EXP] LoRA train start $(date +%T)"
  mkdir -p "SoVITS_weights_${VARIANT}"
  "$PY" - << EOF
import json
with open("GPT_SoVITS/configs/s2.json") as f:
    data = json.load(f)
data["train"]["batch_size"] = 8   # v5 DiT 注意力吃显存：batch 12 fp16 OOM(20.9GB)，8 稳
data["train"]["epochs"] = 8
data["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2G${VARIANT}.pth"
data["train"]["pretrained_s2D"] = ""          # LoRA 路线无判别器
data["train"]["lora_rank"] = 32               # webui 默认档 16/32/64/128
data["train"]["if_save_latest"] = True
data["train"]["if_save_every_weights"] = True
data["train"]["save_every_epoch"] = 4
data["train"]["gpu_numbers"] = "0"
data["train"]["grad_ckpt"] = True             # 梯度检查点，换显存
data["s2_ckpt_dir"] = "$EXP_DIR"
data["data"]["exp_dir"] = "$EXP_DIR"
data["save_weight_dir"] = "SoVITS_weights_${VARIANT}"
data["name"] = "$EXP"
data["version"] = "${VARIANT}"
data["model"]["version"] = "${VARIANT}"
with open("logs/${EXP}_tmp_s2.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print("tmp json written")
EOF
  "$PY" -s GPT_SoVITS/s2_train_v3_lora.py --config "logs/${EXP}_tmp_s2.json"
  echo "== [$EXP] train done $(date +%T)"

  # fixed 导出: savee 存的 HParams config 推理端不可下标访问会炸（v2ProPlus 同款坑），重写为 dict
  "$PY" - << EOF
import sys, json, glob, os
sys.path.insert(0, "GPT_SoVITS")
from collections import OrderedDict
from process_ckpt import load_sovits_new, my_save2
variant = "${VARIANT}"
exp = "${EXP}"
srcs = glob.glob("SoVITS_weights_%s/%s_e*_l32.pth" % (variant, exp))
assert srcs, "no savee output found"
src = max(srcs, key=os.path.getmtime)
print("exporting from", src)
d = load_sovits_new(src)   # savee 产物带版本标签头，须用 load_sovits_new 剥离（torch.load 会 stack underflow）
cfg = json.load(open("GPT_SoVITS/configs/s2.json"))
cfg["model"]["version"] = variant
cfg["model"]["semantic_frame_rate"] = "25hz"
opt = OrderedDict()
opt["weight"] = d["weight"]
opt["config"] = cfg
opt["info"] = d.get("info", "")
opt["lora_rank"] = d["lora_rank"]
dst = "SoVITS_weights_%s/%s_fixed.pth" % (variant, exp)
my_save2(opt, dst, variant)
print("exported", dst)
EOF
done
echo "== ALL DONE $(date +%T)"
ls -la SoVITS_weights_v5turbo SoVITS_weights_v5dev
