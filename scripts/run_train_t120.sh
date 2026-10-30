#!/usr/bin/env bash
# train120 正式轮: 1Ba SoVITS 8 epoch -> 1Bb GPT 15 epoch 串行
# 用法: nohup bash ~/tts-sovits/run_train_t120.sh > ~/tts-sovits/logs/train_t120.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"
export _CUDA_VISIBLE_DEVICES=0

EXP=mar7th_t120_v2pp
LOGDIR="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP"
PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
mkdir -p "$LOGDIR/logs_s2_v2ProPlus" "$LOGDIR/logs_s1_v2ProPlus"
mkdir -p SoVITS_weights_v2Pro GPT_weights_v2Pro   # 预建，避免 savee 相对路径失败

################ 1Ba SoVITS ################
"$PY" - << EOF
import json

exp = "$EXP"
logdir = "$LOGDIR"
with open("GPT_SoVITS/configs/s2v2ProPlus.json") as f:
    data = json.load(f)

data["train"]["batch_size"] = 12
data["train"]["epochs"] = 8
data["train"]["text_low_lr_rate"] = 0.4
data["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
data["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
data["train"]["if_save_latest"] = True
data["train"]["if_save_every_weights"] = True
data["train"]["save_every_epoch"] = 4
data["train"]["gpu_numbers"] = "0"
data["s2_ckpt_dir"] = logdir + "/logs_s2_v2ProPlus"
data["data"]["exp_dir"] = logdir
data["save_weight_dir"] = "SoVITS_weights_v2Pro"
data["name"] = exp
data["version"] = "v2ProPlus"
data["model"]["version"] = "v2ProPlus"

with open(logdir + "/tmp_s2.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print("tmp_s2.json written")
EOF

echo "== 1Ba sovits train start $(date +%T)"
"$PY" -s GPT_SoVITS/s2_train.py --config "$LOGDIR/tmp_s2.json"
echo "== 1Ba sovits train done $(date +%T)"

# 修正导出: 完整 dict config（webui savee 存 HParams 会导致推理端不兼容，此处预防性修正）
"$PY" - << EOF
import sys, json, glob
sys.path.insert(0, "GPT_SoVITS")
from process_ckpt import load_sovits_new, my_save2
from collections import OrderedDict

exp = "$EXP"
with open("logs/%s/logs_s2_v2ProPlus/config.json" % exp) as f:
    cfg = json.load(f)

# 取最新 G_*.pth
g = sorted(glob.glob("logs/%s/logs_s2_v2ProPlus/G_*.pth" % exp), key=lambda p: __import__("os").path.getmtime(p))[-1]
print("exporting from", g)
d = torch_load = None
import torch
d = torch.load(g, map_location="cpu", weights_only=False)
opt = OrderedDict()
opt["weight"] = d["model"]
opt["config"] = cfg
opt["info"] = "epoch_%siteration" % d.get("iteration", "?")
my_save2(opt, "SoVITS_weights_v2Pro/%s_e8_fixed.pth" % exp, "v2ProPlus")
print("exported SoVITS_weights_v2Pro/%s_e8_fixed.pth" % exp)
EOF

################ 1Bb GPT ################
"$PY" - << EOF
import yaml

exp = "$EXP"
logdir = "$LOGDIR"
with open("GPT_SoVITS/configs/s1longer-v2.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

data["train"]["batch_size"] = 12
data["train"]["epochs"] = 15
data["train"]["save_every_n_epoch"] = 5
data["train"]["if_save_every_weights"] = True
data["train"]["if_save_latest"] = True
data["train"]["if_dpo"] = False
data["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
data["train"]["half_weights_save_dir"] = "GPT_weights_v2Pro"
data["train"]["exp_name"] = exp
data["train_semantic_path"] = logdir + "/6-name2semantic.tsv"
data["train_phoneme_path"] = logdir + "/2-name2text.txt"
data["output_dir"] = logdir + "/logs_s1_v2ProPlus"

with open(logdir + "/tmp_s1.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
print("tmp_s1.yaml written")
EOF

echo "== 1Bb gpt train start $(date +%T)"
"$PY" -s GPT_SoVITS/s1_train.py --config_file "$LOGDIR/tmp_s1.yaml"
echo "== 1Bb gpt train done $(date +%T)"

echo "== ALL DONE $(date +%T)"
ls -la SoVITS_weights_v2Pro GPT_weights_v2Pro
