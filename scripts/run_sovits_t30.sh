#!/usr/bin/env bash
# train30 验证轮: 1Ba SoVITS v2ProPlus 微调 + 修正导出
# 数据从 s2_ckpt_dir(exp_dir) 读取: 2-name2text.txt / 4-cnhubert / 5-wav32k / 7-sv_cn
# 用法: nohup bash ~/tts-sovits/run_sovits_t30.sh > ~/tts-sovits/logs/sovits_t30.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"

EXP=mar7th_t30_v2pp
S2DIR="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP"
PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
EPOCHS=12
mkdir -p "$S2DIR/logs_s2_v2ProPlus"
mkdir -p SoVITS_weights_v2Pro GPT_weights_v2Pro   # savee 用相对路径，缺失会使训练崩溃

"$PY" - << EOF
import json, os

exp = "$EXP"
s2dir = "$S2DIR"
with open("GPT_SoVITS/configs/s2v2ProPlus.json") as f:
    data = json.load(f)

data["train"]["batch_size"] = 8          # fp16; 显存 24G 充裕
data["train"]["epochs"] = $EPOCHS             # 30 分钟数据验证轮
data["train"]["text_low_lr_rate"] = 0.4
data["train"]["pretrained_s2G"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
data["train"]["pretrained_s2D"] = "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth"
data["train"]["if_save_latest"] = True
data["train"]["if_save_every_weights"] = True
data["train"]["save_every_epoch"] = 4
data["train"]["gpu_numbers"] = "0"
data["s2_ckpt_dir"] = s2dir + "/logs_s2_v2ProPlus"
data["data"]["exp_dir"] = s2dir
data["save_weight_dir"] = "SoVITS_weights_v2Pro"
data["name"] = exp
data["version"] = "v2ProPlus"
data["model"]["version"] = "v2ProPlus"

with open(os.path.join(s2dir, "tmp_s2.json"), "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print("tmp_s2.json written")
EOF

"$PY" -s GPT_SoVITS/s2_train.py --config "$S2DIR/tmp_s2.json"

# 修正导出: webui savee 在 CLI 下存 HParams/SimpleNamespace config，推理端
# (TTS_infer_pack) 要求嵌套 dict —— 从最新 G_*.pth + config.json 重新打包
"$PY" - << EOF
import glob, json, os, sys
from collections import OrderedDict

import torch

sys.path.insert(0, "GPT_SoVITS")
from process_ckpt import my_save2

exp = "$EXP"
with open("logs/%s/logs_s2_v2ProPlus/config.json" % exp) as f:
    cfg = json.load(f)

g = sorted(glob.glob("logs/%s/logs_s2_v2ProPlus/G_*.pth" % exp),
           key=os.path.getmtime)[-1]
print("exporting from", g)
d = torch.load(g, map_location="cpu", weights_only=False)
opt = OrderedDict()
opt["weight"] = d["model"]
opt["config"] = cfg
opt["info"] = "epoch_%siteration" % d.get("iteration", "?")
out = "SoVITS_weights_v2Pro/%s_e${EPOCHS}_fixed.pth" % exp
my_save2(opt, out, "v2ProPlus")
print("exported", out)
EOF

echo "== sovits train done $(date +%T)"
ls -la "$S2DIR/logs_s2_v2ProPlus" SoVITS_weights_v2Pro
