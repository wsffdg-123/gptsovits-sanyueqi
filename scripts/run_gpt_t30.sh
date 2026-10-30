#!/usr/bin/env bash
# train30 验证轮: 1Bb GPT (s1) 微调
# 用法: nohup bash ~/tts-sovits/run_gpt_t30.sh > ~/tts-sovits/logs/gpt_t30.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"

EXP=mar7th_t30_v2pp
S1DIR="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP"
PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
mkdir -p "$S1DIR/logs_s1_v2ProPlus"

# 对照 webui open1Bb: 基于 s1longer-v2.yaml 注入训练参数
"$PY" - << EOF
import yaml, os

exp = "$EXP"
s1dir = "$S1DIR"
with open("GPT_SoVITS/configs/s1longer-v2.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

data["train"]["batch_size"] = 6       # GPT 阶段显存占用更高，保守
data["train"]["epochs"] = 20          # 30 分钟数据验证轮
data["train"]["save_every_n_epoch"] = 5
data["train"]["if_save_every_weights"] = True
data["train"]["if_save_latest"] = True
data["train"]["if_dpo"] = False
data["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
data["train"]["half_weights_save_dir"] = "GPT_weights_v2Pro"
data["train"]["exp_name"] = exp
data["train_semantic_path"] = s1dir + "/6-name2semantic.tsv"
data["train_phoneme_path"] = s1dir + "/2-name2text.txt"
data["output_dir"] = s1dir + "/logs_s1_v2ProPlus"

with open(s1dir + "/tmp_s1.yaml", "w") as f:
    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
print("tmp_s1.yaml written")
EOF

export _CUDA_VISIBLE_DEVICES=0
"$PY" -s GPT_SoVITS/s1_train.py --config_file "$S1DIR/tmp_s1.yaml"

echo "== gpt train done $(date +%T)"
ls -la "$S1DIR/logs_s1_v2ProPlus" GPT_weights_v2Pro 2>/dev/null
