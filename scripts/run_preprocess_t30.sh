#!/usr/bin/env bash
# train30 验证轮: GPT-SoVITS 预处理 1A -> 1B(hubert+wav32k+sv) -> 1C(semantic)
# 用法: nohup bash ~/tts-sovits/run_preprocess_t30.sh > ~/tts-sovits/logs/preprocess_t30.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export HF_ENDPOINT="https://hf-mirror.com"

EXP=mar7th_t30_v2pp
LIST="$HOME/tts-sovits/data/sovis_lists/train30.list"
WAVDIR="$HOME/tts-sovits/data/mar7th_train30/wavs"
OPT="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP"
PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"
mkdir -p "$OPT" ~/tts-sovits/logs

export inp_text="$LIST" inp_wav_dir="$WAVDIR" exp_name="$EXP" opt_dir="$OPT"
export i_part=0 all_parts=1 _CUDA_VISIBLE_DEVICES=0 is_half=True version=v2ProPlus
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
export sv_path="GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
export pretrained_s2G="GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"
export s2config_path="GPT_SoVITS/configs/s2v2ProPlus.json"

echo "== 1A get-text $(date +%T)"
"$PY" -s GPT_SoVITS/prepare_datasets/1-get-text.py

echo "== 1B hubert+wav32k $(date +%T)"
"$PY" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py

echo "== 1B sv $(date +%T)"
"$PY" -s GPT_SoVITS/prepare_datasets/2-get-sv.py

echo "== 1C semantic $(date +%T)"
"$PY" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py

echo "== preprocess done $(date +%T)"
ls -la "$OPT"
