#!/usr/bin/env bash
# train120 正式轮: GPT-SoVITS 预处理 1A -> 1B(hubert+wav32k+sv) -> 1C(semantic) + 分片合并
# 用法: nohup bash ~/tts-sovits/run_preprocess_t120.sh > ~/tts-sovits/logs/preprocess_t120.log 2>&1 &
set -e
source ~/tts-sovits/env.sh
cd ~/tts-sovits/GPT-SoVITS
export PATH="$HOME/tts-sovits/bin:$PATH"
export HF_HOME="$HOME/tts-sovits/cache/hf"
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS:$HOME/tts-sovits/GPT-SoVITS"

EXP=mar7th_t120_v2pp
LIST="$HOME/tts-sovits/data/sovis_lists/train120.list"
WAVDIR="$HOME/tts-sovits/data/mar7th_train120/wavs"
OPT="$HOME/tts-sovits/GPT-SoVITS/logs/$EXP"
PY="$HOME/tts-sovits/miniconda3/envs/sovits/bin/python"
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

echo "== merge shards $(date +%T)"
cd "$OPT"
cat 2-name2text-0.txt > 2-name2text.txt
{ echo -e "item_name\tsemantic_audio"; cat 6-name2semantic-0.tsv; } > 6-name2semantic.tsv

echo "== verify coverage $(date +%T)"
"$PY" - << "PYEOF"
import os, sys

opt = os.environ["opt_dir"]
names = set()
with open(os.environ["inp_text"], encoding="utf-8") as f:
    for line in f:
        names.add(os.path.basename(line.split("|", 1)[0]).rsplit(".", 1)[0])

dirs = {"3-bert": ".pt", "4-cnhubert": ".pt", "5-wav32k": ".wav", "7-sv_cn": ".pt"}
problems = []
for d, ext in dirs.items():
    have = {f[: -len(ext)] for f in os.listdir(os.path.join(opt, d)) if f.endswith(ext)}
    miss = names - have
    if miss:
        problems.append((d, sorted(miss)[:5] + ([f"... 共{len(miss)}条缺失"] if len(miss) > 5 else [])))

text_lines = [l for l in open(os.path.join(opt, "2-name2text.txt"), encoding="utf-8").read().strip("\n").split("\n") if l]
sem_lines = [l for l in open(os.path.join(opt, "6-name2semantic.tsv"), encoding="utf-8").read().strip("\n").split("\n")[1:] if l]
if len(text_lines) != len(names):
    problems.append(("2-name2text.txt", [f"行数 {len(text_lines)} != 清单 {len(names)}"]))
if len(sem_lines) != len(names):
    problems.append(("6-name2semantic.tsv", [f"行数 {len(sem_lines)} != 清单 {len(names)}"]))

if problems:
    for k, v in problems:
        print(f"INCOMPLETE {k}: {v}", file=sys.stderr)
    sys.exit(1)
print(f"coverage OK: {len(names)} 条全部特征齐备")
PYEOF

echo "== preprocess done $(date +%T)"
