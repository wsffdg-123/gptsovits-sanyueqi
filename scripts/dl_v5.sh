#!/usr/bin/env bash
# 下载 v5 底模（hf-mirror，~1.6GB）到 GPT_SoVITS/pretrained_models/gsv-v5-pretrained/
# 用法: nohup bash ~/tts-sovits/dl_v5.sh > ~/tts-sovits/logs/dl_v5.log 2>&1 &
set -e
D="$HOME/tts-sovits/GPT-SoVITS/GPT_SoVITS/pretrained_models/gsv-v5-pretrained"
mkdir -p "$D"
cd "$D"
BASE=https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main/gsv-v5-pretrained
for f in s2Gv5dev.pth s2Gv5turbo.pth vocoder.pth; do
  echo "== $f =="
  curl -L -C - --retry 3 -o "$f" "$BASE/$f"
  stat -c "%n %s bytes" "$f"
done
echo "V5 DOWNLOAD DONE"
