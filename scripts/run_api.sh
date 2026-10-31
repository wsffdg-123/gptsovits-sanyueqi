#!/usr/bin/env bash
# 三月七 TTS API 服务启动器（定稿配置固化在 mar7th_api.py）
#   启动: nohup bash ~/tts-sovits/run_api.sh > ~/tts-sovits/logs/api.log 2>&1 &
#   停止: kill $(cat ~/tts-sovits/api.pid)
#   就绪: 模型加载 + 2 句预热约 10 秒后开 9880 端口，日志见 ~/tts-sovits/logs/api.log
set -e
source ~/tts-sovits/env.sh   # TORCH_CUDNN_V8_API_DISABLED=1（关键，35s/句 → 2.6s/句）+ ffmpeg PATH

if ss -tln 2>/dev/null | grep -q ':9880 '; then
  echo "端口 9880 已被占用（可能旧实例在跑）。停止旧实例: kill \$(cat ~/tts-sovits/api.pid)" >&2
  exit 1
fi

echo $$ > ~/tts-sovits/api.pid
exec "$TTS_HOME/miniconda3/envs/sovits/bin/python" "$TTS_HOME/mar7th_api.py"
