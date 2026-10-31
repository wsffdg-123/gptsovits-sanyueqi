#!/usr/bin/env bash
# 幂等拉起三月七 TTS API —— 由本地 supervisor 的 tts-tunnel 远端命令在每次隧道（重）连时调用
#   ssh n1 'bash ~/tts-sovits/ensure_api.sh; exec sleep infinity'
# 就绪退出 0；120s 未就绪退出 1（隧道仍保持，由 supervisor 轮询 9880 兜底）
set -u
source ~/tts-sovits/env.sh   # TORCH_CUDNN_V8_API_DISABLED=1 + ffmpeg PATH

if ss -tln 2>/dev/null | grep -q ':9880 '; then
  echo "[ensure_api] already listening"
  exit 0
fi

nohup bash ~/tts-sovits/run_api.sh > ~/tts-sovits/logs/api.log 2>&1 < /dev/null &
echo "[ensure_api] launched (pid $!), waiting for port 9880 ..."
for i in $(seq 1 24); do
  sleep 5
  if ss -tln 2>/dev/null | grep -q ':9880 '; then
    echo "[ensure_api] ready after $((i * 5))s"
    exit 0
  fi
done
echo "[ensure_api] NOT ready after 120s, last log:"
tail -5 ~/tts-sovits/logs/api.log
exit 1
