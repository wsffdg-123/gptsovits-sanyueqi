# 三月七 TTS 训练环境（与 AIC 完全隔离）
# 用法: source ~/tts-sovits/env.sh
export TTS_HOME="$HOME/tts-sovits"
export PATH="$TTS_HOME/bin:$PATH"
export PIP_CACHE_DIR="$TTS_HOME/cache/pip"
export HF_HOME="$TTS_HOME/cache/hf"
export HF_ENDPOINT="https://hf-mirror.com"
alias sovits_py="$TTS_HOME/miniconda3/envs/sovits/bin/python"
