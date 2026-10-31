# -*- coding: utf-8 -*-
"""0.1 卡（显存配额）推理可行性：用 set_per_process_memory_fraction 模拟总显存预算。

用法: python test_memcap.py <cap_mb>
  cap_mb = 进程可用的总显存预算（含 ~300MB CUDA 上下文开销）
  1203 ≈ 0.1 卡（按当前半卡 12032MiB 计）；2456 ≈ 0.1 卡（按全卡 24564MiB 计）
定稿配置（v2ProPlus e8_fixed + e15 + refC + hot）。在 n1 运行。
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")  # cudnn v8 前端在变长 conv 上重建图，强制 legacy API
import sys
import time

CAP_MB = int(sys.argv[1])
GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import torch

total = torch.cuda.get_device_properties(0).total_memory / 2**20
budget = max(256, CAP_MB - 300)  # 预留 CUDA 上下文 ~300MB
torch.cuda.set_per_process_memory_fraction(budget / total, 0)
print(f"GPU 可见 {total:.0f}MiB | 预算 {CAP_MB}MB | torch 分配上限 {budget:.0f}MiB", flush=True)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

SOVITS = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth")
GPT = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
section = {
    "device": "cuda", "is_half": True, "version": "v2ProPlus",
    "t2s_weights_path": GPT, "vits_weights_path": SOVITS,
    "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
}
TEXTS = [
    ("short", "没错，我就是那个三月七"),
    ("long", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
]

try:
    t0 = time.time()
    tts = TTS(TTS_Config({"custom": dict(section)}))
    print(f"模型加载 {time.time() - t0:.1f}s, allocated "
          f"{torch.cuda.memory_allocated() / 2**20:.0f}MiB", flush=True)
    for tid, text in TEXTS:
        t1 = time.time()
        chunks = list(tts.run({
            "text": text, "text_lang": "zh",
            "ref_audio_path": REF, "prompt_text": "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！",
            "prompt_lang": "zh",
            "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
        }))
        sr, audio = chunks[0]
        print(f"{tid}: 合成 {time.time() - t1:.1f}s → {len(audio) / sr:.1f}s 音频", flush=True)
    print(f"峰值 reserved {torch.cuda.max_memory_reserved() / 2**20:.0f}MiB", flush=True)
    print("RESULT PASS")
except torch.OutOfMemoryError as e:
    print(f"RESULT OOM: {str(e).splitlines()[0]}", flush=True)
    sys.exit(1)
