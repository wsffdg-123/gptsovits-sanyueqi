# -*- coding: utf-8 -*-
"""v5 LoRA 微调版推理测试：微调 SoVITS(v5turbo/v5dev) × GPT(e15 微调 / 原版 s1v3) 组合对照。

组合：
  v5turbo_ft           e15 GPT + v5turbo LoRA   ← 部署候选
  v5dev_ft             e15 GPT + v5dev LoRA     ← 质量参照（32步采样，慢）
  v5turbo_ft_baseGPT   原版 s1v3 + v5turbo LoRA ← 归因：GPT 微调对吞字的作用
在 n1 运行: python ~/tts-sovits/v5_finetune_test.py
输出: ~/tts-sovits/outputs/v5_ft/<组名>/<tid>.wav
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")  # cudnn v8 前端在变长 conv 上重建图，强制 legacy API
import os
import sys
import time

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import soundfile as sf
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
REF_TEXT = "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！"
OUT = os.path.expanduser("~/tts-sovits/outputs/v5_ft")

TEXTS = [
    ("baoxiao", "呃…能让布洛妮娅帮咱们报销吗？"),
    ("val_2", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
    ("val_3", "没错，我就是那个三月七"),
    ("val_4", "只是个建议啦——没事，你有我呢！"),
    ("val_5", "开拓者，快看！围墙后边好像有座灯炉诶"),
]

E15 = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
S1V3 = "GPT_SoVITS/pretrained_models/s1v3.ckpt"

COMBOS = [
    ("v5turbo_ft", "v5turbo",
     "SoVITS_weights_v5turbo/mar7th_t120_v5turbo_fixed.pth", E15),
    ("v5dev_ft", "v5dev",
     "SoVITS_weights_v5dev/mar7th_t120_v5dev_fixed.pth", E15),
    ("v5turbo_ft_baseGPT", "v5turbo",
     "SoVITS_weights_v5turbo/mar7th_t120_v5turbo_fixed.pth", S1V3),
]


def build_cfg(version: str, sovits: str, t2s: str) -> TTS_Config:
    section = {
        "device": "cuda", "is_half": True, "version": version,
        "t2s_weights_path": t2s,
        "vits_weights_path": os.path.expanduser(
            os.path.join("~/tts-sovits/GPT-SoVITS", sovits)),
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    }
    # 只传 custom 键！若把 version 键也传进去，TTS_Config.update 会覆盖 default_configs[version]，
    # LoRA 推理端就用 default_configs[version]['vits_weights_path'] 找底模——会拿到 LoRA pth 自己，
    # 导致 SoVITS 随机初始化（音小/时长失控/prompt semantic 乱码）
    return TTS_Config({"custom": dict(section)})


for tag, version, sovits, t2s in COMBOS:
    print(f"\n===== {tag} =====", flush=True)
    t0 = time.time()
    tts = TTS(build_cfg(version, sovits, t2s))
    print(f"模型加载 {time.time() - t0:.1f}s", flush=True)
    outdir = os.path.join(OUT, tag)
    os.makedirs(outdir, exist_ok=True)
    for tid, text in TEXTS:
        t1 = time.time()
        try:
            chunks = list(tts.run({
                "text": text, "text_lang": "zh",
                "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
                "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
            }))
        except Exception as e:
            print(f"{tid}: 异常 {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.time() - t1
        if chunks and chunks[0] is not None:
            sr, audio = chunks[0]
            sf.write(os.path.join(outdir, f"{tid}.wav"), audio, sr)
            print(f"{tid}: 合成 {dt:.2f}s → 音频 {len(audio) / sr:.1f}s "
                  f"({len(audio) / sr / dt:.1f}x 实时)", flush=True)
        else:
            print(f"{tid}: no audio ({dt:.2f}s)", flush=True)
    del tts
    torch.cuda.empty_cache()

print("\nV5 FINETUNE TEST DONE")
