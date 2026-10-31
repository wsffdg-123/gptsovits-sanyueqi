# -*- coding: utf-8 -*-
"""v5 零样本测试：原版 s1v3.ckpt + v5 底模 SoVITS（不训练），对比 v2ProPlus 微调的吞字/音色。

两个变体：
  v5dev   sample_steps=32, cfg_rate=1.30（质量参照）
  v5turbo sample_steps=4,  cfg_rate=0.0 （部署候选，快）
参考音频沿用定稿 refC。在 n1 运行: python ~/tts-sovits/v5_zeroshot.py
输出: ~/tts-sovits/outputs/v5_zeroshot/<variant>/<tid>.wav
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
OUT = os.path.expanduser("~/tts-sovits/outputs/v5_zeroshot")

TEXTS = [
    ("baoxiao", "呃…能让布洛妮娅帮咱们报销吗？"),
    ("val_2", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
    ("val_3", "没错，我就是那个三月七"),
    ("val_4", "只是个建议啦——没事，你有我呢！"),
    ("val_5", "开拓者，快看！围墙后边好像有座灯炉诶"),
]

VARIANTS = ["v5dev", "v5turbo"]


def build_cfg(version: str) -> TTS_Config:
    section = {
        "device": "cuda", "is_half": True, "version": version,
        "t2s_weights_path": "GPT_SoVITS/pretrained_models/s1v3.ckpt",  # 零样本：原版 GPT
        "vits_weights_path": f"GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2G{version}.pth",
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    }
    return TTS_Config({version: dict(section), "custom": dict(section)})


for variant in VARIANTS:
    print(f"\n===== {variant} =====", flush=True)
    t0 = time.time()
    tts = TTS(build_cfg(variant))
    print(f"模型加载 {time.time() - t0:.1f}s", flush=True)
    outdir = os.path.join(OUT, variant)
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

print("\nV5 ZEROSHOT DONE")
