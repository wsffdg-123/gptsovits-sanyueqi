# -*- coding: utf-8 -*-
"""train120 正式轮推理试听：t120 权重合成 val 文本，GPT e15/e10 对比。

在 n1 运行: python ~/tts-sovits/infer_t120.py
输出: ~/tts-sovits/outputs/t120/{e15,e10}/*.wav
"""
import os
import sys

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import soundfile as sf

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

SOVITS = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth")
GPT_DIR = os.path.expanduser("~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/Side0_al_mar7th_101.wav")
REF_TEXT = "哇，这么可爱的狗狗！…哇，你也来了。"
OUT_BASE = os.path.expanduser("~/tts-sovits/outputs/t120")
os.makedirs(OUT_BASE, exist_ok=True)

SENTENCES = [
    "今天也一起加油吧，有很多事情等着我们去做呢。",
    "欸？你说那个新来的转校生吗？我昨天在走廊碰到过一次。",
    "真的是服了，明明说好了要一起来的，结果他自己先跑掉了。",
    "这一路上看到了好多没见过的风景，我都想全部拍下来了！",
]


def synth(gpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    cfg = TTS_Config({
        "v2ProPlus": {
            "device": "cuda",
            "is_half": True,
            "version": "v2ProPlus",
            "t2s_weights_path": gpt_path,
            "vits_weights_path": SOVITS,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "custom": {
            "device": "cuda",
            "is_half": True,
            "version": "v2ProPlus",
            "t2s_weights_path": gpt_path,
            "vits_weights_path": SOVITS,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
    })
    tts = TTS(cfg)
    for i, text in enumerate(SENTENCES, 1):
        chunks = list(tts.run({
            "text": text,
            "text_lang": "zh",
            "ref_audio_path": REF,
            "prompt_text": REF_TEXT,
            "prompt_lang": "zh",
            "top_k": 15,
            "top_p": 1.0,
            "temperature": 1.0,
            "speed_factor": 1.0,
        }))
        if not chunks or chunks[0] is None:
            print(f"{out_dir}/val_{i}: no audio")
            continue
        sr, audio = chunks[0]
        out = os.path.join(out_dir, f"val_{i}.wav")
        sf.write(out, audio, sr)
        print(f"saved {out} ({len(audio)/sr:.1f}s)")


for tag, ckpt in (("e15", "mar7th_t120_v2pp-e15.ckpt"), ("e10", "mar7th_t120_v2pp-e10.ckpt")):
    print(f"== GPT {tag} ==")
    synth(os.path.join(GPT_DIR, ckpt), os.path.join(OUT_BASE, tag))
print("ALL DONE")
