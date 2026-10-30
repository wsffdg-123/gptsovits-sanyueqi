# -*- coding: utf-8 -*-
"""train30 验证轮推理试听：微调权重合成 val 集文本。

在 n1 运行: source ~/tts-sovits/env.sh && python ~/tts-sovits/infer_t30.py
输出: ~/tts-sovits/outputs/t30_e12/*.wav
"""
import os
import sys

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import numpy as np
import soundfile as sf
import torch

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

EXP_E12 = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t30_v2pp_e12_s648_fixed.pth")
GPT_E20 = os.path.expanduser("~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t30_v2pp-e20.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/Side0_al_mar7th_101.wav")
REF_TEXT = "哇，这么可爱的狗狗！…哇，你也来了。"
OUT_DIR = os.path.expanduser("~/tts-sovits/outputs/t30_e12")
os.makedirs(OUT_DIR, exist_ok=True)

# 未参与训练的 val 集句子
SENTENCES = [
    "今天也一起加油吧，有很多事情等着我们去做呢。",
    "欸？你说那个新来的转校生吗？我昨天在走廊碰到过一次。",
    "真的是服了，明明说好了要一起来的，结果他自己先跑掉了。",
]

cfg = TTS_Config({
    "v2ProPlus": {
        "device": "cuda",
        "is_half": True,
        "version": "v2ProPlus",
        "t2s_weights_path": GPT_E20,
        "vits_weights_path": EXP_E12,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "custom": {
        "device": "cuda",
        "is_half": True,
        "version": "v2ProPlus",
        "t2s_weights_path": GPT_E20,
        "vits_weights_path": EXP_E12,
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
        print(f"val_{i}: no audio returned")
        continue
    sr, audio = chunks[0]
    out = os.path.join(OUT_DIR, f"val_{i}.wav")
    sf.write(out, audio, sr)
    print(f"saved {out} ({len(audio)/sr:.1f}s)")

print("ALL DONE")
