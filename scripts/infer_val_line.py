# -*- coding: utf-8 -*-
"""对 val 集有原声的样例做同文本推理，输出对照试听。

用法: python infer_val_line.py [行号1,行号2...]  （默认合成 select/val_bench.md 第 6 句）
输出: ~/tts-sovits/outputs/t120_e15_vs_orig/val_<n>.wav
"""
import os
import sys

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import pandas as pd
import soundfile as sf

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

HERE = os.path.expanduser("~/tts-sovits")
CSV = os.path.join(HERE, "val_selection.csv")  # 上传副本；不存在则提示
if not os.path.exists(CSV):
    raise SystemExit("需要把本机 select/selection.csv 上传为 ~/tts-sovits/val_selection.csv")

SOVITS = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth")
GPT_CKPT = os.path.expanduser("~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/Side0_al_mar7th_101.wav")
REF_TEXT = "哇，这么可爱的狗狗！…哇，你也来了。"
OUT_DIR = os.path.expanduser("~/tts-sovits/outputs/t120_e15_vs_orig")
os.makedirs(OUT_DIR, exist_ok=True)

ROWS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["6"])]

df = pd.read_csv(CSV, encoding="utf-8-sig")
df["sets"] = df["sets"].fillna("").astype(str).str.strip()
val = df[df["sets"].str.contains("val")].sort_values("path").reset_index(drop=True)

cfg = TTS_Config({
    "v2ProPlus": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT_CKPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "custom": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT_CKPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
})
tts = TTS(cfg)

for n in ROWS:
    r = val.iloc[n - 1]
    text = str(r["text"])
    orig = os.path.join(HERE, r["path"])
    chunks = list(tts.run({
        "text": text, "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
        "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
    }))
    if not chunks or chunks[0] is None:
        print(f"#{n}: no audio")
        continue
    sr, audio = chunks[0]
    out = os.path.join(OUT_DIR, f"val_{n}_synth.wav")
    sf.write(out, audio, sr)
    print(f"#{n} [{r['tier']}] {os.path.basename(r['path'])} 原声 {r['dur']:.1f}s")
    print(f"   文本: {text}")
    print(f"   合成: {out} ({len(audio)/sr:.1f}s)")
print("DONE")
