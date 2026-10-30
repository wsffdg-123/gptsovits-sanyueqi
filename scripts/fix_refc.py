# -*- coding: utf-8 -*-
"""refC 定妆验证：baoxiao 句在多组折中参数 × 多 seed 下找稳定点。

在 n1 运行: python ~/tts-sovits/fix_refc.py
输出: ~/tts-sovits/outputs/fix_refc/<参数组>/baoxiao_seed<k>.wav + val_<n>.wav
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
GPT = os.path.expanduser("~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
REF_TEXT = "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！"
OUT = os.path.expanduser("~/tts-sovits/outputs/fix_refc")

BAOXIAO = "呃…能让布洛妮娅帮咱们报销吗？"
OTHERS = [
    ("val_2", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
    ("val_3", "没错，我就是那个三月七"),
    ("val_5", "开拓者，快看！围墙后边好像有座灯炉诶"),
]

PARAMS = {
    "hot": {"top_k": 15, "top_p": 1.0, "temperature": 1.0},
    "mid": {"top_k": 15, "top_p": 0.8, "temperature": 0.8},
    "coolish": {"top_k": 15, "top_p": 0.9, "temperature": 0.7},
}

cfg = TTS_Config({
    "v2ProPlus": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "custom": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
})
tts = TTS(cfg)


def gen(text, seed):
    # torch 采样种子按进程注入；同进程内逐句顺序不可复现，用重试采样不同结果
    chunks = list(tts.run({
        "text": text, "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
        "speed_factor": 1.0, **PARAMS,
    }))
    return chunks[0] if chunks and chunks[0] is not None else None


for ptag in PARAMS:
    outdir = os.path.join(OUT, ptag)
    os.makedirs(outdir, exist_ok=True)
    # baoxiao × 3 次重试（观察崩坏率）
    ok = 0
    for k in range(1, 4):
        r = gen(BAOXIAO, k)
        if r is None:
            continue
        sr, audio = r
        out = os.path.join(outdir, f"baoxiao_try{k}.wav")
        sf.write(out, audio, sr)
        ok += 1
        print(f"{ptag} baoxiao_try{k}: {len(audio)/sr:.1f}s")
    # 其余句子各 1 次
    for tid, text in OTHERS:
        r = gen(text, 0)
        if r is None:
            print(f"{ptag} {tid}: no audio")
            continue
        sr, audio = r
        out = os.path.join(outdir, f"{tid}.wav")
        sf.write(out, audio, sr)
        print(f"{ptag} {tid}: {len(audio)/sr:.1f}s")
print("ALL DONE")
