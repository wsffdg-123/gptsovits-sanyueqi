# -*- coding: utf-8 -*-
"""train120 正式轮推理试听：t120 权重合成 val 清单文本（固定种子抽样）。

在 n1 运行: python ~/tts-sovits/infer_t120.py [--ckpt e15] [--n 6] [--seed 42] [--rows 1,5,9]
默认对比 e15 与 e10 两个 GPT ckpt，同一批固定种子抽样句。
输出: ~/tts-sovits/outputs/t120/{tag}/*.wav
"""
import argparse
import os
import random
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
# 2026-10-31 盲听定稿：refC（嘻嘻本姑娘句，F0≈291Hz 日常平静带笑音）+ hot 参数
# 最接近原音色与元气语气；cool（top_p0.6/temp0.6）音色接近但丢元气感，仅作备选。
# 参考选择策略：日常平静档、4.5-9s、quality>=1.0、F0 295-315Hz、非 _m/_f。
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
REF_TEXT = "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！"  # 盲听定稿 refC+hot
OUT_BASE = os.path.expanduser("~/tts-sovits/outputs/t120")

VAL_LIST = os.path.expanduser("~/tts-sovits/data/sovis_lists/val.list")
TIER_MAP = {"日常平静": 0, "开心活泼": 1, "疑问惊讶": 2, "温柔轻声": 3, "生气紧张": 4}


def load_val():
    rows = []
    with open(VAL_LIST, encoding="utf-8") as f:
        for line in f:
            p, spk, lang, text = line.rstrip("\n").split("|", 3)
            rows.append({"path": p, "text": text})
    tier_csv = os.path.expanduser("~/tts-sovits/val_selection.csv")
    if os.path.exists(tier_csv):
        import pandas as pd
        df = pd.read_csv(tier_csv, encoding="utf-8-sig")
        tier_by_path = dict(zip(df["path"], df["tier"]))
        for r in rows:
            r["tier"] = tier_by_path.get(r["path"], "?")
    else:
        for r in rows:
            r["tier"] = "?"
    return rows


def pick(rows, n, seed):
    rng = random.Random(seed)
    by_tier = {}
    for r in rows:
        by_tier.setdefault(r["tier"], []).append(r)
    tiers = sorted(by_tier, key=lambda t: TIER_MAP.get(t, 9))
    picked = []
    i = 0
    while len(picked) < n:
        t = tiers[i % len(tiers)]
        pool = [r for r in by_tier[t] if r not in picked]
        if pool:
            picked.append(rng.choice(pool))
        i += 1
        if i > n * len(tiers) * 2:
            break
    return picked


def synth(gpt_path: str, out_dir: str, picked):
    os.makedirs(out_dir, exist_ok=True)
    cfg = TTS_Config({
        "v2ProPlus": {
            "device": "cuda", "is_half": True, "version": "v2ProPlus",
            "t2s_weights_path": gpt_path, "vits_weights_path": SOVITS,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "custom": {
            "device": "cuda", "is_half": True, "version": "v2ProPlus",
            "t2s_weights_path": gpt_path, "vits_weights_path": SOVITS,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
    })
    tts = TTS(cfg)
    for k, r in enumerate(picked, 1):
        chunks = list(tts.run({
            "text": r["text"], "text_lang": "zh",
            "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
            "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
        }))
        if not chunks or chunks[0] is None:
            print(f"{out_dir}/val_{k}: no audio")
            continue
        sr, audio = chunks[0]
        out = os.path.join(out_dir, f"val_{k}.wav")
        sf.write(out, audio, sr)
        print(f"saved {out} ({len(audio)/sr:.1f}s) [{r['tier']}] {r['text'][:30]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", type=str, default="e15,e10", help="对比的 GPT ckpt 标签")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=str, default="")
    args = ap.parse_args()

    rows = load_val()
    if args.rows:
        picked = [rows[int(x) - 1] for x in args.rows.split(",")]
    else:
        picked = pick(rows, args.n, args.seed)
    print(f"抽样 {len(picked)} 条 (seed={args.seed}):")
    for k, r in enumerate(picked, 1):
        print(f"  val_{k} [{r['tier']}] {r['text'][:40]}")

    for tag in args.ckpts.split(","):
        tag = tag.strip()
        print(f"== GPT {tag} ==")
        synth(os.path.join(GPT_DIR, f"mar7th_t120_v2pp-{tag}.ckpt"),
              os.path.join(OUT_BASE, tag), picked)
    print("ALL DONE")


if __name__ == "__main__":
    main()
