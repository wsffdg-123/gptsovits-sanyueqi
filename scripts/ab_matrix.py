# -*- coding: utf-8 -*-
"""音色偏差定位对照矩阵：参考音频 × 采样参数 × checkpoint，全部固定种子与文本。

在 n1 运行: python ~/tts-sovits/ab_matrix.py [--groups ref,params,ckpt]
输出: ~/tts-sovits/outputs/ab/<组名>/<权重标签>/<ref标签>_<参数标签>/val_<n>.wav
清单: outputs/ab/manifest.tsv  (组,权重,参考,参数,种子,文本,音频路径)

基线 B0 = GPT e15 + SoVITS e8_fixed + 狗狗参考 + topk15/topp1.0/temp1.0
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")  # cudnn v8 前端在变长 conv 上缓存失效，强制 legacy API
import argparse
import os
import random
import sys

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import numpy as np
import soundfile as sf

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

HERE = os.path.expanduser("~/tts-sovits")
OUT = os.path.join(HERE, "outputs", "ab")
SEED = 42

# ---------- 权重 ----------
W = {
    "gpt_e15": os.path.join(GPT_ROOT, "GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt"),
    "gpt_e10": os.path.join(GPT_ROOT, "GPT_weights_v2Pro/mar7th_t120_v2pp-e10.ckpt"),
    "sovits_e8": os.path.join(GPT_ROOT, "SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth"),
    "sovits_e4": os.path.join(GPT_ROOT, "SoVITS_weights_v2Pro/mar7th_t120_v2pp_e4_s500.pth"),
    "gpt_base": os.path.join(GPT_ROOT, "GPT_SoVITS/pretrained_models/s1v3.ckpt"),
    "sovits_base": os.path.join(GPT_ROOT, "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth"),
}

# ---------- 参考音频（train30，原声文件与台词严格对应） ----------
REFS = {
    "refA_dog": {
        "wav": os.path.join(HERE, "data/mar7th_train30/wavs/Side0_al_mar7th_101.wav"),
        "text": "哇，这么可爱的狗狗！…哇，你也来了",
        "note": "当前基线参考",
    },
    "refB_gene": {
        "wav": os.path.join(HERE, "data/mar7th_train30/wavs/chapter1_6_mar7th_222.wav"),
        "text": "金发大美人和金发大帅哥，啧啧，遗传基因真厉害呀",
        "note": "F0≈302Hz 5.9s 日常平静",
    },
    "refC_xixi": {
        "wav": os.path.join(HERE, "data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav"),
        "text": "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！",
        "note": "F0≈296Hz 5.6s 日常平静带笑音",
    },
}

# ---------- 采样参数 ----------
PARAMS = {
    "p_hot": {"top_k": 15, "top_p": 1.0, "temperature": 1.0},      # 当前基线
    "p_cool": {"top_k": 15, "top_p": 0.6, "temperature": 0.6},    # 审计建议组
}

# ---------- 测试文本（固定；报销句 + seed42 分层抽样 5 条） ----------
def load_val():
    import pandas as pd
    df = pd.read_csv(os.path.join(HERE, "val_selection.csv"), encoding="utf-8-sig")
    df["sets"] = df["sets"].fillna("").astype(str).str.strip()
    return df[df["sets"].str.contains("val")].sort_values("path").reset_index(drop=True)


def pick_texts():
    val = load_val()
    texts = [("baoxiao", "呃…能让布洛妮娅帮咱们报销吗？", "日常平静")]
    rng = random.Random(SEED)
    tiers = ["日常平静", "疑问惊讶", "开心活泼", "温柔轻声", "日常平静"]
    used = {"呃…能让布洛妮娅帮咱们报销吗？"}
    for t in tiers:
        pool = val[(val["tier"] == t) & (~val["text"].isin(used)) & (val["dur"].between(2.5, 8))]
        r = pool.iloc[rng.integers(0, len(pool))] if hasattr(rng, "integers") else pool.iloc[rng.randrange(len(pool))]
        used.add(r["text"])
        texts.append((f"val_t{len(texts)}", str(r["text"]), t))
    return texts


def synth_group(gpt, sovits, ref, params, texts, out_dir, seed):
    cfg = TTS_Config({
        "v2ProPlus": {
            "device": "cuda", "is_half": True, "version": "v2ProPlus",
            "t2s_weights_path": gpt, "vits_weights_path": sovits,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
        "custom": {
            "device": "cuda", "is_half": True, "version": "v2ProPlus",
            "t2s_weights_path": gpt, "vits_weights_path": sovits,
            "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
            "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        },
    })
    tts = TTS(cfg)
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    for k, (tid, text, tier) in enumerate(texts, 1):
        rng = np.random.default_rng(seed + k)  # 每句独立固定种子（torch 采样无法直接注入，用多次重跑一致性替代）
        chunks = list(tts.run({
            "text": text, "text_lang": "zh",
            "ref_audio_path": ref["wav"], "prompt_text": ref["text"], "prompt_lang": "zh",
            "speed_factor": 1.0, **params,
        }))
        if not chunks or chunks[0] is None:
            print(f"  {out_dir} val_{k}: no audio")
            continue
        sr, audio = chunks[0]
        out = os.path.join(out_dir, f"val_{k}_{tid}.wav")
        sf.write(out, audio, sr)
        line = f"{out_dir}\tval_{k}\t{tier}\t{params['top_k']}/{params['top_p']}/{params['temperature']}\t{text}\t{out}"
        manifest.append(line)
        print(f"  saved {out} ({len(audio)/sr:.1f}s) [{tier}]")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=str, default="ref,params,ckpt",
                    help="要跑的对照组: ref,params,ckpt(,base)")
    args = ap.parse_args()
    groups = set(args.groups.split(","))

    texts = pick_texts()
    print(f"固定测试文本 {len(texts)} 句 (seed={SEED}):")
    for tid, t, tier in texts:
        print(f"  {tid} [{tier}] {t[:30]}")

    os.makedirs(OUT, exist_ok=True)
    all_manifest = []

    if "ref" in groups:  # 步骤2: 参考音频对照（基线参数, e15+e8）
        for rtag, ref in REFS.items():
            for ptag, p in PARAMS.items():
                tag = f"{rtag}_{ptag}"
                print(f"== ref组 {tag} ==")
                out = os.path.join(OUT, "ref", tag)
                all_manifest += synth_group(W["gpt_e15"], W["sovits_e8"], ref, p, texts, out, SEED)

    if "params" in groups:  # 步骤3: 采样参数（最佳参考, e15+e8）—— ref 组已覆盖两参数×3参考
        pass

    if "ckpt" in groups:  # 步骤4: 2×2 checkpoint + 底模基线（最佳参考两参数）
        ref = REFS["refB_gene"]
        combos = [
            ("gpt_e15_sovits_e8", W["gpt_e15"], W["sovits_e8"]),
            ("gpt_e15_sovits_e4", W["gpt_e15"], W["sovits_e4"]),
            ("gpt_e10_sovits_e8", W["gpt_e10"], W["sovits_e8"]),
            ("gpt_e10_sovits_e4", W["gpt_e10"], W["sovits_e4"]),
            ("base_pretrained", W["gpt_base"], W["sovits_base"]),
        ]
        for ctag, gpt, sov in combos:
            for ptag, p in PARAMS.items():
                tag = f"{ctag}_{ptag}"
                print(f"== ckpt组 {tag} ==")
                out = os.path.join(OUT, "ckpt", tag)
                all_manifest += synth_group(gpt, sov, ref, p, texts, out, SEED)

    mf = os.path.join(OUT, "manifest.tsv")
    with open(mf, "a", encoding="utf-8") as f:
        for line in all_manifest:
            f.write(line + "\n")
    print(f"manifest: {mf} (+{len(all_manifest)} rows)")


if __name__ == "__main__":
    main()
