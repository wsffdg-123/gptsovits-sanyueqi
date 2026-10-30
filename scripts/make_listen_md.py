# -*- coding: utf-8 -*-
"""生成音色偏差对照试听清单（基于 ab/manifest.tsv 与客观 F0 指标）。"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "outputs", "ab")

F0 = {
    "ckpt/base_pretrained_p_cool": 296, "ckpt/base_pretrained_p_hot": 296,
    "ckpt/gpt_e10_sovits_e4_p_cool": 291, "ckpt/gpt_e10_sovits_e4_p_hot": 291,
    "ckpt/gpt_e10_sovits_e8_p_cool": 294, "ckpt/gpt_e10_sovits_e8_p_hot": 294,
    "ckpt/gpt_e15_sovits_e4_p_cool": 291, "ckpt/gpt_e15_sovits_e4_p_hot": 299,
    "ckpt/gpt_e15_sovits_e8_p_cool": 295, "ckpt/gpt_e15_sovits_e8_p_hot": 294,
    "ref/refA_dog_p_cool": 395, "ref/refA_dog_p_hot": 405,
    "ref/refB_gene_p_cool": 292, "ref/refB_gene_p_hot": 296,
    "ref/refC_xixi_p_cool": 314, "ref/refC_xixi_p_hot": 320,
}
NOTE = {
    "ref/refA_dog_p_hot": "★ 当前基线 B0（音色偏高问题组）",
    "ref/refA_dog_p_cool": "★ B0 仅降温参数",
    "ref/refB_gene_p_hot": "★ 换参考B（hot 参数）",
    "ref/refB_gene_p_cool": "★ 换参考B（cool 参数）",
    "ref/refC_xixi_p_hot": "★ 换参考C（hot 参数）",
    "ref/refC_xixi_p_cool": "★ 换参考C（cool 参数）",
    "ckpt/gpt_e15_sovits_e8_p_hot": "ckpt 对照主组（参考B）",
    "ckpt/base_pretrained_p_hot": "预训练底模基线（未微调）",
}

GROUPS = [
    ("## 第一步：听参考对照（同权重 e15+e8，只有参考不同）", "ref",
     "定位问题最快路径：若 refB/refC 明显比 refA 像原声 → 参考音频选择问题，换参考即可，无需重训。"),
    ("## 第二步：听参数对照（同参考同权重，hot vs cool）", None,
     "cool（top_p0.6/temp0.6）若更稳 → 推理参数调整。注意 cool 也应保持声线不变。"),
    ("## 第三步：听 checkpoint 对照（固定参考B）", "ckpt",
     "2×2+底模：若四组都像、且与底模接近 → 微调没有带来负面变化，问题只在参考选择；"
     "若 e4 明显优于 e8 → SoVITS 后期过拟合；若全部不像底模 → 微调数据或流程问题。"),
]

rows_by_group = {}
with open(os.path.join(OUT, "manifest.tsv"), encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 6:
            continue
        group_dir, idx, tier, params, text, wav = parts
        # n1 绝对路径 → 本地相对路径（…/outputs/ab/<sub>/<group>）
        n1_prefix = "/home/u2023211867/tts-sovits/outputs/ab/"
        rel = group_dir.replace(n1_prefix, "").replace(os.sep, "/")
        wav_local = wav.replace(n1_prefix, "").replace(os.sep, "/")
        rows_by_group.setdefault(rel, []).append((idx, tier, text, wav_local))

lines = ["# 音色偏差定位对照试听", ""]
lines.append("原声基准：报销句（val 第6条）F0≈314Hz / 3.1s；训练数据整体 F0 中位 ≈ 306-327Hz。")
lines.append("参考音频 F0：A狗狗=381Hz，B遗传基因=302Hz，C嘻嘻=291Hz。")
lines.append("合成组 F0 中位数见各表。**F0 是排查线索，最终以盲听相似度为准。**\n")

for title, filt, guide in GROUPS:
    lines.append(title + "\n")
    lines.append(guide + "\n")
    for rel in sorted(rows_by_group):
        if filt and not rel.startswith(filt):
            continue
        if not filt and rel.startswith("ckpt"):
            continue
        f0 = F0.get(rel, "?")
        note = NOTE.get(rel, "")
        lines.append(f"### {rel}  (组F0≈{f0}Hz) {('—— ' + note) if note else ''}\n")
        for idx, tier, text, wav_local in sorted(rows_by_group[rel]):
            lines.append(f"- {idx} [{tier}] {text}  →  [{os.path.basename(wav_local)}]({wav_local})")
        lines.append("")

with open(os.path.join(OUT, "LISTEN.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", os.path.join(OUT, "LISTEN.md"))
