# -*- coding: utf-8 -*-
"""生成 val 集对照表：Markdown 总表 + 试听对照说明。

输出 select/val_bench.md
"""
import os
import sys
import urllib.parse

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "select", "val_bench.md")

# 三月七审查页与本地服务端口
BASE_URL = "http://127.0.0.1:6270"

DEMO = {
    "t30_e12": {
        "weights": "SoVITS mar7th_t30_v2pp_e12_s648_fixed + GPT mar7th_t30_v2pp-e20",
        "sentences": [
            "今天也一起加油吧，有很多事情等着我们去做呢。",
            "欸？你说那个新来的转校生吗？我昨天在走廊碰到过一次。",
            "真的是服了，明明说好了要一起来的，结果他自己先跑掉了。",
        ],
        "local_dir": "outputs/t30_e12",
    },
    "t120_e15": {
        "weights": "SoVITS mar7th_t120_v2pp_e8_fixed + GPT mar7th_t120_v2pp-e15",
        "sentences": [
            "今天也一起加油吧，有很多事情等着我们去做呢。",
            "欸？你说那个新来的转校生吗？我昨天在走廊碰到过一次。",
            "真的是服了，明明说好了要一起来的，结果他自己先跑掉了。",
            "这一路上看到了好多没见过的风景，我都想全部拍下来了！",
        ],
        "local_dir": "outputs/t120/e15",
    },
    "t120_e10": {
        "weights": "SoVITS mar7th_t120_v2pp_e8_fixed + GPT mar7th_t120_v2pp-e10",
        "sentences": [
            "今天也一起加油吧，有很多事情等着我们去做呢。",
            "欸？你说那个新来的转校生吗？我昨天在走廊碰到过一次。",
            "真的是服了，明明说好了要一起来的，结果他自己先跑掉了。",
            "这一路上看到了好多没见过的风景，我都想全部拍下来了！",
        ],
        "local_dir": "outputs/t120/e10",
    },
}


def esc(s):
    return str(s).replace("|", "\\|")


def main():
    df = pd.read_csv(os.path.join(HERE, "select", "selection.csv"), encoding="utf-8-sig")
    df["sets"] = df["sets"].fillna("").astype(str).str.strip()
    val = df[df["sets"].str.contains("val")].sort_values("path").reset_index(drop=True)

    lines = []
    lines.append("# val 验证集对照表\n")
    lines.append(f"- 共 **{len(val)} 条 / {val['dur'].sum()/60:.1f} 分钟**，"
                 f"与所有训练集（train30/train120/full）文件级且文本级零重叠（scripts/verify_lists.py 程序化验证）")
    lines.append("- 用途：声线相似度与发音自然度评测。评测文本必须来自本清单——模型训练时从未见过")
    tier_cnt = val["tier"].value_counts()
    lines.append("- 情绪分层：" + "、".join(f"{k} {v}条" for k, v in tier_cnt.items()) + "\n")

    lines.append("## 明细（110 条）\n")
    lines.append("| # | 文件 | 时长s | 分层 | 台词 | 试听 |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in val.iterrows():
        wav_url = BASE_URL + "/" + "/".join(urllib.parse.quote(p) for p in r["path"].split("/"))
        lines.append(f"| {i+1} | {os.path.basename(r['path'])} | {r['dur']:.1f} | "
                     f"{esc(r['tier'])} | {esc(r['text'])} | [▶]({wav_url}) |")
    lines.append("")

    lines.append("## 试听文件对照（模型从未见过下列句子）\n")
    lines.append("推理脚本手写的 4 句测试文本**不在 val 清单内**，同样满足'训练未见'条件；"
                 "参考音频 `Side0_al_mar7th_101.wav`（哇，这么可爱的狗狗！…哇，你也来了）取自 train30，属于训练数据——"
                 "这是 v2ProPlus 参考音频机制的预期用法（推理时给出声线锚点），不代表模型记住了参考句。\n")
    lines.append("| 试听组 | 权重组合 | 句子数 | 本地文件 |")
    lines.append("|---|---|---|---|")
    for tag, info in DEMO.items():
        sents = "<br>".join(esc(f"{i+1}. {s}") for i, s in enumerate(info["sentences"]))
        files = "<br>".join(f"{i+1}.wav" for i in range(len(info["sentences"])))
        lines.append(f"| {tag} | {esc(info['weights'])} | {len(info['sentences'])} | "
                     f"`{info['local_dir']}/` {files} |\n")
    lines.append("**试听句子全文**\n")
    seen = []
    for info in DEMO.values():
        for s in info["sentences"]:
            if s not in seen:
                seen.append(s)
    for i, s in enumerate(seen, 1):
        lines.append(f"{i}. {s}")
    lines.append("")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"written {OUT}: {len(val)} rows")


if __name__ == "__main__":
    main()
