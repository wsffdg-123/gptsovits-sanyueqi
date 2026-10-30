# -*- coding: utf-8 -*-
"""抽样各分组文本，判断战斗语音占比与可疑命名文件。"""
import collections
import glob
import os
import random
import wave

random.seed(42)
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "三月七")

BATTLE_HINTS = (
    "上吧", "进攻", "攻击", "接招", "看招", "出手", "讨伐", "开战", "迎战",
    "撤退", "小心", "别小看", "吃我一", "尝尝", "结束战斗", "胜利", "赢了",
    "输了", "可恶", "疼疼疼", "呜哇", "援护", "掩护我", "必杀", "终结",
)


def main():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "**", "*.wav"), recursive=True):
        lab = path[:-4] + ".lab"
        text = open(lab, encoding="utf-8").read().strip()
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / w.getframerate()
        rows.append((os.path.relpath(path, ROOT).replace(os.sep, "/"), dur, text))

    # 1) 抽样各组文本
    groups = collections.defaultdict(list)
    for rel, dur, text in rows:
        stem = os.path.basename(rel).rsplit(".", 1)[0]
        # 取文件名第一段作为来源组：Side0_al / archive / chapter1_1 / side2_lc1 ...
        parts = stem.split("_")
        # chapterN_M / sideN_x / SideN_x / archive / 其它|带变量 目录
        if parts[0].lower().startswith("side"):
            g = "side*" if parts[1] not in ("al",) else "Side_al"
        elif parts[0].startswith("chapter"):
            g = parts[0].split("mar7th")[0] + "*" if "mar7th" in stem else parts[0]
            g = "_".join(parts[:2]) if parts[0] != "chapter" else "chapter*"
        else:
            g = parts[0]
        groups[g].append((rel, dur, text))

    print("== 各组抽样 (n=minutes | 5 random texts)")
    for g in sorted(groups):
        items = groups[g]
        m = sum(d for _, d, _ in items) / 60
        print("\n[%s]  n=%d  %.1f min" % (g, len(items), m))
        for rel, dur, text in random.sample(items, min(4, len(items))):
            print("   %5.1fs %s" % (dur, text[:60]))

    # 2) 战斗语音扫描
    print("\n== 战斗提示词扫描")
    hits = [(rel, dur, text) for rel, dur, text in rows if any(h in text for h in BATTLE_HINTS)]
    print("hits:", len(hits), "total %.1f min" % (sum(d for _, d, _ in hits) / 60))
    for rel, dur, text in random.sample(hits, min(10, len(hits))):
        print("   %5.1fs %s | %s" % (dur, rel.split("/")[-1], text[:40]))

    # 3) 可疑命名
    print("\n== 可疑命名文件")
    sus = [r for r in rows if any(k in r[0] for k in ("fake", "jia", "sanyueqi"))]
    for rel, dur, text in sus:
        print("   %5.1fs %s | %s" % (dur, rel, text[:40]))

    # 4) Side0_al 与 archive 全文抽样
    print("\n== Side0_al 文本前 8 条")
    for rel, dur, text in sorted([r for r in rows if r[0].startswith("Side0_al")], key=lambda x: x[0])[:8]:
        print("   %5.1fs %s" % (dur, text[:60]))


if __name__ == "__main__":
    main()
