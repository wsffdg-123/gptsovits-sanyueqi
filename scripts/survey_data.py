# -*- coding: utf-8 -*-
"""勘察三月七语音数据：分组统计、文本质量、命名变体。"""
import collections
import glob
import os
import statistics
import wave

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "三月七")


def main():
    rows = []
    for path in glob.glob(os.path.join(ROOT, "**", "*.wav"), recursive=True):
        lab = path[:-4] + ".lab"
        text = open(lab, encoding="utf-8").read().strip() if os.path.exists(lab) else ""
        with wave.open(path, "rb") as w:
            dur = w.getnframes() / w.getframerate()
            params = (w.getnchannels(), w.getframerate(), w.getsampwidth())
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        top = rel.split("/")[0] if "/" in rel else "(root)"
        rows.append((rel, top, dur, text, params))

    params = collections.Counter(r[4] for r in rows)
    print("wav params (ch,rate,width):", dict(params))

    bytop = collections.defaultdict(lambda: [0, 0.0])
    for r in rows:
        bytop[r[1]][0] += 1
        bytop[r[1]][1] += r[2]
    print()
    print("%-28s %5s %8s" % ("group", "n", "minutes"))
    for g, (n, d) in sorted(bytop.items(), key=lambda x: -x[1][1]):
        print("%-28s %5d %8.1f" % (g, n, d / 60))

    print()
    sus = [r for r in rows if len(r[3]) <= 3]
    print("text<=3 chars:", len(sus), "total %.1f min" % (sum(r[2] for r in sus) / 60))
    dots = [r for r in rows if r[3] and set(r[3]) <= set(".…、，。！？!?,. ")]
    print("pure-punctuation texts:", len(dots), "total %.1f min" % (sum(r[2] for r in dots) / 60))
    for r in sorted(dots, key=lambda x: -x[2])[:5]:
        print("   %5.1fs %s  text=%r" % (r[2], r[0], r[3]))

    mf = [r for r in rows if r[0].split("/")[-1].rsplit(".", 1)[0].endswith(("_m", "_f"))]
    print()
    print("_m/_f variant files:", len(mf), "total %.1f min" % (sum(r[2] for r in mf) / 60))
    for r in mf[:8]:
        print("   %5.1fs %s  text=%r" % (r[2], r[0], r[3][:40]))

    battle_like = [r for r in rows if r[3] and len(r[3]) <= 12 and ("！" in r[3] or "!" in r[3])]
    print()
    print("short exclamatory lines (<=12 chars with !):", len(battle_like), "total %.1f min" % (sum(r[2] for r in battle_like) / 60))
    ex_groups = collections.Counter(r[1] for r in battle_like)
    print("their groups:", dict(ex_groups.most_common(10)))


if __name__ == "__main__":
    main()
