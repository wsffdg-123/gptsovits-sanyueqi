# -*- coding: utf-8 -*-
"""核查生成清单：嵌套性、验证集泄漏、格式、mar7thf 处理。"""
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
L = os.path.join(HERE, "select", "lists")


def load(name):
    rows = []
    with open(os.path.join(L, name), encoding="utf-8") as f:
        for line in f:
            p, spk, lang, text = line.rstrip("\n").split("|", 3)
            rows.append((p, text))
    return rows


def main():
    t30 = dict(load("train30.list"))
    t120 = dict(load("train120.list"))
    full = dict(load("full.list"))
    val = dict(load("val.list"))
    print("sizes:", len(t30), len(t120), len(full), len(val))

    print("train30 ⊆ train120:", set(t30) <= set(t120))
    print("train120 ⊆ full:", set(t120) <= set(full))
    print("val ∩ train120:", len(set(val) & set(t120)))
    print("val ∩ train30:", len(set(val) & set(t30)))
    print("val ∩ full:", len(set(val) & set(full)))

    # 文本级泄漏
    v_texts = set(t for _, t in val.items())
    leak = [p for p, t in t120.items() if t in v_texts]
    print("val/train120 文本重复:", len(leak), leak[:5])

    # 同文本多条 take 在 train120 内？
    import collections
    dup = {t: c for t, c in collections.Counter(t120.values()).items() if c > 1}
    print("train120 内同文本条数:", len(dup), list(dup.items())[:5])

    # mar7thf 与 fake 是否混入
    for name, d in (("train30", t30), ("train120", t120), ("full", full)):
        f1 = [p for p in d if "fake" in p.lower() or "jiasanyue" in p.lower()]
        f2 = [p for p in d if "mar7thf" in p.lower()]
        print(name, "fake:", len(f1), "mar7thf:", len(f2), f2[:3])

    # 路径存在性 + NICKNAME 残留
    import glob
    missing = [p for p in full if not os.path.exists(os.path.join(HERE, p))]
    print("缺失路径:", len(missing), missing[:5])
    nick = [(p, t) for p, t in full.items() if "{" in t or "}" in t]
    print("残留花括号:", len(nick), nick[:5])

    # 空文本
    empty = [p for p, t in full.items() if len(t) < 2]
    print("短文本(<2字):", len(empty), empty[:5])


if __name__ == "__main__":
    main()
