# -*- coding: utf-8 -*-
"""核查生成清单：嵌套性、验证集泄漏、格式、仿冒混入。

所有约束违规时以非零退出（供 pack_for_n1.py 打包前调用）。
用法: python scripts/verify_lists.py
"""
import collections
import os
import sys

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
    failures = []

    t30 = dict(load("train30.list"))
    t120 = dict(load("train120.list"))
    full = dict(load("full.list"))
    val = dict(load("val.list"))
    print("sizes:", len(t30), len(t120), len(full), len(val))

    if not set(t30) <= set(t120):
        failures.append(f"train30 ⊄ train120: {sorted(set(t30) - set(t120))[:5]}")
    print("train30 ⊆ train120:", set(t30) <= set(t120))
    if not set(t120) <= set(full):
        failures.append(f"train120 ⊄ full: {sorted(set(t120) - set(full))[:5]}")
    print("train120 ⊆ full:", set(t120) <= set(full))

    for name, d in (("train30", t30), ("train120", t120), ("full", full)):
        inter = set(val) & set(d)
        print(f"val ∩ {name}:", len(inter))
        if inter:
            failures.append(f"val ∩ {name} 文件级重叠 {len(inter)} 条: {sorted(inter)[:5]}")

    # 文本级泄漏
    v_texts = set(t for _, t in val.items())
    leak = [p for p, t in t120.items() if t in v_texts]
    print("val/train120 文本重复:", len(leak), leak[:5])
    if leak:
        failures.append(f"val/train120 文本级泄漏 {len(leak)} 条: {leak[:5]}")

    # 同文本多条 take 在 train120 内？
    dup = {t: c for t, c in collections.Counter(t120.values()).items() if c > 1}
    print("train120 内同文本条数:", len(dup), list(dup.items())[:5])
    if dup:
        failures.append(f"train120 内同文本重复 {len(dup)} 组: {list(dup.items())[:3]}")

    # mar7thf 与 fake 是否混入
    for name, d in (("train30", t30), ("train120", t120), ("full", full)):
        f1 = [p for p in d if "fake" in p.lower() or "jiasanyue" in p.lower() or "mar7thf" in p.lower()]
        print(name, "fake/mar7thf:", len(f1), f1[:3])
        if f1:
            failures.append(f"{name} 混入仿冒文件 {len(f1)} 条: {f1[:3]}")

    # 路径存在性 + 花括号残留
    missing = [p for p in full if not os.path.exists(os.path.join(HERE, p))]
    print("缺失路径:", len(missing), missing[:5])
    if missing:
        failures.append(f"full 中 {len(missing)} 条路径不存在: {missing[:5]}")
    nick = [(p, t) for p, t in full.items() if "{" in t or "}" in t]
    print("残留花括号:", len(nick), nick[:5])
    if nick:
        failures.append(f"full 中 {len(nick)} 条残留占位符: {nick[:3]}")

    # 空文本
    empty = [p for p, t in full.items() if len(t) < 2]
    print("短文本(<2字):", len(empty), empty[:5])
    if empty:
        failures.append(f"full 中 {len(empty)} 条短文本: {empty[:5]}")

    if failures:
        print("\n== 校验失败 ==")
        for f_ in failures:
            print("FAIL:", f_)
        sys.exit(1)
    print("\n全部约束通过")


if __name__ == "__main__":
    main()
