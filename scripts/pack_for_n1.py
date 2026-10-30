# -*- coding: utf-8 -*-
"""为 n1 训练打包：把清单中引用的 wav 复制成 GPT-SoVITS 期望的平铺结构并打包。

输出 dist/:
  mar7th_train30.tar.zst? -> mar7th_train30.tar.gz  (wavs/ + train.list)
嵌套关系保证 train120 包含 train30 全部条目；上传 n1 后各清单独立解包引用同一 wavs。
实际打包两个 tar: train30 自己的条目 + train120 相对 train30 的增量，服务器侧合并。
"""
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LISTS = os.path.join(HERE, "select", "lists")
DIST = os.path.join(HERE, "dist")


def read_list(name):
    rows = []
    with open(os.path.join(LISTS, name), encoding="utf-8") as f:
        for line in f:
            p, spk, lang, text = line.rstrip("\n").split("|", 3)
            rows.append((p, spk, lang, text))
    return rows


def main():
    os.makedirs(DIST, exist_ok=True)
    t30 = read_list("train30.list")
    t120 = read_list("train120.list")
    t30set = {p for p, *_ in t30}
    incr = [r for r in t120 if r[0] not in t30set]

    # train120 包 = t30 子集目录 + 增量；直接打全量 train120 更简单可靠
    for name, rows in (("train30", t30), ("train120", t120)):
        out = os.path.join(DIST, f"mar7th_{name}.tar.gz")
        with tarfile.open(out, "w:gz") as tar:
            manifest = []
            for p, spk, lang, text in rows:
                src = os.path.join(HERE, p.replace("/", os.sep))
                if not os.path.exists(src):
                    print("MISSING:", p)
                    continue
                arc = "wavs/" + os.path.basename(p)
                tar.add(src, arcname=arc)
                manifest.append(f"{arc}|{spk}|{lang}|{text}")
            mpath = os.path.join(DIST, f"{name}.list")
            with open(mpath, "w", encoding="utf-8") as f:
                f.write("\n".join(manifest) + "\n")
            tar.add(mpath, arcname=f"{name}.list")
        size = os.path.getsize(out) / 1e6
        print(f"{out}: {len(rows)} 条, {size:.1f} MB")


if __name__ == "__main__":
    main()
