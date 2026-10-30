# -*- coding: utf-8 -*-
"""在 n1 上为 GPT-SoVITS 训练准备清单：改写路径、抽样验证集。

在 n1 运行:  python prepare_lists.py
输入: ~/tts-sovits/data/mar7th_train{30,120}/wavs + train{30,120}.list
输出: ~/tts-sovits/data/sovis_lists/{train30,train120,val}.list  (绝对路径指向解包 wavs)
"""
import os
import random

BASE = os.path.expanduser("~/tts-sovits/data")
OUT = os.path.join(BASE, "sovis_lists")


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(42)
    for name in ("train30", "train120"):
        src = os.path.join(BASE, f"mar7th_{name}", f"{name}.list")
        rows = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                arc, spk, lang, text = line.rstrip("\n").split("|", 3)
                wav = os.path.join(BASE, f"mar7th_{name}", arc)  # arc = wavs/xxx.wav
                rows.append(f"{wav}|{spk}|{lang}|{text}")
        with open(os.path.join(OUT, f"{name}.list"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print(name, len(rows))
    # val 随独立 val 清单而来；本机 val.list 已有，这里从 train120.list 引用的 wavs 池抽样没有意义，
    # 直接复用本地已上传的 val（如果有）
    val_src = os.path.join(BASE, "mar7th_val", "val.list")
    if os.path.exists(val_src):
        rows = []
        with open(val_src, encoding="utf-8") as f:
            for line in f:
                arc, spk, lang, text = line.rstrip("\n").split("|", 3)
                wav = os.path.join(BASE, "mar7th_val", arc)
                rows.append(f"{wav}|{spk}|{lang}|{text}")
        with open(os.path.join(OUT, "val.list"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")
        print("val", len(rows))
    else:
        print("val: 未上传（可选）")


if __name__ == "__main__":
    main()
