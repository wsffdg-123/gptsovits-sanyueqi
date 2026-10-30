# -*- coding: utf-8 -*-
"""为 n1 训练打包：把清单中引用的 wav 复制成 GPT-SoVITS 期望的平铺结构并打包。

打包前自动运行 scripts/verify_lists.py（嵌套/泄漏校验，非零退出即中止）。
缺失 wav 视为错误立即失败（不静默跳过）；报告每个包实际写入条数。

输出 dist/:
  mar7th_train30.tar.gz / mar7th_train120.tar.gz / mar7th_val.tar.gz
  每包含 wavs/ + <name>.list（仅含实际打包成功的条目）
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


def pack(name, rows):
    out = os.path.join(DIST, f"mar7th_{name}.tar.gz")
    packed = []
    with tarfile.open(out, "w:gz") as tar:
        for p, spk, lang, text in rows:
            src = os.path.join(HERE, p.replace("/", os.sep))
            if not os.path.isfile(src):
                raise FileNotFoundError(f"{name}: 清单文件缺失 on disk: {p}")
            arc = "wavs/" + os.path.basename(p)
            tar.add(src, arcname=arc)
            packed.append(f"{arc}|{spk}|{lang}|{text}")
        mpath = os.path.join(DIST, f"{name}.list")
        with open(mpath, "w", encoding="utf-8") as f:
            f.write("\n".join(packed) + "\n")
        tar.add(mpath, arcname=f"{name}.list")
    print(f"{out}: 实际打包 {len(packed)}/{len(rows)} 条, {os.path.getsize(out)/1e6:.1f} MB")
    if len(packed) != len(rows):
        raise RuntimeError(f"{name}: 打包条数 {len(packed)} != 清单 {len(rows)}")


def main():
    # 打包前强制校验清单约束（嵌套/泄漏），非零退出即中止
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "verify_lists.py")])
    if r.returncode != 0:
        raise SystemExit(f"verify_lists.py 校验失败 (exit {r.returncode})，中止打包")

    os.makedirs(DIST, exist_ok=True)
    pack("train30", read_list("train30.list"))
    pack("train120", read_list("train120.list"))
    pack("val", read_list("val.list"))


if __name__ == "__main__":
    main()
