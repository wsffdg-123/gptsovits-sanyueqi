# -*- coding: utf-8 -*-
"""三月七 TTS API 冒烟测试/预热：向已启动的服务发固定句子，输出延迟与音频时长。

用法（n1 上）:       python ~/tts-sovits/api_smoke.py
用法（Windows 直连）: python scripts/api_smoke.py --host 10.160.17.102
     若 9880 不可达，先开隧道: ssh -L 9880:127.0.0.1:9880 n1，再 --host 127.0.0.1
可选 --save DIR 保存 wav 试听，--repeat N 观察采样抖动。
退出码: 0 = 全部通过，1 = 有失败。
"""
import argparse
import io
import json
import sys
import time
import pathlib
import urllib.error
import urllib.parse
import urllib.request
import wave

SENTS = [
    ("短句", "没错，我就是那个三月七"),
    ("中句", "开拓者，快看！围墙后边好像有座灯炉诶"),
    ("长句", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--save", default="")
    ap.add_argument("--repeat", type=int, default=1, help="每组句子重复次数")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            print(f"[health] {json.loads(r.read().decode('utf-8'))}")
    except Exception as e:
        print(f"[health] 服务未就绪: {e}")
        return 1

    savedir = pathlib.Path(args.save) if args.save else None
    if savedir:
        savedir.mkdir(parents=True, exist_ok=True)

    fails = 0
    for k in range(max(1, args.repeat)):
        for tag, text in SENTS:
            url = f"{base}/tts?" + urllib.parse.urlencode({"text": text})
            t0 = time.time()
            try:
                with urllib.request.urlopen(url, timeout=120) as r:
                    ctype = r.headers.get("Content-Type", "")
                    synth_s = r.headers.get("X-Synth-Seconds", "?")
                    data = r.read()
            except urllib.error.HTTPError as e:
                print(f"[失败] {tag}: HTTP {e.code} {e.read().decode('utf-8', 'replace')}")
                fails += 1
                continue
            except Exception as e:
                print(f"[失败] {tag}: {e}")
                fails += 1
                continue
            dt = time.time() - t0
            if "json" in ctype:
                print(f"[失败] {tag}: 服务返回错误 {data.decode('utf-8', 'replace')}")
                fails += 1
                continue
            dur = 0.0
            try:
                with wave.open(io.BytesIO(data)) as w:
                    dur = w.getnframes() / w.getframerate()
            except Exception:
                pass
            name = tag if max(1, args.repeat) == 1 else f"{tag}_r{k + 1}"
            print(f"[通过] {name}: 合成 {dt:.2f}s (服务端 {synth_s}s) → 音频 {dur:.1f}s "
                  f"({dur / dt:.1f}x 实时, {len(data) / 1024:.0f}KB)")
            if savedir:
                (savedir / f"{name}.wav").write_bytes(data)
    print("SMOKE", "PASS" if fails == 0 else f"FAIL({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
