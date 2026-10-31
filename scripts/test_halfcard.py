# -*- coding: utf-8 -*-
"""半卡（12GB 总显存，可用约 10GB）推理测试。

测: 模型加载耗时、逐句合成延迟、显存峰值(torch 侧)、是否 OOM。
配置 = 定稿配置: GPT e15 + SoVITS e8_fixed + refC + top_k15/top_p1.0/temp1.0
在 n1 运行: python ~/tts-sovits/test_halfcard.py
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")  # cudnn v8 前端在变长 conv 上缓存失效，强制 legacy API
import os
import sys
import time

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import torch

print("== 半卡推理测试 ==")
print(f"GPU: {torch.cuda.get_device_name(0)}")
total = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"可见显存: {total:.1f} GiB")

SOVITS = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth")
GPT = os.path.expanduser("~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
REF_TEXT = "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！"

TEXTS = [
    ("baoxiao", "呃…能让布洛妮娅帮咱们报销吗？"),
    ("val_2", "嘿嘿，乔瓦尼先生说得对。决胜乐园的徽章已经都集齐了…接下来就只等大赛开幕咯！"),
    ("val_3", "没错，我就是那个三月七"),
    ("val_4", "只是个建议啦——没事，你有我呢！"),
    ("val_5", "开拓者，快看！围墙后边好像有座灯炉诶"),
    ("val_6", "岩明先生说了，接渡使返乡时会带礼物送给信赖的同僚…"),
]

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

cfg = TTS_Config({
    "v2ProPlus": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "custom": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
})

t0 = time.time()
tts = TTS(cfg)
load_s = time.time() - t0
torch.cuda.synchronize()
peak_alloc = torch.cuda.max_memory_allocated() / 2**20
peak_rsv = torch.cuda.max_memory_reserved() / 2**20
print(f"模型加载: {load_s:.1f}s | 显存峰值 alloc={peak_alloc:.0f}MB reserved={peak_rsv:.0f}MB")

lat = []
for k, (tid, text) in enumerate(TEXTS):
    t1 = time.time()
    chunks = list(tts.run({
        "text": text, "text_lang": "zh",
        "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
        "top_k": 15, "top_p": 1.0, "temperature": 1.0, "speed_factor": 1.0,
    }))
    dt = time.time() - t1
    if chunks and chunks[0] is not None:
        sr, audio = chunks[0]
        lat.append((tid, dt, len(audio) / sr))
        print(f"{tid}: 合成 {dt:.2f}s -> 音频 {len(audio)/sr:.1f}s ({len(audio)/sr/dt:.1f}x 实时)")
    else:
        print(f"{tid}: no audio ({dt:.2f}s)")
    torch.cuda.synchronize()

pa = torch.cuda.max_memory_allocated() / 2**20
pr = torch.cuda.max_memory_reserved() / 2**20
print(f"推理后显存峰值: alloc={pa:.0f}MB reserved={pr:.0f}MB (可见上限 {total*1024:.0f}MB)")
ok = [l for l in lat if l[2] / l[1] >= 1.0]
print(f"成功率: {len(lat)}/{len(TEXTS)} | 平均延迟 {sum(l[1] for l in lat)/len(lat):.2f}s | "
      f"实时率达标(≥1x): {len(ok)}/{len(lat)}")
print("HALFCARD TEST DONE")
