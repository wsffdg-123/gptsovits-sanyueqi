# -*- coding: utf-8 -*-
"""三月七 TTS HTTP API —— 在 GPT-SoVITS TTS_infer_pack 之上的封装。

定稿配置全部固化在服务端（2026-10-31 盲听结论）：
  权重   = t120 GPT e15 + SoVITS e8_fixed
  参考音频 = refC chapter1_2_mar7th_236.wav（含提示文本）
  采样参数 = hot: top_k 15 / top_p 1.0 / temperature 1.0
客户端只需传 text，其余参数可逐请求覆盖（top_k/top_p/temperature/speed_factor/seed）。

接口：
  GET  /tts?text=...            → audio/wav（失败返回 400/500 JSON {"message": ...}）
  POST /tts  {"text": "..."}    → audio/wav
  GET  /health                  → 权重/参考/默认参数自检

n1 启动: nohup bash ~/tts-sovits/run_api.sh > ~/tts-sovits/logs/api.log 2>&1 &
停止:   kill $(cat ~/tts-sovits/api.pid)   （pid 由 run_api.sh 写入）
"""
import os
os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")  # cudnn v8 前端在变长 conv 上重建图，强制 legacy API（35s/句 → 2.6s/句）
import io
import sys
import threading
import time

GPT_ROOT = os.path.expanduser("~/tts-sovits/GPT-SoVITS")
sys.path.insert(0, os.path.join(GPT_ROOT, "GPT_SoVITS"))
sys.path.insert(0, GPT_ROOT)
os.chdir(GPT_ROOT)

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from GPT_SoVITS.TTS_infer_pack.TTS import TTS_Config, TTS

SOVITS = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth")
GPT_CKPT = os.path.expanduser(
    "~/tts-sovits/GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e15.ckpt")
REF = os.path.expanduser(
    "~/tts-sovits/data/mar7th_train30/wavs/chapter1_2_mar7th_236.wav")
REF_TEXT = "嘻嘻，明察秋毫，心细如发——说的就是本姑娘我啦！"
DEFAULTS = {"top_k": 15, "top_p": 1.0, "temperature": 1.0}  # hot
MAX_TEXT_CHARS = 1000

cfg = TTS_Config({
    "v2ProPlus": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT_CKPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
    "custom": {
        "device": "cuda", "is_half": True, "version": "v2ProPlus",
        "t2s_weights_path": GPT_CKPT, "vits_weights_path": SOVITS,
        "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    },
})

print(cfg, flush=True)
tts = TTS(cfg)
_infer_lock = threading.Lock()  # 单卡串行：一次只跑一条合成


def synth(text: str, overrides: dict) -> bytes:
    params = dict(DEFAULTS)
    params.update({k: v for k, v in overrides.items() if v is not None})
    seed = params.pop("seed", None)
    if seed is not None and int(seed) >= 0:
        torch.manual_seed(int(seed))  # 同 api_v2：请求级种子
    with _infer_lock:
        chunks = list(tts.run({
            "text": text, "text_lang": "zh",
            "ref_audio_path": REF, "prompt_text": REF_TEXT, "prompt_lang": "zh",
            "speed_factor": float(params.get("speed_factor") or 1.0),
            "top_k": int(params["top_k"]), "top_p": float(params["top_p"]),
            "temperature": float(params["temperature"]),
        }))
    if not chunks or chunks[0] is None:
        raise RuntimeError("合成失败：未返回音频（可能采样崩坏，重试一次或调低 top_p）")
    sr, audio = chunks[0]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


app = FastAPI(title="三月七 TTS (GPT-SoVITS v2ProPlus)")


def parse_overrides(req: dict) -> dict:
    over = {}
    for k in ("top_k", "top_p", "temperature", "speed_factor", "seed"):
        v = req.get(k)
        if v in (None, ""):
            continue
        try:
            over[k] = int(v) if k in ("top_k", "seed") else float(v)
        except (TypeError, ValueError):
            raise ValueError(f"参数 {k}={v} 不是数字")
    sf_ = over.get("speed_factor")
    if sf_ is not None and not (0.5 <= sf_ <= 3.0):
        raise ValueError("speed_factor 须在 0.5–3.0")
    return over


async def _serve(req: dict):
    text = (req.get("text") or "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"message": "text 不能为空"})
    if len(text) > MAX_TEXT_CHARS:
        return JSONResponse(status_code=400, content={"message": f"text 超过 {MAX_TEXT_CHARS} 字，请分段"})
    try:
        over = parse_overrides(req)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"message": str(e)})
    t0 = time.time()
    try:
        wav = await run_in_threadpool(synth, text, over)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"{type(e).__name__}: {e}"})
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Synth-Seconds": f"{time.time() - t0:.2f}"})


@app.get("/tts")
async def tts_get(request: Request):
    return await _serve(dict(request.query_params))


@app.post("/tts")
async def tts_post(request: Request):
    try:
        req = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"message": "请求体不是合法 JSON"})
    if not isinstance(req, dict):
        return JSONResponse(status_code=400, content={"message": "请求体必须是 JSON 对象"})
    return await _serve(req)


@app.get("/health")
async def health():
    return {
        "status": "ok", "version": "v2ProPlus",
        "sovits": os.path.basename(SOVITS), "gpt": os.path.basename(GPT_CKPT),
        "ref": os.path.basename(REF), "defaults": DEFAULTS,
    }


def warmup():
    """端口开放前跑 2 句预热：首句含 cudnn 各形状算法搜索（~5s），之后稳态 1.4-2.2s/句。"""
    for i, text in enumerate(["你好呀。", "嘻嘻，说的就是本姑娘我啦！"], 1):
        t0 = time.time()
        try:
            synth(text, {})
            print(f"[warmup {i}/2] {time.time() - t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[warmup {i}/2] 失败: {e}", flush=True)


if __name__ == "__main__":
    warmup()
    uvicorn.run(app, host="0.0.0.0", port=9880, log_level="info")
