"""venv 级补丁：torchaudio 2.11 的 load 依赖 torchcodec（本环境无匹配 CUDA 轮子）。

对本地 wav 文件改走 soundfile 解码（GPT-SoVITS 预处理只读标准 PCM wav，
5-wav32k 目录产物为 32kHz/16bit PCM，soundfile 完全覆盖）。
不装到 torchcodec 上，import torchcodec 本身仍保持可用性原样。
"""
import os

import torch


def _patched_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                  channels_first=True, format=None, buffer_size=4096,
                  backend=None):
    import soundfile as sf

    if isinstance(uri, (str, os.PathLike)):
        data, sr = sf.read(os.fspath(uri), dtype="float32",
                           start=frame_offset if frame_offset > 0 else 0,
                           stop=frame_offset + num_frames if num_frames > 0 else -1,
                           always_2d=True)
        t = torch.from_numpy(data.T)  # (channel, time)
        return (t if channels_first else t.t()), sr
    # 非路径输入（文件对象等）回退原始实现
    import torchaudio
    return torchaudio.load.__wrapped__(uri, frame_offset=frame_offset,
                                       num_frames=num_frames,
                                       normalize=normalize,
                                       channels_first=channels_first,
                                       format=format, buffer_size=buffer_size,
                                       backend=backend)


def _apply():
    try:
        import torchaudio
    except Exception:
        return
    if getattr(torchaudio.load, "_sf_patched", False):
        return
    import functools

    torchaudio.load.__wrapped__ = torchaudio.load
    torchaudio.load = functools.wraps(torchaudio.load)(_patched_load)
    torchaudio.load._sf_patched = True


_apply()
