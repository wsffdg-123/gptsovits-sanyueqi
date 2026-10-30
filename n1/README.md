# n1 训练环境与产物档案

本目录保存三月七 TTS 在 n1 服务器（10.160.17.102，RTX 4090 D 24G）上训练环境的
关键档案，用于故障排查与完整复现。训练可执行代码、数据、权重均在 n1 的
`~/tts-sovits/` 下（与 `~/AIC` 完全隔离）。

## n1 目录布局

```
~/tts-sovits/
├── miniconda3/envs/sovits/    Python 3.12 + torch 2.11.0+cu128 + GPT-SoVITS 依赖
├── GPT-SoVITS/                代码（见下方版本）+ pretrained_models 4.6GB 底模
├── data/
│   ├── sovis_lists/           训练清单（绝对路径，prepare_lists.py 生成）
│   ├── mar7th_train30/        410 条 (wavs/ + train30.list)
│   ├── mar7th_train120/       1459 条 (wavs/ + train120.list)
│   └── mar7th_val/            110 条独立验证集
├── bin/                       静态 ffmpeg/ffprobe (BtbN)
├── cache/                     pip/HF 缓存（独立于 AIC）
├── logs/                      各阶段 nohup 日志（本地档案见 train_logs/）
└── env.sh                     环境变量（source 后使用）
```

## GPT-SoVITS 版本

- 仓库: https://github.com/RVC-Boss/GPT-SoVITS.git
- commit: `b4570b8` fix(streaming): 支持API v2语义流式文本切分 (#2850)（2026-10-30 main）
- 底模: modelscope `XXXXRT/GPT-SoVITS-Pretrained` 的 pretrained_models.zip（4.6GB，
  含 v2ProPlus s2G/s2D、s1v3.ckpt、chinese-hubert-base、chinese-roberta-wwm-ext-large、
  sv 模型）。重拉方法见官方 install.sh，用 modelscope 源。

## 环境搭建要点（重现时按序）

1. 系统 Python 无 venv/pip，先装 Miniconda（tuna 镜像）到 `~/tts-sovits/miniconda3`，
   `conda create -n sovits python=3.12`（conda 自带 Python.h；系统缺 python3-dev，
   C 扩展装不进系统 Python）
2. torch: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`
3. GPT-SoVITS requirements: tuna pypi 源（opencc/pyopenjtalk/jieba_fast 需本地编译）
4. ffmpeg: conda-forge 装 `ffmpeg=7`（共享库，供 torchcodec），另加静态
   ffmpeg/ffprobe 到 `bin/`（供 load_audio 子进程调用）
5. torchcodec: **pip 轮子与 torch 2.11+cu128 不兼容**（官方仅 cuda129/130，
   0.8.1 又是旧 torch ABI）。解决方案是 sitecustomize 补丁，见下
6. `source env.sh` 后 `python -c "import torch; print(torch.cuda.is_available())"` 应为 True

## torchcodec 兼容补丁

`sitecustomize/sitecustomize.py`（与 `scripts/sitecustomize.py` 一致）部署在
`~/tts-sovits/miniconda3/envs/sovits/lib/python3.12/site-packages/sitecustomize.py`。
原理：Python 启动时自动 import sitecustomize，把 `torchaudio.load` 换成 soundfile
解码实现（本地 PCM wav 场景完全覆盖），绕开 torchcodec。删除该文件即回退原状。
注意：若日后升级 torchaudio 大版本，需确认补丁逻辑仍适配（load 签名可能变化）。

## 数据来源与刷新

Windows 本机 `E:\App\QQbot\tts`（审查页面 http://127.0.0.1:6270/select/review.html
可修正文本/移出条目）→ `scripts/pack_for_n1.py` 打包 → scp 上传解压。
数据更新后重跑训练前先删 n1 上对应 `logs/mar7th_*` 目录（预处理产物有缓存跳过
逻辑，旧特征不会自动失效）。

## 训练流程（本地 scripts/ 与 n1 ~/tts-sovits/ 同名脚本一一对应）

```bash
# 0. 确认 GPU 空闲（有任务先协调，不要盲目启动）
nvidia-smi --query-compute-apps=pid --format=csv,noheader

# 1. 数据清单（上传数据后跑一次）
python ~/tts-sovits/prepare_lists.py

# 2. train30 验证轮（约 25 分钟端到端）
nohup bash ~/tts-sovits/run_preprocess_t30.sh > ~/tts-sovits/logs/preprocess_t30.log 2>&1 &
nohup bash ~/tts-sovits/run_sovits_t30.sh      > ~/tts-sovits/logs/sovits_t30.log 2>&1 &
nohup bash ~/tts-sovits/run_gpt_t30.sh         > ~/tts-sovits/logs/gpt_t30.log 2>&1 &

# 2'. train120 正式轮（预处理+训练一键串行，约 20 分钟）
nohup bash ~/tts-sovits/run_train_t120.sh > ~/tts-sovits/logs/train_t120.log 2>&1 &

# 3. 推理试听（改脚本内权重路径/句子）
python ~/tts-sovits/infer_t120.py
```

脚本内置的关键修正（排障时优先怀疑这几处）：
- 所有 prepare_datasets/s1_train/s2_train 需要
  `PYTHONPATH=GPT-SoVITS/GPT_SoVITS:GPT-SoVITS`（text 模块解析）
- 预处理后必须合并分片：`2-name2text.txt` 与带表头的 `6-name2semantic.tsv`
  （run_preprocess_*.sh 已内置）
- 权重导出目录 `SoVITS_weights_v2Pro/`、`GPT_weights_v2Pro/` 需预先 mkdir
  （savee 用相对路径，目录缺失会让训练进程崩溃）
- SoVITS 权重用于推理前，config 必须是嵌套 dict（run_train_t120.sh 的
  e8_fixed 导出步骤已处理；webui 正常保存无此问题）

## 实际训练配置存档

- `configs_t120/tmp_s2.json` — SoVITS 实际生效配置（batch 12 / 8 epoch / fp16）
- `configs_t120/tmp_s1.yaml` — GPT 实际生效配置（batch 12 / 15 epoch / save_every 5）
- `configs_t30/` 同理（batch 8/6，12/20 epoch，验证轮）
- `train_logs/` — 预处理与训练的 nohup 日志存档

## 产物清单（n1，2026-10-30）

| 文件 | 说明 |
|---|---|
| `GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e8_fixed.pth` | **主推理权重**（t120 SoVITS e8，dict config 修正版） |
| `GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t120_v2pp_e{4,8}_s{500,1000}.pth` | webui savee 原始导出（config 为 HParams，勿直接用于 TTS_infer_pack） |
| `GPT-SoVITS/SoVITS_weights_v2Pro/mar7th_t30_v2pp_e12_s648_fixed.pth` | t30 验证轮权重 |
| `GPT-SoVITS/GPT_weights_v2Pro/mar7th_t120_v2pp-e{5,10,15}.ckpt` | t120 GPT 三个轮数 |
| `GPT-SoVITS/GPT_weights_v2Pro/mar7th_t30_v2pp-e{5,10,15,20}.ckpt` | t30 GPT 四个轮数 |
| `GPT-SoVITS/logs/mar7th_t{30,120}_v2pp/logs_s2_v2ProPlus/{G,D}_*.pth` | SoVITS 完整训练态（含 optimizer，续训用） |
| `outputs/t30_e12/` `outputs/t120/{e10,e15}/` | 试听 wav（本地副本在 E:\App\QQbot\tts\outputs\） |

## 训练耗时参考（4090D）

| 阶段 | t30 (410条) | t120 (1459条) |
|---|---|---|
| 预处理 | 4 分钟 | 6 分钟 |
| SoVITS | 12ep / 5 分钟 | 8ep / 6 分钟 |
| GPT | 20ep / 13 分钟 | 15ep / 3.5 分钟 |
| 显存峰值 | 9.2G (s2) / 7.4G (s1) | 12.4G (s2) |
