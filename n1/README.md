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

## 推理配置定稿（2026-10-31 盲听矩阵结论）

96 条对照矩阵（3 参考 × 2 参数 × 权重组合 + 底模基线，脚本 `ab_matrix.py`，
试听指南 `outputs/ab/LISTEN.md`）盲听结论：

- **参考音频是音色偏差的主导因素，不是训练问题**：参考A（狗狗句 Side0_al_mar7th_101，
  F0≈381Hz 偏高亢）导致合成整体 F0 抬到 400Hz；换参考后立即回到 292–320Hz（原声 314Hz）。
- **checkpoint 无退化**：GPT e10/e15 × SoVITS e4/e8 四组与预训练底模 F0 高度一致
  （291–299Hz），SoVITS 8 epoch 无过拟合，无需重训。
- **定稿配置**：参考C = `chapter1_2_mar7th_236.wav`（"嘻嘻，明察秋毫，心细如发——
  说的就是本姑娘我啦！"，5.6s，F0≈291Hz，日常平静带笑音）+
  `top_k=15 / top_p=1.0 / temperature=1.0`。已固化到 infer_t30/t120/val_line.py。
- 参考音频选择策略：日常平静档、4.5–9s、quality≥1.0、F0≈295–315Hz（贴近训练中位 306–327Hz
  的下半区）、台词与录音逐字一致。
- cool 参数（top_p0.6/temp0.6）整体略稳但对短语气句（"呃…"开头）有采样重复崩坏风险；
  low-top_p 下的崩坏与个别句子的人名语义（如"布洛妮娅"）处理欠佳属已知限制，暂不追究。

对照与验证脚本：`ab_matrix.py`（矩阵）、`fix_refc.py`（参数点复验）、
`infer_val_line.py`（同文本原声/合成对照）。

## 半卡推理与 cudnn v8 性能修复（2026-10-31）

n1 GPU 曾以半卡状态运行（可见显存 12032MiB，Virtualization Mode None，无 vGPU）。
测试结论（test_halfcard.py，定稿配置 refC+hot+e15/e8）：

- 显存峰值仅 **1.7GB**（torch alloc），半卡余量巨大；6/6 成功
- **性能修复**：初测 35s/句（0.1x 实时）。定位：GPU kernel 总计仅 0.75s，
  aten::cudnn_convolution 的 **CPU 侧调用耗时 40.8s**（378 次 × 108ms）——
  cudnn v8 前端在 ConvTranspose1d 变长形状下每次调用重建执行图（缓存永不命中）。
  解法：export TORCH_CUDNN_V8_API_DISABLED=1（已写入 env.sh 与全部推理脚本，
  torch 导入前 os.environ.setdefault 生效）
- 修复后半卡推理：**平均 2.6s/句，5/6 ≥1x 实时**，长句 7.1s 音频 2.2s 合成（3.3x 实时）；
  模型加载 3.7s
- **显存修正（2026-10-31 复测）**：稳态/短句 ~2.2GB；**长句推理峰值可达 torch alloc 6.4GB /
  reserved 7.7GB**（test_halfcard 6/6 实测计数器）——此前记录的"峰值 1.7GB"是模型加载后的
  数值而非推理峰值。半卡 12GB 下与他人共存（~2.2GB）仍安全；部署估算请按 8GB 起步
- **0.1 卡结论（test_memcap.py 两档实测）**：1203MB（半卡基数）模型加载即 OOM；
  2456MB（全卡基数）模型加载 OK（1.25GB）但推理 OOM（长句注意力大画布单笔 ~4.6GB 申请，
  expandable_segments 无效）。**0.1 卡不可用，最低可行档 ≈ 半卡 12GB**
- 部署建议：api_v2 常驻进程启动后先跑 2-3 句 warmup（首句含 cudnn 各形状
  算法搜索 ~5s），之后稳态 1.4-2.2s/句
- 另注意：SV 说话人向量（compute_embedding3）在每次 run() 都对参考音频重算；
  同一参考连续对话时上游可缓存（健康 GPU 上单次 ~0.3s，影响小）

## API 封装：mar7th_api.py（2026-10-31）

定稿配置全部固化在服务端，客户端只需传 `text`。没有用仓库自带 api_v2：该 commit
（b4570b8）的 api_v2 无默认参考音频回退（`ref_audio_path is required`），每次请求
都得带权重/参考 5 个字段，封装反而漏到客户端。故基于 `TTS_infer_pack.TTS` 直写薄服务。

| 项 | 值 |
|---|---|
| 权重 | t120 GPT e15 + SoVITS e8_fixed |
| 参考音频 | refC `chapter1_2_mar7th_236.wav`（含提示文本，服务端默认） |
| 默认采样 | hot：top_k 15 / top_p 1.0 / temperature 1.0（可逐请求覆盖） |
| 端口 | 0.0.0.0:9880（n1 主机防火墙未放行，外部须走 ssh 隧道） |
| 显存 | 常驻 ~2GB（半卡 12032MiB 环境实测） |

```bash
# 启动（模型加载 ~4s + 预热 2 句，~10s 后端口开放即已热）
nohup bash ~/tts-sovits/run_api.sh > ~/tts-sovits/logs/api.log 2>&1 &
# 冒烟/预热（n1 上；Windows 侧 python scripts/api_smoke.py --host 127.0.0.1 走隧道）
~/tts-sovits/miniconda3/envs/sovits/bin/python ~/tts-sovits/api_smoke.py --save ~/tts-sovits/outputs/api_smoke
# 停止
kill $(cat ~/tts-sovits/api.pid)
```

接口：
- `GET /tts?text=...` / `POST /tts {"text": ...}` → `audio/wav`；响应头
  `X-Synth-Seconds` 为服务端合成耗时。可选覆盖参数 `top_k/top_p/temperature/speed_factor(0.5-3)/seed`；
  失败返回 400/500 JSON `{"message": ...}`（空文本、超 1000 字、参数非法、采样崩坏无音频）
- `GET /health` → 权重/参考/默认参数自检

调用路径（2026-10-31 实测）：
- Windows → n1 直连 9880 被主机防火墙拒绝（10061），**须开隧道**：
  `ssh -N -L 9880:127.0.0.1:9880 n1`，然后请求 `http://127.0.0.1:9880`
- **bot 已接入（2026-10-31）**：隧道由 `ops/supervisor.ps1` 的 `tts-tunnel` 目标
  常驻管理（BatchMode/ExitOnForwardFailure，断线自动重启，面板可启停），
  核心插件 `send_message` 的 voice 项经 `voice_tts.py` 调用，无需手动开隧道
- 隧道实测（QQbot 调用路径）：短句 1.3s、中句 1.3s、长句 1.8s（6.9s 音频，3.8x 实时）
- 客户端注意：单并发（GPU 内部已串行，bot 一次只发一句）；HTTP 超时建议 ≥60s；
  Git Bash 的 curl 发中文会变 GBK，测试用 python urllib（api_smoke.py 即示例）

本地脚本 `scripts/{mar7th_api.py,run_api.sh,api_smoke.py}` 与 n1 `~/tts-sovits/` 同名对应。

**隧道自愈接线（2026-10-31）**：本地 `ops/supervisor.ps1` 的 tts-tunnel 目标去掉了 `-N`，
改为远端命令 `bash ~/tts-sovits/ensure_api.sh; exec sleep infinity`——隧道每次（重）连先幂等
拉起 n1 API（`ensure_api.sh`：端口在即秒过；否则 nohup 启动并等就绪，上限 120s），再保持隧道。
服务器重启/重分配后整条链自愈：supervisor 10s 轮询发现端口断 → 重连隧道 → 远端命令拉起 API，
实测恢复周期 ~60-75s（期间请求失败，bot 侧需容错重试）。

## v5 零样本测试（2026-10-31）

背景：v2ProPlus 定稿仍有吞字，尝试升级 v5。该 commit（b4570b8）原生支持
`v5dev` / `v5turbo`（`V5_VERSIONS = {"v5", "v5dev", "v5turbo"}`），t2s 不变仍用
s1v3.ckpt，仅 SoVITS + vocoder 换新。官方宣称零样本（不微调）音色相似度显著提升、
新 vocoder 消除高频镜像伪影。

- 底模下载：`dl_v5.sh`（hf-mirror，~1.6GB → `GPT_SoVITS/pretrained_models/gsv-v5-pretrained/`：
  s2Gv5dev.pth 766MB / s2Gv5turbo.pth 766MB / vocoder.pth 58MB；训练底模 s2D 另需确认来源）
- 零样本 = 原版 s1v3.ckpt + v5 底模，参考音频 refC，hot 采样参数；脚本 `v5_zeroshot.py`，
  日志 `train_logs/v5_zeroshot.log`，试听 `outputs/v5_zeroshot/LISTEN.md`
- 实测速度（全卡 4090D）：v5dev（32步/cfg1.30）加载 5.4s、稳态 2.0–4.0s/句（慢）；
  **v5turbo（4步/cfg0.0）加载 3.4s、稳态 1.6–2.2s/句，与 v2ProPlus 相当**
- 决策：盲听吞字/音色后定——v5turbo 零样本达标即免训练上线；差一口气再走 v5 微调

## v5 LoRA 微调（2026-10-31）

零样本之后接续微调。关键结论：

- **路线**：v3/v4/v5 走 `s2_train_v3_lora.py`（纯 CFM LoRA，无判别器，**不需要 s2D**）；
  LoRA rank 32（webui 默认档 16/32/64/128），底模 = gsv-v5-pretrained/s2G<variant>.pth
- **训练**：batch 12 OOM（DiT 注意力 20.9GB fp16）→ **batch 8 + grad_ckpt=True**（6.8GB，3.4it/s），
  1459 条每变体约 7 分钟；预处理产物直接复用 logs/mar7th_t120_v2pp（v5 用 LoaderV4 读同一套 2/4/5/6）
- **GPT 不重训**：t2s 各版本通用，复用 mar7th_t120_v2pp-e15.ckpt
- **导出**：savee 产物带版本标签头，重读必须 `load_sovits_new`（torch.load 直接读会
  unpickling stack underflow）；且 savee 存 HParams config 推理端不可下标访问，
  须重写为 dict（run_train_t120_v5.sh 的 fixed 导出步骤）
- **推理配置大坑（已排障）**：`TTS_Config(dict)` 的 `configs_.update(configs)` 会用传入的
  version 键**覆盖 default_configs[version]**；LoRA 推理端从这个结构解析底模
  （`path_sovits = default_configs[model_version]['vits_weights_path']`）——若把 version 键
  一并传入且 vits 指向 LoRA pth，底模就成了 LoRA 自己，SoVITS 实为随机初始化
  （症状：音量 1/3、时长 2–5 倍、prompt semantic 全乱）。**config dict 只传 `custom` 键**。
  （v2ProPlus 非 LoRA 不受影响——它直接加载 weights_path，不走底模解析）
- **脚本**：`run_train_t120_v5.sh`（训练+导出）、`v5_finetune_test.py`（三组合对照：
  e15/v5turbo-lora、e15/v5dev-lora、s1v3/v5turbo-lora），日志 `train_logs/v5_finetune_test.log`，
  试听 `outputs/v5_ft/LISTEN.md`
- 修复后速度：v5turbo 稳态 0.8–2.3s/句；v5dev 2.3–4.4s/句

## 训练耗时参考（4090D）

| 阶段 | t30 (410条) | t120 (1459条) |
|---|---|---|
| 预处理 | 4 分钟 | 6 分钟 |
| SoVITS | 12ep / 5 分钟 | 8ep / 6 分钟 |
| GPT | 20ep / 13 分钟 | 15ep / 3.5 分钟 |
| 显存峰值 | 9.2G (s2) / 7.4G (s1) | 12.4G (s2) |
