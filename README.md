# gptsovits-sanyueqi

《崩坏：星穹铁道》三月七中文语音的 GPT-SoVITS v2ProPlus 微调全流程：数据筛选、
审查工具、训练与推理脚本。目标是为聊天机器人长期稳定克隆角色日常声线。

## 内容结构

```
scripts/            全部流水线脚本（数据筛选 / 审查服务 / 打包上传 / 训练 / 推理）
select/             筛选产物：四个训练清单 .list、report.md、明细列说明 COLUMNS.md
n1/                 n1 服务器训练档案：环境搭建、实际生效配置、训练日志、坑与修正
outputs/            试听 wav（t30 验证轮 / t120 正式轮，val 集文本合成）
```

## 快速了解

- **数据**：4995 条 / 388.5 分钟游戏语音，筛选出嵌套的
  train30(410 条/29.4min) ⊆ train120(1459 条/117.6min) ⊆ full(4437 条)，外加
  110 条独立验证集（与训练集文件级+文本级零重叠）。分层目标
  日常 40% / 活泼 25% / 惊讶 15% / 轻声 10% / 紧张 10%，贪心覆盖采样实现。
- **审查工具**：`scripts/serve_review.py` 起本地页（默认
  http://127.0.0.1:6270/select/review.html），按清单/分层/分组筛选、逐条试听、
  文本修正与移出训练集，修正实时同步到清单文件。
- **训练**：GPT-SoVITS v2ProPlus 微调，在 RTX 4090D (n1 服务器) 上完成。
  t120 正式轮：预处理 6 分钟 + SoVITS 8 epoch 6 分钟 + GPT 15 epoch 3.5 分钟。
- **试听**：`outputs/t120/e15/`（正式轮）与 `outputs/t30_e12/`（验证轮）为
  验证集文本合成结果，可对比数据规模与轮数效果。

## 复现入口

1. 数据筛选：`python scripts/select_data.py`（需 miniconda pytorch 环境 + pypinyin 等，见脚本头注释）
2. 审查与人工清洗：`python scripts/serve_review.py --port 6270 --open`
3. 上传 n1：`python scripts/pack_for_n1.py` → scp → `n1/README.md` 的训练流程
4. 训练环境细节（torchcodec 兼容补丁、权重导出修正等坑）：`n1/README.md`

## 许可与来源

- 语音数据来自游戏《崩坏：星穹铁道》公开剧情语音，仅用于个人研究与非 commercial
  的角色扮演机器人，版权归 miHoYo/HoYoverse 所有。
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS（MIT），训练时点 commit b4570b8。
