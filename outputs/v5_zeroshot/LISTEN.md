# v5 零样本试听（10 条）

**v5 零样本 = 原版 s1v3.ckpt（未微调 GPT）+ v5 底模 SoVITS，不训练**，参考音频沿用定稿 refC。
背景：v2ProPlus 微调版仍有吞字，官方 V5 Release Notes 称「SoVITS 不微调即显著提升音色相似度」。

实测速度（RTX 4090 D 全卡）：

| 变体 | 采样 | 加载 | 稳态延迟 |
|---|---|---|---|
| v5dev | 32 步 / cfg 1.30 | 5.4s | 2.0–4.0s/句（慢） |
| v5turbo | 4 步 / cfg 0.0 | 3.4s | **1.6–2.2s/句（与 v2ProPlus 相当）** |

## v5turbo（部署候选）

- baoxiao（吞字观察句）：[wav](v5turbo/baoxiao.wav)
- [val_2](v5turbo/val_2.wav) 长句 · [val_3](v5turbo/val_3.wav) 短句 · [val_4](v5turbo/val_4.wav) · [val_5](v5turbo/val_5.wav)

## v5dev（质量参照）

- baoxiao：[wav](v5dev/baoxiao.wav)
- [val_2](v5dev/val_2.wav) · [val_3](v5dev/val_3.wav) · [val_4](v5dev/val_4.wav) · [val_5](v5dev/val_5.wav)

## v2ProPlus 定稿对照（t120 e15/e8 微调 + refC + hot）

- baoxiao：[try1](../fix_refc/hot/baoxiao_try1.wav) · [try2](../fix_refc/hot/baoxiao_try2.wav) · [try3](../fix_refc/hot/baoxiao_try3.wav)
- [val_2](../fix_refc/hot/val_2.wav) · [val_3](../fix_refc/hot/val_3.wav) · [val_5](../fix_refc/hot/val_5.wav)

## 判定要点

1. **吞字**：v5 两变体 vs v2ProPlus 定稿，逐句数有没有少字/吞字（尤其 baoxiao 的「布洛妮娅」和长句 val_2）
2. **音色相似度**：零样本（无微调）v5 vs 微调过的 v2ProPlus，哪个更像三月七
3. **电声质量**：v5 新 vocoder 是否消除高频镜像/混叠毛刺（官方宣称的主要改进）
4. 若 v5turbo 零样本已达标 → 直接换零样本上线，免训练；若音色差一口气 → 再走 v5 微调
