# v5 LoRA 微调试听（15 条，3 组合）

微调：SoVITS v5 LoRA（rank 32，8 epoch，batch 8 + grad_ckpt），GPT 复用 e15 不重训。
预处理/清单与 v2ProPlus 完全同源（train120 1459 条）。

**排障记录**：第一轮测试音频全异常（音量 1/3、时长 2–5 倍）——根因是 `TTS_Config(dict)`
的 `configs_.update(configs)` 会把传入的 version 键覆盖 `default_configs[version]`，
LoRA 推理端从那里解析底模 → 拿到 LoRA pth 自己 → SoVITS 实际为随机初始化 + LoRA。
修复：config dict 只传 `custom` 键，不传版本键。e4/e8 的"过拟合"结论作废，属同一 bug。

实测速度（修复后）：v5turbo 稳态 0.8–2.3s/句（与 v2ProPlus 相当）；v5dev 2.3–4.4s/句。

## 组合1: v5turbo_ft（e15 GPT + v5turbo LoRA）★ 部署候选

- baoxiao（吞字观察句）：[wav](v5turbo_ft/baoxiao.wav)
- [val_2](v5turbo_ft/val_2.wav) 长句 · [val_3](v5turbo_ft/val_3.wav) 短句 · [val_4](v5turbo_ft/val_4.wav) · [val_5](v5turbo_ft/val_5.wav)

## 组合2: v5dev_ft（e15 GPT + v5dev LoRA，32 步采样质量参照）

- baoxiao：[wav](v5dev_ft/baoxiao.wav)
- [val_2](v5dev_ft/val_2.wav) · [val_3](v5dev_ft/val_3.wav) · [val_4](v5dev_ft/val_4.wav) · [val_5](v5dev_ft/val_5.wav)

## 组合3: v5turbo_ft_baseGPT（原版 s1v3 + v5turbo LoRA）— 归因 GPT 微调的贡献

- baoxiao：[wav](v5turbo_ft_baseGPT/baoxiao.wav)
- [val_2](v5turbo_ft_baseGPT/val_2.wav) · [val_3](v5turbo_ft_baseGPT/val_3.wav) · [val_4](v5turbo_ft_baseGPT/val_4.wav) · [val_5](v5turbo_ft_baseGPT/val_5.wav)

## 对照

- v5 零样本（同句子）：[LISTEN.md](../v5_zeroshot/LISTEN.md)
- v2ProPlus 定稿微调（当前 API 线上配置）：baoxiao [try1](../fix_refc/hot/baoxiao_try1.wav) ·
  [val_2](../fix_refc/hot/val_2.wav) · [val_3](../fix_refc/hot/val_3.wav) · [val_5](../fix_refc/hot/val_5.wav)

## 判定要点

1. **吞字**：v5 三组合 vs v2ProPlus 定稿逐句对比（baoxiao「布洛妮娅」、val_2 长句最易暴露）；
   组合1 vs 组合3 的差异 = GPT 微调（e15）对内容稳定性的贡献
2. **音色**：v5 微调 vs v2ProPlus 微调 vs v5 零样本，谁更像三月七
3. **电声**：v5 新 vocoder 的高频毛刺是否消失
4. 若组合1 胜出 → 换 API 权重为 v5turbo LoRA（改 mar7th_api.py 两行 + 重启）
