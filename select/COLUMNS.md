# select/ 明细数据列说明

完整明细 CSV 未入库（体积与隐私考量），本地生成：
`python scripts/select_data.py`（特征缓存命中时数十秒）。此处记录各列含义，便于解读清单与日志。

## selection.csv（4995 行 × 每条语音）

| 列 | 含义 |
|---|---|
| path | wav 相对路径（相对 tts/） |
| stem / group | 文件名 / 来源组（chapterN_M、archive、sideN_x…） |
| dur / sr / ch | 时长秒 / 采样率 / 声道 |
| raw_text | .lab 原始文本（含 {NICKNAME} 等占位标记） |
| text | 清洗后文本（占位符替换、性别分支对齐录音） |
| norm | 归一化文本（去标点，用于查重/泄漏检查） |
| variant_mf | 是否 _m/_f 双版本录音 |
| fake / memory / battle | 仿冒（fakemar7th/jiasanyue/mar7thf）/ 记忆体（sanyueqi）/ 战斗（_Ag*）标记 |
| peak / speech_db / floor_db / snr_db / silence_ratio | 音频电平与静音统计 |
| f0_median / f0_range_st / f0_std / f0_slope / f0_term_delta | 基频中位数(Hz) / 起伏(半音) / 波动 / 走势 / 句尾趋势 |
| voiced_dur | 有效发声时长 |
| rate | 语速（字/秒） |
| f0_med_z … slope_z | 各特征鲁棒 z 分数（MAD） |
| cluster | 韵律 KMeans 聚类（k=12，-1 为 F0 缺失） |
| tier | 情绪分层：日常平静/开心活泼/疑问惊讶/温柔轻声/生气紧张 |
| quality | 质量分 0–1（时长/SNR/削波/静音惩罚） |
| p_inis / p_fins / p_tones / p_sylls | 拼音声母/韵母/声调/音节三元组列表 |
| hanzi_n | 汉字数 |
| excl | 排除原因（空=合格）：低采样率/无有效文本/仿冒/过短/质量分为0/同文本重复(take spare)/验证集/人工移出 |
| sets | 所属清单（full train120 train30 val，空格分隔） |
| edited / man_removed | 人工修正 / 人工移出标记 |

## features.csv

selection 的特征提取子集（path + 音频特征列），作特征缓存：删除即重算。

## 生成清单（lists/，已入库）

`path|speaker|lang|text` 格式，speaker 固定 mar7th：
- train30.list（410 条 / 29.4 分钟）快速验证
- train120.list（1459 条 / 117.6 分钟）主训练集，含 train30
- full.list（4435 条 / 351.1 分钟）全量高质量
- val.list（110 条 / 8.0 分钟）独立验证集，与训练集文件级+文本级零重叠

嵌套关系 train30 ⊆ train120 ⊆ full 由 scripts/verify_lists.py 程序化验证。
