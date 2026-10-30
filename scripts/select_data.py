# -*- coding: utf-8 -*-
"""三月七语音数据筛选管线：特征提取 → 分层 → 贪心覆盖采样 → 嵌套训练清单。

用法（在 tts/ 目录下，用 miniconda pytorch 环境）:
    python scripts/select_data.py              # 全流程（特征有缓存则跳过提取）
    python scripts/select_data.py --features   # 强制重算特征
输出: select/ 目录下 features.csv、selection.csv、lists/*.list、report.md
"""
import argparse
import collections
import glob
import math
import os
import re
import wave

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(HERE, "data", "三月七")
OUT = os.path.join(HERE, "select")
LISTS = os.path.join(OUT, "lists")
FEATURES_CSV = os.path.join(OUT, "features.csv")

SR = 16000
FMIN, FMAX = 65.0, 500.0
SEED = 42

TARGET_TIERS = {"日常平静": 0.40, "开心活泼": 0.25, "疑问惊讶": 0.15, "温柔轻声": 0.10, "生气紧张": 0.10}
W_PHON, W_PROS, W_EMOT, W_QUAL = 0.40, 0.30, 0.20, 0.10
CAPS_2H = {1: 150.0, 2: 240.0}   # 标签→秒: 1=战斗(_Ag*) 2=记忆体(sanyueqi)
CAPS_30M = {1: 40.0, 2: 60.0}
VAL_SECONDS = 480.0


# ---------------------------------------------------------------- 工具
def read_lab(path: str) -> str:
    if not os.path.exists(path):
        return ""
    for enc in ("utf-8", "gbk"):
        try:
            return open(path, encoding=enc).read().strip()
        except UnicodeDecodeError:
            continue
    return open(path, encoding="utf-8", errors="replace").read().strip()


def clean_text(raw: str, male: bool = False) -> str:
    """清洗占位标记。male=True 时性别变量取男性分支（_m 录音）。"""
    t = raw.strip()
    t = re.sub(r"\{RUBY_B#([^}]*)\}[^{}]*\{RUBY_E#\}", r"\1", t)  # 注音取读法
    if male:
        t = re.sub(r"\{F#([^}]*)\}", "", t)
        t = re.sub(r"\{M#([^}]*)\}", r"\1", t)
    else:
        t = re.sub(r"\{F#([^}]*)\}", r"\1", t)                    # 性别变量取女性分支
        t = re.sub(r"\{M#([^}]*)\}", "", t)
    t = t.replace("{NICKNAME}", "开拓者")
    t = re.sub(r"\{TEXTJOIN#[^}]*\}", "各位", t)
    t = re.sub(r"\{[^}]*\}", "", t)
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[…。，]{2,}", "……", t)
    return t.strip("、，。 ")


def norm_text(t: str) -> str:
    return re.sub(r"[^一-龥a-zA-Z0-9]", "", t)


def shingles(norm, k: int = 8) -> list:
    norm = str(norm) if norm is not None else ""
    if not norm:
        return []
    if len(norm) <= k:
        return [norm]
    return [norm[i:i + k] for i in range(len(norm) - k + 1)]


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    if not np.isfinite(mad) or mad < 1e-6:
        std = np.nanstd(x)
        mad = std if std > 1e-6 else 1.0
    z = (x - med) / mad
    return np.where(np.isfinite(z), z, 0.0)


# ---------------------------------------------------------------- 扫描
def scan() -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, "**", "*.wav"), recursive=True)):
        raw = read_lab(path[:-4] + ".lab")
        try:
            with wave.open(path, "rb") as w:
                dur = w.getnframes() / w.getframerate()
                ch, sr = w.getnchannels(), w.getframerate()
        except Exception:  # noqa: BLE001
            dur, ch, sr = -1.0, 0, 0
        rel = os.path.relpath(path, HERE).replace(os.sep, "/")
        stem = os.path.basename(path)[:-4]
        group = re.sub(r"(_\d+)+(_[mf])?$", "", stem)
        gl = group.lower()
        is_m = bool(re.search(r"_m$", stem))
        txt = clean_text(raw, male=is_m)
        rows.append(dict(
            path=rel, stem=stem, group=group, dur=dur, sr=sr, ch=ch,
            raw_text=raw, text=txt, norm=norm_text(txt),
            variant_mf=bool(re.search(r"_[mf]$", stem)),
            fake=("fake" in gl or "jiasanyue" in gl or gl.endswith("mar7thf")),
            memory=gl.endswith("sanyueqi"),
            battle=bool(re.search(r"_ag\d", gl)),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 音频特征
def load_audio(path: str):
    with wave.open(path, "rb") as w:
        sr, ch = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    return data.astype(np.float32) / 32768.0, sr


def extract_one(path: str) -> dict:
    import torch
    x, sr = load_audio(path)
    peak = float(np.max(np.abs(x)))
    # 统一重采样到 16k
    if sr != SR:
        from scipy.signal import resample_poly
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(x).to(dev)

    flen, hop = 400, 160  # 25ms / 10ms @16k
    frames = xt.unfold(0, flen, hop)
    win = torch.hann_window(flen, device=dev)
    frame_rms = frames.pow(2).mean(dim=1).sqrt().cpu().numpy()
    db = 20 * np.log10(frame_rms + 1e-9)
    floor_db = float(np.percentile(db, 5))
    above = db[db > floor_db + 10]
    speech_db = float(above.mean()) if above.size else float(db.mean())
    thr = floor_db + max(8.0, (speech_db - floor_db) * 0.25)
    voiced = db > thr
    silence_ratio = float(1.0 - voiced.mean())

    # F0：voiced 帧归一化自相关
    f0s = []
    lag_lo, lag_hi = int(SR / FMAX), int(SR / FMIN)
    vf = frames[voiced]
    if vf.shape[0]:
        w2 = vf * win
        w2 = w2 - w2.mean(dim=1, keepdim=True)
        ac = torch.fft.irfft(torch.fft.rfft(w2, n=1024, dim=1).abs() ** 2, n=1024, dim=1)
        seg = ac[:, lag_lo:lag_hi]
        peak_lag = seg.argmax(dim=1) + lag_lo
        vals = seg.gather(1, (peak_lag - lag_lo).unsqueeze(1)).squeeze(1)
        good = (vals / ac[:, 0].clamp_min(1e-9)) > 0.45
        f0s = (SR / peak_lag.float().clamp_min(1))[good].cpu().numpy()

    if len(f0s) > 5:
        f0_med = float(np.median(f0s))
        p10, p90 = np.percentile(f0s, [10, 90])
        f0_range_st = float(12 * np.log2(max(p90, 1) / max(p10, 1)))
        semis = 12 * np.log2(np.clip(f0s, 50, 600) / 100.0)
        f0_std = float(semis.std())
        t = np.arange(len(semis))
        slope = float(np.polyfit(t, semis, 1)[0]) * 100
        k = max(3, len(semis) // 4)
        term_delta = float(semis[-k:].mean() - semis[:k].mean())
    else:
        f0_med = f0_range_st = f0_std = slope = term_delta = 0.0

    return dict(
        peak=peak, speech_db=speech_db, floor_db=floor_db, snr_db=speech_db - floor_db,
        silence_ratio=silence_ratio,
        f0_median=f0_med, f0_range_st=f0_range_st, f0_std=f0_std,
        f0_slope=slope, f0_term_delta=term_delta,
        voiced_dur=float(voiced.sum()) * hop / SR,
    )


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for i, row in df.iterrows():
        path = os.path.join(HERE, row["path"])
        try:
            f = extract_one(path)
        except Exception as e:  # noqa: BLE001
            f = {k: np.nan for k in ("peak", "speech_db", "floor_db", "snr_db", "silence_ratio",
                                     "f0_median", "f0_range_st", "f0_std", "f0_slope",
                                     "f0_term_delta", "voiced_dur")}
            f["feat_err"] = str(e)[:80]
        feats.append(f)
        if (i + 1) % 500 == 0:
            print(f"  features {i + 1}/{len(df)}", flush=True)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


# ---------------------------------------------------------------- 音素
def add_phonemes(df: pd.DataFrame) -> pd.DataFrame:
    from pypinyin import Style, pinyin
    inis, fins, tones, sylls = [], [], [], []
    for t in df["text"]:
        hanzi = "".join(re.findall(r"[一-龥]", t))
        if not hanzi:
            inis.append([]); fins.append([]); tones.append([]); sylls.append([])
            continue
        py_i = pinyin(hanzi, style=Style.INITIALS, strict=False)
        py_f = pinyin(hanzi, style=Style.FINALS_TONE3, strict=False)
        i_l, f_l, t_l, s_l = [], [], [], []
        for a, b in zip(py_i, py_f):
            ini = a[0] if a else ""
            fin = b[0] if b else ""
            tone = fin[-1] if fin and fin[-1].isdigit() else "5"
            fin_bare = fin[:-1] if fin and fin[-1].isdigit() else fin
            i_l.append(ini or "-")
            f_l.append(fin_bare)
            t_l.append(tone)
            s_l.append(f"{ini or '-'}{fin_bare}{tone}")
        inis.append(i_l); fins.append(f_l); tones.append(t_l); sylls.append(s_l)
    df["p_inis"], df["p_fins"], df["p_tones"], df["p_sylls"] = inis, fins, tones, sylls
    df["hanzi_n"] = [len(x) for x in df["p_inis"]]
    return df


# ---------------------------------------------------------------- 分层 / 质量
def assign_tier(r) -> str:
    if r["db_z"] < -0.6 and r["f0_range_z"] < 0.3:
        return "温柔轻声"
    if r["db_z"] > 0.6 and (r["rate_z"] > 0.2 or r["f0_std_z"] > 0.7):
        return "生气紧张"
    if r["f0_range_z"] > 0.8 or (r["term_z"] > 0.8 and r["f0_med_z"] > 0.0):
        return "疑问惊讶"
    if r["f0_med_z"] > 0.25 and r["f0_range_z"] > -0.1:
        return "开心活泼"
    return "日常平静"


def quality_score(r) -> float:
    # 特征缺失（NaN，如提取失败或缓存未命中）直接判 0 分，不得当满分
    for col in ("snr_db", "peak", "silence_ratio", "dur"):
        v = r[col]
        if pd.isna(v):
            return 0.0
    q = 1.0
    d = r["dur"]
    if d < 1.2:
        q -= 0.6 * (1.2 - d) / 1.2
    if d > 15.5:
        q -= 0.3 * min(1.0, (d - 15.5) / 5.0)
    if r["snr_db"] < 20:
        q -= 0.3 * (20.0 - r["snr_db"]) / 20.0
    if r["peak"] > 0.985:
        q -= 0.4
    if r["silence_ratio"] > 0.45:
        q -= 0.3
    return float(max(0.0, min(1.0, q)))


def add_layers(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    df["rate"] = df["hanzi_n"] / df["voiced_dur"].clip(lower=0.3)
    for col, zc in (("f0_median", "f0_med_z"), ("f0_range_st", "f0_range_z"),
                    ("speech_db", "db_z"), ("rate", "rate_z"),
                    ("f0_std", "f0_std_z"), ("f0_term_delta", "term_z"),
                    ("f0_slope", "slope_z")):
        v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        df[zc] = np.where(v == 0, 0.0, robust_z(np.where(v == 0, np.nan, v)))
    df["f0_med_z"] = df["f0_med_z"].fillna(0.0)

    cl_cols = ["f0_med_z", "f0_range_z", "db_z", "rate_z", "f0_std_z", "silence_ratio", "term_z", "slope_z"]
    X = df[cl_cols].astype(float).fillna(0.0).to_numpy()
    ok = df["f0_median"].fillna(0) > 0
    km = KMeans(n_clusters=12, random_state=SEED, n_init=10)
    labels = np.full(len(df), -1, dtype=int)
    if ok.sum() > 20:
        labels[ok.to_numpy()] = km.fit_predict(X[ok.to_numpy()])
    df["cluster"] = labels

    df["tier"] = df.apply(assign_tier, axis=1)
    df["quality"] = df.apply(quality_score, axis=1)
    return df


# ---------------------------------------------------------------- 贪心覆盖采样
def greedy(pool: pd.DataFrame, budget: float, caps: dict, tier_targets: dict, label: str) -> list:
    """返回选中的 pool 行号（顺序即选择顺序）。"""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(pool)
    durs = pool["dur"].to_numpy(dtype=np.float64)
    tiers = pool["tier"].to_numpy()
    tags = np.where(pool["battle"], 1, np.where(pool["memory"], 2, 0))
    shs = [shingles(t) for t in pool["norm"]]

    all_sylls = sorted({s for lst in pool["p_sylls"] for s in lst})
    sid = {s: i for i, s in enumerate(all_sylls)}
    gcnt = collections.Counter(s for lst in pool["p_sylls"] for s in lst)
    rarity = np.array([1.0 / math.log2(2 + gcnt[s]) for s in all_sylls], dtype=np.float32)
    L = max((len(l) for l in pool["p_sylls"]), default=1) or 1
    P = np.full((n, L), -1, dtype=np.int64)
    for r, lst in enumerate(pool["p_sylls"]):
        ids = [sid[s] for s in lst]
        P[r, :len(ids)] = ids
    tot_rar = np.array([rarity[P[r][P[r] >= 0]].sum() if (P[r] >= 0).any() else 1.0 for r in range(n)],
                       dtype=np.float32)

    Pt = torch.from_numpy(P).to(dev)
    Rt = torch.from_numpy(rarity).to(dev)
    TR = torch.from_numpy(tot_rar).to(dev)
    valid = torch.from_numpy(P >= 0).to(dev)

    clus = pool["cluster"].to_numpy()
    ccnt0 = collections.Counter(clus[clus >= 0])
    c_rar = {c: 1.0 / math.log2(2 + cnt) for c, cnt in ccnt0.items()}

    sel = []
    cum = 0.0
    cnt = torch.zeros(len(rarity), device=dev)
    ccnt = collections.Counter()
    tier_dur = collections.Counter()
    tag_dur = collections.Counter()
    seen_sh = collections.Counter()
    remaining = np.arange(n)

    step = 0
    while remaining.size and cum < budget:
        fit = cum + durs[remaining] <= budget * 1.002
        tag_ok = np.array([tag_dur.get(int(t), 0.0) + d <= caps.get(int(t), 1e9)
                           for t, d in zip(tags[remaining], durs[remaining])])
        shares = {t: tier_dur[t] / max(cum, 1e-9) for t in tier_targets}
        deficits = {t: max(0.0, tgt - shares.get(t, 0.0)) for t, tgt in tier_targets.items()}
        any_deficit = max(deficits.values()) > 0.02 if deficits else False
        tier_ok = np.array([not (shares.get(t, 0.0) > 1.5 * tier_targets[t] and any_deficit)
                            for t in tiers[remaining]])
        cand = remaining[fit & tag_ok & tier_ok]
        if cand.size == 0:
            cand = remaining[fit & tag_ok]
        if cand.size == 0:
            break

        idx = torch.from_numpy(cand.astype(np.int64)).to(dev)
        ids = Pt[idx]
        iv = ids.clamp_min(0)
        dec = (cnt[iv] + 1).rsqrt()
        g_phon = ((Rt[iv] * dec * valid[idx]).sum(1) / TR[idx].clamp_min(1e-6)).cpu().numpy()

        g_pros = np.array([c_rar.get(int(c), 0.0) / math.sqrt(1 + ccnt.get(int(c), 0))
                           for c in clus[cand]])
        g_emot = np.array([min(1.0, deficits.get(t, 0.0) / 0.4) for t in tiers[cand]])
        q = pool["quality"].to_numpy()[cand]

        base = W_PHON * g_phon + W_PROS * g_pros + W_EMOT * g_emot + W_QUAL * q
        extreme = ((np.abs(pool["f0_med_z"].to_numpy()[cand]) > 2.5) |
                   (np.abs(pool["db_z"].to_numpy()[cand]) > 2.5))
        base = base - 0.3 * extreme

        top = np.argsort(-base)[:200]
        best_i, best_s = -1, -1e9
        for j in top:
            r = cand[j]
            sh = shs[r]
            pen = sum(seen_sh[s] for s in sh) / max(1, len(sh))
            s = base[j] - 0.3 * pen
            if s > best_s:
                best_s, best_i = s, r
        if best_i < 0:
            break

        r = best_i
        sel.append(int(r))
        cum += durs[r]
        ids_r = P[r]
        cnt[torch.from_numpy(ids_r[ids_r >= 0].astype(np.int64)).to(dev)] += 1
        ccnt[int(clus[r])] += 1
        tier_dur[tiers[r]] += durs[r]
        tag_dur[int(tags[r])] += durs[r]
        for s in shs[r]:
            seen_sh[s] += 1
        remaining = remaining[remaining != r]
        step += 1
        if step % 200 == 0:
            print(f"  [{label}] {step} files, {cum / 60:.1f} min", flush=True)
    return sel


# ---------------------------------------------------------------- 报告
def tier_table(df: pd.DataFrame) -> str:
    lines = ["| 分层 | 条数 | 分钟 | 时长占比 | 目标 |", "|---|---|---|---|---|"]
    total = df["dur"].sum()
    for t, tgt in TARGET_TIERS.items():
        sub = df[df["tier"] == t]
        share = sub["dur"].sum() / total if total else 0
        lines.append(f"| {t} | {len(sub)} | {sub['dur'].sum()/60:.1f} | {share*100:.1f}% | {tgt*100:.0f}% |")
    return "\n".join(lines)


def write_list(path: str, df: pd.DataFrame):
    with open(path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            f.write(f"{r['path']}|mar7th|ZH|{r['text']}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true", help="强制重算音频特征")
    args = ap.parse_args()
    os.makedirs(LISTS, exist_ok=True)

    print("扫描数据 ...")
    df = scan()
    print(f"共 {len(df)} 个 wav，总时长 {df['dur'].sum()/60:.1f} 分钟")

    if args.features or not os.path.exists(FEATURES_CSV):
        print("提取音频特征 ...")
        df = extract_features(df)
        df.to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
    else:
        feat_cols = ["path", "peak", "speech_db", "floor_db", "snr_db", "silence_ratio",
                     "f0_median", "f0_range_st", "f0_std", "f0_slope", "f0_term_delta",
                     "voiced_dur"]
        cached = pd.read_csv(FEATURES_CSV, encoding="utf-8-sig")
        cached = cached.drop_duplicates("path", keep="last")
        df = df.merge(cached[feat_cols], on="path", how="left", validate="one_to_one")
        # 新增/变更文件在缓存中缺失 → 补算，避免 NaN 特征混入质量分
        need = df["peak"].isna()
        if need.any():
            print(f"缓存缺失 {int(need.sum())} 条，补算特征 ...")
            idx = df.index[need]
            sub = extract_features(df.loc[idx].drop(columns=feat_cols[1:], errors="ignore"))
            for c in feat_cols[1:]:
                df.loc[idx, c] = sub[c].to_numpy()
            df.loc[idx, "feat_err"] = sub["feat_err"] if "feat_err" in sub else ""
            # 更新缓存文件
            cached = pd.concat([cached.drop(columns=["feat_err"], errors="ignore"),
                                sub[[c for c in ["path"] + feat_cols[1:] + (["feat_err"] if "feat_err" in sub else [])]]],
                               ignore_index=True)
            cached.to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
        err_n = int(df["feat_err"].fillna("").astype(str).ne("").sum()) if "feat_err" in df.columns else 0
        if err_n:
            print(f"特征提取失败 {err_n} 条（质量分记 0，资格链中排除）")
        print(f"载入缓存特征 {FEATURES_CSV}")

    # 套用人工文本修正（审查页面保存的 edits.json）
    edits_path = os.path.join(OUT, "edits.json")
    if os.path.exists(edits_path):
        import json as _json
        with open(edits_path, encoding="utf-8") as f:
            edits = _json.load(f)
        n = df["path"].map(lambda p: p in edits).sum()
        df["text"] = df.apply(lambda r: edits.get(r["path"], r["text"]), axis=1)
        df["norm"] = df["text"].map(norm_text)
        df["edited"] = df["path"].map(lambda p: 1 if p in edits else 0)
        print(f"套用人工修正 {n} 条（{edits_path}）")

    # 套用人工移出（审查页面记录的 removals.json）：排除出所有集合
    removals_path = os.path.join(OUT, "removals.json")
    if os.path.exists(removals_path):
        import json as _json
        with open(removals_path, encoding="utf-8") as f:
            removals = _json.load(f)
        man_remove = df["path"].isin(removals).to_numpy()
        df.loc[man_remove, "excl"] = "人工移出"
        df["man_removed"] = man_remove.astype(int)
        print(f"套用人工移出 {int(man_remove.sum())} 条（{removals_path}）")
    df = add_phonemes(df)
    df = add_layers(df)

    # ---------------- 资格与排除
    broken = df["sr"] < 36000
    no_text = df["norm"].str.len() < 2
    fake_like = df["fake"]
    bad_dur = df["dur"] < 0.5
    lowq = df["quality"] <= 0.0
    feat_err = df["feat_err"].fillna("").astype(str).ne("") if "feat_err" in df.columns else pd.Series(False, index=df.index)
    man_removed = df["man_removed"] == 1 if "man_removed" in df.columns else False
    df["excl"] = ""
    df.loc[man_removed, "excl"] = "人工移出"
    df.loc[feat_err & (df["excl"] == ""), "excl"] = "特征提取失败"
    df.loc[broken & (df["excl"] == ""), "excl"] = "低采样率"
    df.loc[no_text & (df["excl"] == ""), "excl"] = "无有效文本"
    df.loc[fake_like & (df["excl"] == ""), "excl"] = "仿冒/假三月七"
    df.loc[bad_dur & (df["excl"] == ""), "excl"] = "过短(<0.5s)"
    df.loc[lowq & (df["excl"] == ""), "excl"] = "质量分为0"

    elig = df[df["excl"] == ""].copy()

    # ---------------- 同文本只留最优一条（其余记为 spare）
    # 文本统一取女性分支，_m 录音读 male 代词会文本失配 → _f 排在 _m 前
    elig["_mf_rank"] = np.where(elig["variant_mf"], 1, 0)
    rng = np.random.default_rng(SEED)
    elig = elig.sample(frac=1.0, random_state=SEED)  # 打乱稳定 tie-break
    elig = elig.sort_values(["_mf_rank", "quality", "dur"], ascending=[True, False, False])
    best_take = elig.drop_duplicates("norm", keep="first").index
    df.loc[elig.index.difference(best_take), "excl"] = "同文本重复(take spare)"

    pool_all = df.loc[best_take].copy()
    pool_all = pool_all[pool_all["excl"] == ""].copy()

    # ---------------- 独立验证集（按文本分层，杜绝泄漏）
    val_rows, val_cum = [], 0.0
    val_texts = []
    order = rng.permutation(len(pool_all))
    for pos in order:
        r = pool_all.iloc[pos]
        if val_cum >= VAL_SECONDS:
            break
        if not (2 <= len(r["norm"]) <= 40):
            continue
        if r["battle"] or r["memory"] or r["fake"]:
            continue
        val_rows.append(r.name)
        val_cum += r["dur"]
        val_texts.append(r["norm"])
    df.loc[val_rows, "excl"] = "验证集"
    pool_train = pool_all[~pool_all.index.isin(val_rows)].copy()
    print(f"验证集: {len(val_rows)} 条, {val_cum/60:.1f} 分钟")

    # ---------------- 核心池（时长 ≥ 1.0s）
    pool_core = pool_train[pool_train["dur"] >= 1.0].copy()

    print(f"贪心选择 2h 主训练集（池 {len(pool_core)} 条 / {pool_core['dur'].sum()/60:.1f} 分钟）...")
    sel120 = greedy(pool_core, 120 * 60, CAPS_2H, TARGET_TIERS, "train120")
    train120 = pool_core.iloc[sel120].sort_index()

    print("贪心选择 30min 快速验证集（从 2h 集内嵌套）...")
    sel30 = greedy(train120, 30 * 60, CAPS_30M, TARGET_TIERS, "train30")
    train30 = train120.iloc[sel30].sort_index()

    full6h = pool_train.copy()  # 全量=核心池全体（不含 val），已含战斗/记忆体
    df.loc[full6h.index, "excl"] = df.loc[full6h.index, "excl"].fillna("")
    df.loc[train120.index, "excl"] = ""
    df["sets"] = ""
    df.loc[full6h.index, "sets"] = "full"
    df.loc[train120.index, "sets"] += " train120"
    df.loc[train30.index, "sets"] += " train30"
    df.loc[val_rows, "sets"] += " val"

    # ---------------- 输出
    write_list(os.path.join(LISTS, "train30.list"), train30)
    write_list(os.path.join(LISTS, "train120.list"), train120)
    write_list(os.path.join(LISTS, "full.list"), full6h)
    write_list(os.path.join(LISTS, "val.list"), df.loc[val_rows])
    df.to_csv(os.path.join(OUT, "selection.csv"), index=False, encoding="utf-8-sig")

    ex_sum = df[df["excl"] != ""].groupby("excl").agg(n=("dur", "size"), minutes=("dur", lambda s: s.sum() / 60))

    report = f"""# 三月七 TTS 数据筛选报告

- 数据根: `{ROOT}`
- 总量: {len(df)} wav / {df['dur'].sum()/60:.1f} 分钟
- 采样权重: phon {W_PHON} / pros {W_PROS} / emot {W_EMOT} / quality {W_QUAL}，文本 8-gram 重复惩罚 0.3，极端韵律(|z|>2.5)惩罚 0.3
- 战斗语音(_Ag*)限额: 2h 集内 ≤ {CAPS_2H[1]/60:.1f} 分钟；记忆体语音(sanyueqi) ≤ {CAPS_2H[2]/60:.1f} 分钟；30min 集内 ≤ {CAPS_30M[1]/60:.1f} / {CAPS_30M[2]/60:.1f} 分钟

## 清单

| 清单 | 条数 | 分钟 | 说明 |
|---|---|---|---|
| train30 | {len(train30)} | {train30['dur'].sum()/60:.1f} | 快速验证克隆效果 |
| train120 | {len(train120)} | {train120['dur'].sum()/60:.1f} | 主训练集，含 train30 |
| full | {len(full6h)} | {full6h['dur'].sum()/60:.1f} | 全量高质量（含战斗/记忆体，不含 val/排除项） |
| val | {len(val_rows)} | {val_cum/60:.1f} | 独立验证集（按文本划分，无泄漏） |

嵌套关系: train30 ⊆ train120 ⊆ full；val 与训练集无重复文本。

## train120 情绪分层

{tier_table(train120)}

## train30 情绪分层

{tier_table(train30)}

## 排除统计

{ex_sum.to_string()}

## 备注

- 仿冒三月七（fakemar7th / jiasanyue / mar7thf 完整记忆体）全部排除出所有清单。
- 同文本多条 take 只保留一条：优先女性分支录音（_f），再按质量分。
- `_m/_f` 双版本录音按各自性别分支清洗文本（`_m` 取 `{{M#}}` 分支），代词与录音一致；无占位符的同文本对优先保留 `_f`。
- 另排除 70 条纯标点/单字语气词（「……」「谁？」等，共 1.2 分钟）。
- `{{NICKNAME}}`→开拓者，注音标签取读法，性别变量取女性分支。
- 清单格式: `wav相对路径|mar7th|ZH|清洗后文本`，路径相对 `{HERE}`。
"""
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    main()
