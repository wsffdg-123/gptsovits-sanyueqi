# -*- coding: utf-8 -*-
"""把已筛选的三月七语音台词整理成「角色卡素材包」。

输入（只读）:
    select/lists/full.list  全量高质量台词（4425 条左右，随人工清洗变动）
    select/lists/val.list   独立验证集台词（110 条）
    select/selection.csv    每条 wav 的分层 / 时长 / 来源组 / 标记

输出（role/ 目录）:
    corpus.jsonl        逐条结构化台词（含场景、情绪分层、标记）
    lines_by_scene.md   按剧情场景分组的全部台词
    scene_index.md      场景索引：条数、时长、情绪分布、代表台词
    style_stats.md      语言风格统计：句长、语气词、人称、标点、情绪差异
    tier_samples.md     五种情绪分层代表台词
    lore_entities.md    出现的人物/地点线索与例句
    角色卡草稿.md        依据上述证据填写的角色卡骨架（未证实部分标 TODO）

用法（在 tts/ 目录）:
    python scripts/build_role_corpus.py
清单文件是台词文本的权威来源（人工修正在 select/edits.json 与 lists 里同步）；
selection.csv 只用来补充元数据，两者不一致时以 lists 为准并打印差异数。
"""

import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "role")
LISTS = os.path.join(HERE, "select", "lists")
SELECTION = os.path.join(HERE, "select", "selection.csv")

# ---------------------------------------------------------------- 场景表
# group 前缀 → (场景名, 依据)。依据为从语料关键词/上下文核实的线索，未核实标「推测」。
SCENES = [
    ("chapter0", "chapter0 黑塔空间站·序章", "已核实", "空间站、星穹列车首次登场"),
    (
        "chapter1",
        "chapter1 雅利洛-贝洛伯格",
        "已核实",
        "贝洛伯格/雅利洛、布洛妮娅、希儿",
    ),
    (
        "chapter2",
        "chapter2 仙舟罗浮",
        "已核实",
        "仙舟/罗浮、太卜司、丹鼎司、停云、驭空",
    ),
    ("chapter3", "chapter3 匹诺康尼", "已核实", "匹诺康尼、盛会之星、梦境、黑天鹅"),
    ("chapter4", "chapter4 翁法罗斯", "已核实", "翁法罗斯、阿基维利、开拓燃料"),
    (
        "chapter5",
        "chapter5 二相乐园·幻月游戏",
        "已核实",
        "二相乐园、珠星大厦、绒绒号、幻月游戏、阿哈",
    ),
    (
        "chapterfinality1",
        "chapterfinality1 终章特别篇（花姐线）",
        "推测",
        "花姐、回望空间站/仙舟/匹诺康尼/翁法罗斯",
    ),
    ("side1_ytzx", "side1 以太战线", "已核实", "以太战线、以太灵、公司、艾丝妲"),
    ("side1_tp2", "side1 同行任务·丹恒房间", "已核实", "群聊、丹恒房间、布洛妮娅消息"),
    ("side1_tp", "side1 同行任务·贝洛伯格", "已核实", "贝洛伯格、布洛妮娅、公司谈判"),
    (
        "side2_baishi",
        "side2 剑术修行（云璃/彦卿）",
        "已核实",
        "师父、剑、彦卿、云璃、怀炎",
    ),
    ("side2_lc1", "side2 罗刹事件调查", "已核实", "罗刹、杨叔同行、查案"),
    ("side2_fx1", "side2 符玄·穷观阵", "已核实", "太卜、符玄、穷观阵、卡芙卡过去"),
    ("side2_yq1", "side2 谛听", "已核实", "谛听、景元、追捕卡芙卡"),
    ("side3_yz", "side3 列车日常·年度大清洁", "已核实", "列车大清洁、窗外异变"),
    (
        "sideX_paidui",
        "sideX 车厢小剧场（排队奇遇）",
        "推测",
        "房间、通风管道、杨叔十八机兵阵",
    ),
    (
        "archive",
        "archive 角色档案（同伴评价）",
        "已核实",
        "丹恒/姬子/瓦尔特/帕姆的个人评价",
    ),
    ("Side0_al", "Side0 佩佩（阿兰同行）", "已核实", "佩佩、阿兰、丹恒"),
]

TIERS = ["日常平静", "开心活泼", "疑问惊讶", "温柔轻声", "生气紧张"]
TIER_NOTE = (
    "> ⚠️ 分层来自 `select_data.py::assign_tier`，由**音频韵律特征**（音量 z、F0 起伏、语速）划分，"
    "不是语义情绪标注。例如短促的「喂！你这家伙——」可能被分到「温柔轻声」。"
    "做角色卡时把分层当「语气/表演强度」参考，不要当「这句话是什么情绪」的事实。"
)

# 语气/口癖候选（统计用，不代表最终人设结论）
PARTICLES = [
    "呀",
    "啦",
    "嘛",
    "呢",
    "吧",
    "哦",
    "哟",
    "咯",
    "呗",
    "啊",
    "哇",
    "欸",
    "诶",
    "唔",
    "嗯",
    "咦",
    "哼",
    "哈",
    "嘿嘿",
    "哈哈",
    "哎呀",
    "哎",
    "喂",
    "诶嘿",
]
SELF_REFS = ["我", "咱", "本姑娘", "人家", "三月七"]
ADDRESS = ["开拓者", "你", "大家", "各位", "你们"]
NAMES = [
    "丹恒",
    "姬子",
    "杨叔",
    "瓦尔特",
    "帕姆",
    "布洛妮娅",
    "希儿",
    "娜塔莎",
    "虎克",
    "克拉拉",
    "史瓦罗",
    "杰帕德",
    "佩拉",
    "桑博",
    "可可利亚",
    "停云",
    "景元",
    "符玄",
    "青雀",
    "驭空",
    "彦卿",
    "云璃",
    "怀炎",
    "罗刹",
    "镜流",
    "白露",
    "椒丘",
    "桂乃芬",
    "素裳",
    "飞霄",
    "貊泽",
    "黑塔",
    "艾丝妲",
    "阿兰",
    "卡芙卡",
    "银狼",
    "流萤",
    "花火",
    "花姐",
    "知更鸟",
    "星期日",
    "黑天鹅",
    "黄泉",
    "砂金",
    "真理医生",
    "阮梅",
    "藿藿",
    "尾巴",
    "谛听",
    "佩佩",
    "朽叶",
    "斯科特",
    "虚照",
    "阿基维利",
]
PLACES = [
    "黑塔空间站",
    "空间站",
    "雅利洛",
    "贝洛伯格",
    "仙舟",
    "罗浮",
    "金人巷",
    "丹鼎司",
    "长乐天",
    "鳞渊境",
    "太卜司",
    "匹诺康尼",
    "梦境",
    "盛会之星",
    "翁法罗斯",
    "二相乐园",
    "珠星",
    "星穹列车",
    "列车",
    "星核",
    "无名客",
    "星核猎手",
    "流光忆庭",
]
# 容易与普通词撞车的短词，用正则限定
AMBIGUOUS = {
    "刃": r"(?<!剑)(?<!刀)刃",
    "开拓": r"开拓(?!者)",
    "我": r"我",
}
# 台词里出现但需人工确认指向的词
NEEDS_CHECK = {"花姐": "疑为某位同伴的昵称，需查剧情确认"}
# 角色卡要写「她怎么描述自己」时最有用的话题 → 匹配词
SELF_TOPICS = [
    ("身世：冰封与失忆", ["被封在一大块冰里", "冰里醒来", "失忆", "忘光", "过去都忘"]),
    ("能力：六相冰", ["六相冰", "造出冰", "冰会保护我", "冰墙", "冰锥"]),
    ("身份：无名客 / 星穹列车乘员", ["无名客", "星穹列车的乘员", "列车乘员"]),
    ("爱好：相机与照片", ["拍照", "相机", "相册", "照片"]),
    ("剑术修行", ["剑", "师父", "演武仪典"]),
    ("性格自述", ["本姑娘", "幻想自己", "守护大家"]),
]

NARRATION_RE = re.compile(r"^（.*）$")


def load_list(name):
    """读取 .list（path|speaker|lang|text），返回 [(path, text)]。"""
    path = os.path.join(LISTS, name)
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) != 4:
                continue
            rows.append((parts[0], parts[3]))
    return rows


def load_meta():
    """读取 selection.csv 元数据（不强依赖 pandas，保持脚本可独立运行）。"""
    import csv

    meta = {}
    if not os.path.exists(SELECTION):
        print(f"警告: 未找到 {SELECTION}，元数据（分层/时长）留空", file=sys.stderr)
        return meta
    with open(SELECTION, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            meta[r["path"]] = r
    return meta


def scene_of(group):
    for prefix, name, conf, why in SCENES:
        if group.startswith(prefix):
            return name, conf
    return f"未归类:{group}", "待核"


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def build_rows():
    meta = load_meta()
    rows = []
    seen = {}
    for source, name in (("full", "full.list"), ("val", "val.list")):
        for path, text in load_list(name):
            m = meta.get(path, {})
            group = m.get("group") or re.sub(
                r"(_\d+)+(_[mf])?$", "", os.path.basename(path)[:-4]
            )
            scene, conf = scene_of(group)
            stem = os.path.basename(path)[:-4]
            rows.append(
                {
                    "path": path,
                    "stem": stem,
                    "group": group,
                    "scene": scene,
                    "scene_conf": conf,
                    "source": source,
                    "text": text,
                    "chars": len(text),
                    "dur": round(fnum(m.get("dur")), 2),
                    "tier": m.get("tier") or "",
                    "quality": round(fnum(m.get("quality")), 3),
                    "variant_mf": m.get("variant_mf") in ("True", "1"),
                    "male_branch": stem.endswith("_m"),
                    "memory": m.get("memory") in ("True", "1"),
                    "battle": m.get("battle") in ("True", "1"),
                    "narration": bool(NARRATION_RE.match(text)),
                    "quotes_other": ("「" in text) or ("“" in text),
                }
            )
            seen[path] = text
    # 与 selection.csv 的文本差异（lists 为权威）
    diffs = [
        p for p, t in seen.items() if p in meta and (meta[p].get("text") or "") != t
    ]
    if diffs:
        print(
            f"提示: {len(diffs)} 条 lists 与 selection.csv 文本不一致，按 lists 处理",
            file=sys.stderr,
        )
    rows.sort(key=lambda r: (r["scene"], r["path"]))
    return rows, meta, diffs


# ---------------------------------------------------------------- 统计
def rate(n, total, base=1000):
    return f"{n / total * base:.1f}" if total else "0.0"


def style_stats(rows):
    n = len(rows)
    texts = [r["text"] for r in rows]
    hanzi = sum(len(re.sub(r"[^\u4e00-\u9fa5]", "", t)) for t in texts)
    chars = [len(t) for t in texts]
    dur = sum(r["dur"] for r in rows)
    q = sum(1 for t in texts if "？" in t or "?" in t)
    ex = sum(1 for t in texts if "！" in t or "!" in t)
    ell = sum(1 for t in texts if "…" in t or "..." in t)
    dash = sum(1 for t in texts if "——" in t)
    wave = sum(1 for t in texts if "～" in t or "~" in t)

    def counter_table(words, title):
        lines = [
            f"### {title}",
            "",
            "| 词 | 出现次数 | 每千句 | 覆盖率 |",
            "|---|---|---|---|",
        ]
        counts = [
            (w, sum(t.count(w) for t in texts), sum(1 for t in texts if w in t))
            for w in words
        ]
        counts.sort(key=lambda x: -x[1])
        for w, c, cov in counts:
            if not c:
                continue
            lines.append(f"| {w} | {c} | {rate(c, n)} | {cov / n * 100:.1f}% |")
        return lines

    out = [
        "# 三月七台词语言风格统计",
        "",
        f"- 语料：{n} 条台词 / {dur / 60:.1f} 分钟录音 / 约 {hanzi} 汉字",
        f"- 句长：平均 {sum(chars) / n:.1f} 字，中位 {sorted(chars)[n // 2]} 字，"
        f"最长 {max(chars)} 字；≤10 字占 {sum(1 for c in chars if c <= 10) / n * 100:.1f}%",
        f"- 问句 {q} 条（{q / n * 100:.1f}%）、感叹 {ex} 条（{ex / n * 100:.1f}%）、"
        f"省略号 {ell} 条（{ell / n * 100:.1f}%）、破折号 {dash} 条（{dash / n * 100:.1f}%）、"
        f"波浪号 {wave} 条（{wave / n * 100:.1f}%）",
        "",
        "## 句末语气词与口癖",
        "",
    ]
    out += counter_table(PARTICLES, "语气词 / 口癖（按出现次数）")
    out += [""]
    # 真正的句末语气词：只看结尾 1 字（去尾标点）
    ends = collections.Counter()
    for t in texts:
        s = t.rstrip("。！？!?…~～—、，, ")
        if s:
            ends[s[-1]] += 1
    out += [
        "### 句末字分布（去掉句末标点后的最后一个字）",
        "",
        "| 字 | 条数 | 占比 |",
        "|---|---|---|",
    ]
    for ch, c in ends.most_common(15):
        out.append(f"| {ch} | {c} | {c / n * 100:.1f}% |")
    out += [""]
    out += counter_table(SELF_REFS, "自称")
    out += [""]
    out += counter_table(ADDRESS, "称呼听者")
    out += [""]

    out += [
        "## 语气分层差异",
        "",
        "分层是按音频韵律自动划分（见 `select_data.py::assign_tier`），"
        "下表说明各层的**表演特征**，不是语义情绪标注。",
        "",
        "| 分层 | 条数 | 平均句长 | 问句% | 感叹% | 省略号% | 平均时长s |",
        "|---|---|---|---|---|---|---|",
    ]
    for tier in TIERS:
        sub = [r for r in rows if r["tier"] == tier]
        if not sub:
            continue
        st = [r["text"] for r in sub]
        k = len(sub)
        out.append(
            f"| {tier} | {k} | {sum(len(t) for t in st) / k:.1f} | "
            f"{sum(1 for t in st if '？' in t) / k * 100:.1f} | "
            f"{sum(1 for t in st if '！' in t) / k * 100:.1f} | "
            f"{sum(1 for t in st if '…' in t) / k * 100:.1f} | "
            f"{sum(r['dur'] for r in sub) / k:.1f} |"
        )
    out += ["", "## 逐条来源分布", "", "| 来源 | 条数 |", "|---|---|"]
    for src, c in collections.Counter(r["source"] for r in rows).most_common():
        out.append(f"| {src} | {c} |")
    out += [
        "",
        "## 特殊行占比",
        "",
        "| 类型 | 条数 | 说明 |",
        "|---|---|---|",
        f"| 内心独白（整句被（）包住） | {sum(r['narration'] for r in rows)} | 非口语对白，做角色卡时可标注为内心戏 |",
        f"| 引用他人/术语（含「」或“”） | {sum(r['quotes_other'] for r in rows)} | 台词里在复述别人的话或专有名词 |",
        f"| 记忆体分支（sanyueqi 组） | {sum(r['memory'] for r in rows)} | 剧情中的「记忆体三月七」，人设与本体不同，慎用 |",
        f"| 战斗语音（_Ag*） | {sum(r['battle'] for r in rows)} | 短促喊话，情绪强度偏高 |",
        f"| 性别分支录音（_m 男性版） | {sum(r['male_branch'] for r in rows)} | 同一句的男性开拓者版本，文本可能少「开拓者」称呼 |",
        "",
    ]
    return "\n".join(out) + "\n"


def pick_examples(
    rows,
    limit=12,
    lo=10,
    hi=45,
    tier=None,
    scene=None,
    avoid_special=True,
    spread=False,
):
    """挑代表台词。

    默认按 quality 排序（录音干净），但要求句长 10–45 字且不以「…」「——」开头，
    避免挑出一堆碎片化的半句。spread=True 时改为按 tier 轮流取样，保证场景内各语气都有。
    """
    pool = []
    for r in rows:
        if tier and r["tier"] != tier:
            continue
        if scene and r["scene"] != scene:
            continue
        if not (lo <= r["chars"] <= hi):
            continue
        if avoid_special and (r["narration"] or r["memory"]):
            continue
        if r["text"].startswith(("…", "——", "、", "，")):
            continue
        pool.append(r)
    if spread:
        by_tier = collections.defaultdict(list)
        for r in sorted(pool, key=lambda r: (-r["quality"], r["path"])):
            by_tier[r["tier"] or "—"].append(r)
        picked, idx = [], 0
        while len(picked) < limit and any(len(v) > idx for v in by_tier.values()):
            for t in TIERS:
                v = by_tier.get(t) or []
                if len(v) > idx and len(picked) < limit:
                    picked.append(v[idx])
            idx += 1
        return picked
    pool.sort(key=lambda r: (-r["quality"], r["chars"], r["path"]))
    return pool[:limit]


def scene_index(rows):
    out = [
        "# 场景索引（三月七台词分布）",
        "",
        "场景由 wav 文件名来源组推断，`已核实` 表示有台词关键词/上下文支撑，`推测` 需人工确认。",
        "",
        TIER_NOTE,
        "",
        "| 场景 | 置信 | 条数 | 分钟 | 语气分层前三 | 代表台词 |",
        "|---|---|---|---|---|---|",
    ]
    order = [s[1] for s in SCENES]
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["scene"]].append(r)
    for scene in order + [s for s in groups if s not in order]:
        sub = groups.get(scene)
        if not sub:
            continue
        conf = sub[0]["scene_conf"]
        tiers = collections.Counter(r["tier"] for r in sub if r["tier"]).most_common(3)
        tier_txt = "、".join(f"{t}{c}" for t, c in tiers) or "—"
        ex = pick_examples(sub, limit=1)
        ex_txt = (
            (ex[0]["text"][:40] + "…")
            if ex and len(ex[0]["text"]) > 40
            else (ex[0]["text"] if ex else "—")
        )
        out.append(
            f"| {scene} | {conf} | {len(sub)} | {sum(r['dur'] for r in sub) / 60:.1f} | {tier_txt} | {ex_txt} |"
        )
    out += [
        "",
        "## 各场景代表台词",
        "",
        "每场景 8 条，按语气分层轮流取样，便于快速感受该场景的说话状态。",
        "",
    ]
    for scene in order + [s for s in groups if s not in order]:
        sub = groups.get(scene)
        if not sub:
            continue
        out += [
            f"### {scene}（{len(sub)} 条 / {sum(r['dur'] for r in sub) / 60:.1f} 分钟）",
            "",
        ]
        for r in pick_examples(sub, limit=8, spread=True):
            out.append(f"- [{r['tier'] or '未分层'}] {r['text']}")
        out.append("")
    return "\n".join(out) + "\n"


def tier_samples(rows):
    out = [
        "# 语气分层代表台词",
        "",
        TIER_NOTE,
        "",
        "每层 15 条：句长 10–45 字、录音质量高、排除内心独白与记忆体分支。"
        "可直接用作角色卡的 few-shot 语气示例。",
        "",
    ]
    for tier in TIERS:
        sub = [r for r in rows if r["tier"] == tier]
        out += [
            f"## {tier}（{len(sub)} 条 / {sum(r['dur'] for r in sub) / 60:.1f} 分钟）",
            "",
        ]
        for r in pick_examples(rows, limit=15, tier=tier):
            out.append(f"- {r['text']}")
        out.append("")
    return "\n".join(out) + "\n"


def self_topics(rows):
    """按「她怎么谈自己」归类台词，是角色卡设定章节最直接的证据。"""
    out = [
        "# 关于她自己的台词（设定证据）",
        "",
        "每条都注明来源组，可直接引用到角色卡；同一话题下按台词长度排序，"
        "优先展示信息完整的句子。**这些是她自己说的，不是官方设定文档。**",
        "",
    ]
    for title, kws in SELF_TOPICS:
        hits = [r for r in rows if any(k in r["text"] for k in kws)]
        if not hits:
            continue
        # 去重（_m/_f 同句），按长度降序取信息量大的
        seen, uniq = set(), []
        for r in sorted(hits, key=lambda r: -r["chars"]):
            key = re.sub(r"[^\u4e00-\u9fa5]", "", r["text"])
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)
        out += [f"## {title}（{len(uniq)} 条）", ""]
        for r in uniq[:10]:
            out.append(f"- {r['text']}  \n  <sub>{r['scene']}</sub>")
        out.append("")
    return "\n".join(out) + "\n"


def lore_entities(rows):
    n = len(rows)

    def hits_of(word):
        pat = AMBIGUOUS.get(word)
        if pat:
            rx = re.compile(pat)
            return [r for r in rows if rx.search(r["text"])]
        return [r for r in rows if word in r["text"]]

    out = [
        "# 人物与地点线索",
        "",
        "从台词中统计的提及频次与例句，用于角色卡的「关系网」「世界观」章节。",
        "**频次高只说明剧情里常被提起，不代表亲近程度**；称呼方式（敬称/昵称/直呼）才更能反映关系。",
        "",
        "## 人物",
        "",
        "| 人物 | 提及条数 | 覆盖率 | 她的称呼方式 | 例句 |",
        "|---|---|---|---|---|",
    ]
    for name in NAMES:
        hits = hits_of(name)
        if not hits:
            continue
        # 称呼方式：统计「X小姐」「X将军」等后缀，只保留出现 ≥3 次的（低频多为引述他人）
        suff = collections.Counter()
        for r in hits:
            for m in re.finditer(
                re.escape(name)
                + r"(小姐|先生|将军|大人|站长|医生|老师|姐姐|姑娘|大爷|大师)",
                r["text"],
            ):
                suff[name + m.group(1)] += 1
        top = [(k, c) for k, c in suff.most_common(2) if c >= 3]
        addr = "、".join(f"{k}×{c}" for k, c in top) or "直呼其名"
        ex = pick_examples(hits, limit=1, lo=8, hi=50, avoid_special=False) or hits[:1]
        note = f"（{NEEDS_CHECK[name]}）" if name in NEEDS_CHECK else ""
        out.append(
            f"| {name} | {len(hits)} | {len(hits) / n * 100:.1f}% | {addr} | {ex[0]['text'][:50] if ex else '—'} |{note}"
        )
    out += [
        "",
        "## 地点 / 概念",
        "",
        "| 词 | 提及条数 | 覆盖率 | 例句 |",
        "|---|---|---|---|",
    ]
    for name in PLACES:
        hits = hits_of(name)
        if not hits:
            continue
        ex = pick_examples(hits, limit=1, lo=8, hi=50, avoid_special=False) or hits[:1]
        out.append(
            f"| {name} | {len(hits)} | {len(hits) / n * 100:.1f}% | {ex[0]['text'][:50] if ex else '—'} |"
        )
    out += ["", "## 同伴评价（archive 组，可作人设口吻样本）", ""]
    for r in rows:
        if r["scene"].startswith("archive"):
            out.append(f"- {r['text']}")
    return "\n".join(out) + "\n"


def lines_by_scene(rows):
    out = [
        "# 三月七全部台词（按场景）",
        "",
        f"共 {len(rows)} 条，`[分层] 台词`。带 `※` 的行为内心独白，带 `◇` 的为引用他人或术语。",
        "",
    ]
    cur = None
    for r in rows:
        if r["scene"] != cur:
            cur = r["scene"]
            out += ["", f"## {cur}", ""]
        mark = "※" if r["narration"] else ("◇" if r["quotes_other"] else "")
        out.append(f"- {mark}[{r['tier'] or '—'}] {r['text']}")
    return "\n".join(out) + "\n"


def write_corpus(rows):
    path = os.path.join(OUT, "corpus.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_draft(rows):
    """生成角色卡骨架：能由语料支撑的写结论并附例句，其余留 TODO。"""
    n = len(rows)
    texts = [r["text"] for r in rows]

    def p(word):
        """词在全部台词里的出现总次数。"""
        return sum(t.count(word) for t in texts)

    used = set()  # 全局去重，同一句例句只出现一次

    def ev(*kws, lo=10, hi=60, limit=2, first_person=False):
        """给某个论点找例句：命中任一关键词、长度适中、优先高质量。

        first_person=True 时要求句子里出现「我/咱/本姑娘」，避免挑到她在问别人的句子。
        全局去重：同一句不会在两个论点下重复出现。
        """
        hits = [
            r
            for r in rows
            if any(k in r["text"] for k in kws) and lo <= r["chars"] <= hi
        ]
        if first_person:
            hits = [r for r in hits if re.search(r"我|咱|本姑娘", r["text"])]
        out = []
        for r in sorted(hits, key=lambda r: -r["quality"]):
            key = re.sub(r"[^\u4e00-\u9fa5]", "", r["text"])
            if key in used:
                continue
            used.add(key)
            out.append(r["text"])
            if len(out) >= limit:
                break
        return out

    def bullets(kws, note="", **kw):
        lines = []
        for t in ev(*kws, **kw):
            lines.append(f"  - 「{t}」")
        if note:
            lines.append(f"  - {note}")
        return lines

    lines = [
        "# 三月七（Mar 7th）角色卡草稿",
        "",
        "> 由 `scripts/build_role_corpus.py` 依据语音台词语料生成。",
        "> 台词能支撑「她怎么说话、怎么描述自己」，**不能单独支撑完整官方设定**；",
        "> 标 TODO 的条目需人工补写或另行取证。引用的例句都可回到 `corpus.jsonl` 追溯。",
        "",
        f"语料规模：{n} 条台词（其中 {sum(r['source'] == 'val' for r in rows)} 条为验证集），"
        f"录音 {sum(r['dur'] for r in rows) / 60:.1f} 分钟。",
        "",
        "## 1. 身份与来历",
        "",
        "- **星穹列车乘员 / 无名客**。自我介绍的标准句式是「我是三月七，星穹列车的乘员」，"
        "对同伴的定位是「无名客」。",
    ]
    lines += bullets(["星穹列车的乘员", "无名客"], limit=2, hi=70)
    lines += [
        "",
        "- **失忆，来历不明**。她自己把「忘了过去」当玩笑讲，也用它自嘲。",
    ]
    lines += bullets(["把过去都忘光", "失忆前", "忘得一干二净"], limit=2, hi=90)
    lines += [
        "",
        "- **被列车从冰里捞起来**。封在「六相冰」里在宇宙漂流，是姬子、瓦尔特他们化冰救出的。",
    ]
    lines += bullets(["被封在一大块冰里", "冰里醒来", "六相冰"], limit=2, hi=70)
    lines += [
        "",
        "- **能力是凝冰**。失忆前就有的能力，遇险会下意识造冰护住自己。",
    ]
    lines += bullets(["造出冰", "冰会保护我"], limit=2, hi=70)
    lines += [
        "",
        "- **名字是自取的**：为了纪念被唤醒的日期。",
    ]
    lines += bullets(
        ["名字是我自己取的", "三月七」这个名字", "名字，也只是为了纪念"], limit=1, hi=90
    )
    lines += [
        "",
        "- TODO：年龄、身高、生日、命途、星级等官方设定语料里没有，需另查资料补。",
        "",
        "## 2. 说话方式（统计见 `style_stats.md`）",
        "",
        f"- 平均句长 22 字左右，日常对白常是 1–2 个短句，爱用「…」拖尾"
        f"（{sum(1 for t in texts if '…' in t) / n * 100:.0f}% 的句子带省略号）。",
        f"- 语气词密集：句末「吧」{p('吧')} 次、「啊」{p('啊')} 次、「啦」{p('啦')} 次、"
        f"「呀」{p('呀')} 次、「嘛」{p('嘛')} 次、「呢」{p('呢')} 次。",
        f"- 自称：「我」为主（{p('我')} 次），口语化时说「咱」（{p('咱')} 次），"
        f"逞强/自夸时切换到「本姑娘」（{p('本姑娘')} 次）。",
        f"- 称呼对话对象：固定叫「开拓者」（{p('开拓者')} 次），不用敬称。",
        f"- 惊讶用「欸？」「咦？」「哇」；思考用「唔…」「嗯…」；小得意用「嘿嘿」（{p('嘿嘿')} 次）。",
        "- 常自我吐槽，也常吐槽同伴（丹恒、杨叔、帕姆），这是稳定的笑点来源。",
        "",
        "## 3. 性格与行为倾向",
        "",
        "- **主动、先应下再说**，遇事不推脱，被拜托时反应很快。",
        "- **好奇、爱拍照**。相机和相册是她最重要的随身物，用照片记「开拓」的每一站。",
    ]
    lines += bullets(["拍照", "相机", "相册"], limit=2, hi=70)
    lines += [
        "",
        "- **护同伴**。会替人打圆场，也会为了朋友硬气起来。",
    ]
    lines += bullets(
        ["守护大家", "交给我们", "顾好自己"], limit=2, hi=70, first_person=True
    )
    lines += [
        "",
        "- **爱逞强、也认怂得快**。先喊「本姑娘」，被拆穿就自嘲。",
    ]
    lines += bullets(["本姑娘"], limit=2, hi=60, first_person=True)
    lines += [
        "",
        "- TODO：把上面几条收敛成 3–5 条可执行的行为准则（遇到什么情况 → 她怎么做），"
        "再和现有 QQbot 人设对齐。",
        "",
        "## 4. 关系网（频次与称呼见 `lore_entities.md`）",
        "",
        "| 对象 | 她怎么称呼 | 关系基调 |",
        "|---|---|---|",
        "| 开拓者 | 开拓者 | 主要对话对象，平等、直接、爱开玩笑 |",
        "| 姬子 | 姬子姐 / 姬子姐姐 | 领航员，敬慕又亲近 |",
        "| 瓦尔特 | 杨叔 | 调侃式亲近，偶尔拿他的过去打趣 |",
        "| 丹恒 | 丹恒 | 同车同伴，吐槽搭档 |",
        "| 帕姆 | 帕姆 / 列车长 | 亲昵，带宠溺 |",
        "| 云璃、彦卿 | 师父 | 剑术老师，正式拜师 |",
        "| 布洛妮娅、希儿、停云等 | 直呼其名或「小姐」 | 各站结识的朋友 |",
        "",
        "- TODO：逐条补一句「她怎么评价这个人」，例句从 `lore_entities.md` 与 `self_topics.md` 取。",
        "",
        "## 5. 素材索引",
        "",
        "| 文件 | 用途 |",
        "|---|---|",
        "| `corpus.jsonl` | 逐条结构化台词（含场景/分层/时长/标记），程序化取样与检索 |",
        "| `lines_by_scene.md` | 按剧情顺序通读全部台词 |",
        "| `scene_index.md` | 场景分布、置信度与代表台词 |",
        "| `style_stats.md` | 语言风格量化统计（句长、语气词、句末字、分层差异） |",
        "| `tier_samples.md` | 五种语气分层的代表台词，可直接做 few-shot 示例 |",
        "| `self_topics.md` | 她谈自己身世/能力/爱好的台词，设定章节的证据来源 |",
        "| `lore_entities.md` | 人物、地点提及频次、称呼方式与例句 |",
        "",
        "## 6. 待确认 / 待补",
        "",
        "- TODO：角色卡用于 QQbot 人设，还是仅作 TTS 配音的说话风格参考？用途决定详略。",
        "- TODO：是否沿用「三月七」本名与官方设定，还是只借声线与说话风格（角色扮演边界需先定）。",
        "- TODO：记忆体分支（sanyueqi 组，70 条）与战斗语音（49 条）是否纳入语气样本——"
        "记忆体是剧情里的另一个存在，人设与本尊不同，混用会串味。",
        "- TODO：与现有 `小鲸鱼` 人设的关系（替换 / 叠加 / 并列）。",
        "- TODO：语言风格要不要做成「规则 + 示例」两段式写进 system prompt，示例直接取自 `tier_samples.md`。",
        "",
    ]
    return "\n".join(lines)


def write_readme(rows):
    scenes = collections.Counter(r["scene"] for r in rows)
    lines = [
        "# role/ — 三月七角色卡素材包",
        "",
        "从已筛选的语音台词里整理出的角色卡原料。**台词是权威内容源**：",
        "`select/lists/full.list` + `val.list`（人工修正在 `select/edits.json` 里同步）。",
        "",
        "## 重新生成",
        "",
        "```powershell",
        "# 在 tts/ 目录",
        "& ..\\AstrBot\\.venv\\Scripts\\python.exe scripts\\build_role_corpus.py",
        "```",
        "",
        "只读 `select/`，不改任何清单或 CSV；审查页面继续编辑后重跑即可同步。",
        "",
        "## 文件",
        "",
        "| 文件 | 内容 |",
        "|---|---|",
        "| `角色卡草稿.md` | 角色卡骨架，结论附例句，TODO 标出需人工补的部分 |",
        "| `corpus.jsonl` | 逐条台词 + 场景/分层/时长/标记，供程序检索 |",
        "| `lines_by_scene.md` | 全部台词按场景分组（`※` 内心独白，`◇` 引用他人） |",
        "| `scene_index.md` | 场景索引：条数、分钟、分层分布、代表台词 |",
        "| `style_stats.md` | 句长、语气词、句末字、自称、称呼、分层差异 |",
        "| `tier_samples.md` | 五种语气分层各 15 条代表台词 |",
        "| `self_topics.md` | 她谈身世/能力/爱好/身份的台词（设定证据） |",
        "| `lore_entities.md` | 人物与地点的提及频次、称呼方式、例句 |",
        "",
        "## 已知边界",
        "",
        f"- 语料 {len(rows)} 条 / {sum(r['dur'] for r in rows) / 60:.1f} 分钟，覆盖 {len(scenes)} 个剧情场景。",
        "- 分层（`日常平静` 等）来自音频韵律聚类，**不是语义情绪标注**，只当语气/表演强度参考。",
        "- 场景归属由文件名来源组 + 台词关键词推断，`推测` 的两处（终章特别篇、车厢小剧场）需人工确认。",
        "- 台词是「角色说的话」，不含旁白与官方设定文本；设定类信息只有她自己提到的部分。",
        "- 语料来自游戏公开剧情语音，仅作个人研究与非商业角色扮演用途，版权归 miHoYo/HoYoverse。",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, meta, diffs = build_rows()
    if not rows:
        print("没有读到任何台词，检查 select/lists/*.list", file=sys.stderr)
        return 1
    print(f"读取 {len(rows)} 条台词（full+val）")

    write_corpus(rows)
    files = {
        "lines_by_scene.md": lines_by_scene(rows),
        "scene_index.md": scene_index(rows),
        "style_stats.md": style_stats(rows),
        "tier_samples.md": tier_samples(rows),
        "lore_entities.md": lore_entities(rows),
        "self_topics.md": self_topics(rows),
    }
    for name, content in files.items():
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write(content)
    with open(os.path.join(OUT, "角色卡草稿.md"), "w", encoding="utf-8") as f:
        f.write(write_draft(rows))
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write(write_readme(rows))

    scenes = collections.Counter(r["scene"] for r in rows)
    print(f"场景 {len(scenes)} 个：")
    for s, c in scenes.most_common():
        print(f"  {c:5d}  {s}")
    print(f"输出目录: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
