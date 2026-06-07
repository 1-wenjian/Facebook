"""规则驱动的产品推荐 + 早晚流程 + 成分冲突检测。"""
import json
from collections import defaultdict

# 早 / 晚 流程使用顺序
ROUTINE_ORDER = ["洁面", "化妆水", "精华", "乳液", "面霜", "眼霜", "防晒"]

# 已知成分搭配冲突 / 注意事项
CONFLICT_PAIRS = [
    (
        ["视黄醇", "A醇", "Retinol", "维A"],
        ["果酸", "水杨酸", "AHA", "BHA", "乙醇酸", "杏仁酸"],
        "视黄醇/A醇 与 酸类（AHA/BHA）同时使用易刺激，建议错峰（早晚分开或隔天交替）",
    ),
    (
        ["视黄醇", "A醇", "Retinol"],
        ["维生素C", "VC", "抗坏血酸"],
        "视黄醇与维生素C建议早晚分用：VC 早间抗氧化、视黄醇晚间修护",
    ),
    (
        ["维生素C", "VC", "抗坏血酸"],
        ["烟酰胺"],
        "高浓度 VC 与高浓度烟酰胺同时使用部分敏感肌会刺痛/泛红，建议错峰试用",
    ),
]


def load_products(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_any(haystack_list, needles):
    for h in haystack_list or []:
        for n in needles:
            if n in h:
                return True
    return False


def _allergy_safe(product, allergens):
    if not allergens:
        return True
    ing_text = " ".join(product.get("key_ingredients", []))
    contra_text = " ".join(product.get("contraindications", []))
    for a in allergens:
        if a and (a in ing_text or a in contra_text):
            return False
    return True


def _score(product, concern_types, skin_type, budget):
    score = 0.0
    targets = product.get("target_concerns", []) or []
    # 命中肌肤问题
    for ct in concern_types:
        if ct in targets:
            score += 3
        else:
            for t in targets:
                if ct and (ct in t or t in ct):
                    score += 1.5
                    break
    # 肤质匹配
    sts = product.get("suitable_skin_types", []) or []
    if skin_type and skin_type != "未知":
        if skin_type in sts:
            score += 2
        elif "多种肤质" in sts:
            score += 1
    elif "多种肤质" in sts:
        score += 0.5
    # 评分加权
    rating = product.get("rating") or 0
    try:
        score += float(rating) * 0.4
    except (TypeError, ValueError):
        pass
    # 预算硬过滤
    try:
        price = float(product.get("price") or 0)
        if budget and price > budget * 1.2:
            score -= 5  # 超预算严重降权
        elif budget and price > budget:
            score -= 2
    except (TypeError, ValueError):
        pass
    return score


def _step_bucket(product):
    """把 usage_step 归一到流程桶。"""
    step = (product.get("usage_step") or "").strip()
    cat = (product.get("category") or "").strip()
    for key in ROUTINE_ORDER:
        if key in step or key in cat:
            return key
    if "水" in step or "水" in cat:
        return "化妆水"
    if "乳" in step or "乳" in cat:
        return "乳液"
    return None


def recommend(products, concerns, skin_type, budget, allergens):
    """返回 [(step, product), ...]，按 ROUTINE_ORDER 排好序。"""
    concern_types = [c.get("type", "") for c in concerns if c.get("type")]
    # 若模型没识别出问题，给"日常护理"兜底
    if not concern_types:
        concern_types = ["日常护理"]

    by_step = defaultdict(list)
    for p in products:
        if not _allergy_safe(p, allergens):
            continue
        bucket = _step_bucket(p)
        if not bucket:
            continue
        s = _score(p, concern_types, skin_type, budget)
        by_step[bucket].append((s, p))

    picks = []
    for step in ROUTINE_ORDER:
        candidates = by_step.get(step) or []
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_p = candidates[0]
        if top_score <= -3:
            continue
        picks.append((step, top_p))
    return picks


def detect_conflicts(products):
    ingredients_flat = []
    for p in products:
        for ing in p.get("key_ingredients", []) or []:
            ingredients_flat.append((p["name"], ing))

    msgs = []
    for set_a, set_b, msg in CONFLICT_PAIRS:
        hit_a = [(n, ing) for n, ing in ingredients_flat
                 if any(k in ing for k in set_a)]
        hit_b = [(n, ing) for n, ing in ingredients_flat
                 if any(k in ing for k in set_b)]
        if hit_a and hit_b:
            msgs.append(f"{msg}（涉及：{hit_a[0][1]} × {hit_b[0][1]}）")
    return msgs


def build_routines(picks):
    morning, evening = [], []
    for step, p in picks:
        times = p.get("usage_time", []) or []
        in_morning = "早" in times or not times or step in ("洁面", "防晒", "化妆水", "乳液")
        in_evening = "晚" in times or not times or step in ("洁面", "化妆水", "精华", "面霜", "眼霜")
        # 防晒只在早间
        if step == "防晒":
            in_evening = False
        if in_morning:
            morning.append((step, p))
        if in_evening:
            evening.append((step, p))

    def _key(item):
        try:
            return ROUTINE_ORDER.index(item[0])
        except ValueError:
            return 99
    morning.sort(key=_key)
    evening.sort(key=_key)
    return morning, evening


def explain_pick(product, concerns, skin_type):
    targets = product.get("target_concerns", []) or []
    sts = product.get("suitable_skin_types", []) or []
    hit_concerns = [c.get("type") for c in concerns
                    if c.get("type") and (c.get("type") in targets
                                          or any(c["type"] in t or t in c["type"] for t in targets))]
    parts = []
    if hit_concerns:
        parts.append(f"针对你的「{ '、'.join(dict.fromkeys(hit_concerns)) }」问题")
    if skin_type and skin_type != "未知" and (skin_type in sts or "多种肤质" in sts):
        parts.append(f"适用于{skin_type}")
    ings = product.get("key_ingredients", []) or []
    if ings:
        parts.append(f"核心成分含 {ings[0]}")
    if not parts:
        parts.append("品类常见日常护理款")
    return "；".join(parts) + "。"


def format_report(report, final_skin_type):
    concerns = report.get("concerns", [])
    sev_label = {"mild": "轻度", "moderate": "中度", "severe": "重度"}
    lines = [f"### 📋 肌肤分析报告", ""]
    lines.append(f"- **肤质判断**：{final_skin_type}")
    if not concerns:
        lines.append("- **可见问题**：暂未识别明显问题，状态整体良好。")
    else:
        lines.append("- **可见问题**：")
        for c in concerns:
            sev = sev_label.get(c.get("severity", ""), c.get("severity", ""))
            area = c.get("area", "")
            t = c.get("type", "")
            tail = f"（{area}）" if area else ""
            lines.append(f"  - {t} · {sev}{tail}")
    summary = report.get("summary", "")
    if summary:
        lines.append("")
        lines.append(f"> {summary}")
    return "\n".join(lines)


def render_routine(steps):
    if not steps:
        return "_（无）_"
    rows = []
    for i, (step, p) in enumerate(steps, 1):
        rows.append(f"{i}. **{step}** — {p['brand']} · {p['name'][:30]}")
    return "\n".join(rows)
