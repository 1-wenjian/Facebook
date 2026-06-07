"""推荐工程模块。

实现「三模块协作规范 v1.0」第 6 / 7 / 8 / 9 / 10 / 11 / 15 / 16 章。
输入：analyze_skin 标准输出 + 用户偏好 + 产品库
输出：标准推荐 JSON（含 profile_summary / recommendations / routine / fallback）
"""
import json
from typing import Iterable

from contracts import (
    SCHEMA_VERSION, ROUTINE_ORDER, ALLOWED_SKIN_TYPES, ALLOWED_VISIBLE_CONCERNS,
    STATUS_SUCCESS, STATUS_PARTIAL,
    ERR_REC_INVALID_INPUT, ERR_REC_NO_MATCH, ERR_REC_BUDGET_TOO_LOW,
    CONFLICT_RULES, SENSITIVE_TIPS,
    normalize_skin_type, normalize_concern, std_envelope, new_request_id,
)


def load_products(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return _clean_products(raw)


def _clean_products(raw: Iterable[dict]) -> list[dict]:
    """规范第 6.9 节：数据清洗。"""
    cleaned: list[dict] = []
    seen_ids: set[str] = set()
    for p in raw:
        pid = (p.get("product_id") or "").strip()
        if not pid or pid in seen_ids:
            continue
        try:
            price = float(p.get("price") or 0)
        except (TypeError, ValueError):
            price = 0.0
        if price <= 0:
            continue
        seen_ids.add(pid)
        cleaned.append({
            "product_id": pid,
            "name": p.get("name") or "",
            "brand": (p.get("brand") or "未知品牌").strip() or "未知品牌",
            "category": p.get("category") or "",
            "price": price,
            "volume": p.get("volume") or "",
            "key_ingredients": list(p.get("key_ingredients") or []),
            "suitable_skin_types": list(p.get("suitable_skin_types") or []),
            "target_concerns": list(p.get("target_concerns") or []),
            "contraindications": list(p.get("contraindications") or []),
            "usage_step": p.get("usage_step") or "",
            "usage_time": list(p.get("usage_time") or []),
            "rating": p.get("rating"),
            "description": p.get("description") or "",
        })
    return cleaned


def _build_profile(analysis: dict, preferences: dict) -> dict:
    """规范第 6.6 节：信息合并优先级（用户自述 > 模型估计）。"""
    user_skin_raw = (preferences.get("self_reported_skin_type") or "").strip()
    user_skin = normalize_skin_type(user_skin_raw) if user_skin_raw else ""
    model_skin = (analysis.get("skin_type_estimate") or {}).get("value", "不确定")

    if user_skin and user_skin in ALLOWED_SKIN_TYPES and user_skin != "不确定":
        base_skin_type = user_skin
        source = "user"
    else:
        base_skin_type = model_skin if model_skin in ALLOWED_SKIN_TYPES else "不确定"
        source = "model"

    user_sensitive = bool(preferences.get("sensitive", False))
    vs = analysis.get("visible_sensitivity") or {}
    model_sensitive = vs.get("detected") and (vs.get("confidence") or 0) >= 0.5
    sensitive = user_sensitive or model_sensitive

    skin_tags = [base_skin_type] if base_skin_type != "不确定" else []
    if sensitive:
        skin_tags.append("敏感性")

    concerns = analysis.get("concerns") or []
    primary_concerns = sorted(
        [{"name": c["name"], "severity": c.get("severity", 0),
          "confidence": float(c.get("confidence", 0.0))} for c in concerns],
        key=lambda x: (x["severity"], x["confidence"]),
        reverse=True,
    )

    return {
        "base_skin_type": base_skin_type,
        "skin_type_source": source,
        "skin_tags": skin_tags,
        "sensitive": sensitive,
        "primary_concerns": primary_concerns,
        "budget_total": float(preferences.get("budget_total") or 500),
        "allergies": [str(a) for a in (preferences.get("allergies") or [])],
        "excluded_brands": [str(b) for b in (preferences.get("excluded_brands") or [])],
        "preferred_brands": [str(b) for b in (preferences.get("preferred_brands") or [])],
        "routine_mode": preferences.get("routine_mode") or "标准",
    }


def _passes_hard_filters(product: dict, profile: dict) -> bool:
    """规范第 7.1 节：硬过滤。"""
    if product["brand"] in profile["excluded_brands"]:
        return False

    ing_text = " ".join(product["key_ingredients"])
    contra_text = " ".join(product["contraindications"])
    for allergen in profile["allergies"]:
        if allergen and (allergen in ing_text or allergen in contra_text):
            return False

    skin_type = profile["base_skin_type"]
    suitable = product["suitable_skin_types"]
    if skin_type != "不确定":
        if skin_type not in suitable and "多种肤质" not in suitable:
            return False
    return True


def _confidence_factor(confidence: float) -> float:
    """规范第 7.3 节"""
    if confidence >= 0.70:
        return 1.0
    if confidence >= 0.50:
        return 0.5
    return 0.0


def _concern_score(product: dict, concerns: list) -> tuple[float, list[str]]:
    """规范第 7.3 节：肌肤问题匹配（满分 45）。"""
    targets = set(product["target_concerns"])
    total_w, matched_w = 0.0, 0.0
    matched_names: list[str] = []
    for c in concerns:
        f = _confidence_factor(float(c.get("confidence", 0)))
        w = c.get("severity", 0) * f
        total_w += w
        if c["name"] in targets:
            matched_w += w
            matched_names.append(c["name"])
    if total_w == 0:
        return 0.0, matched_names
    return 45.0 * matched_w / total_w, matched_names


def _skin_type_score(product: dict, skin_type: str) -> float:
    """规范第 7.4 节（满分 20）"""
    suitable = product["suitable_skin_types"]
    if skin_type in suitable:
        return 20.0
    if "多种肤质" in suitable:
        return 14.0
    return 0.0


def _budget_score(price: float, total_budget: float) -> float:
    """规范第 7.5 节（满分 15）—— 按 4 件均价做参照"""
    if total_budget <= 0:
        return 0.0
    ideal = total_budget / 4
    if price <= ideal:
        return 15.0
    if price <= total_budget * 0.4:
        return 12.0
    if price <= total_budget * 0.7:
        return 6.0
    return 0.0


def _rating_score(product: dict) -> float:
    """规范第 7.6 节（满分 10）"""
    try:
        r = float(product.get("rating") or 0)
    except (TypeError, ValueError):
        r = 0.0
    return max(0.0, min(10.0, r / 5.0 * 10.0))


def _info_score(product: dict) -> float:
    """信息完整度（满分 5）"""
    score = 0.0
    if product["usage_step"]:
        score += 1.5
    if product["usage_time"]:
        score += 1.5
    if product["key_ingredients"]:
        score += 1.0
    if product["volume"]:
        score += 1.0
    return min(5.0, score)


def _calculate_score(product: dict, profile: dict) -> dict:
    """规范第 7.7 节：四层评分总分"""
    concerns = profile["primary_concerns"]
    cs, matched_concerns = _concern_score(product, concerns)
    sts = _skin_type_score(product, profile["base_skin_type"])
    bs = _budget_score(product["price"], profile["budget_total"])
    rs = _rating_score(product)
    info_s = _info_score(product)
    scene_s = 5.0 if (product["usage_step"] and product["usage_time"]) else 2.5
    pref_bonus = 3.0 if product["brand"] in profile["preferred_brands"] else 0.0

    total = cs + sts + bs + rs + info_s + scene_s + pref_bonus
    total = max(0.0, min(100.0, total))

    return {
        "match_score": round(total, 2),
        "score_breakdown": {
            "concern_match": round(cs, 2),
            "skin_type_match": round(sts, 2),
            "budget_match": round(bs, 2),
            "rating_score": round(rs, 2),
            "scene_match": round(scene_s, 2),
            "information_score": round(info_s, 2),
        },
        "matched_concerns": matched_concerns,
    }


def _step_bucket(product: dict) -> str | None:
    """归一到流程桶。"""
    step = (product["usage_step"] or "").strip()
    cat = (product["category"] or "").strip()
    for key in ROUTINE_ORDER:
        if key in step or key in cat:
            return key
    if "水" in step or "水" in cat:
        return "化妆水"
    if "乳" in step or "乳" in cat:
        return "乳液"
    return None


def _select_steps(routine_mode: str) -> list[str]:
    """规范第 8.2 / 8.3 节"""
    if routine_mode == "精简":
        return ["洁面", "面霜", "防晒"]
    return ["洁面", "化妆水", "精华", "面霜", "防晒"]


def _generate_reason(product: dict, scored: dict, profile: dict) -> str:
    """规范第 10.2 节模板"""
    parts: list[str] = []
    matched = scored["matched_concerns"]
    if matched:
        parts.append(f"可覆盖你当前关注的「{'、'.join(matched)}」问题")
    suitable = product["suitable_skin_types"]
    if profile["base_skin_type"] in suitable:
        parts.append(f"适合{profile['base_skin_type']}肤质")
    elif "多种肤质" in suitable:
        parts.append("适用多种肤质")
    if product["usage_step"]:
        parts.append(f"可用于{product['usage_step']}步骤")
    if not parts:
        parts.append("品类常见日常护理款")
    return "；".join(parts) + "。"


def _detect_conflicts(picked: list[dict]) -> list[str]:
    """规范第 11.2 节"""
    flat: list[tuple[str, str]] = []
    for p in picked:
        for ing in p["key_ingredients"]:
            flat.append((p["name"], ing))

    msgs: list[str] = []
    seen_messages: set[str] = set()
    for rule in CONFLICT_RULES:
        set_a, set_b = rule["ingredients"]
        hit_a = [(n, i) for n, i in flat if any(k in i for k in set_a)]
        hit_b = [(n, i) for n, i in flat if any(k in i for k in set_b)]
        if hit_a and hit_b and rule["message"] not in seen_messages:
            msgs.append(f"{rule['message']}（涉及：{hit_a[0][1]} × {hit_b[0][1]}）")
            seen_messages.add(rule["message"])
    return msgs


def _build_routines(picked: list[dict]) -> tuple[list, list]:
    morning, evening = [], []
    for p in picked:
        step = p["_step_bucket"]
        times = p.get("usage_time", [])
        in_morning = ("早" in times) or (not times) or step in ("洁面", "防晒", "化妆水", "乳液")
        in_evening = ("晚" in times) or (not times) or step in ("洁面", "化妆水", "精华", "面霜", "眼霜")
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
    morning_seq = [{"order": i + 1, "step": s, "product_id": p["product_id"]}
                   for i, (s, p) in enumerate(morning)]
    evening_seq = [{"order": i + 1, "step": s, "product_id": p["product_id"]}
                   for i, (s, p) in enumerate(evening)]
    return morning_seq, evening_seq


def _budget_adjust(picked_by_step: dict, profile: dict) -> tuple[list[dict], dict]:
    """规范第 9 节：超预算调整 + fallback"""
    budget = profile["budget_total"]
    selected_steps = _select_steps(profile["routine_mode"])

    candidate_picks = []
    for step in selected_steps:
        if step in picked_by_step and picked_by_step[step]:
            candidate_picks.append(picked_by_step[step][0])

    total = sum(p["price"] for p in candidate_picks)
    fallback = {"applied": False, "reason": ""}

    if total <= budget:
        return candidate_picks, fallback

    sorted_picks = sorted(candidate_picks, key=lambda p: p["price"], reverse=True)
    for over in sorted_picks:
        step = over["_step_bucket"]
        if step in ("洁面", "面霜", "防晒"):
            continue
        alternatives = picked_by_step.get(step, [])
        replacement = None
        for alt in alternatives[1:]:
            if alt["price"] < over["price"]:
                replacement = alt
                break
        if replacement:
            candidate_picks = [replacement if p["product_id"] == over["product_id"] else p
                               for p in candidate_picks]
            total = sum(p["price"] for p in candidate_picks)
            if total <= budget:
                fallback = {"applied": True, "reason": "已替换为同类目内更低价产品"}
                return candidate_picks, fallback

    final = [p for p in candidate_picks
             if p["_step_bucket"] in ("洁面", "面霜", "防晒")]
    total = sum(p["price"] for p in final)
    if total > budget:
        fallback = {"applied": True, "reason": "当前预算不足以从数据库中组成完整护肤方案"}
    else:
        fallback = {"applied": True, "reason": "已精简为洁面/保湿/防晒三件套以贴合预算"}
    return final, fallback


def recommend_products(analysis: dict, preferences: dict,
                       products: list[dict], request_id: str | None = None) -> dict:
    """规范第 6.5 / 14 / 15 节标准函数签名 + 标准输出。"""
    request_id = request_id or analysis.get("request_id") or new_request_id()

    if not isinstance(products, list) or not products:
        return _err_envelope(request_id, ERR_REC_INVALID_INPUT, "产品数据库为空")

    if analysis.get("status") != STATUS_SUCCESS:
        return _err_envelope(request_id, ERR_REC_INVALID_INPUT,
                             "肌肤分析未成功，不进行推荐")

    profile = _build_profile(analysis, preferences or {})

    # 硬过滤 + 评分
    scored: list[tuple[float, dict, dict]] = []
    by_step: dict[str, list[dict]] = {}
    for p in products:
        if not _passes_hard_filters(p, profile):
            continue
        bucket = _step_bucket(p)
        if not bucket:
            continue
        sc = _calculate_score(p, profile)
        scored.append((sc["match_score"], p, sc))
        enriched = {**p, "_step_bucket": bucket, "_score": sc}
        by_step.setdefault(bucket, []).append(enriched)

    if not scored:
        return _err_envelope(request_id, ERR_REC_NO_MATCH,
                             "未找到匹配产品，可放宽预算或减少限制后重试",
                             profile=profile)

    for step in by_step:
        by_step[step].sort(key=lambda x: x["_score"]["match_score"], reverse=True)

    picked, fallback = _budget_adjust(by_step, profile)

    if not picked:
        return _err_envelope(request_id, ERR_REC_BUDGET_TOO_LOW,
                             "预算过低，无法组成方案", profile=profile,
                             fallback=fallback)

    recommendations = []
    for p in picked:
        sc = p["_score"]
        reason = _generate_reason(p, sc, profile)
        warnings = list(p["contraindications"])
        if profile["sensitive"] and "首次使用建议进行局部耐受测试" not in warnings:
            warnings.append("首次使用建议进行局部耐受测试")
        recommendations.append({
            "product_id": p["product_id"],
            "name": p["name"],
            "brand": p["brand"],
            "category": p["category"],
            "usage_step": p["usage_step"],
            "price": p["price"],
            "volume": p["volume"],
            "rating": p["rating"],
            "match_score": sc["match_score"],
            "score_breakdown": sc["score_breakdown"],
            "matched_concerns": sc["matched_concerns"],
            "key_ingredients": p["key_ingredients"],
            "reason": reason,
            "usage_time": p["usage_time"],
            "warnings": warnings,
        })

    morning, evening = _build_routines(picked)
    total_price = round(sum(p["price"] for p in picked), 2)

    global_warnings = list(_detect_conflicts(picked))
    if profile["sensitive"]:
        global_warnings.extend(SENSITIVE_TIPS)

    status = STATUS_PARTIAL if fallback["applied"] else STATUS_SUCCESS
    env = std_envelope(request_id, status, error=None)
    env.update({
        "profile_summary": {
            "base_skin_type": profile["base_skin_type"],
            "skin_type_source": profile["skin_type_source"],
            "skin_tags": profile["skin_tags"],
            "primary_concerns": profile["primary_concerns"],
            "budget_total": profile["budget_total"],
        },
        "recommendations": recommendations,
        "routine": {"morning": morning, "evening": evening},
        "total_price": total_price,
        "global_warnings": global_warnings,
        "fallback": fallback,
    })
    return env


def _err_envelope(request_id: str, code: str, message: str,
                  profile: dict | None = None,
                  fallback: dict | None = None) -> dict:
    env = std_envelope(request_id, "partial" if code == ERR_REC_BUDGET_TOO_LOW else "error",
                       error={"code": code, "message": message})
    env.update({
        "profile_summary": {
            "base_skin_type": (profile or {}).get("base_skin_type", "不确定"),
            "skin_type_source": (profile or {}).get("skin_type_source", "model"),
            "skin_tags": (profile or {}).get("skin_tags", []),
            "primary_concerns": (profile or {}).get("primary_concerns", []),
            "budget_total": (profile or {}).get("budget_total", 0),
        },
        "recommendations": [],
        "routine": {"morning": [], "evening": []},
        "total_price": 0.0,
        "global_warnings": [],
        "fallback": fallback or {"applied": True, "reason": message},
    })
    return env
