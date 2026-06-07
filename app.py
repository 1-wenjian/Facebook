"""肌肤分析与护肤产品推荐助手 · Gradio 应用入口。

对应「三模块协作规范 v1.0」第 4 章 前端模块。
"""
import io
import base64
import os
import gradio as gr
from PIL import Image

from analyzer import analyze_skin
from contracts import (
    MEDICAL_DISCLAIMER, ROUTINE_MODES, ALLOWED_SKIN_TYPES,
    STATUS_SUCCESS, STATUS_PARTIAL, STATUS_INVALID_IMAGE,
    new_request_id,
)
from recommender import load_products, recommend_products

PRODUCTS = load_products("products.json")
PRODUCT_INDEX = {p["product_id"]: p for p in PRODUCTS}

TITLE = "🪞 肌肤分析与护肤产品推荐助手"
INTRO = (
    "上传一张清晰的面部正面照，AI 会识别可见的肌肤状态，"
    f"并从京东 {len(PRODUCTS)} 款护肤品中为你拼一套早晚护肤流程。"
)

SKIN_TYPE_CHOICES = ["（让 AI 判断）"] + [t for t in ALLOWED_SKIN_TYPES if t != "不确定"]


def _to_data_url(pil_img: Image.Image) -> str:
    img = pil_img if pil_img.mode == "RGB" else pil_img.convert("RGB")
    if max(img.size) > 1280:
        img = img.copy()
        img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _bad_image_message(analysis: dict) -> str:
    err = analysis.get("error") or {}
    msg = err.get("message", "图片不适合分析")
    issues = (analysis.get("image_quality") or {}).get("issues") or []
    extra = "\n".join(f"- {i}" for i in issues) if issues else ""
    return (
        f"### ⚠️ {msg}\n\n"
        "为了得到准确的判断，请上传：\n"
        "- 一张正面、自然光下的面部照片\n"
        "- 单人、无重度妆容、无明显遮挡\n"
        "- 距离适中、对焦清晰\n"
        + (f"\n模型检测到的问题：\n{extra}\n" if extra else "")
    )


def _format_report(analysis: dict, profile_summary: dict) -> str:
    sev_label = {0: "未检测", 1: "轻度", 2: "中度", 3: "明显"}
    lines = ["### 📋 肌肤分析报告", ""]
    src = "你自填" if profile_summary["skin_type_source"] == "user" else "AI 判断"
    lines.append(f"- **肤质**：{profile_summary['base_skin_type']} _（{src}）_")

    ste = analysis.get("skin_type_estimate") or {}
    if profile_summary["skin_type_source"] == "user" and ste.get("value"):
        lines.append(f"  - 模型估计：{ste['value']}（置信度 {float(ste.get('confidence',0)):.0%}）")

    tags = profile_summary.get("skin_tags") or []
    if tags:
        lines.append(f"- **标签**：{ ' · '.join(tags) }")

    concerns = profile_summary.get("primary_concerns") or []
    if not concerns:
        lines.append("- **可见问题**：未识别明显问题")
    else:
        lines.append("- **可见问题**：")
        for c in concerns:
            sev = sev_label.get(c.get("severity", 0), str(c.get("severity")))
            conf = float(c.get("confidence", 0))
            lines.append(f"  - {c['name']} · {sev}（置信度 {conf:.0%}）")

    vs = analysis.get("visible_sensitivity") or {}
    if vs.get("detected"):
        lines.append(f"- **敏感迹象**：检测到（{sev_label.get(vs.get('severity', 0))}，置信度 {float(vs.get('confidence',0)):.0%}）")

    sf = analysis.get("safety_flags") or {}
    if sf.get("recommend_medical_consultation"):
        reasons = "；".join(sf.get("reasons") or []) or "建议就医评估"
        lines.append(f"\n> ⚕️ **建议就医**：{reasons}")

    summary = analysis.get("summary", "")
    if summary:
        lines.append("")
        lines.append(f"> {summary}")

    iq = analysis.get("image_quality") or {}
    issues = iq.get("issues") or []
    if issues:
        lines.append(f"\n> ⚠️ 图片质量提示：{ '；'.join(issues) }，结果可能不准。")
    return "\n".join(lines)


def _format_picks(recommendations: list, fallback: dict, total_price: float, budget: float) -> str:
    if not recommendations:
        reason = (fallback or {}).get("reason") or "暂无匹配产品"
        return f"### 🛍️ 推荐产品\n\n_{reason}_"

    lines = ["### 🛍️ 为你挑选的产品", ""]
    if fallback and fallback.get("applied"):
        lines.append(f"> ⚠️ {fallback.get('reason')}\n")

    lines.append(f"**总价**：¥{total_price:.2f} / 预算 ¥{budget:.0f}\n")

    for r in recommendations:
        sb = r["score_breakdown"]
        time_label = "、".join(r.get("usage_time") or []) or "随时"
        ings = "、".join(r.get("key_ingredients") or []) or "（未公开）"
        warnings_md = ""
        if r.get("warnings"):
            warnings_md = "\n  - ⚠️ " + "；".join(r["warnings"])
        lines.append(
            f"**【{r['usage_step'] or r['category']}】{r['name']}**  \n"
            f"- 品牌：{r['brand']} · 价格：¥{r['price']} · 评分：{r.get('rating', '-')}/5 · 使用：{time_label}\n"
            f"- 核心成分：{ings}\n"
            f"- **匹配分**：{r['match_score']}/100"
            f"（问题 {sb['concern_match']} + 肤质 {sb['skin_type_match']} + 预算 {sb['budget_match']}"
            f" + 评分 {sb['rating_score']} + 场景 {sb['scene_match']} + 信息 {sb['information_score']}）\n"
            f"- 推荐理由：{r['reason']}{warnings_md}\n"
        )
    return "\n".join(lines)


def _format_routines(routine: dict) -> str:
    def _render(seq: list, title: str) -> str:
        if not seq:
            return f"### {title}\n_（无）_"
        rows = [f"### {title}"]
        for item in seq:
            p = PRODUCT_INDEX.get(item["product_id"], {})
            name = p.get("name", "")[:30]
            brand = p.get("brand", "")
            rows.append(f"{item['order']}. **{item['step']}** — {brand} · {name}")
        return "\n".join(rows)
    return _render(routine.get("morning", []), "🌅 早间流程") + "\n\n" + _render(routine.get("evening", []), "🌙 晚间流程")


def _format_warnings(global_warnings: list) -> str:
    if not global_warnings:
        return "### ⚠️ 成分搭配与注意事项\n\n_无特别提示。_"
    lines = ["### ⚠️ 成分搭配与注意事项", ""]
    lines.extend(f"- {w}" for w in global_warnings)
    return "\n".join(lines)


def run(image, skin_type_pick, sensitive, age_band,
        budget_total, allergens, excluded_brands, routine_mode):
    if image is None:
        return "请先上传一张面部照片。", "", "", ""

    request_id = new_request_id()

    try:
        data_url = _to_data_url(image)
    except Exception as e:
        return f"❌ 图片处理失败：{e}", "", "", ""

    analysis = analyze_skin(data_url, request_id=request_id)

    if analysis["status"] == STATUS_INVALID_IMAGE:
        return _bad_image_message(analysis), "", "", ""

    if analysis["status"] != STATUS_SUCCESS:
        err = analysis.get("error") or {}
        return f"❌ 分析失败（{err.get('code','')}）：{err.get('message','请稍后重试')}", "", "", ""

    preferences = {
        "self_reported_skin_type": "" if skin_type_pick == SKIN_TYPE_CHOICES[0] else skin_type_pick,
        "sensitive": bool(sensitive),
        "budget_total": float(budget_total or 500),
        "allergies": [a.strip() for a in (allergens or "").replace("，", ",").split(",") if a.strip()],
        "excluded_brands": [b.strip() for b in (excluded_brands or "").replace("，", ",").split(",") if b.strip()],
        "routine_mode": routine_mode or "标准",
    }

    rec = recommend_products(analysis, preferences, PRODUCTS, request_id=request_id)

    if rec["status"] not in (STATUS_SUCCESS, STATUS_PARTIAL):
        err = rec.get("error") or {}
        return (_format_report(analysis, rec["profile_summary"]),
                f"❌ 推荐失败（{err.get('code','')}）：{err.get('message','')}",
                "", "")

    report_md = _format_report(analysis, rec["profile_summary"])
    if age_band:
        report_md += f"\n- **年龄段**：{age_band}（仅参考，不参与肌肤判断）"

    picks_md = _format_picks(rec["recommendations"], rec.get("fallback"),
                             rec["total_price"], preferences["budget_total"])
    routine_md = _format_routines(rec["routine"])
    warn_md = _format_warnings(rec["global_warnings"])

    return report_md, picks_md, routine_md, warn_md


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="上传面部照片", type="pil", height=300)
                skin_type_pick = gr.Dropdown(
                    label="你的肤质（自述优先于 AI 判断）",
                    choices=SKIN_TYPE_CHOICES,
                    value=SKIN_TYPE_CHOICES[0],
                )
                sensitive = gr.Checkbox(label="敏感肌（自述）", value=False)
                age_band = gr.Dropdown(
                    label="年龄段（可选）",
                    choices=["", "18 以下", "18-25", "26-35", "36-45", "46+"],
                    value="",
                )
                budget_total = gr.Number(label="整套预算（¥）", value=500, precision=0)
                allergens = gr.Textbox(
                    label="已知过敏成分（逗号分隔）",
                    placeholder="例如：酒精, 香精, 视黄醇",
                )
                excluded_brands = gr.Textbox(
                    label="排除品牌（逗号分隔）",
                    placeholder="例如：某品牌",
                )
                routine_mode = gr.Radio(
                    label="方案复杂度",
                    choices=ROUTINE_MODES,
                    value="标准",
                )
                submit = gr.Button("🔍 开始分析并推荐", variant="primary")
            with gr.Column(scale=2):
                with gr.Tab("肌肤报告"):
                    report_out = gr.Markdown()
                with gr.Tab("推荐产品"):
                    picks_out = gr.Markdown()
                with gr.Tab("早晚流程"):
                    routine_out = gr.Markdown()
                with gr.Tab("成分提示"):
                    warn_out = gr.Markdown()

        submit.click(
            run,
            inputs=[image, skin_type_pick, sensitive, age_band,
                    budget_total, allergens, excluded_brands, routine_mode],
            outputs=[report_out, picks_out, routine_out, warn_out],
        )

        gr.Markdown("---\n> ⚕️ **免责声明**：" + MEDICAL_DISCLAIMER)
        gr.Markdown(
            "**产品边界**：本工具仅做日常护肤产品推荐与流程拼装，"
            "**不识别医疗级皮肤疾病**（湿疹、玫瑰痤疮、银屑病等），"
            "**不替代皮肤科医生诊断**；对成人、面部、单人、自然光照片效果最好。"
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
