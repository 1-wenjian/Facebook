"""肌肤分析与护肤产品推荐助手 · Gradio 应用入口。"""
import io
import base64
import os
import gradio as gr
from PIL import Image

from analyzer import analyze_skin, AnalyzeError
from recommender import (
    load_products, recommend, detect_conflicts, build_routines,
    explain_pick, format_report, render_routine,
)

PRODUCTS = load_products("products.json")

TITLE = "🪞 肌肤分析与护肤产品推荐助手"
INTRO = (
    "上传一张清晰的面部正面照，AI 会识别可见的肌肤状态，"
    "并从京东 297 款护肤品中为你拼一套早晚护肤流程。"
)

DISCLAIMER = (
    "> ⚕️ **免责声明**：本工具基于多模态 AI 对照片的视觉判断，仅供日常护肤参考，"
    "**不构成医疗诊断或治疗建议**。如出现持续性痤疮、皮疹、红肿、过敏等问题，"
    "请及时咨询专业皮肤科医生。推荐产品来自京东公开商品数据，使用前请先在耳后或手腕做敏感测试。"
)

SKIN_TYPE_CHOICES = ["（让 AI 判断）", "干性", "油性", "混合性", "中性", "敏感性"]


def _to_data_url(pil_img: Image.Image) -> str:
    img = pil_img
    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > 1280:
        img = img.copy()
        img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _bad_photo_message(report):
    return (
        "### ⚠️ 这张图似乎不适合做肌肤分析\n\n"
        "为了得到准确的判断，请上传：\n"
        "- 一张正面、自然光下的面部照片\n"
        "- 单人、无重度妆容、无明显遮挡\n"
        "- 距离适中、对焦清晰\n\n"
        f"模型补充说明：{report.get('reason', '（无）')}"
    )


def run(image, skin_type_pick, age_band, budget, allergens):
    if image is None:
        return "请先上传一张面部照片。", "", "", ""

    try:
        data_url = _to_data_url(image)
        report = analyze_skin(data_url)
    except AnalyzeError as e:
        return f"❌ 分析失败：{e}", "", "", ""
    except Exception as e:
        return f"❌ 系统异常：{type(e).__name__}: {e}", "", "", ""

    if not report.get("is_face"):
        return _bad_photo_message(report), "", "", ""

    iq = report.get("image_quality", "good")
    quality_warn = ""
    if iq and iq != "good":
        quality_warn = f"\n\n> ⚠️ 图片质量提示：`{iq}`，分析结果可能不准确，建议重拍。"

    final_skin = (
        skin_type_pick
        if skin_type_pick and skin_type_pick != SKIN_TYPE_CHOICES[0]
        else report.get("skin_type", "未知")
    )

    allergen_list = []
    if allergens:
        for piece in allergens.replace("，", ",").split(","):
            piece = piece.strip()
            if piece:
                allergen_list.append(piece)

    try:
        budget_val = float(budget) if budget else 0
    except (TypeError, ValueError):
        budget_val = 0

    picks = recommend(PRODUCTS, report.get("concerns", []), final_skin, budget_val, allergen_list)
    conflicts = detect_conflicts([p for _s, p in picks])
    morning, evening = build_routines(picks)

    report_md = format_report(report, final_skin) + quality_warn
    if age_band:
        report_md += f"\n- **年龄段**：{age_band}（仅作为推荐参考，不参与肌肤判断）"

    picks_md = ["### 🛍️ 为你挑选的产品", ""]
    if not picks:
        picks_md.append("_当前预算/过敏成分约束下没有完全匹配的产品。可以放宽预算或减少过敏成分后再试。_")
    else:
        for step, p in picks:
            why = explain_pick(p, report.get("concerns", []), final_skin)
            ings = p.get("key_ingredients", []) or []
            picks_md.append(
                f"**【{step}】{p['name']}**  \n"
                f"- 品牌：{p['brand']} · 价格：¥{p.get('price','-')} · 评分：{p.get('rating','-')}/5\n"
                f"- 核心成分：{'、'.join(ings) if ings else '（未公开）'}\n"
                f"- 推荐理由：{why}\n"
            )

    routine_md = (
        "### 🌅 早间流程\n" + render_routine(morning)
        + "\n\n### 🌙 晚间流程\n" + render_routine(evening)
    )

    conflict_md = ""
    if conflicts:
        conflict_md = "### ⚠️ 成分搭配提示\n" + "\n".join(f"- {c}" for c in conflicts)

    return report_md, "\n".join(picks_md), routine_md, conflict_md


def build_ui():
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {TITLE}\n{INTRO}")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="上传面部照片", type="pil", height=320, sources=["upload", "webcam"])
                skin_type_pick = gr.Dropdown(
                    label="你的肤质（可选）",
                    choices=SKIN_TYPE_CHOICES,
                    value=SKIN_TYPE_CHOICES[0],
                )
                age_band = gr.Dropdown(
                    label="年龄段（可选）",
                    choices=["", "18 以下", "18-25", "26-35", "36-45", "46+"],
                    value="",
                )
                budget = gr.Number(label="单件预算上限（¥，可选）", value=300, precision=0)
                allergens = gr.Textbox(
                    label="已知过敏 / 不想用的成分（逗号分隔，可选）",
                    placeholder="例如：酒精, 香精, 视黄醇",
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
                    conflict_out = gr.Markdown()

        submit.click(
            run,
            inputs=[image, skin_type_pick, age_band, budget, allergens],
            outputs=[report_out, picks_out, routine_out, conflict_out],
        )

        gr.Markdown("---\n" + DISCLAIMER)
        gr.Markdown(
            "**产品取舍说明**：本工具仅做日常护肤产品推荐与流程拼装，"
            "不识别医疗级皮肤疾病（如银屑病、玫瑰痤疮、湿疹等），"
            "不替代皮肤科医生诊断；对成人、面部、单人、自然光照片效果最好。"
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
