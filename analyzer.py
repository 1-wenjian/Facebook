"""Qwen3-VL 视觉分析：把面部照片转成结构化肌肤报告。

环境变量：
    SKIN_API_KEY       必填，魔搭个人访问令牌（部署在魔搭 Studio 用这个名）
    MS_API_KEY         兼容
    MODELSCOPE_API_KEY 兼容（本地开发）
    SKIN_MODEL         可选，默认 Qwen/Qwen3-VL-30B-A3B-Instruct
"""
import os
import json
import re
import requests

API_BASE = "https://api-inference.modelscope.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
TIMEOUT = 60

CONCERN_VOCAB = [
    "干燥缺水", "松弛", "细纹", "屏障受损", "色斑", "暗沉",
    "出油", "泛红敏感", "痘痘", "黑头", "毛孔粗大", "痘印", "闭口",
]
SKIN_TYPE_VOCAB = ["干性", "油性", "混合性", "中性", "敏感性", "未知"]

SYSTEM_PROMPT = f"""你是一位经验丰富的护肤顾问，擅长从面部照片中识别常见肌肤状况。
你只能输出严格的 JSON，不要任何解释性文字、不要 Markdown 代码块包裹。

JSON 字段定义：
- is_face: bool，照片中是否是单人、清晰可见的人脸正面
- image_quality: 字符串，从以下取值：good / blurry / low_light / multiple_faces / not_face
- reason: 字符串，如果 is_face 为 false 或图片质量异常，简短说明原因；否则空串
- skin_type: 字符串，从以下取值之一：{" / ".join(SKIN_TYPE_VOCAB)}
- concerns: 数组，每项 {{ "type": <下列词之一>, "severity": "mild" / "moderate" / "severe", "area": "额头" / "脸颊" / "下巴" / "鼻子" / "T区" / "全脸" }}
  type 必须从以下词表中选：{" / ".join(CONCERN_VOCAB)}
- summary: 字符串，用 1-2 句中文总结整体肌肤状态

请遵循：
1. 严格基于照片可见信息判断，避免过度推断。
2. 若无法清晰判断肤质，skin_type 用 "未知"。
3. concerns 只列出图像中能观察到的，可以为空数组。
4. 不要做任何医疗诊断、不要点评长相或种族。
"""

USER_PROMPT = "请分析这张照片，按 system 中定义的 JSON 格式返回肌肤状态报告。直接输出 JSON。"


class AnalyzeError(Exception):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise AnalyzeError(f"模型未返回 JSON：{text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise AnalyzeError(f"JSON 解析失败：{e}；原文：{text[:200]}")


def analyze_skin(image_data_url: str) -> dict:
    api_key = (
        os.environ.get("SKIN_API_KEY")
        or os.environ.get("MS_API_KEY")
        or os.environ.get("MODELSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise AnalyzeError("未配置 SKIN_API_KEY 环境变量")

    model = (
        os.environ.get("SKIN_MODEL")
        or os.environ.get("MS_MODEL")
        or os.environ.get("MODELSCOPE_MODEL")
        or DEFAULT_MODEL
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        r = requests.post(API_BASE, json=payload, headers=headers, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise AnalyzeError(f"网络请求失败：{e}")

    if r.status_code != 200:
        raise AnalyzeError(f"API 返回 {r.status_code}：{r.text[:300]}")

    try:
        body = r.json()
        content = body["choices"][0]["message"]["content"]
    except (KeyError, ValueError) as e:
        raise AnalyzeError(f"返回结构异常：{e}；原文：{r.text[:200]}")

    report = _extract_json(content)
    report.setdefault("is_face", False)
    report.setdefault("image_quality", "good")
    report.setdefault("skin_type", "未知")
    report.setdefault("concerns", [])
    report.setdefault("summary", "")
    report.setdefault("reason", "")
    return report
