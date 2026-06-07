"""Qwen3-VL 视觉分析模块。

实现「三模块协作规范 v1.0」第 5 章 大模型识别模块。
输出严格 JSON，符合第 5.8 / 5.9 节的标准结构。

环境变量：
    SKIN_API_KEY        必填，魔搭个人访问令牌
    MS_API_KEY          兼容
    MODELSCOPE_API_KEY  兼容（本地开发）
    SKIN_MODEL          可选，默认 Qwen/Qwen3-VL-30B-A3B-Instruct
"""
import os
import json
import re
import requests

from contracts import (
    SCHEMA_VERSION, ALLOWED_SKIN_TYPES, ALLOWED_VISIBLE_CONCERNS,
    STATUS_SUCCESS, STATUS_INVALID_IMAGE, STATUS_ERROR, STATUS_TIMEOUT,
    ERR_IMG_NO_FACE, ERR_IMG_MULTI_FACE, ERR_IMG_BLUR, ERR_IMG_TOO_DARK,
    ERR_MODEL_TIMEOUT, ERR_MODEL_BAD_JSON, ERR_MODEL_API_ERROR,
    normalize_skin_type, normalize_concern, new_request_id, std_envelope,
)

API_BASE = "https://api-inference.modelscope.cn/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
TIMEOUT = 60
MAX_RETRY = 1  # 第 4.5 节：超时允许重试一次

SYSTEM_PROMPT = f"""你是一位经验丰富的护肤顾问，擅长从面部照片中识别常见肌肤状况。
你**只能**输出严格的 JSON，不要任何解释性文字、不要 Markdown 代码块包裹。

## 输出字段定义

- image_quality.usable: bool — 是否为可分析的清晰单人面部正面照
- image_quality.face_count: int — 检测到的人脸数量
- image_quality.quality_score: float 0~1 — 图像质量评分
- image_quality.issues: array[string] — 问题列表，取值：未检测到清晰人脸 / 多张人脸 / 图片模糊 / 图片过暗 / 面部遮挡严重
- skin_type_estimate.value: string — 从词表中选：{" / ".join(ALLOWED_SKIN_TYPES)}
- skin_type_estimate.confidence: float 0~1
- concerns: array，每项：
  - name: 从词表中选：{" / ".join(ALLOWED_VISIBLE_CONCERNS)}
  - severity: int 0~3（0=未检测到，1=轻度，2=中度，3=明显）
  - confidence: float 0~1
  - regions: array[string]，从「额头 / 脸颊 / 下巴 / 鼻子 / 鼻翼 / T区 / 全脸」中选
  - evidence: string — 简短证据描述（10-30 字）
- visible_sensitivity.detected: bool
- visible_sensitivity.severity: int 0~3
- visible_sensitivity.confidence: float 0~1
- visible_sensitivity.regions: array[string]
- safety_flags.recommend_medical_consultation: bool — 是否建议就医（仅在看到明显病变迹象时为 true）
- safety_flags.reasons: array[string]
- summary: string — 1~2 句中文整体总结

## 严格规则

1. **不输出 Markdown**、不输出代码块。
2. **不推荐产品**，不提及任何品牌/商品名称。
3. **不诊断疾病**，不使用"湿疹/玫瑰痤疮/银屑病/皮炎"等医学术语。
4. 肤质和肌肤问题**必须**来自上述固定词表，不得自创"混油"等自由文本。
5. 不确定时降低 confidence，不得强行判断；可以输出空 concerns 数组。
6. 图片不合格时（非人脸/多人/模糊/过暗）：image_quality.usable=false，concerns=[]，summary=""。
7. 严重程度（severity）必须为整数 0/1/2/3，不要用字符串。
"""

USER_PROMPT = "请按 system 中定义的 JSON 格式分析这张照片，输出严格 JSON。"


class AnalyzeError(Exception):
    """analyzer 内部异常（封装后转 std envelope 返回）"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    # 去除可能的 ```json ... ``` 包裹
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise AnalyzeError(f"模型未返回 JSON：{text[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise AnalyzeError(f"JSON 解析失败：{e}；原文：{text[:200]}")


def _normalize_severity(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return max(0, min(3, int(v)))
    if isinstance(v, str):
        mapping = {"mild": 1, "moderate": 2, "severe": 3,
                   "轻度": 1, "中度": 2, "明显": 3, "重度": 3}
        return mapping.get(v.strip(), 0)
    return 0


def _normalize_concerns(raw_concerns) -> list:
    out = []
    for c in raw_concerns or []:
        if not isinstance(c, dict):
            continue
        name = normalize_concern(c.get("name") or c.get("type") or "")
        if name not in ALLOWED_VISIBLE_CONCERNS:
            continue
        out.append({
            "name": name,
            "severity": _normalize_severity(c.get("severity")),
            "confidence": float(c.get("confidence") or 0.5),
            "regions": [str(r) for r in (c.get("regions") or [c.get("area")] if c.get("area") else c.get("regions") or [])],
            "evidence": str(c.get("evidence") or "")[:120],
        })
    return out


def _call_api(payload: dict, headers: dict, attempt: int = 0) -> dict:
    try:
        r = requests.post(API_BASE, json=payload, headers=headers, timeout=TIMEOUT)
    except requests.Timeout:
        if attempt < MAX_RETRY:
            return _call_api(payload, headers, attempt + 1)
        raise AnalyzeError("API 超时")
    except requests.RequestException as e:
        raise AnalyzeError(f"网络请求失败：{e}")

    if r.status_code != 200:
        raise AnalyzeError(f"API 返回 {r.status_code}：{r.text[:300]}")
    try:
        body = r.json()
        return {"content": body["choices"][0]["message"]["content"], "raw": body}
    except (KeyError, ValueError, TypeError) as e:
        raise AnalyzeError(f"返回结构异常：{e}；原文：{r.text[:200]}")


def analyze_skin(image_data_url: str, request_id: str | None = None) -> dict:
    """对应规范第 5.7 节标准函数签名（image_path 这里改用 data_url 以适配 Gradio）。"""
    request_id = request_id or new_request_id()

    api_key = (
        os.environ.get("SKIN_API_KEY")
        or os.environ.get("MS_API_KEY")
        or os.environ.get("MODELSCOPE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        return _error_envelope(request_id, ERR_MODEL_API_ERROR, "未配置 SKIN_API_KEY 环境变量")

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
        "max_tokens": 1000,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        api_res = _call_api(payload, headers)
    except AnalyzeError as e:
        msg = str(e)
        if "超时" in msg:
            return _error_envelope(request_id, ERR_MODEL_TIMEOUT, "模型调用超时")
        return _error_envelope(request_id, ERR_MODEL_API_ERROR, msg)

    try:
        report = _extract_json(api_res["content"])
    except AnalyzeError as e:
        return _error_envelope(request_id, ERR_MODEL_BAD_JSON, str(e))

    return _build_response(report, request_id)


def _build_response(report: dict, request_id: str) -> dict:
    iq = report.get("image_quality") or {}
    usable = bool(iq.get("usable", True))
    face_count = int(iq.get("face_count") or 0)
    issues = iq.get("issues") or []

    image_quality = {
        "usable": usable,
        "face_count": face_count,
        "quality_score": float(iq.get("quality_score") or 0.0),
        "issues": [str(i) for i in issues],
    }

    if not usable:
        return _invalid_image_envelope(request_id, image_quality, issues)

    ste = report.get("skin_type_estimate") or {}
    skin_type = normalize_skin_type(ste.get("value", "不确定"))
    skin_type_estimate = {
        "value": skin_type,
        "confidence": float(ste.get("confidence") or 0.0),
    }

    concerns = _normalize_concerns(report.get("concerns"))

    vs = report.get("visible_sensitivity") or {}
    visible_sensitivity = {
        "detected": bool(vs.get("detected", False)),
        "severity": _normalize_severity(vs.get("severity")),
        "confidence": float(vs.get("confidence") or 0.0),
        "regions": [str(r) for r in (vs.get("regions") or [])],
    }

    sf = report.get("safety_flags") or {}
    safety_flags = {
        "recommend_medical_consultation": bool(sf.get("recommend_medical_consultation", False)),
        "reasons": [str(r) for r in (sf.get("reasons") or [])],
    }

    env = std_envelope(request_id, STATUS_SUCCESS, error=None)
    env.update({
        "image_quality": image_quality,
        "skin_type_estimate": skin_type_estimate,
        "concerns": concerns,
        "visible_sensitivity": visible_sensitivity,
        "safety_flags": safety_flags,
        "summary": str(report.get("summary") or ""),
    })
    return env


def _invalid_image_envelope(request_id: str, image_quality: dict, issues: list) -> dict:
    if "未检测到清晰人脸" in issues or image_quality["face_count"] == 0:
        code, msg = ERR_IMG_NO_FACE, "未检测到清晰的单人面部，请重新上传正面照"
    elif "多张人脸" in issues or image_quality["face_count"] > 1:
        code, msg = ERR_IMG_MULTI_FACE, "检测到多张人脸，请上传单人照片"
    elif "图片模糊" in issues:
        code, msg = ERR_IMG_BLUR, "图片不够清晰，请重新拍摄"
    elif "图片过暗" in issues:
        code, msg = ERR_IMG_TOO_DARK, "光线过暗，请在自然光下拍摄"
    else:
        code, msg = ERR_IMG_NO_FACE, "图片不适合做肌肤分析"

    env = std_envelope(request_id, STATUS_INVALID_IMAGE, error={"code": code, "message": msg})
    env.update({
        "image_quality": image_quality,
        "skin_type_estimate": {"value": "不确定", "confidence": 0.0},
        "concerns": [],
        "visible_sensitivity": {"detected": False, "severity": 0, "confidence": 0.0, "regions": []},
        "safety_flags": {"recommend_medical_consultation": False, "reasons": []},
        "summary": "",
    })
    return env


def _error_envelope(request_id: str, code: str, message: str) -> dict:
    env = std_envelope(request_id, STATUS_ERROR, error={"code": code, "message": message})
    env.update({
        "image_quality": {"usable": False, "face_count": 0, "quality_score": 0.0, "issues": []},
        "skin_type_estimate": {"value": "不确定", "confidence": 0.0},
        "concerns": [],
        "visible_sensitivity": {"detected": False, "severity": 0, "confidence": 0.0, "regions": []},
        "safety_flags": {"recommend_medical_consultation": False, "reasons": []},
        "summary": "",
    })
    return env
