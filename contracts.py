"""统一接口契约：固定词表、错误码、词表映射、冲突规则。

对应「三模块协作规范 v1.0」第 5 / 6 / 11 / 12 / 17 章。
"""

SCHEMA_VERSION = "1.0"

# 第 5.3 节：肤质固定词表（敏感单独存放，不作为肤质）
ALLOWED_SKIN_TYPES = ["油性", "干性", "混合性", "中性", "不确定"]

# 第 5.4 节：可见肌肤问题固定词表
ALLOWED_VISIBLE_CONCERNS = [
    "痘痘", "痘印", "闭口", "黑头", "毛孔粗大", "出油",
    "干燥缺水", "泛红敏感", "暗沉", "色斑", "细纹", "松弛",
]

# 第 4.1 节：护肤流程模式
ROUTINE_MODES = ["精简", "标准"]

# 第 6.7 节：词表映射（用户/模型自由表达 -> 标准词表）
SKIN_TYPE_MAP = {
    "油皮": "油性", "偏油": "油性",
    "干皮": "干性", "偏干": "干性",
    "混油": "混合性", "混干": "混合性", "混合": "混合性",
    "正常": "中性",
    "未知": "不确定", "不知道": "不确定",
    "所有肤质": "多种肤质",
}

CONCERN_MAP = {
    "缺水": "干燥缺水", "干燥": "干燥缺水",
    "泛红": "泛红敏感", "敏感": "泛红敏感",
    "毛孔": "毛孔粗大",
    "油脂": "出油", "油光": "出油",
    "皱纹": "细纹",
    "衰老": "松弛", "下垂": "松弛",
    "痤疮": "痘痘", "粉刺": "闭口",
}

CATEGORY_MAP = {
    "精华": "精华液", "保湿霜": "面霜",
    "日霜": "面霜", "晚霜": "面霜",
    "防晒霜": "防晒",
}

# 流程顺序（早 + 晚的步骤库）
ROUTINE_ORDER = ["洁面", "化妆水", "精华", "乳液", "面霜", "眼霜", "防晒"]

# 第 17.1 节：状态枚举
STATUS_SUCCESS = "success"
STATUS_PARTIAL = "partial"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_INVALID_IMAGE = "invalid_image"
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"

# 第 17.2 节：错误码
ERR_IMG_NO_FACE = "IMG_NO_FACE"
ERR_IMG_MULTI_FACE = "IMG_MULTI_FACE"
ERR_IMG_BLUR = "IMG_BLUR"
ERR_IMG_TOO_DARK = "IMG_TOO_DARK"
ERR_IMG_OCCLUDED = "IMG_OCCLUDED"
ERR_MODEL_TIMEOUT = "MODEL_TIMEOUT"
ERR_MODEL_BAD_JSON = "MODEL_BAD_JSON"
ERR_MODEL_API_ERROR = "MODEL_API_ERROR"
ERR_REC_INVALID_INPUT = "REC_INVALID_INPUT"
ERR_REC_NO_MATCH = "REC_NO_MATCH"
ERR_REC_BUDGET_TOO_LOW = "REC_BUDGET_TOO_LOW"
ERR_PRODUCT_DATA_ERROR = "PRODUCT_DATA_ERROR"

# 第 11.2 节：成分搭配冲突规则
CONFLICT_RULES = [
    {
        "ingredients": [{"视黄醇", "A醇", "Retinol", "维A"},
                        {"果酸", "AHA", "乙醇酸", "杏仁酸"}],
        "message": "视黄醇/A醇 与 果酸（AHA）同时使用易刺激，建议错峰（早晚分开或隔天交替）",
    },
    {
        "ingredients": [{"视黄醇", "A醇", "Retinol", "维A"},
                        {"水杨酸", "BHA"}],
        "message": "视黄醇/A醇 与 水杨酸（BHA）叠加可能增加刺激，敏感肌建议错开",
    },
    {
        "ingredients": [{"果酸", "AHA", "乙醇酸"}, {"水杨酸", "BHA"}],
        "message": "两种去角质成分（AHA × BHA）同时使用可能导致过度刺激",
    },
    {
        "ingredients": [{"视黄醇", "A醇", "Retinol"},
                        {"维生素C", "VC", "抗坏血酸"}],
        "message": "视黄醇与维生素C建议早晚分用（VC 早间抗氧化 + 防晒，视黄醇 晚间修护）",
    },
    {
        "ingredients": [{"维生素C", "VC", "抗坏血酸"}, {"烟酰胺"}],
        "message": "高浓度 VC 与高浓度烟酰胺同时使用，敏感肌可能刺痛/泛红，建议错峰",
    },
]

# 第 11.3 节：敏感用户通用提示
SENSITIVE_TIPS = [
    "敏感肌建议逐个引入新产品，避免一次同时更换全部护肤品。",
    "首次使用建议进行局部耐受测试（耳后或手腕 24h 无反应再上脸）。",
    "若出现持续刺痛、红肿或脱皮，应停止使用并咨询专业人士。",
]

# 医疗免责声明（第 1.4 节）
MEDICAL_DISCLAIMER = (
    "本工具仅提供非医疗级护肤参考，不构成医疗诊断、治疗建议或处方建议。"
    "若存在持续性红肿、疼痛、破损、感染等情况，请及时咨询专业医生。"
)


def normalize_skin_type(value: str) -> str:
    """模型/用户输入 -> 标准词表"""
    if not value:
        return "不确定"
    if value in ALLOWED_SKIN_TYPES:
        return value
    return SKIN_TYPE_MAP.get(value, "不确定")


def normalize_concern(value: str) -> str:
    """concern 名称 -> 标准词表（找不到映射返回原值）"""
    if not value:
        return ""
    if value in ALLOWED_VISIBLE_CONCERNS:
        return value
    return CONCERN_MAP.get(value, value)


def normalize_category(value: str) -> str:
    if not value:
        return ""
    return CATEGORY_MAP.get(value, value)


def new_request_id(prefix: str = "REQ") -> str:
    """生成 REQ_YYYYMMDD_xxxxxx 形式的 request_id"""
    import time
    import uuid
    return f"{prefix}_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:6].upper()}"


def std_envelope(request_id: str, status: str, error: dict | None = None) -> dict:
    """规范第 12.1 节：所有响应的公共字段"""
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": status,
        "error": error,
    }
