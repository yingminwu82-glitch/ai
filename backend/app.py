"""
AI 美甲试戴后端 v2
====================

按需求重做：
1. Chatbot 风格：用户可对话、可选款式编号
2. 指甲检测与分割：基于 LLM + OpenCV
3. 先用"白模"调试定位（DEBUG_WHITE_MODE 开关）
4. 然后用训练集 14 款 PNG 透视变换贴到指甲
5. 可选：自动推荐 (LLM 根据手图选款)
"""
import os
import io
import re
import json
import base64
import logging
import urllib.request
import ssl
from typing import Optional, Any

import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import dashscope
from dashscope import MultiModalConversation

# ==================== 配置 ====================
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_CONFIGURED = bool(DASHSCOPE_API_KEY)
if DASHSCOPE_CONFIGURED:
    dashscope.api_key = DASHSCOPE_API_KEY

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STYLES_DIR = os.path.join(PROJECT_ROOT, 'assets', 'styles')
STYLE_MANIFEST_PATH = os.path.join(PROJECT_ROOT, 'assets', 'styles_manifest.json')
TIP_DIRS = [
    os.path.join(PROJECT_ROOT, 'assets', 'nail-tips-v4'),
    os.path.join(PROJECT_ROOT, 'assets', 'nail-tips-generated'),
    os.path.join(PROJECT_ROOT, 'assets', 'nail-tips'),
    os.path.join(PROJECT_ROOT, 'assets', 'nail-tips-v3'),
    os.path.join(PROJECT_ROOT, 'assets', 'nail-tips-v2'),
]

GENERATIVE_STYLE_IDS = {20}

# 优先使用千问图像编辑做真实试戴；本地 OpenCV 贴图仅作为失败兜底。
USE_QWEN_IMAGE_EDIT = str(os.getenv("USE_QWEN_IMAGE_EDIT", "1")).strip().lower() not in {"0", "false", "no", "off"}
QWEN_IMAGE_EDIT_MODELS = [
    m.strip()
    for m in os.getenv("QWEN_IMAGE_EDIT_MODELS", "qwen-image-2.0-pro,qwen-image-edit-plus,qwen-image-edit").split(",")
    if m.strip()
]
try:
    QWEN_IMAGE_EDIT_TIMEOUT = int(os.getenv("QWEN_IMAGE_EDIT_TIMEOUT", "180"))
except (TypeError, ValueError):
    QWEN_IMAGE_EDIT_TIMEOUT = 180

# 调试模式: True = 画白模, False = 用真实款式贴图
DEBUG_WHITE_MODE = False

# 默认用干净甲面渲染。直接裁客户款式图会混入皮肤/背景，效果不稳定。
# B 级路线: 默认开 — 25 款式 × 5 手指的真拍贴图库
USE_REAL_TIP_TEXTURE = True

# v10: 用 SAM 在用户手图上抠真实指甲 mask 校正 box
USE_SAM_CORRECTION = True

# ==================== 款式库 ====================
_BASE_14_STYLES = [
    {
        "id": 1, "name": "奶白裸色", "color_hex": "#f5e9dc",
        "description": "温柔低饱和奶白色，简约日常",
        "tags": ["温柔", "日常", "裸色"],
    },
    {
        "id": 2, "name": "米白法式", "color_hex": "#f4e9d8",
        "description": "经典法式微笑线，显手纤长",
        "tags": ["优雅", "法式", "通勤"],
    },
    {
        "id": 3, "name": "哥特棋盘", "color_hex": "#1a1a1a",
        "description": "红黑棋盘格，哥特甜心风",
        "tags": ["个性", "撞色", "潮流"],
    },
    {
        "id": 4, "name": "裸粉温柔", "color_hex": "#f0d4c2",
        "description": "裸粉色调，温柔感拉满",
        "tags": ["温柔", "粉色", "少女"],
    },
    {
        "id": 5, "name": "纯白光感", "color_hex": "#ffffff",
        "description": "纯白甲油，清爽简约",
        "tags": ["干净", "白色", "简约"],
    },
    {
        "id": 6, "name": "深酒红", "color_hex": "#5a1f1f",
        "description": "深酒红色，气场全开",
        "tags": ["气场", "深色", "成熟"],
    },
    {
        "id": 7, "name": "奶茶琥珀", "color_hex": "#c8956d",
        "description": "奶茶色琥珀感，温润如玉",
        "tags": ["温润", "奶茶色", "复古"],
    },
    {
        "id": 8, "name": "银灰闪粉", "color_hex": "#c0c0c0",
        "description": "银灰闪粉，未来感",
        "tags": ["闪", "银色", "未来"],
    },
    {
        "id": 9, "name": "薄荷绿", "color_hex": "#a8d5ba",
        "description": "薄荷绿色，清新自然",
        "tags": ["清新", "绿色", "春夏"],
    },
    {
        "id": 10, "name": "樱花粉", "color_hex": "#ffc0cb",
        "description": "樱花粉，少女心爆棚",
        "tags": ["粉色", "少女", "春夏"],
    },
    {
        "id": 11, "name": "焦糖棕", "color_hex": "#8b4513",
        "description": "焦糖棕色，复古质感",
        "tags": ["复古", "棕色", "秋冬季"],
    },
    {
        "id": 12, "name": "黑色花朵", "color_hex": "#0a0a0a",
        "description": "黑色花朵图案，极简艺术",
        "tags": ["艺术", "黑色", "图案"],
    },
    {
        "id": 13, "name": "薰衣草紫", "color_hex": "#b8a9c9",
        "description": "薰衣草紫色，温柔浪漫",
        "tags": ["紫色", "浪漫", "春夏"],
    },
    {
        "id": 14, "name": "极简线条", "color_hex": "#f0f0f0",
        "description": "极简线条，干净利落",
        "tags": ["极简", "线条", "现代"],
    },
]

# 旧兜底：如果 styles_manifest.json 不存在，仍可显示 25 个编号。
_EXTRA_11_STYLES = [
    {"id": i, "name": f"训练款式 {i:02d}", "color_hex": "#d9b8ad", "description": "来自评测数据的真实款式", "tags": ["训练集"]}
    for i in range(15, 26)
]

def load_style_library() -> list:
    """Load 25 real evaluation styles when the generated manifest exists."""
    fallback = _BASE_14_STYLES + _EXTRA_11_STYLES
    fallback_by_id = {s["id"]: dict(s) for s in fallback}
    if not os.path.exists(STYLE_MANIFEST_PATH):
        return fallback
    try:
        with open(STYLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        styles = []
        for item in manifest.get("styles", []):
            try:
                sid = int(item.get("id", 0))
            except (TypeError, ValueError):
                sid = 0
            if sid <= 0:
                continue
            base = fallback_by_id.get(sid, {
                "id": sid,
                "name": f"训练款式 {sid:02d}",
                "color_hex": item.get("color_hex", "#d9b8ad"),
                "description": "来自评测数据的真实款式",
                "tags": ["训练集"],
            })
            merged = dict(base)
            if sid > 14:
                merged["name"] = item.get("name") or merged["name"]
                merged["color_hex"] = item.get("color_hex") or merged["color_hex"]
                merged["description"] = item.get("description") or merged["description"]
                merged["tags"] = item.get("tags") or merged["tags"]
            merged["image"] = item.get("image") or f"/assets/styles/style-{sid:02d}.png"
            merged["source_url"] = item.get("source_url", "")
            merged["tip_count"] = item.get("tip_count", 0)
            styles.append(merged)
        return sorted(styles, key=lambda s: s["id"]) or fallback
    except Exception as e:
        print(f"failed to load style manifest: {e}")
        return fallback


STYLE_LIBRARY = load_style_library()
TARGET_TOTAL_STYLES = 25  # 客户要求的总数

# ==================== Logging ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# ==================== Flask ====================
app = Flask(__name__)
CORS(app)
FRONTEND_DIR = os.path.join(PROJECT_ROOT, 'frontend')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')


# ==================== 工具 ====================
def safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def safe_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_bool(v, default=False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_tuning(source) -> dict:
    return {
        "width_scale": safe_float(source.get("width_scale"), 1.0),
        "length_scale": safe_float(source.get("length_scale"), 1.0),
        "offset_scale": safe_float(source.get("offset_scale"), 0.0),
        "opacity": safe_float(source.get("opacity"), 0.92),
    }


def repair_truncated_array(text: str) -> str:
    """尝试修复 qwen-vl-plus 返回的截断 JSON 数组"""
    # 找到最后一个完整的对象
    matches = []
    depth = 0
    obj_start = None
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                obj_start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and obj_start is not None:
                matches.append((obj_start, i + 1))
                obj_start = None
    if not matches:
        return text
    # 用最后一个完整对象收尾
    last_obj_end = matches[-1][1]
    return text[:last_obj_end] + "]"


def parse_json_loose(text: str) -> Any:
    """宽松 JSON 解析：剥离 markdown、修复截断"""
    if not text:
        return None
    # 剥离 ```json ... ```
    m = re.search(r'```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```', text)
    if m:
        text = m.group(1)
    else:
        # 抓 ``` 到 ``` 中间
        m = re.search(r'```[^\n]*\n([\s\S]*?)```', text)
        if m:
            text = m.group(1)
        else:
            # 抓裸数组
            m = re.search(r'\[[\s\S]*\]', text)
            if m:
                text = m.group(0)
            else:
                m = re.search(r'\{[\s\S]*\}', text)
                if m:
                    text = m.group(0)

    # 修复常见 typo: "y2=" 应为 "y2":  (LLM 偶尔错)
    text = re.sub(r'"(y2|x2|y|x)"\s*=\s*', r'"\1": ', text)
    # 修复 y2 后面没引号
    text = re.sub(r'(\d+)\s*"\s*\}', r'\1}', text)

    # 尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 截断修复: 找到最后一个 } 然后补 ]
    if text.count('[') > text.count(']'):
        repaired = repair_truncated_array(text)
        # repaired 已含 ]
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e:
            log.debug(f"repaired parse failed: {e}, text={repaired[-100:]}")

    log.warning(f"parse_json_loose failed: {text[:200]}")
    return None


def call_qwen_vl(content: list, max_retries: int = 2) -> Optional[str]:
    """调用 qwen-vl-plus，带重试"""
    if not DASHSCOPE_CONFIGURED:
        log.warning("DASHSCOPE_API_KEY not configured")
        return None
    for attempt in range(max_retries):
        try:
            resp = MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=[{"role": "user", "content": content}],
            )
            if resp.status_code != 200:
                log.warning(f"qwen-vl status: {resp.status_code}, message: {resp.message}")
                continue
            msg = resp.output.choices[0].message
            # content 可能是 list 或 str
            if isinstance(msg.content, list):
                texts = [c.get('text', '') for c in msg.content if c.get('text')]
                return '\n'.join(texts)
            return msg.content
        except Exception as e:
            log.warning(f"qwen-vl attempt {attempt+1} failed: {e}")
    return None


def bgr_to_data_uri(img_bgr: np.ndarray, max_long: int = 1600, quality: int = 92) -> str:
    """Encode an image for multimodal API calls while keeping the original aspect ratio."""
    h, w = img_bgr.shape[:2]
    work = img_bgr
    long_edge = max(h, w)
    if long_edge > max_long:
        scale = max_long / long_edge
        work = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", work, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("image encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def file_to_data_uri(path: str, max_long: int = 1400) -> Optional[str]:
    """Load a local style reference image as a data URI."""
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return bgr_to_data_uri(img, max_long=max_long, quality=90)
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def download_image_as_data_uri(url: str, timeout: int = 90) -> Optional[str]:
    """DashScope image-edit URLs expire quickly, so return a stable data URI to the frontend."""
    if not url:
        return None
    if url.startswith("data:image/"):
        return url
    req = urllib.request.Request(url, headers={"User-Agent": "nail-tryon/1.0"})
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "image/png").split(";")[0].strip() or "image/png"
    if not ctype.startswith("image/"):
        ctype = "image/png"
    return f"data:{ctype};base64," + base64.b64encode(data).decode()


def extract_image_uri_from_qwen_response(resp) -> Optional[str]:
    """Handle the slightly different response shapes used by DashScope image models."""
    candidates = []

    def add_candidate(v):
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://") or v.startswith("data:image/")):
            candidates.append(v)

    try:
        msg = resp.output.choices[0].message
        content = msg.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    for key in ("image", "url", "image_url", "output_image", "result_image"):
                        add_candidate(item.get(key))
                else:
                    add_candidate(getattr(item, "image", None))
                    add_candidate(getattr(item, "url", None))
        elif isinstance(content, str):
            add_candidate(content)
    except Exception:
        pass

    try:
        for item in getattr(resp.output, "results", []) or []:
            add_candidate(getattr(item, "url", None))
            if isinstance(item, dict):
                add_candidate(item.get("url"))
                add_candidate(item.get("image"))
    except Exception:
        pass

    if candidates:
        return candidates[0]

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        if isinstance(obj, str):
            add_candidate(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth + 1)
        else:
            if hasattr(obj, "__dict__"):
                walk({k: v for k, v in obj.__dict__.items() if not k.startswith("_")}, depth + 1)

    walk(resp)
    return candidates[0] if candidates else None


def style_reference_path(style_id: int) -> Optional[str]:
    path = os.path.join(STYLES_DIR, f"style-{style_id:02d}.png")
    return path if os.path.exists(path) else None


def build_qwen_tryon_prompt(style_id: int, nails_count: int = 0) -> str:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    if style_id == 20:
        target_style = (
            "半透明琥珀棕凝胶甲，圆角延长甲片，有白色猫脸、小星星和金箔点缀。"
            "整体像客户参考图里的 AI 美甲效果，温润透亮、不是贴纸。"
        )
    else:
        target_style = (
            f"参考图1的美甲款式：#{style['id']} {style['name']}。"
            f"{style.get('description', '')} 请提取它的颜色、图案、质感和装饰元素，重新生成到手图指甲上。"
        )
    count_hint = f"图中检测到约 {nails_count} 个可见指甲。" if nails_count else "请识别图中所有可见指甲。"
    return f"""你是专业美甲试戴图像编辑师。输入中如果有两张图，图1是款式参考，最后一张图是必须保留的用户手部照片。

任务：
- 只编辑最后一张手部照片中的真实指甲/甲床区域，{count_hint}
- 保持手指形状、皮肤纹理、掌纹、背景、光照、阴影、构图和清晰度不变
- 将美甲自然生成在每个可见指甲上：要沿真实甲面曲率弯曲贴合，符合手指透视和遮挡关系
- 需要真实凝胶甲质感：半透明层次、甲面高光、边缘厚度、甲沟阴影、轻微反光和环境光
- 不能出现平面贴图、方块边缘、黑色硬边、悬浮感、错位、覆盖皮肤或改变手指
- 如果原甲较短，可以生成自然的圆角延长甲，但必须从甲床连续生长出来

目标款式：
{target_style}

输出一张真实照片风格的完整手部试戴图，不要拼图，不要文字，不要边框。"""


def qwen_image_tryon(hand_bgr: np.ndarray, style_id: int, nails_count: int = 0) -> Optional[dict]:
    """Use Qwen image editing to regenerate the nails instead of planar compositing."""
    if not DASHSCOPE_CONFIGURED or not USE_QWEN_IMAGE_EDIT or not QWEN_IMAGE_EDIT_MODELS:
        return None
    hand_uri = bgr_to_data_uri(hand_bgr, max_long=1600, quality=92)
    content = []
    ref_path = style_reference_path(style_id)
    ref_uri = file_to_data_uri(ref_path) if ref_path else None
    if ref_uri:
        content.append({"image": ref_uri})
    # Last image controls output aspect ratio for Qwen image editing.
    content.append({"image": hand_uri})
    content.append({"text": build_qwen_tryon_prompt(style_id, nails_count=nails_count)})

    last_error = ""
    for model in QWEN_IMAGE_EDIT_MODELS:
        try:
            log.info("qwen image edit tryon: model=%s style=%s", model, style_id)
            resp = MultiModalConversation.call(
                model=model,
                messages=[{"role": "user", "content": content}],
                request_timeout=QWEN_IMAGE_EDIT_TIMEOUT,
            )
            if getattr(resp, "status_code", 0) != 200:
                last_error = f"{getattr(resp, 'status_code', '')} {getattr(resp, 'message', '')}".strip()
                log.warning("qwen image edit status failed: model=%s %s", model, last_error)
                continue
            image_uri = extract_image_uri_from_qwen_response(resp)
            if not image_uri:
                last_error = "no image url in response"
                log.warning("qwen image edit no output image: model=%s", model)
                continue
            result_uri = download_image_as_data_uri(image_uri)
            if result_uri:
                return {"result_image": result_uri, "model": model}
        except Exception as e:
            last_error = str(e)
            log.warning("qwen image edit failed: model=%s error=%s", model, e)
    if last_error:
        log.warning("qwen image edit fallback to local render: %s", last_error)
    return None


# ==================== 指甲定位 (MediaPipe Hands) ====================

# 尝试加载 MediaPipe（可选依赖）
try:
    import mediapipe as mp
    MP_HANDS = mp.solutions.hands
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False
    log.warning("mediapipe not available, falling back to LLM locate")

# MediaPipe landmark 顺序
# 0:腕, 1-4:拇指, 5-8:食指, 9-12:中指, 13-16:无名指, 17-20:小指
# 每指: 端点指尖 -> DIP -> PIP -> MCP
# 例如食指: 5(MCP), 6(PIP), 7(DIP), 8(端点)
FINGER_TIPS = {
    "拇指": (4, 3, 2),       # 端点, IP, MCP
    "食指": (8, 7, 6),       # 端点, DIP, PIP
    "中指": (12, 11, 10),
    "无名指": (16, 15, 14),
    "小指": (20, 19, 18),
}


def estimate_nail_box(tip_xy, dip_xy, pip_xy, finger_name, h_img, w_img):
    """根据指尖+DIP 关键点估计甲片外接矩形。

    客户示例是“甲片试戴/延长甲”效果，不是只给短甲床上色。
    因此以指尖为锚点，让甲片覆盖甲床并略微伸出指尖。
    """
    import math
    dx = dip_xy[0] - tip_xy[0]
    dy = dip_xy[1] - tip_xy[1]
    finger_len = math.hypot(dx, dy)
    if finger_len < 5:
        return None
    angle = math.atan2(dy, dx)
    # MediaPipe gives us fingertip -> DIP, not a true nail box. The default mode
    # is short-nail polish try-on, so keep the prompt close to the visible nail
    # bed instead of creating extension-length tips.
    if finger_name == "拇指":
        offset = 0.22
        nail_w = finger_len * 0.60
        nail_len = finger_len * 0.76
    else:
        offset = 0.20
        nail_w = finger_len * 0.50
        nail_len = finger_len * 0.70
    cx = tip_xy[0] + offset * finger_len * math.cos(angle)
    cy = tip_xy[1] + offset * finger_len * math.sin(angle)
    # 限制在图内
    cx = clamp(int(cx), 0, w_img - 1)
    cy = clamp(int(cy), 0, h_img - 1)
    w = clamp(int(nail_w), 15, w_img // 2)
    h = clamp(int(nail_len), 15, h_img // 2)
    return (cx, cy, w, h)


def locate_nails_mediapipe(hand_bgr: np.ndarray) -> list:
    """用 MediaPipe 定位指甲 (v9: 双手支持, 最多 10 指甲)
    返回: nail dict list, 每个含 hand_id (0/1) + handedness (Left/Right)
    """
    if not MP_AVAILABLE:
        return []
    h, w = hand_bgr.shape[:2]
    img_rgb = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2RGB)
    nails = []
    with MP_HANDS.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4) as hands:
        result = hands.process(img_rgb)
        if not result.multi_hand_landmarks:
            return []

        # v9: 保留所有手 (最多 2), 各自生成 5 指甲 (最多 10)
        import math as _m
        for hand_idx, hl in enumerate(result.multi_hand_landmarks):
            # handedness (可能不准, 仅参考)
            try:
                handedness = result.multi_handedness[hand_idx].classification[0].label
            except Exception:
                handedness = "Unknown"

            # 保留 3D landmark (z 是 wrist-relative 深度, 越负 = 越靠相机)
            landmarks_3d = [(lm.x, lm.y, lm.z) for lm in hl.landmark]
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hl.landmark]
            for finger_name, (tip_id, dip_id, pip_id) in FINGER_TIPS.items():
                tip_xy = landmarks[tip_id]
                dip_xy = landmarks[dip_id]
                pip_xy = landmarks[pip_id]
                # 曲率 (3D): tip 相对 DIP 的 z 偏移, 归一化到 0-1
                tip_z = landmarks_3d[tip_id][2]
                dip_z = landmarks_3d[dip_id][2]
                z_diff = dip_z - tip_z
                curvature = clamp((z_diff - 0.0) * 25.0, 0.0, 1.0)
                # 手指弯曲度
                tx, ty = landmarks_3d[tip_id][0], landmarks_3d[tip_id][1]
                dx_, dy_, dz_ = landmarks_3d[dip_id][0] - tx, landmarks_3d[dip_id][1] - ty, landmarks_3d[dip_id][2] - tip_z
                px_, py_, pz_ = landmarks_3d[pip_id][0] - landmarks_3d[dip_id][0], landmarks_3d[pip_id][1] - landmarks_3d[dip_id][1], landmarks_3d[pip_id][2] - dip_z
                v1len = _m.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) or 1e-6
                v2len = _m.sqrt(px_ * px_ + py_ * py_ + pz_ * pz_) or 1e-6
                cos_a = (dx_ * px_ + dy_ * py_ + dz_ * pz_) / (v1len * v2len)
                bend_angle = _m.degrees(_m.acos(max(-1.0, min(1.0, cos_a))))
                bend_factor = clamp((bend_angle - 5.0) / 30.0, 0.0, 1.0)
                curvature = max(curvature, bend_factor)
                box = estimate_nail_box(tip_xy, dip_xy, pip_xy, finger_name, h, w)
                if box:
                    cx, cy, bw, bh = box
                    nails.append({
                        "source": "mediapipe",
                        "finger": finger_name,
                        "hand_id": hand_idx,
                        "handedness": handedness,
                        "cx": cx, "cy": cy,
                        "w": bw, "h": bh,
                        "tip_xy": list(tip_xy),
                        "dip_xy": list(dip_xy),
                        "curvature": round(curvature, 3),
                    })
    return nails


# ===== LLM fallback 定位 =====
LOCATE_PROMPT = """观察这张手部照片，精准定位所有可见的指甲甲床（图片尺寸约 {w} x {h} 像素）。

要求：
- 对每个可见指甲甲床，给出**外接矩形** (x, y, x2, y2)，矩形**紧紧包裹甲面**，不包含指节皮肤
- 最多 5 个指甲
- 如果指甲被遮挡或不可见，跳过
- finger 字段用 "拇指"/"食指"/"中指"/"无名指"/"小指" 命名

严格按 JSON 数组输出（不要其他文字、代码块标记）：
[
  {{"finger":"拇指","x":109,"y":476,"x2":183,"y2":555}},
  ...
]"""


def locate_nails_llm(hand_data_uri: str, target_w: int, target_h: int) -> list:
    """调 qwen-vl-plus 定位指甲（fallback）"""
    prompt = LOCATE_PROMPT.format(w=target_w, h=target_h)
    raw = call_qwen_vl([
        {"image": hand_data_uri},
        {"text": prompt},
    ])
    if not raw:
        return []
    parsed = parse_json_loose(raw)
    if not isinstance(parsed, list):
        return []
    nails = []
    for n in parsed:
        try:
            x = safe_int(n.get("x"))
            y = safe_int(n.get("y"))
            x2 = safe_int(n.get("x2"))
            y2 = safe_int(n.get("y2"))
            w = x2 - x
            h = y2 - y
            if w < 10 or h < 10:
                continue
            cx = (x + x2) / 2
            cy = (y + y2) / 2
            cy = cy - h * 0.10
            w = int(w * 1.15)
            h = int(h * 1.25)
            nails.append({
                "source": "llm",
                "finger": str(n.get("finger", "?")),
                "cx": int(cx), "cy": int(cy),
                "w": w, "h": h,
            })
        except Exception:
            continue
    return nails


def locate_nails(hand_bgr: np.ndarray, hand_data_uri: str = None) -> list:
    """主定位函数：优先 MediaPipe，失败则用 LLM"""
    # 1. MediaPipe（用原图）
    nails = locate_nails_mediapipe(hand_bgr)
    if nails:
        log.info(f"mediapipe detected {len(nails)} nails")
        return nails
    # 2. LLM fallback
    if hand_data_uri:
        h, w = hand_bgr.shape[:2]
        nails = locate_nails_llm(hand_data_uri, 1000, int(h * 1000 / w))
        log.info(f"llm fallback detected {len(nails)} nails")
        return nails
    return []


# ==================== 自动推荐 ====================
ANALYZE_PROMPT = """你是专业美甲顾问。仔细观察用户的手部照片，从以下几个维度进行分析：

1. **肤色色调**：冷色（偏冷白、粉白）、中性、暖色（偏黄、偏黑）
2. **手型特点**：手指纤细、手指粗短、手指修长、手指短粗
3. **指甲现状**：长款、圆款、短款、有涂甲、裸甲
4. **背景暗示场景**：是在美甲店、家中、户外？反映什么场合/氛围？

然后从以下 {style_count} 款美甲中推荐 **3 款**（主推荐 1 + 备选 2）：
{style_catalog}

输出 JSON：
{
  "analysis": {
    "skin_tone": "冷白 | 中性 | 暖黄 | 偏黑",
    "hand_type": "纤细 | 粗短 | 修长 | 短粗",
    "nail_status": "长款 | 圆款 | 短款 | 有涂甲 | 裸甲",
    "scene": "美甲店 | 家中 | 户外 | 商务 | 休闲"
  },
  "recommendations": [
    {"style_id": 1, "reason": "适合肤色的具体理由（20字内）"},
    {"style_id": 7, "reason": "备选款理由（20字内）"},
    {"style_id": 9, "reason": "备选款理由（20字内）"}
  ]
}

严格按 JSON 输出，不要任何额外文字："""


def build_style_catalog() -> str:
    return "\n".join(
        f"{s['id']}. {s['name']} - {s.get('description', '')}"
        for s in STYLE_LIBRARY
    )


def build_analyze_prompt() -> str:
    return (
        ANALYZE_PROMPT
        .replace("{style_count}", str(len(STYLE_LIBRARY)))
        .replace("{style_catalog}", build_style_catalog())
    )


def recommend_style(hand_data_uri: str) -> dict:
    """分析手部 + 推荐 3 款"""
    raw = call_qwen_vl([
        {"image": hand_data_uri},
        {"text": build_analyze_prompt()},
    ])
    if not raw:
        return {
            "analysis": {"skin_tone": "中性", "hand_type": "纤细", "nail_status": "裸甲", "scene": "日常"},
            "recommendations": [
                {"style_id": 1, "reason": "温柔裸色适合多数肤色"},
                {"style_id": 7, "reason": "奶茶色温润显气质"},
                {"style_id": 9, "reason": "清新自然不挑场合"},
            ],
        }
    parsed = parse_json_loose(raw)
    if not isinstance(parsed, dict):
        return {
            "analysis": {"skin_tone": "中性", "hand_type": "纤细", "nail_status": "裸甲", "scene": "日常"},
            "recommendations": [
                {"style_id": 1, "reason": "温柔裸色适合多数肤色"},
                {"style_id": 7, "reason": "奶茶色温润显气质"},
                {"style_id": 9, "reason": "清新自然不挑场合"},
            ],
        }
    
    analysis = parsed.get("analysis", {}) or {}
    recs = parsed.get("recommendations", [])
    if not isinstance(recs, list):
        recs = []
    
    # 清理 recs
    cleaned_recs = []
    for r in recs[:3]:
        if not isinstance(r, dict):
            continue
        sid = safe_int(r.get("style_id"), 1)
        sid = clamp(sid, 1, len(STYLE_LIBRARY))
        cleaned_recs.append({
            "style_id": sid,
            "reason": str(r.get("reason", "推荐"))[:50],
        })
    
    # 不足 3 个补齐
    while len(cleaned_recs) < 3:
        sid = (cleaned_recs[-1]["style_id"] + 1) if cleaned_recs else 1
        cleaned_recs.append({"style_id": sid, "reason": "备选推荐"})
    
    return {
        "analysis": {
            "skin_tone": str(analysis.get("skin_tone", "中性")),
            "hand_type": str(analysis.get("hand_type", "纤细")),
            "nail_status": str(analysis.get("nail_status", "裸甲")),
            "scene": str(analysis.get("scene", "日常")),
        },
        "recommendations": cleaned_recs,
    }


FIT_TREND_PROMPT = """你是专业美甲顾问。请基于用户手部照片和当前试戴款式，输出款式适配度和流行趋势判断。

当前款式：#{style_id} {style_name}
款式说明：{style_desc}
手部分析：{analysis}

输出 JSON：
{{
  "fit_score": 88,
  "fit_text": "一句话说明这款为什么适合当前手型、肤色、甲型，40字内",
  "trend_text": "一句话说明当前流行趋势或场景建议，40字内"
}}

严格按 JSON 输出，不要任何额外文字："""


def generate_fit_trend(hand_data_uri: str, style_id: int, analysis: dict = None) -> dict:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    fallback = {
        "fit_score": 86,
        "fit_text": f"{style['name']}和当前手型适配度较高，视觉上更显干净修长。",
        "trend_text": "低饱和、光泽感和细节纹理仍是日常试戴里的主流方向。",
    }
    prompt = FIT_TREND_PROMPT.format(
        style_id=style["id"],
        style_name=style["name"],
        style_desc=style.get("description", ""),
        analysis=json.dumps(analysis or {}, ensure_ascii=False),
    )
    raw = call_qwen_vl([
        {"image": hand_data_uri},
        {"text": prompt},
    ])
    parsed = parse_json_loose(raw or "")
    if not isinstance(parsed, dict):
        return fallback
    score = clamp(safe_int(parsed.get("fit_score"), fallback["fit_score"]), 1, 100)
    return {
        "fit_score": score,
        "fit_text": str(parsed.get("fit_text") or fallback["fit_text"])[:120],
        "trend_text": str(parsed.get("trend_text") or fallback["trend_text"])[:120],
    }


# ==================== 渲染 ====================
def find_tip_path(style_id: int, finger_en: str = "middle") -> Optional[str]:
    """Find a real nail tip across generated, legacy, and v4 naming schemes."""
    positions = [finger_en, "middle", "index", "ring", "thumb", "pinky"]
    names = []
    for pos in positions:
        names.extend([
            f"tip-{style_id:02d}-{pos}1.png",
            f"tip-{style_id:02d}-{pos}2.png",
            f"tip-{style_id:02d}-{pos}-h0.png",
        ])
    for directory in TIP_DIRS:
        for name in names:
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
    return None


def load_style_image_as_tip(style_id: int, w: int, h: int) -> Optional[np.ndarray]:
    """Fallback texture crop from the full evaluation style image."""
    style_path = os.path.join(STYLES_DIR, f"style-{style_id:02d}.png")
    img = cv2.imread(style_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    ih, iw = img.shape[:2]
    # The central upper part usually contains the showcased nails in the evaluation images.
    x0 = int(iw * 0.28)
    x1 = int(iw * 0.72)
    y0 = int(ih * 0.18)
    y1 = int(ih * 0.62)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = img
    return cv2.resize(crop, (max(1, w), max(1, h)), interpolation=cv2.INTER_CUBIC)


def load_style_thumbnail(style_id: int) -> Optional[np.ndarray]:
    """从 nail-tips 加载款式缩略图（裁好的单指甲贴图）"""
    path = find_tip_path(style_id, "middle")
    if not path:
        return load_style_image_as_tip(style_id, 96, 120)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return img


def extract_nail_only(tip_bgr: np.ndarray) -> np.ndarray:
    """
    从贴图中提取指甲区域 (v4: 边缘检测 + 凸包 + 中心优先评分)
    核心依据：
    - 指甲 vs 皮肤 有 明锐的 边缘过渡 (颜色 / 纹理)
    - 指甲位置 居中 且 面积 10-50%
    策略：
    1. 转灰度, 拉普拉斯 取 边缘
    2. Canny 边缘 + 闭包填充
    3. 连通域评分: 面积 (10-50%) + 中心距离 (近优) + 凸性
    4. fallback：中心椭圆
    """
    h, w = tip_bgr.shape[:2]
    if h < 20 or w < 20:
        return tip_bgr

    # 1. 转灰度
    gray = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2GRAY)
    # 轻高斯去噪
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Canny 边缘
    edges = cv2.Canny(gray_blur, 30, 100)
    # 闭包填充边缘
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel3, iterations=2)
    # 漫水填充 闭包 后的 内部
    fill_mask = edges_closed.copy()
    bgd = np.zeros((h + 2, w + 2), np.uint8)
    # 从 4 角 灌背景
    cv2.floodFill(fill_mask, bgd, (0, 0), 255)
    cv2.floodFill(fill_mask, bgd, (w - 1, 0), 255)
    cv2.floodFill(fill_mask, bgd, (0, h - 1), 255)
    cv2.floodFill(fill_mask, bgd, (w - 1, h - 1), 255)
    # 填充区域 = 未被背景填充的 = 内部区域
    filled = cv2.bitwise_not(fill_mask)

    # 3. 合并: 边缘闭包 区域  ∪ 内部填充区域
    nail_mask = cv2.bitwise_or(edges_closed, filled)
    # 形态学清理
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_OPEN, kernel5)
    nail_mask = cv2.morphologyEx(nail_mask, cv2.MORPH_CLOSE, kernel5)

    # 4. 连通域: 评分 选 最佳
    contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_area = h * w
    cx_img, cy_img = w / 2.0, h / 2.0
    best = None
    best_score = -1
    for c in contours:
        a = cv2.contourArea(c)
        if a < total_area * 0.05 or a > total_area * 0.55:
            continue
        # 中心距离 (近优)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        ccx = M["m10"] / M["m00"]
        ccy = M["m01"] / M["m00"]
        dist = ((ccx - cx_img) ** 2 + (ccy - cy_img) ** 2) ** 0.5
        max_dist = (cx_img ** 2 + cy_img ** 2) ** 0.5
        center_score = 1.0 - dist / max_dist
        # 面积适中优 (目标 ~30%)
        area_score = 1.0 - abs(a / total_area - 0.30) / 0.30
        # 凸性 (凸包面积 / contour 面积, 越接近 1 越平滑)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull) or 1
        convex_score = a / hull_area
        # 总分
        score = center_score * 0.5 + area_score * 0.3 + convex_score * 0.2
        if score > best_score:
            best_score = score
            best = c

    nail_mask = np.zeros_like(nail_mask)
    if best is not None and best_score > 0.1:
        hull = cv2.convexHull(best)
        cv2.drawContours(nail_mask, [hull], -1, 255, -1)

    # 5. 边缘羽化
    nail_mask = cv2.GaussianBlur(nail_mask, (3, 3), 0)

    # 6. fallback
    if cv2.countNonZero(nail_mask) < h * w * 0.08:
        nail_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(nail_mask, (w // 2, h // 2),
                    (int(w * 0.35), int(h * 0.40)),
                    0, 0, 360, 255, -1)
    return nail_mask


# ==================== SAM 指甲分割 (v7: 预训练 MobileSAM/ViT-B) ====================
SAM_MODEL = None
SAM_PREDICTOR = None
SAM_AVAILABLE = False
SAM_CHECKPOINT = os.path.join(PROJECT_ROOT, 'models', 'sam_vit_b.pth')

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    if os.path.exists(SAM_CHECKPOINT):
        SAM_MODEL = sam_model_registry['vit_b'](checkpoint=SAM_CHECKPOINT)
        SAM_MODEL.to('cpu')
        SAM_PREDICTOR = SamPredictor(SAM_MODEL)
        SAM_AVAILABLE = True
        log.info(f"SAM loaded from {SAM_CHECKPOINT}")
    else:
        log.warning(f"SAM checkpoint not found: {SAM_CHECKPOINT}")
except Exception as e:
    log.warning(f"SAM not available: {e}")


def extract_nail_sam(tip_bgr: np.ndarray) -> Optional[np.ndarray]:
    """用 SAM 在整张图作 box 提示, 返回 评分最高 且 覆盖率 适中 (15-60%) 的 mask
    返回: mask (uint8 0-255) 或 None (失败)
    """
    if not SAM_AVAILABLE:
        return None
    h, w = tip_bgr.shape[:2]
    if h < 20 or w < 20:
        return None
    try:
        img_rgb = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2RGB)
        SAM_PREDICTOR.set_image(img_rgb)
        input_box = np.array([0, 0, w, h])
        masks, scores, _ = SAM_PREDICTOR.predict(box=input_box, multimask_output=True)
        # 选 评分高 + 覆盖率适中 (15-60%) 的
        best_idx, best_score = -1, -1
        for i, (m, s) in enumerate(zip(masks, scores)):
            cov = m.sum() / m.size
            if 0.15 <= cov <= 0.60 and s > best_score:
                best_score = s
                best_idx = i
        # 如果都不符合, 退到 评分最高
        if best_idx == -1:
            best_idx = int(np.argmax(scores))
        mask = (masks[best_idx] * 255).astype(np.uint8)
        # 形态学清理
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
        # 高斯羽化
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return mask
    except Exception as e:
        log.warning(f"SAM extract failed: {e}")
        return None


def load_nail_mask_for_tip(tip_bgr: np.ndarray) -> np.ndarray:
    """返回指甲区域的 mask (v7: 优先 SAM, fallback GrabCut)
    返回: uint8 0-255
    """
    # 优先用 SAM (预训练 ViT-B, ~8.5/10 质量)
    sam_mask = extract_nail_sam(tip_bgr)
    if sam_mask is not None and cv2.countNonZero(sam_mask) > 0:
        return sam_mask
    # fallback: opencv 算法
    return extract_nail_only(tip_bgr)


def sam_segment_nail_in_hand(hand_bgr: np.ndarray, cx: int, cy: int, w: int, hh: int) -> Optional[np.ndarray]:
    """v10: 用 SAM 在用户手图 box 内抠真实指甲 mask
    输入: 手图 + MediaPipe 估算的 box (cx, cy, w, hh)
    返回: full-size mask (h_img, w_img, uint8 0-255), 指甲 = 255, 其它 = 0
    失败: 返回 None
    """
    if not SAM_AVAILABLE:
        return None
    h_img, w_img = hand_bgr.shape[:2]
    # Give SAM enough context to include the full nail bed, while keeping nearby
    # fingers out of the prompt as much as possible.
    pad = 0.38
    x0 = max(0, int(cx - w * (0.5 + pad)))
    y0 = max(0, int(cy - hh * (0.5 + pad)))
    x1 = min(w_img, int(cx + w * (0.5 + pad)))
    y1 = min(h_img, int(cy + hh * (0.5 + pad)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    roi = hand_bgr[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    try:
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        SAM_PREDICTOR.set_image(roi_rgb)
        # box 提示: MediaPipe 估算的 box (在 roi 内)
        bx0 = int(w * pad)
        by0 = int(hh * pad)
        bx1 = int(w * pad + w)
        by1 = int(hh * pad + hh)
        input_box = np.array([bx0, by0, bx1, by1])
        masks, scores, _ = SAM_PREDICTOR.predict(box=input_box, multimask_output=True)
        # 选 评分高 + 面积接近 MediaPipe 估算 (1.5x-0.4x 指甲框面积) + 中心在 box 内 的 mask
        import math
        target_area = w * hh
        best_idx, best_score = -1, -1.0
        for i, (m, s) in enumerate(zip(masks, scores)):
            m_area = m.sum()
            area_ratio = m_area / max(target_area, 1)
            # Bare nails can be small inside a generous prompt, but very tiny
            # masks are usually fingertip highlights or texture fragments.
            if not (0.12 <= area_ratio <= 2.8):
                continue
            if s < 0.7:
                continue
            ys, xs = np.where(m)
            if len(ys) == 0:
                continue
            m_cx, m_cy = xs.mean(), ys.mean()
            dist_to_box_c = math.hypot(m_cx - (bx0 + bx1) / 2, m_cy - (by0 + by1) / 2)
            if dist_to_box_c > max(w, hh) * 0.65:
                continue
            # 综合评分: 评分 * 0.6 + 面积接近度 * 0.4
            fit = 1.0 - min(abs(1.0 - area_ratio), 1.0)
            combined = s * 0.6 + fit * 0.4
            if combined > best_score:
                best_score = combined
                best_idx = i
        if best_idx == -1:
            best_idx = int(np.argmax(scores))
        mask_roi = (masks[best_idx] * 255).astype(np.uint8)
        # 形态学清理
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel3)
        # 凸包填补
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 30:
                hull = cv2.convexHull(largest)
                mask_roi = np.zeros_like(mask_roi)
                cv2.fillConvexPoly(mask_roi, hull, 255)
        # 放回 full size
        full_mask = np.zeros((h_img, w_img), np.uint8)
        mask_roi = cv2.GaussianBlur(mask_roi, (3, 3), 0)
        full_mask[y0:y1, x0:x1] = mask_roi
        if cv2.countNonZero(full_mask) < 30:
            return None
        return full_mask
    except Exception as e:
        log.warning(f"v10 SAM in-hand failed: {e}")
        return None


def make_nail_alpha_mask(w: int, h: int, curvature: float = 0.0, finger: str = "中指") -> np.ndarray:
    """Stable target nail alpha mask with a rounded almond silhouette.

    v10-fix8: 取消锐角 taper, 用纯椭圆 (顶端不变尖, 看起来是椭圆形指甲不是锐角)
    之前 taper 把顶端缩到 30-45% 锐角 → 看起来是 "锐角矩形", 贴上去像方框
    finger: 拇指/食指/中指/无名指/小指 - 调整指甲比例
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2
    # v10-fix8: 加大椭圆 (接近全屏)
    # rx 比例 0.46-0.50, ry 比例 0.49 (让椭圆几乎占满指甲 4 边)
    finger_ratio = {
        "拇指": 0.50, "食指": 0.48, "中指": 0.48, "无名指": 0.47, "小指": 0.46,
    }.get(finger, 0.48)
    rx = max(3, int(w * (finger_ratio - curvature * 0.02)))
    ry = max(4, int(h * 0.49))
    cy = int(h * 0.50)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    ksize = max(3, int(min(w, h) * 0.08)) // 2 * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def make_rounded_tip_mask(w: int, h: int) -> np.ndarray:
    """Rounded-square artificial nail silhouette used for AI-style try-on."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pad_x = max(1, int(w * 0.05))
    pad_y = max(1, int(h * 0.03))
    radius = max(3, int(min(w, h) * 0.34))
    x0, y0 = pad_x, pad_y
    x1, y1 = w - pad_x - 1, h - pad_y - 1
    cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
    cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
    for x in (x0 + radius, x1 - radius):
        for y in (y0 + radius, y1 - radius):
            cv2.circle(mask, (x, y), radius, 255, -1)
    ksize = max(3, int(min(w, h) * 0.06)) // 2 * 2 + 1
    return cv2.GaussianBlur(mask, (ksize, ksize), 0)


def draw_star(img: np.ndarray, center: tuple, radius: int, color: tuple, thickness: int = -1):
    x, y = center
    pts = np.array([
        (x, y - radius),
        (x + max(1, radius // 4), y - max(1, radius // 4)),
        (x + radius, y),
        (x + max(1, radius // 4), y + max(1, radius // 4)),
        (x, y + radius),
        (x - max(1, radius // 4), y + max(1, radius // 4)),
        (x - radius, y),
        (x - max(1, radius // 4), y - max(1, radius // 4)),
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color) if thickness < 0 else cv2.polylines(img, [pts], True, color, thickness)


def draw_cat_face(img: np.ndarray, mask: np.ndarray, w: int, h: int, side: str = "left"):
    """Small cream cat decal clipped to the nail mask."""
    decal = np.zeros_like(img)
    cx = int(w * (0.35 if side == "left" else 0.58))
    cy = int(h * 0.66)
    rw = max(5, int(w * 0.28))
    rh = max(5, int(h * 0.15))
    cream = (222, 232, 236)
    outline = (96, 88, 72)
    cv2.ellipse(decal, (cx, cy), (rw, rh), 0, 0, 360, cream, -1)
    ears = np.array([
        [[cx - rw, cy - rh // 2], [cx - rw // 2, cy - rh - max(3, h // 16)], [cx - rw // 5, cy - rh // 3]],
        [[cx + rw, cy - rh // 2], [cx + rw // 2, cy - rh - max(3, h // 16)], [cx + rw // 5, cy - rh // 3]],
    ], dtype=np.int32)
    cv2.fillPoly(decal, list(ears), cream)
    cv2.ellipse(decal, (cx, cy), (rw, rh), 0, 0, 360, outline, max(1, w // 38))
    eye_y = cy - max(1, rh // 8)
    cv2.circle(decal, (cx - rw // 3, eye_y), max(1, w // 42), outline, -1)
    cv2.circle(decal, (cx + rw // 3, eye_y), max(1, w // 42), outline, -1)
    cv2.line(decal, (cx, eye_y + max(1, h // 40)), (cx, eye_y + max(3, h // 18)), outline, max(1, w // 50))
    for direction in (-1, 1):
        cv2.line(decal, (cx + direction * rw // 8, cy + rh // 4),
                 (cx + direction * rw // 2, cy + rh // 6), outline, max(1, w // 60))
    decal_mask = (mask > 32).astype(np.uint8)[:, :, None]
    np.copyto(img, decal, where=(decal_mask.astype(bool) & (decal > 0)))


def synthesize_ai_amber_cat_tip(w: int, h: int, finger: str = "中指", nail_index: int = 0,
                                curvature: float = 0.65) -> np.ndarray:
    """Reference-style generated nail: translucent amber gel, curved gloss, decals."""
    w, h = max(8, int(w)), max(12, int(h))
    mask = make_rounded_tip_mask(w, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.50) / max(w * 0.50, 1)
    ny = (yy - h * 0.48) / max(h * 0.55, 1)
    dome = np.clip(1.0 - (nx ** 2 * 0.95 + ny ** 2 * 0.55), 0, 1)
    side_shadow = np.clip(np.abs(nx), 0, 1)
    vertical = yy / max(h - 1, 1)

    base = np.zeros((h, w, 3), dtype=np.float32)
    amber = np.array([58, 118, 168], dtype=np.float32)   # BGR, translucent caramel
    honey = np.array([82, 152, 198], dtype=np.float32)
    base[:] = amber
    base = base * (0.74 + 0.28 * dome[:, :, None]) + honey * (0.18 * (1 - vertical)[:, :, None])
    base -= side_shadow[:, :, None] * np.array([12, 24, 34], dtype=np.float32)

    # Milky translucent inner glow.
    glow = np.zeros_like(base)
    cv2.ellipse(glow, (int(w * 0.45), int(h * 0.34)), (max(3, int(w * 0.22)), max(3, int(h * 0.18))),
                -18, 0, 360, (34, 62, 92), -1)
    glow = cv2.GaussianBlur(glow, (0, 0), max(1.0, min(w, h) * 0.08))
    base += glow * 0.38

    img = np.clip(base, 0, 255).astype(np.uint8)

    # Gold foil and tiny stars, varied by finger.
    gold = (62, 178, 235)
    if finger in {"拇指", "小指"}:
        pts = np.array([
            [int(w * 0.56), int(h * 0.58)],
            [int(w * 0.77), int(h * 0.52)],
            [int(w * 0.70), int(h * 0.70)],
            [int(w * 0.47), int(h * 0.72)],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], gold)
        cv2.polylines(img, [pts], True, (90, 210, 250), max(1, w // 45))
    if finger in {"中指", "小指"}:
        draw_star(img, (int(w * 0.42), int(h * 0.50)), max(2, min(w, h) // 13), gold)
        draw_star(img, (int(w * 0.62), int(h * 0.43)), max(2, min(w, h) // 18), (90, 215, 250))
    if finger in {"食指", "无名指"}:
        draw_cat_face(img, mask, w, h, side="left" if finger == "食指" else "right")

    # Curved gel highlights and edge thickness.
    highlight = np.zeros_like(img)
    cv2.ellipse(highlight, (int(w * 0.33), int(h * 0.22)),
                (max(3, int(w * 0.20)), max(2, int(h * 0.055))),
                -18, 0, 360, (255, 255, 245), -1)
    cv2.ellipse(highlight, (int(w * 0.72), int(h * 0.36)),
                (max(2, int(w * 0.08)), max(2, int(h * 0.20))),
                10, 0, 360, (255, 245, 225), -1)
    highlight = cv2.GaussianBlur(highlight, (0, 0), max(1.0, min(w, h) * 0.025))
    img = cv2.addWeighted(highlight, 0.36, img, 1.0, 0)

    edge = cv2.Canny(mask, 80, 160)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    edge_blur = cv2.GaussianBlur(edge.astype(np.float32) / 255.0, (5, 5), 0)
    img = np.clip(img.astype(np.float32) * (1 - edge_blur[:, :, None] * 0.08), 0, 255).astype(np.uint8)

    alpha = (mask.astype(np.float32) * 0.84).astype(np.uint8)
    return np.dstack([img, alpha])


def match_lighting(tip_bgr: np.ndarray, hand_bgr: np.ndarray, cx: int, cy: int, w: int, hh: int) -> np.ndarray:
    """HSV 光照匹配: 将贴图色调调整到用户手部光照范围
    策略: 拿手部周围皮肤 V 通道均值, 调整贴图 V 通道
    """
    h_img, w_img = hand_bgr.shape[:2]
    x0 = max(0, cx - w); y0 = max(0, cy - hh)
    x1 = min(w_img, cx + w); y1 = min(h_img, cy + hh)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return tip_bgr
    skin_roi = hand_bgr[y0:y1, x0:x1]
    skin_hsv = cv2.cvtColor(skin_roi, cv2.COLOR_BGR2HSV).astype(np.float32)
    skin_v_mean = skin_hsv[:, :, 2].mean()
    # 贴图 HSV
    tip_hsv = cv2.cvtColor(tip_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    tip_v_mean = tip_hsv[:, :, 2].mean()
    if tip_v_mean < 1:
        return tip_bgr
    # 缩放 V 通道到与皮肤匹配
    scale = skin_v_mean / tip_v_mean
    scale = clamp(scale, 0.6, 1.4)  # 限制调整范围
    tip_hsv[:, :, 2] = np.clip(tip_hsv[:, :, 2] * scale, 0, 255)
    out = cv2.cvtColor(tip_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def prepare_tip_texture_for_render(tip_bgr: np.ndarray, style_id: int, target_w: int, target_h: int,
                                   curvature: float = 0.0, finger: str = "中指") -> np.ndarray:
    """Turn an evaluation close-up into a clean nail-polish texture.

    The generated tip assets are often cropped from product photos, so they may
    contain neighboring nails, skin, shadows, and photo edges. Rendering the whole
    rectangle makes the result look like a sticker. This isolates the likely nail
    area, fills non-nail pixels, and blends it with a synthetic polish base so the
    final texture behaves like a real nail surface.
    """
    h, w = tip_bgr.shape[:2]
    if h < 8 or w < 8:
        return cv2.resize(tip_bgr, (max(1, target_w), max(1, target_h)), interpolation=cv2.INTER_CUBIC)

    mask = extract_nail_only(tip_bgr)
    mask_bin = (mask > 96).astype(np.uint8) * 255
    coverage = cv2.countNonZero(mask_bin) / float(h * w)

    texture = tip_bgr
    if 0.08 <= coverage <= 0.78:
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            x, y, bw, bh = cv2.boundingRect(largest)
            pad_x = max(2, int(bw * 0.10))
            pad_y = max(2, int(bh * 0.10))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + bw + pad_x)
            y1 = min(h, y + bh + pad_y)
            crop = tip_bgr[y0:y1, x0:x1].copy()
            crop_mask = mask_bin[y0:y1, x0:x1]
            outside = cv2.bitwise_not(crop_mask)
            if cv2.countNonZero(outside) > 0:
                texture = cv2.inpaint(crop, outside, 3, cv2.INPAINT_TELEA)
            else:
                texture = crop

    texture = cv2.resize(texture, (max(1, target_w), max(1, target_h)), interpolation=cv2.INTER_CUBIC)
    texture = cv2.bilateralFilter(texture, 7, 55, 55)
    synthetic = synthesize_style_tip(style_id, max(1, target_w), max(1, target_h), curvature=curvature, finger=finger)
    short_side = min(target_w, target_h)
    texture_alpha = 0.08 if short_side < 42 else 0.16 if short_side < 70 else 0.24
    return cv2.addWeighted(texture, texture_alpha, synthetic, 1.0 - texture_alpha, 0)


def reinforce_style_color_if_needed(tip_bgr: np.ndarray, style_id: int, curvature: float = 0.0) -> np.ndarray:
    """Prevent skin/background-heavy crops from becoming skin-colored stickers."""
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    base_bgr = np.array(hex_to_bgr(style.get("color_hex", "#d9b8ad")), dtype=np.float32)
    base_luma = float(0.114 * base_bgr[0] + 0.587 * base_bgr[1] + 0.299 * base_bgr[2])
    mean = tip_bgr.reshape(-1, 3).mean(axis=0).astype(np.float32)
    tip_luma = float(0.114 * mean[0] + 0.587 * mean[1] + 0.299 * mean[2])
    color_gap = np.linalg.norm(mean - base_bgr)

    if base_luma < 95 and tip_luma > 105 and color_gap > 70:
        h, w = tip_bgr.shape[:2]
        synthetic = synthesize_style_tip(style_id, w, h, curvature=curvature)
        return cv2.addWeighted(synthetic, 0.72, tip_bgr, 0.28, 0)
    return tip_bgr


def hex_to_bgr(hex_str: str) -> tuple:
    h = hex_str.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def synthesize_style_tip(style_id: int, w: int, h: int, curvature: float = 0.0, finger: str = "中指") -> np.ndarray:
    """从 STYLE_LIBRARY 的色卡生成标准甲片贴图 (v4: 3D 曲率感知 shading)
    curvature: 0-1, 越大 = 指甲越凸 -> 高光越窄越集中 + 边缘阴影越强
    finger: 拇指/食指/中指/无名指/小指 - 高光位置跟手指类型变
    """
    s = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    base_color = hex_to_bgr(s['color_hex'])
    name = s['name']
    is_french = '法式' in name or 'French' in name
    is_glitter = '亮片' in name or '闪' in name

    # 按手指调高光位置 — 统一光源 (整只手高光方向一致, 只微调以适应指甲形状)
    # 统一从左上方打光: 高光 在指甲 上 偏左 位置
    # 小指 (窄) 偏左多一点, 拇指 (宽) 偏中间
    finger_hl = {
        "拇指":  (0.50, 0.32),  # 居中偏上 (指甲最宽, 不偏)
        "食指":  (0.48, 0.30),  # 偏左上一点
        "中指":  (0.50, 0.30),  # 居中偏上
        "无名指": (0.50, 0.30),  # 居中偏上
        "小指":  (0.50, 0.30),  # 居中偏上 (不偏左, 避免太靠边)
    }.get(finger, (0.50, 0.30))

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = base_color

    # 1. 中心高亮，边缘暗（3D 球面感），强度随曲率变化
    cx, cy = w // 2, int(h * 0.5)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    # 曲率影响: 曲率大时高光衰减更陡 (更亮中心 + 更暗边缘)
    falloff = 1.0 + curvature * 1.8
    shade = (1 - dist / max_dist) * 25 * falloff - (12 + curvature * 8)
    img_f = img.astype(np.float32) + shade[:, :, None]
    img = np.clip(img_f, 0, 255).astype(np.uint8)

    # 2. 顶部高光弧 (曲率大时变窄变亮, 位置跟手指类型)
    overlay = img.copy()
    hl_w = int(w * (0.38 - curvature * 0.15))
    hl_h = int(h * (0.13 - curvature * 0.04))
    hl_cx = int(w * finger_hl[0])
    hl_cy = int(h * finger_hl[1])
    cv2.ellipse(overlay, (hl_cx, hl_cy),
                (max(4, hl_w), max(3, hl_h)), 0, 0, 360, (255, 255, 255), -1)
    hl_alpha = 0.32 + curvature * 0.18
    img = cv2.addWeighted(overlay, hl_alpha, img, 1 - hl_alpha, 0)

    # 3. 二次小高光 (曲率大时往中心偏, 位置跟手指类型)
    overlay2 = img.copy()
    sx = int(w * (finger_hl[0] - curvature * 0.05))
    sy = int(h * (finger_hl[1] + 0.10))
    sr = max(2, int(w * (0.06 + curvature * 0.03)))
    cv2.circle(overlay2, (sx, sy), sr, (255, 255, 255), -1)
    img = cv2.addWeighted(overlay2, 0.18 + curvature * 0.12, img, 1 - (0.18 + curvature * 0.12), 0)

    # 4. 月牙线 (cuticle) - 指甲根部弧线 (曲率大时更明显)
    overlay3 = img.copy()
    cv2.ellipse(overlay3, (w // 2, int(h * 0.85)),
                (int(w * 0.42), int(h * 0.10)),
                0, 0, 180, (210, 210, 210), max(1, h // 30))
    cuticle_alpha = 0.4 + curvature * 0.15
    img = cv2.addWeighted(overlay3, cuticle_alpha, img, 1 - cuticle_alpha, 0)

    # 5. 底部深度 (曲率大时颜色更深, 模拟根部凹陷)
    overlay4 = img.copy()
    depth = 25 + int(curvature * 20)
    cv2.ellipse(overlay4, (w // 2, int(h * 0.88)),
                (int(w * 0.42), int(h * 0.12)),
                0, 0, 360,
                (max(0, base_color[0] - depth),
                 max(0, base_color[1] - depth),
                 max(0, base_color[2] - depth)), -1)
    img = cv2.addWeighted(overlay4, 0.20 + curvature * 0.10, img, 1 - (0.20 + curvature * 0.10), 0)

    # 6. 法式白边
    if is_french:
        cv2.ellipse(img, (w // 2, int(h * 0.85)),
                    (int(w * 0.40), int(h * 0.13)),
                    0, 0, 360, (250, 245, 240), -1)

    # 7. 亮片
    if is_glitter:
        np.random.seed(style_id)
        for _ in range(int(w * h * 0.005)):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            cv2.circle(img, (x, y), 1, (255, 255, 240), -1)

    # 8. 边缘描边 (曲率大时更深)
    edge_depth = 30 + int(curvature * 20)
    cv2.ellipse(img, (w // 2, h // 2),
                (int(w * 0.44), int(h * 0.45)), 0, 0, 360,
                (max(0, base_color[0] - edge_depth),
                 max(0, base_color[1] - edge_depth),
                 max(0, base_color[2] - edge_depth)), 1)
    return img


def synthesize_white_tip(w: int, h: int) -> np.ndarray:
    """白模：白色椭圆+高光，便于调试定位"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # 椭圆主体（指甲形状：稍圆稍长）
    cx, cy = w // 2, int(h * 0.5)
    rx, ry = int(w * 0.42), int(h * 0.45)
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (245, 240, 235), -1)
    # 顶部高光（弧形白条）
    cv2.ellipse(img, (cx, int(h * 0.3)), (int(rx * 0.4), int(ry * 0.15)),
                0, 0, 360, (255, 255, 255), -1)
    # 边缘深色描边
    cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (180, 160, 150), 2)
    return img


def warp_style_to_nail(style_img: np.ndarray, target_w: int, target_h: int, finger: str = "middle") -> np.ndarray:
    """透视变换：把款式贴图 warp 到真实指甲形状
    v10-fix8: 渲染贴图后, alpha mask 是严格椭圆, 看起来是椭圆形指甲 (不是矩形)
    关键: warp 本身不变成椭圆 (cv2 透视只能矩形变矩形),
          但 alpha mask 乘椭圆后, 边缘按椭圆渐变, 用户看到椭圆形
    """
    sh, sw = style_img.shape[:2]
    src_pts = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    # v10-fix4/8: dst 4 角内收 0.01 (几乎贴边, 让贴图占满 4 角)
    finger_warp = {
        "拇指": dict(top_pad_y=0.02, side_pad_x=0.01, bottom_pad_y=0.04, side_curve=0.04),
        "食指": dict(top_pad_y=0.02, side_pad_x=0.01, bottom_pad_y=0.04, side_curve=0.05),
        "中指": dict(top_pad_y=0.02, side_pad_x=0.01, bottom_pad_y=0.04, side_curve=0.06),
        "无名指": dict(top_pad_y=0.02, side_pad_x=0.01, bottom_pad_y=0.04, side_curve=0.07),
        "小指": dict(top_pad_y=0.02, side_pad_x=0.01, bottom_pad_y=0.04, side_curve=0.08),
    }
    p = finger_warp.get(finger, finger_warp["中指"])
    cx, cy = target_w / 2, target_h / 2
    dst_pts = np.array([
        [target_w * p["side_pad_x"],            target_h * p["top_pad_y"]],                          # 左上
        [target_w * (1 - p["side_pad_x"]),     target_h * p["top_pad_y"]],                          # 右上
        [target_w * (1 - p["side_pad_x"] - p["side_curve"]), target_h * (1 - p["bottom_pad_y"])],   # 右下 (内收多一点, 模拟甲根)
        [target_w * (p["side_pad_x"] + p["side_curve"]),    target_h * (1 - p["bottom_pad_y"])],   # 左下
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(style_img, M, (target_w, target_h))
    return warped


def rotation_for_vertical_tip(angle: float) -> float:
    """Return rotation degrees for a tip whose source long axis is vertical."""
    import math
    deg = math.degrees(angle) - 90.0
    while deg < -90.0:
        deg += 180.0
    while deg > 90.0:
        deg -= 180.0
    return deg


def rotate_image_bound(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate image without clipping transparent corners."""
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2.0 - center[0]
    M[1, 2] += new_h / 2.0 - center[1]
    if img.shape[2] == 4:
        border = (0, 0, 0, 0)
    else:
        border = (0, 0, 0)
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def grabcut_nail_mask(hand_bgr: np.ndarray, cx: int, cy: int, w: int, hh: int) -> np.ndarray:
    """GrabCut 抠出真实指甲轮廓
    输入：手图 + 指甲框 (cx, cy, w, hh)
    返回：单指甲 mask (h_img, w_img, uint8 0-255)
    """
    h_img, w_img = hand_bgr.shape[:2]
    # 框 (扩展 1.2x 用于 GrabCut 上下文)
    pad_x = int(w * 0.4)
    pad_y = int(hh * 0.3)
    x0 = max(0, cx - w // 2 - pad_x)
    y0 = max(0, cy - hh // 2 - pad_y)
    x1 = min(w_img, cx + w // 2 + pad_x)
    y1 = min(h_img, cy + hh // 2 + pad_y)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None

    roi = hand_bgr[y0:y1, x0:x1].copy()
    rh, rw = roi.shape[:2]

    # 初始化: 边缘 1px 是 BG, 里面 5px 是 PR_BGD, 中心椭圆是 PR_FGD
    mask = np.full((rh, rw), cv2.GC_PR_BGD, dtype=np.uint8)
    # 边缘 5px 肯定是背景
    cv2.rectangle(mask, (0, 0), (rw, 5), cv2.GC_BGD, -1)  # 顶部 5px
    cv2.rectangle(mask, (0, rh - 5), (rw, rh), cv2.GC_BGD, -1)  # 底部 5px
    cv2.rectangle(mask, (0, 0), (5, rh), cv2.GC_BGD, -1)  # 左 5px
    cv2.rectangle(mask, (rw - 5, 0), (rw, rh), cv2.GC_BGD, -1)  # 右 5px
    # 中心指甲 (60% 宽, 60% 高) 置为 FG
    cx_roi, cy_roi = rw // 2, rh // 2
    rx_roi = int(rw * 0.30)
    ry_roi = int(rh * 0.30)
    cv2.ellipse(mask, (cx_roi, cy_roi), (rx_roi, ry_roi), 0, 0, 360, cv2.GC_FGD, -1)
    # 周边 20% 是 PR_FGD (可能指甲)
    rx_outer = int(rw * 0.45)
    ry_outer = int(rh * 0.45)
    cv2.ellipse(mask, (cx_roi, cy_roi), (rx_outer, ry_outer), 0, 0, 360, cv2.GC_PR_FGD, 1)

    # 肤色检测作为先验
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # 指甲区域: 低饱和度 (指甲中心, 颜色不鲜艳)
    nail_hint = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 160)) & \
                (hsv[:, :, 1] >= 5) & (hsv[:, :, 1] <= 80) & \
                (hsv[:, :, 2] >= 80) & (hsv[:, :, 2] <= 245)
    # 增强中心指甲 hint (距离中心越近权重越高)
    Y, X = np.ogrid[:rh, :rw]
    dist = np.sqrt((X - cx_roi) ** 2 + (Y - cy_roi) ** 2).astype(np.float32)
    max_dist = max(rx_outer, ry_outer)
    center_weight = 1.0 - np.clip(dist / max_dist, 0, 1)
    # 中心 hint 一定为 FG
    strong_hint = nail_hint & (center_weight > 0.5)
    mask[strong_hint] = cv2.GC_FGD
    # 中心弱 hint 置为 PR_FGD
    weak_hint = nail_hint & (center_weight > 0.2) & (center_weight <= 0.5)
    mask[weak_hint] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return None

    # 生成输出 mask
    nail_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 只保留最大连通域
    contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    nail_mask = np.zeros_like(nail_mask)
    largest = max(contours, key=cv2.contourArea)
    # 凸包填补 (指甲边界可能不连续)
    hull = cv2.convexHull(largest)
    cv2.drawContours(nail_mask, [hull], -1, 255, -1)
    # 限制在中心 50% 范围内 (避免溢出) — 用 AND 不是覆盖
    bound_mask = np.zeros_like(nail_mask)
    cv2.ellipse(bound_mask, (cx_roi, cy_roi), (rx_outer, ry_outer), 0, 0, 360, 255, -1)
    nail_mask = cv2.bitwise_and(nail_mask, bound_mask)

    # 平滑边缘
    nail_mask = cv2.GaussianBlur(nail_mask, (5, 5), 0)
    # 贴回原图大小
    full_mask = np.zeros((h_img, w_img), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = nail_mask
    return full_mask


def get_finger_angle(nail: dict) -> float:
    """获取手指方向角 (弧度) — 从 tip 指向 dip"""
    if 'tip_xy' in nail and 'dip_xy' in nail:
        tx, ty = nail['tip_xy']
        dx, dy = nail['dip_xy']
        import math
        return math.atan2(dy - ty, dx - tx)
    return 0.0  # 默认垂直 (从上到下)


def warp_tip_to_nail(tip_bgr: np.ndarray, dst_corners: np.ndarray, dst_size: tuple) -> np.ndarray:
    """透视变换: 把贴图 warp 到目标指甲 4 角点 (支持 BGR 或 BGRA)
    返回: 跟输入同通道数
    """
    sh, sw = tip_bgr.shape[:2]
    h_img, w_img = dst_size
    src_pts = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_corners.astype(np.float32))
    if tip_bgr.shape[2] == 4:
        # BGRA 贴图: 拆分 BGR + A 分别 warp, 再合并
        bgr = tip_bgr[:, :, :3]
        alpha = tip_bgr[:, :, 3]
        warped_bgr = cv2.warpPerspective(bgr, M, (w_img, h_img))
        warped_a = cv2.warpPerspective(alpha, M, (w_img, h_img))
        warped = np.dstack([warped_bgr, warped_a])
        return warped
    else:
        warped = cv2.warpPerspective(tip_bgr, M, (w_img, h_img))
        return warped


def apply_highlight(warped: np.ndarray, nail_mask: np.ndarray, finger_angle: float) -> np.ndarray:
    """高光跟随: 在贴图上根据手指方向叠动态高光, 模拟指甲曲面反光"""
    h_img, w_img = warped.shape[:2]
    # 高光方向: 垂直手指的反方向 (假设光从上方, 手指向上, 高光在上半部)
    # finger_angle = 0 时手指朝右, 高光在右上
    # finger_angle = pi/2 时手指朝下, 高光在左上
    import math
    # 计算高光位置 (沿手指方向偏 30%, 垂直偏 30%)
    hx = int(w_img * 0.5 + math.cos(finger_angle) * w_img * 0.15 - math.sin(finger_angle) * h_img * 0.20)
    hy = int(h_img * 0.5 + math.sin(finger_angle) * w_img * 0.15 + math.cos(finger_angle) * h_img * 0.20)
    hx = clamp(hx, 0, w_img - 1)
    hy = clamp(hy, 0, h_img - 1)

    # 高光 1: 椭圆弧形 (主光)
    highlight = np.zeros_like(warped, dtype=np.float32)
    rx = max(8, int(w_img * 0.20))
    ry = max(5, int(h_img * 0.10))
    cv2.ellipse(highlight, (hx, hy), (rx, ry), 0, 0, 360, (255, 255, 255), -1)
    highlight = cv2.GaussianBlur(highlight, (15, 15), 0)
    # 只在 mask 内显示
    highlight_mask = (nail_mask / 255.0)[:, :, None]
    highlight = highlight * highlight_mask

    # alpha 混合 (高光透明度 35%)
    out = warped.astype(np.float32) * 1.0 + highlight * 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def render_tryon(hand_bgr: np.ndarray, style_id: int, nails: list, white_mode: bool = None,
                 tuning: dict = None) -> np.ndarray:
    """主渲染 v6: GrabCut 真实轮廓 + 3D 透视 + 高光跟随"""
    if white_mode is None:
        white_mode = DEBUG_WHITE_MODE
    tuning = tuning or {}
    width_scale = clamp(float(tuning.get("width_scale", 1.0)), 0.55, 1.45)
    length_scale = clamp(float(tuning.get("length_scale", 1.0)), 0.55, 1.65)
    offset_scale = clamp(float(tuning.get("offset_scale", 0.0)), -0.45, 0.45)
    opacity = clamp(float(tuning.get("opacity", 0.92)), 0.35, 1.0)
    output = hand_bgr.copy()
    h_img, w_img = hand_bgr.shape[:2]

    for nail_index, nail in enumerate(nails):
        cx, cy = nail["cx"], nail["cy"]
        w, hh = nail["w"], nail["h"]
        if cx < 0 or cy < 0 or cx >= w_img or cy >= h_img:
            continue
        w = clamp(w, 20, w_img // 2)
        hh = clamp(hh, 20, h_img // 2)
        angle = get_finger_angle(nail)
        # Positive offset moves toward the hand, negative offset moves past the fingertip.
        import math
        cx = int(cx + math.cos(angle) * hh * offset_scale)
        cy = int(cy + math.sin(angle) * hh * offset_scale)
        w = clamp(int(w * width_scale), 12, w_img // 2)
        hh = clamp(int(hh * length_scale), 12, h_img // 2)
        use_generated_surface = style_id in GENERATIVE_STYLE_IDS and not white_mode

        # === v10: 用 SAM 抠真实指甲 mask (修 v9m 偏 TIP 问题) ===
        real_nail_mask = None
        v10_used = False
        if USE_SAM_CORRECTION and not white_mode and SAM_AVAILABLE and not use_generated_surface:
            real_nail_mask = sam_segment_nail_in_hand(hand_bgr, cx, cy, w, hh)
        if real_nail_mask is not None and cv2.countNonZero(real_nail_mask) > 50:
            ys, xs = np.where(real_nail_mask > 0)
            real_cx = int(xs.mean())
            real_cy = int(ys.mean())
            span_x = int(xs.max() - xs.min() + 1)
            span_y = int(ys.max() - ys.min() + 1)
            real_w = max(12, int(span_x * 1.05))
            real_h = max(14, int(span_y * 1.08))
            # 用椭圆拟合 (更适合指甲形状)
            import math
            pts = np.column_stack([xs.astype(np.int32), ys.astype(np.int32)])
            mask_angle = None
            if len(pts) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pts)
                    # ellipse = ((cx, cy), (w, h), angle)
                    mask_angle = math.radians(ellipse[2] - 90)
                    # fitEllipse sizes are full-axis lengths. Keep them full;
                    # halving them makes the rendered nail look like a tiny tile.
                    axis_a, axis_b = ellipse[1]
                    sam_w = int(min(axis_a, axis_b))
                    sam_h = int(max(axis_a, axis_b))
                    real_w = max(real_w, int(sam_w * 1.05), int(w * 0.70))
                    real_h = max(real_h, int(sam_h * 1.05), int(hh * 0.72))
                except Exception:
                    pass
            # sanity check: mask 面积 vs MediaPipe 估算 ellipse 面积
            # SAM 抠出来的 mask 总是小于 MediaPipe 估算 (因为估算保守), 下限放低到 0.15
            mask_area = float(cv2.countNonZero(real_nail_mask))
            mp_area = float((w * 0.5) * (hh * 0.5) * 3.14159)  # estimated ellipse area
            area_ratio = mask_area / max(mp_area, 1)
            finger_name = nail.get("finger", "?")
            # Only reject masks that are clearly implausible or have drifted to
            # nearby skin/background. Small bare nails still need to be accepted.
            drift = math.hypot(real_cx - cx, real_cy - cy)
            if 0.12 <= area_ratio <= 4.0 and drift < max(80, hh * 0.75) and real_w >= 12 and real_h >= 12:
                if mask_angle is not None:
                    nail['mask_angle'] = mask_angle
                cx, cy = real_cx, real_cy
                # SAM refines the nail bed; it must not turn a short natural nail
                # into an extension-length sticker.
                w = max(10, min(int(real_w), int(w * 1.10)))
                hh = max(12, min(int(real_h), int(hh * 1.12)))
                v10_used = True
                log.info(f"  v10 {finger_name}: mp=({nail['cx']},{nail['cy']},{nail['w']},{nail['h']}) -> sam=({real_cx},{real_cy},{real_w},{real_h}) final=({cx},{cy},{w},{hh}) area={area_ratio:.2f} drift={drift:.0f}")
            else:
                log.warning(f"  v10 {finger_name} FALLBACK: area_ratio={area_ratio:.2f} drift={drift:.0f} w={real_w} h={real_h} (mp=({nail['cx']},{nail['cy']},{nail['w']},{nail['h']}))")
                real_nail_mask = None  # 禁用 step 4

        # 1. 贴图 (v7: 优先真贴图, fallback 合成)
        curvature = nail.get("curvature", 0.0)
        real_tip_bgr = None
        real_tip_mask = None
        if USE_REAL_TIP_TEXTURE and not white_mode:
            # 选 该手指位置  (thumb/index/middle/ring/pinky)
            finger = nail.get("finger", "middle")
            finger_en = {"拇指": "thumb", "食指": "index", "中指": "middle",
                         "无名指": "ring", "小指": "pinky"}.get(finger, "middle")
            path = find_tip_path(style_id, finger_en)
            if path:
                real_tip_bgr = cv2.imread(path)
            if real_tip_bgr is None:
                real_tip_bgr = load_style_image_as_tip(style_id, max(1, int(w)), max(1, int(hh)))
        if white_mode:
            tip_w = max(1, int(w * 1.00))
            tip_h = max(1, int(hh * 1.00))
            tip = synthesize_white_tip(tip_w, tip_h)
        elif use_generated_surface:
            finger_name = nail.get("finger", "中指")
            # Customer reference is an AI-generated artificial nail, not a
            # texture pasted inside the original nail mask. Let the generated
            # surface extend beyond the bare nail bed and curve as one cap.
            tip_w = max(1, int(w * (1.18 if finger_name == "拇指" else 1.08)))
            tip_h = max(1, int(hh * (1.46 if finger_name == "拇指" else 1.36)))
            cy = int(cy + math.sin(angle) * hh * 0.12)
            cx = int(cx + math.cos(angle) * hh * 0.12)
            tip = synthesize_ai_amber_cat_tip(tip_w, tip_h, finger=finger_name,
                                              nail_index=nail_index, curvature=max(curvature, 0.65))
            tip[:, :, :3] = match_lighting(tip[:, :, :3], hand_bgr, cx, cy, tip_w, tip_h)
            opacity = max(opacity, 0.90)
        elif real_tip_bgr is not None:
            tip_h, tip_w = max(1, int(hh)), max(1, int(w))
            finger_name = nail.get("finger", "中指")
            # Clean product-photo crops into a nail-only polish texture. The
            # final shape is still controlled by target alpha / real nail mask.
            tip = prepare_tip_texture_for_render(real_tip_bgr, style_id, tip_w, tip_h, curvature, finger_name)
            tip = match_lighting(tip, hand_bgr, cx, cy, w, hh)
            tip = reinforce_style_color_if_needed(tip, style_id, curvature)
            tip = cv2.cvtColor(tip, cv2.COLOR_BGR2BGRA)
            tip[:, :, 3] = make_nail_alpha_mask(tip_w, tip_h, curvature, finger=finger_name)
        else:
            tip_w = max(1, int(w * 1.00))
            tip_h = max(1, int(hh * 1.00))
            finger_name = nail.get("finger", "中指")
            tip = synthesize_style_tip(style_id, tip_w, tip_h, curvature=curvature, finger=finger_name)
            tip = cv2.cvtColor(tip, cv2.COLOR_BGR2BGRA)
            tip[:, :, 3] = make_nail_alpha_mask(tip_w, tip_h, curvature, finger=finger_name)

        # 2. 计算贴图变换 (B 级: 手指感知透视 + 旋转)
        tip_h, tip_w = tip.shape[:2] if len(tip.shape) == 3 else (tip.shape[0], tip.shape[1])
        # 2a. 手指感知透视 warp (模拟指甲弧面, 按手指类型用不同参数)
        finger_name = nail.get("finger", "中指")
        # synthesize_style_tip 输出的贴图本身就是 竖向 (h>w), 直接做透视 warp
        tip_warped = warp_style_to_nail(tip, tip_w, tip_h, finger=finger_name)
        # 重新生成 alpha (根据手指类型)
        new_alpha = make_nail_alpha_mask(tip_w, tip_h, curvature, finger=finger_name)
        # 2b. 旋转到手指方向。贴图源长轴是竖直的，所以只旋转到相对竖轴的偏差。
        if tip_warped.shape[2] == 4:
            tip_warped[:, :, 3] = new_alpha
        rot_deg = rotation_for_vertical_tip(angle)
        if abs(rot_deg) > 2:
            tip = rotate_image_bound(tip_warped, rot_deg)
        else:
            tip = tip_warped
        warped = tip  # 别名, 后面逻辑不变
        # 同步更新 tip_w/tip_h 为新尺寸
        tip_h, tip_w = tip.shape[:2]

        # 4. v9m: 简化 -> 不需要 GrabCut, 直接 椭圆 限制
        # 在 output 上 (cx, cy) 画椭圆, 作为贴图区域的 mask
        ksize = max(3, int(min(w, hh) * 0.04)) // 2 * 2 + 1
        # 椭圆: 以 box 中心为心, 宽 w*0.85, 高 hh*0.95, 旋转手指方向
        # 5. 在 warped (tip 贴图) 上 生成 mask
        tip_h, tip_w = warped.shape[:2]
        if warped.shape[2] == 4:
            # BGRA: 用 alpha 通道 作为 mask
            alpha_w = warped[:, :, 3].astype(np.float32) / 255.0
            # v10-fix5: 取消椭圆 AND 限制! 之前 alpha*ellipse_f 把覆盖率从 79% 压到 48%
            # 改: 直接用 alpha (make_nail_alpha_mask 已生成指甲形状, 覆盖率 79%)
            a3_tip = (alpha_w * opacity)[:, :, None]
        else:
            # BGR 合成色: 用 椭圆 mask (但加大到 90% 短边)
            short_dim = min(tip_w, tip_h)
            rx_mask = max(2, int(short_dim * 0.49))
            ry_mask = max(2, int(short_dim * 0.49))
            ellipse_mask_tip = np.zeros((tip_h, tip_w), np.uint8)
            cv2.ellipse(ellipse_mask_tip, (tip_w // 2, tip_h // 2),
                        (rx_mask, ry_mask), 0, 0, 360, 255, -1)
            ellipse_f = cv2.GaussianBlur(ellipse_mask_tip.astype(np.float32) / 255.0, (ksize, ksize), 0)
            a3_tip = (ellipse_f * opacity)[:, :, None]

        # 6. 把 warped 贴到 output 区域 (cx-tip_w/2:cx+tip_w/2, cy-tip_h/2:cy+tip_h/2)
        x0 = max(0, cx - tip_w // 2)
        y0 = max(0, cy - tip_h // 2)
        x1 = min(w_img, cx + tip_w // 2 + (tip_w % 2))
        y1 = min(h_img, cy + tip_h // 2 + (tip_h % 2))
        # 调整 a3_tip 到 对应的 区域
        ax0 = max(0, tip_w // 2 - cx)
        ay0 = max(0, tip_h // 2 - cy)
        ax1 = ax0 + (x1 - x0)
        ay1 = ay0 + (y1 - y0)
        a3_region = a3_tip[ay0:ay1, ax0:ax1]

        # === v10-fix1: 真实 mask + 椭圆软混合 (避免硬切) ===
        if real_nail_mask is not None and cv2.countNonZero(real_nail_mask) > 50:
            real_local = real_nail_mask[y0:y1, x0:x1]
            region_h, region_w = real_local.shape[:2]
            if region_h > 0 and region_w > 0:
                # Real nail mask is the boundary contract: outside it, no polish.
                # Keep only a tiny feather for natural anti-aliasing.
                feather_ksize = max(3, int(min(region_w, region_h) * 0.06) // 2 * 2 + 1)
                feather_ksize = min(feather_ksize, 7)
                feathered = cv2.GaussianBlur(real_local, (feather_ksize, feather_ksize), 0)
                real_region = feathered.astype(np.float32) / 255.0
                if real_region.shape == a3_region.shape[:2]:
                    a3_region = (real_region * opacity)[:, :, None]

        if a3_region.size > 0 and (y1 - y0) > 0 and (x1 - x0) > 0:
            output_f = output[y0:y1, x0:x1].astype(np.float32)
            warped_f = warped[ay0:ay1, ax0:ax1, :3].astype(np.float32)
            blended = output_f * (1 - a3_region) + warped_f * a3_region
            # === 缝隙阴影 (B 级): 指甲与皮肤的凹槽造成 3 层暗影 ===
            if real_nail_mask is not None and cv2.countNonZero(real_nail_mask) > 50:
                real_local = real_nail_mask[y0:y1, x0:x1]
                region_h, region_w = real_local.shape[:2]
                # 1) 边缘环 (侧甲沟) — 暗影 25%
                ksize = max(3, int(min(region_w, region_h) * 0.08) // 2 * 2 + 1)
                kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                dilated_edge = cv2.dilate(real_local, kernel_edge)
                edge_ring = cv2.subtract(dilated_edge, real_local).astype(np.float32) / 255.0
                edge_ring = cv2.GaussianBlur(edge_ring, (5, 5), 0)
                # 2) 甲根弧线 (cuticle) — 指甲下方那条弧, 暗影 30%
                cuticle_shadow = np.zeros_like(real_local, dtype=np.float32)
                # 指甲下部 15% 区为 cuticle 区
                cuticle_y_start = int(region_h * 0.85)
                cuticle_band = real_local[cuticle_y_start:, :].astype(np.float32) / 255.0
                # 在 cuticle 上做一个"上深下浅"梯度
                cuticle_band_h = cuticle_band.shape[0]
                for yi in range(cuticle_band_h):
                    fade = 1.0 - (yi / max(cuticle_band_h, 1))  # 越靠指甲越深
                    cuticle_band[yi, :] *= fade
                cuticle_shadow[cuticle_y_start:, :] = cuticle_band
                cuticle_shadow = cv2.GaussianBlur(cuticle_shadow, (7, 7), 0)
                # 3) 指甲与皮肤交界的微环 (软化边缘提拉) — 暗影 18%
                ksize2 = max(3, int(min(region_w, region_h) * 0.04) // 2 * 2 + 1)
                kernel_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize2, ksize2))
                eroded = cv2.erode(real_local, kernel_in)
                inner_ring = cv2.subtract(real_local, eroded).astype(np.float32) / 255.0
                inner_ring = cv2.GaussianBlur(inner_ring, (3, 3), 0)
                # 总暗影 = 边缘环 * 0.25 + cuticle * 0.30 + inner_ring * 0.18
                total_shadow = edge_ring * 0.25 + cuticle_shadow * 0.30 + inner_ring * 0.18
                total_shadow = np.clip(total_shadow, 0, 0.5)  # 上限 50% 暗
                # 应用暗影 (指甲区与手部区都受甲沟阴影影响, 但仅在贴图边缘位置明显)
                blended = blended * (1 - total_shadow[:, :, None])
                # === 指甲面高光 (B+ 级): 沿手指方向叠高光, 模拟曲面反光 ===
                # 高光在指甲上部 35% 位置 (假设光从上方), 指甲 3D 凸起上亮度最高
                hl_alpha_local = np.zeros((region_h, region_w), dtype=np.float32)
                hl_y = int(region_h * 0.35)  # 上部 35% 处
                hl_w_radius = max(4, int(region_w * 0.18))
                hl_h_radius = max(3, int(region_h * 0.08))
                # 只在 mask 范围内画高光
                cv2.ellipse(hl_alpha_local, (region_w // 2, hl_y),
                            (hl_w_radius, hl_h_radius),
                            0, 0, 360, 1.0, -1)
                # 限在 mask 内
                hl_alpha_local = hl_alpha_local * (real_local.astype(np.float32) / 255.0)
                # 羽化 (加大核 7→11 让高光边缘更柔和)
                hl_alpha_local = cv2.GaussianBlur(hl_alpha_local, (11, 11), 0)
                # 18% 高光增强 (lighten blend, 不压缩)
                hl_brightness = 0.22 * hl_alpha_local
                blended = blended * (1 - hl_brightness[:, :, None]) + 255.0 * hl_brightness[:, :, None]
                # === 指甲半透明 (B+ 级): 甲尖 (h*0.0-0.30) 透出底层肤色 ===
                # 采样原图甲尖区颜色作为透出底色
                tip_y_end = int(region_h * 0.30)
                if tip_y_end > 0:
                    # 采集原图指甲上 30% 区作为肤色 (output 在叠贴图前是原图)
                    skin_sample = output_f[0:tip_y_end, :, :]
                    # mask 在这区
                    tip_mask = real_local[0:tip_y_end, :].astype(np.float32) / 255.0
                    # 越靠甲尖越透 (30% 透明度, 递减)
                    for yi in range(tip_y_end):
                        fade = 1.0 - (yi / tip_y_end)  # 越靠尖越透
                        tip_mask[yi, :] *= fade
                    tip_mask = cv2.GaussianBlur(tip_mask, (3, 3), 0)
                    tip_alpha = (tip_mask * 0.30)[:, :, None]  # 上限 30% 透
                    # 在甲尖区混合肤色
                    blended[0:tip_y_end, :, :] = blended[0:tip_y_end, :, :] * (1 - tip_alpha) + skin_sample * tip_alpha
            output[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
        # 继续下一个 nail

    return output


# ==================== 路由 ====================
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route("/<path:filename>")
def static_file(filename):
    # 优先尝试 assets (测试图、款式图)
    if filename.startswith('assets/'):
        try:
            return send_from_directory(PROJECT_ROOT, filename)
        except Exception:
            pass
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "styles_available": len(STYLE_LIBRARY),
        "target_total": TARGET_TOTAL_STYLES,
        "debug_white_mode": DEBUG_WHITE_MODE,
        "curvature_aware": True,
        "qwen_image_edit": USE_QWEN_IMAGE_EDIT,
        "qwen_image_edit_models": QWEN_IMAGE_EDIT_MODELS,
        "qwen_image_edit_timeout": QWEN_IMAGE_EDIT_TIMEOUT,
        "real_tip_texture": USE_REAL_TIP_TEXTURE,
        "sam_available": SAM_AVAILABLE,
        "sam_correction": USE_SAM_CORRECTION,
        "version": "v11-ai-image-edit",
        "style_manifest": os.path.exists(STYLE_MANIFEST_PATH),
        "tip_dirs": [d for d in TIP_DIRS if os.path.isdir(d)],
    })


@app.route("/api/styles")
def list_styles():
    return jsonify({
        "styles": STYLE_LIBRARY,
        "available": len(STYLE_LIBRARY),
        "target": TARGET_TOTAL_STYLES,
        "note": f"当前 {len(STYLE_LIBRARY)} 款可用, 目标 {TARGET_TOTAL_STYLES} 款 (优先使用评测 Excel 同步的真实款式资产)",
    })


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """自动推荐：用户上传手图，返回推荐款"""
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    hand_bytes = request.files["hand_image"].read()
    arr = np.frombuffer(hand_bytes, dtype=np.uint8)
    hand_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if hand_bgr is None:
        return jsonify({"error": "invalid image"}), 400
    h_orig, w_orig = hand_bgr.shape[:2]

    # resize to 1000 wide
    scale = 1.0
    if w_orig > 1000:
        scale = 1000 / w_orig
        new_w = 1000
        new_h = int(h_orig * scale)
        hand_for_api = cv2.resize(hand_bgr, (new_w, new_h))
    else:
        hand_for_api = hand_bgr
    _, buf = cv2.imencode('.jpg', hand_for_api, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    try:
        rec = recommend_style(data_uri)
    except Exception as e:
        log.exception("recommend failed")
        return jsonify({"error": str(e)}), 500

    # 补上 style 详情
    recs = rec.get("recommendations", [])
    for r in recs:
        sid = r.get("style_id", 1)
        if 1 <= sid <= len(STYLE_LIBRARY):
            r["style"] = STYLE_LIBRARY[sid - 1]
    
    return jsonify(rec)


@app.route("/api/recommend_tryon", methods=["POST"])
def api_recommend_tryon():
    """客户要求的一键流程：手图 -> AI 推荐款式 -> 渲染试戴图 -> 适配度/趋势文本。"""
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    white_mode = parse_bool(request.form.get("white_mode"), False)
    tuning = parse_tuning(request.form)

    hand_bytes = request.files["hand_image"].read()
    arr = np.frombuffer(hand_bytes, dtype=np.uint8)
    hand_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if hand_bgr is None:
        return jsonify({"error": "invalid image"}), 400
    h_orig, w_orig = hand_bgr.shape[:2]

    scale = 1.0
    if w_orig > 1000:
        scale = 1000 / w_orig
        hand_for_api = cv2.resize(hand_bgr, (1000, int(h_orig * scale)))
    else:
        hand_for_api = hand_bgr
    _, buf = cv2.imencode('.jpg', hand_for_api, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    try:
        rec = recommend_style(data_uri)
    except Exception:
        log.exception("recommend_tryon recommend failed")
        rec = {
            "analysis": {"skin_tone": "中性", "hand_type": "纤细", "nail_status": "裸甲", "scene": "日常"},
            "recommendations": [{"style_id": 1, "reason": "温柔裸色适合多数肤色"}],
        }

    recs = rec.get("recommendations", []) or [{"style_id": 1, "reason": "默认推荐"}]
    style_id = clamp(safe_int(recs[0].get("style_id"), 1), 1, len(STYLE_LIBRARY))
    for r in recs:
        sid = safe_int(r.get("style_id"), 1)
        if 1 <= sid <= len(STYLE_LIBRARY):
            r["style"] = STYLE_LIBRARY[sid - 1]

    try:
        nails = locate_nails(hand_bgr, data_uri)
    except Exception:
        log.exception("recommend_tryon locate failed")
        nails = []

    if nails and scale != 1.0 and any(n.get("source") == "llm" for n in nails):
        inv = 1.0 / scale
        for n in nails:
            n["cx"] = int(n["cx"] * inv)
            n["cy"] = int(n["cy"] * inv)
            n["w"] = int(n["w"] * inv)
            n["h"] = int(n["h"] * inv)

    render_engine = "local-opencv"
    qwen_result = None
    if not white_mode:
        try:
            qwen_result = qwen_image_tryon(hand_bgr, style_id, nails_count=len(nails))
        except Exception:
            log.exception("recommend_tryon qwen image edit failed")
    if qwen_result:
        result_b64 = qwen_result["result_image"]
        render_engine = f"qwen-image-edit:{qwen_result['model']}"
    else:
        try:
            output_bgr = render_tryon(hand_bgr, style_id, nails, white_mode=white_mode, tuning=tuning) if nails else hand_bgr
            ok, out_buf = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("encode failed")
            result_b64 = "data:image/jpeg;base64," + base64.b64encode(out_buf.tobytes()).decode()
        except Exception:
            log.exception("recommend_tryon render failed")
            _, fb = cv2.imencode(".jpg", hand_bgr)
            result_b64 = "data:image/jpeg;base64," + base64.b64encode(fb.tobytes()).decode()

    try:
        advice = generate_fit_trend(data_uri, style_id, rec.get("analysis", {}))
    except Exception:
        log.exception("recommend_tryon advice failed")
        advice = generate_fit_trend("", style_id, rec.get("analysis", {}))

    return jsonify({
        "result_image": result_b64,
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": len(nails),
        "debug_white_mode": white_mode,
        "analysis": rec.get("analysis", {}),
        "recommendations": recs,
        "fit_score": advice["fit_score"],
        "fit_text": advice["fit_text"],
        "trend_text": advice["trend_text"],
        "tuning": tuning,
        "render_engine": render_engine,
    })


@app.route("/api/tryon", methods=["POST"])
def api_tryon():
    """主试戴：用户上传手图 + style_id，返回渲染图"""
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    style_id = safe_int(request.form.get("style_id"), 1)
    style_id = clamp(style_id, 1, len(STYLE_LIBRARY))
    white_mode = parse_bool(request.form.get("white_mode"), DEBUG_WHITE_MODE)
    tuning = parse_tuning(request.form)

    hand_bytes = request.files["hand_image"].read()
    arr = np.frombuffer(hand_bytes, dtype=np.uint8)
    hand_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if hand_bgr is None:
        return jsonify({"error": "invalid image"}), 400
    h_orig, w_orig = hand_bgr.shape[:2]

    # resize to 1000 wide for API
    scale = 1.0
    if w_orig > 1000:
        scale = 1000 / w_orig
        new_w = 1000
        new_h = int(h_orig * scale)
        hand_for_api = cv2.resize(hand_bgr, (new_w, new_h))
    else:
        hand_for_api = hand_bgr
    _, buf = cv2.imencode('.jpg', hand_for_api, [cv2.IMWRITE_JPEG_QUALITY, 90])
    b64 = base64.b64encode(buf.tobytes()).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    # 1. 定位指甲 (MediaPipe 优先)
    try:
        nails = locate_nails(hand_bgr, data_uri)
    except Exception as e:
        log.exception("locate failed")
        nails = []

    # MediaPipe 返回的坐标已经是原图坐标，不需要缩回
    # LLM fallback 返回的是 1000 宽图坐标，要缩回
    if nails and scale != 1.0 and any(n.get("source") == "llm" for n in nails):
        # LLM fallback 返回的是 1000 宽图坐标；MediaPipe 已经是原图坐标。
        inv = 1.0 / scale
        for n in nails:
            n["cx"] = int(n["cx"] * inv)
            n["cy"] = int(n["cy"] * inv)
            n["w"] = int(n["w"] * inv)
            n["h"] = int(n["h"] * inv)

    log.info(f"style={style_id}, nails found: {len(nails)}")
    for n in nails:
        log.info(f"  {n['finger']}: cx={n['cx']}, cy={n['cy']}, w={n['w']}, h={n['h']}")

    # 2. 渲染：优先 AI 图像编辑，失败时退回本地贴图
    render_engine = "local-opencv"
    qwen_result = None
    if not white_mode:
        try:
            qwen_result = qwen_image_tryon(hand_bgr, style_id, nails_count=len(nails))
        except Exception:
            log.exception("tryon qwen image edit failed")
    if qwen_result:
        result_b64 = qwen_result["result_image"]
        render_engine = f"qwen-image-edit:{qwen_result['model']}"
    else:
        try:
            if nails:
                output_bgr = render_tryon(hand_bgr, style_id, nails, white_mode=white_mode, tuning=tuning)
            else:
                output_bgr = hand_bgr  # 没定位到指甲则返回原图
            ok, out_buf = cv2.imencode(".jpg", output_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise RuntimeError("encode failed")
            result_b64 = "data:image/jpeg;base64," + base64.b64encode(out_buf.tobytes()).decode()
        except Exception as e:
            log.exception("render failed")
            _, fb = cv2.imencode(".jpg", hand_bgr)
            result_b64 = "data:image/jpeg;base64," + base64.b64encode(fb.tobytes()).decode()

    return jsonify({
        "result_image": result_b64,
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": len(nails),
        "debug_white_mode": white_mode,
        "tuning": tuning,
        "render_engine": render_engine,
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Chatbot: 用户问问题，可基于已选款式/已渲染图上下文"""
    data = request.get_json(silent=True) or {}
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400
    style_id = safe_int(data.get("style_id"), 0)
    history = data.get("history", [])  # [{role, content}, ...]
    hand_b64 = data.get("hand_image")  # 可选

    style_info = ""
    if 1 <= style_id <= 14:
        s = STYLE_LIBRARY[style_id - 1]
        style_info = f"\n\n[当前已选款式] #{s['id']} {s['name']} - {s['description']}"

    history_text = ""
    for h in history[-6:]:  # 最近 3 轮
        role = h.get("role", "user")
        content = str(h.get("content", ""))[:300]
        history_text += f"\n{role}: {content}"

    system_prompt = f"""你是 AI 美甲试戴顾问，回复简洁友好，30-80 字。
- 用户可能问款式推荐、搭配、换款、场合等
- 不要 emoji 太多
- 不要客套开场白
- 不要"AI"、"智能"等自夸词
{style_info}{history_text}"""

    content = []
    if hand_b64:
        content.append({"image": hand_b64})
    content.append({"text": system_prompt + f"\n\n用户: {question}\n\n顾问:"})

    raw = call_qwen_vl(content, max_retries=2)
    if not raw:
        return jsonify({"error": "AI no response", "reply": "服务暂时不可用，请稍后再试。"})

    # 简单提取回复（去除前缀）
    reply = raw.strip()
    for prefix in ["顾问:", "答:", "回复:"]:
        if reply.startswith(prefix):
            reply = reply[len(prefix):].strip()
    return jsonify({"reply": reply[:500], "style_id": style_id})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    log.info(f"Starting Flask on :{port}, debug_white_mode={DEBUG_WHITE_MODE}")
    app.run(host="0.0.0.0", port=port, debug=False)
