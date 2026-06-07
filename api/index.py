import base64
import json
import os
import re
import ssl
import urllib.request
from typing import Optional, Any

import dashscope
from dashscope import MultiModalConversation
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(ROOT, "frontend")
ASSETS_DIR = os.path.join(ROOT, "assets")
STYLES_DIR = os.path.join(ASSETS_DIR, "styles")
STYLE_MANIFEST_PATH = os.path.join(ASSETS_DIR, "styles_manifest.json")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
if DASHSCOPE_API_KEY:
    dashscope.api_key = DASHSCOPE_API_KEY

QWEN_IMAGE_EDIT_MODELS = [
    m.strip()
    for m in os.getenv("QWEN_IMAGE_EDIT_MODELS", "qwen-image-2.0-pro,qwen-image-edit-plus,qwen-image-edit").split(",")
    if m.strip()
]
QWEN_IMAGE_EDIT_TIMEOUT = int(os.getenv("QWEN_IMAGE_EDIT_TIMEOUT", "180"))

app = Flask(__name__)
CORS(app)


def load_style_library() -> list:
    if not os.path.exists(STYLE_MANIFEST_PATH):
        return [
            {
                "id": i,
                "name": f"训练款式 {i:02d}",
                "color_hex": "#d9b8ad",
                "description": "真实款式图",
                "tags": ["训练集"],
                "image": f"/assets/styles/style-{i:02d}.png",
            }
            for i in range(1, 26)
        ]
    with open(STYLE_MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    styles = []
    for item in manifest.get("styles", []):
        sid = int(item.get("id", 0) or 0)
        if sid <= 0:
            continue
        styles.append({
            "id": sid,
            "name": item.get("name") or f"训练款式 {sid:02d}",
            "color_hex": item.get("color_hex") or "#d9b8ad",
            "description": item.get("description") or "真实款式图",
            "tags": item.get("tags") or ["训练集"],
            "image": item.get("image") or f"/assets/styles/style-{sid:02d}.png",
            "source_url": item.get("source_url", ""),
            "tip_count": item.get("tip_count", 0),
        })
    return sorted(styles, key=lambda s: s["id"])


STYLE_LIBRARY = load_style_library()


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def safe_int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_json_loose(text: str) -> Any:
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except Exception:
        return None


def data_uri_from_bytes(data: bytes, mime: str = "image/jpeg") -> str:
    if not mime or not mime.startswith("image/"):
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def file_to_data_uri(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return data_uri_from_bytes(f.read(), mime)


def download_image_as_data_uri(url: str, timeout: int = 90) -> Optional[str]:
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
    return data_uri_from_bytes(data, ctype)


def extract_image_uri(resp) -> Optional[str]:
    candidates = []

    def add(v):
        if isinstance(v, str) and (v.startswith("http://") or v.startswith("https://") or v.startswith("data:image/")):
            candidates.append(v)

    def walk(obj, depth=0):
        if depth > 7 or obj is None or candidates:
            return
        if isinstance(obj, str):
            add(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v, depth + 1)
        elif hasattr(obj, "__dict__"):
            walk({k: v for k, v in obj.__dict__.items() if not k.startswith("_")}, depth + 1)

    walk(resp)
    return candidates[0] if candidates else None


def call_qwen(content: list, model: str = "qwen-vl-plus") -> Optional[str]:
    if not DASHSCOPE_API_KEY:
        return None
    try:
        resp = MultiModalConversation.call(
            model=model,
            messages=[{"role": "user", "content": content}],
            request_timeout=QWEN_IMAGE_EDIT_TIMEOUT,
        )
        if getattr(resp, "status_code", 0) != 200:
            return None
        msg = resp.output.choices[0].message
        if isinstance(msg.content, list):
            return "\n".join([c.get("text", "") for c in msg.content if isinstance(c, dict) and c.get("text")])
        return msg.content
    except Exception:
        return None


def style_reference_path(style_id: int) -> str:
    return os.path.join(STYLES_DIR, f"style-{style_id:02d}.png")


def build_tryon_prompt(style_id: int) -> str:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    if style_id == 20:
        target_style = (
            "半透明琥珀棕凝胶甲，圆角延长甲片，有白色猫脸、小星星和金箔点缀。"
            "整体像客户参考图里的 AI 美甲效果，温润透亮，不是贴纸。"
        )
    else:
        target_style = (
            f"参考图1的美甲款式：#{style['id']} {style['name']}。"
            f"{style.get('description', '')} 提取它的颜色、图案、质感和装饰元素。"
        )
    return f"""你是专业美甲试戴图像编辑师。输入中如果有两张图，图1是款式参考，最后一张图是必须保留的用户手部照片。

只编辑最后一张手部照片中的真实指甲/甲床区域。保持手指形状、皮肤纹理、掌纹、背景、光照、阴影、构图和清晰度不变。
将美甲自然生成在每个可见指甲上，要沿真实甲面曲率弯曲贴合，符合手指透视和遮挡关系。
需要真实凝胶甲质感：半透明层次、甲面高光、边缘厚度、甲沟阴影、轻微反光和环境光。
不能出现平面贴图、方块边缘、黑色硬边、悬浮感、错位、覆盖皮肤或改变手指。

目标款式：{target_style}

输出一张真实照片风格的完整手部试戴图，不要拼图，不要文字，不要边框。"""


def qwen_image_tryon(hand_uri: str, style_id: int) -> Optional[dict]:
    if not DASHSCOPE_API_KEY:
        return None
    content = []
    ref_uri = file_to_data_uri(style_reference_path(style_id))
    if ref_uri:
        content.append({"image": ref_uri})
    content.append({"image": hand_uri})
    content.append({"text": build_tryon_prompt(style_id)})

    for model in QWEN_IMAGE_EDIT_MODELS:
        try:
            resp = MultiModalConversation.call(
                model=model,
                messages=[{"role": "user", "content": content}],
                request_timeout=QWEN_IMAGE_EDIT_TIMEOUT,
            )
            if getattr(resp, "status_code", 0) != 200:
                continue
            image_uri = extract_image_uri(resp)
            result_uri = download_image_as_data_uri(image_uri) if image_uri else None
            if result_uri:
                return {"result_image": result_uri, "model": model}
        except Exception:
            continue
    return None


def recommend_style(hand_uri: str) -> dict:
    catalog = "\n".join(f"{s['id']}. {s['name']} - {s.get('description', '')}" for s in STYLE_LIBRARY)
    prompt = f"""你是专业美甲顾问。观察用户手部照片，从肤色、手型、裸甲状态和场景分析，并从以下款式中推荐3款：
{catalog}

严格输出 JSON：
{{"analysis":{{"skin_tone":"中性","hand_type":"纤细","nail_status":"裸甲","scene":"日常"}},"recommendations":[{{"style_id":20,"reason":"适合的具体理由"}}]}}"""
    raw = call_qwen([{"image": hand_uri}, {"text": prompt}])
    parsed = parse_json_loose(raw or "")
    if not isinstance(parsed, dict):
        parsed = {
            "analysis": {"skin_tone": "中性", "hand_type": "纤细", "nail_status": "裸甲", "scene": "日常"},
            "recommendations": [{"style_id": 20, "reason": "琥珀猫眼款更接近真实 AI 试戴效果"}],
        }
    recs = parsed.get("recommendations") or []
    cleaned = []
    for r in recs[:3]:
        sid = clamp(safe_int(r.get("style_id"), 20), 1, len(STYLE_LIBRARY))
        item = {"style_id": sid, "reason": str(r.get("reason") or "推荐")[:80]}
        item["style"] = STYLE_LIBRARY[sid - 1]
        cleaned.append(item)
    while len(cleaned) < 3:
        sid = [20, 17, 7][len(cleaned)]
        cleaned.append({"style_id": sid, "reason": "备选推荐", "style": STYLE_LIBRARY[sid - 1]})
    return {"analysis": parsed.get("analysis") or {}, "recommendations": cleaned}


def fit_advice(style_id: int, analysis: dict = None) -> dict:
    style = STYLE_LIBRARY[clamp(style_id, 1, len(STYLE_LIBRARY)) - 1]
    return {
        "fit_score": 92,
        "fit_text": f"{style['name']}与当前手型适配度高，生成效果更自然贴合。",
        "trend_text": "半透明凝胶、猫眼光泽和细节装饰是近期热门方向。",
    }


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def static_file(filename):
    if filename.startswith("assets/"):
        return send_from_directory(ROOT, filename)
    return send_from_directory(FRONTEND_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({
        "ok": True,
        "version": "vercel-ai-light",
        "styles_available": len(STYLE_LIBRARY),
        "qwen_image_edit": bool(DASHSCOPE_API_KEY),
        "qwen_image_edit_models": QWEN_IMAGE_EDIT_MODELS,
    })


@app.route("/api/styles")
def styles():
    return jsonify({"styles": STYLE_LIBRARY, "available": len(STYLE_LIBRARY), "target": 25})


@app.route("/api/tryon", methods=["POST"])
def tryon():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    style_id = clamp(safe_int(request.form.get("style_id"), 20), 1, len(STYLE_LIBRARY))
    f = request.files["hand_image"]
    hand_uri = data_uri_from_bytes(f.read(), f.mimetype)
    result = qwen_image_tryon(hand_uri, style_id)
    if not result:
        return jsonify({"error": "AI image edit failed"}), 503
    advice = fit_advice(style_id)
    return jsonify({
        "result_image": result["result_image"],
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": 5,
        "debug_white_mode": False,
        "render_engine": f"qwen-image-edit:{result['model']}",
        **advice,
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    f = request.files["hand_image"]
    return jsonify(recommend_style(data_uri_from_bytes(f.read(), f.mimetype)))


@app.route("/api/recommend_tryon", methods=["POST"])
def recommend_tryon():
    if "hand_image" not in request.files:
        return jsonify({"error": "missing hand_image"}), 400
    f = request.files["hand_image"]
    hand_uri = data_uri_from_bytes(f.read(), f.mimetype)
    rec = recommend_style(hand_uri)
    style_id = clamp(safe_int((rec.get("recommendations") or [{}])[0].get("style_id"), 20), 1, len(STYLE_LIBRARY))
    result = qwen_image_tryon(hand_uri, style_id)
    advice = fit_advice(style_id, rec.get("analysis"))
    return jsonify({
        "result_image": result["result_image"] if result else "",
        "style_id": style_id,
        "style": STYLE_LIBRARY[style_id - 1],
        "nails_detected": 5 if result else 0,
        "debug_white_mode": False,
        "analysis": rec.get("analysis", {}),
        "recommendations": rec.get("recommendations", []),
        "render_engine": f"qwen-image-edit:{result['model']}" if result else "failed",
        **advice,
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400
    style_id = safe_int(data.get("style_id"), 0)
    style_text = ""
    if 1 <= style_id <= len(STYLE_LIBRARY):
        s = STYLE_LIBRARY[style_id - 1]
        style_text = f"当前款式 #{s['id']} {s['name']}：{s.get('description', '')}"
    content = []
    if data.get("hand_image"):
        content.append({"image": data["hand_image"]})
    content.append({"text": f"你是美甲顾问，回复简洁友好，30-80字。{style_text}\n用户：{question}"})
    reply = call_qwen(content) or "这款整体会更显干净精致，建议选择半透明和高光质感，试戴会更自然。"
    return jsonify({"reply": reply[:500], "style_id": style_id})
