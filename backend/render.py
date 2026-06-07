"""
指甲试戴渲染管线
- detect_nails(): 检测手图中可能的指甲区域（启发式）
- warp_style_to_nails(): 把款式图贴到指甲上
- render_tryon(): 主入口
"""
import io
import base64
import urllib.request
from typing import List, Tuple
import cv2
import numpy as np


# ========== 工具 ==========
def load_image_from_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def load_image_from_url(url: str) -> np.ndarray:
    """下载公网 URL 图片（处理 data: URI 和 http）"""
    if url.startswith("data:"):
        # data:image/png;base64,xxx
        b64 = url.split(",", 1)[1]
        data = base64.b64decode(b64)
        return load_image_from_bytes(data)
    elif url.startswith("http"):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
        return load_image_from_bytes(data)
    else:
        raise ValueError(f"unsupported url: {url[:50]}")


def encode_png_to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("failed to encode png")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


# ========== 指甲检测 ==========
def detect_nails(hand_bgr: np.ndarray, max_nails: int = 5) -> List[Tuple[int, int, int, int]]:
    """
    在手图中检测可能的指甲区域，返回 [(x, y, w, h), ...]
    算法：肤色分割 + 边缘 + 形态学 + 候选矩形筛选
    """
    h, w = hand_bgr.shape[:2]

    # 1. 转 HSV 提取肤色区域
    hsv = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2HSV)
    # 肤色 HSV 范围（较宽，包含各种肤色）
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
    # 第二段肤色（避开红区）
    lower_skin2 = np.array([160, 20, 50], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # 形态学：闭运算填洞 + 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # 2. 在肤色区域内找"指甲候选"——更亮、更白的区域
    gray = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2GRAY)
    # OTSU 自适应阈值
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 只保留肤色内的亮区
    nail_candidates = cv2.bitwise_and(bright, skin_mask)

    # 形态学清理
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    nail_candidates = cv2.morphologyEx(nail_candidates, cv2.MORPH_CLOSE, kernel2)
    nail_candidates = cv2.morphologyEx(nail_candidates, cv2.MORPH_OPEN, kernel2)

    # 3. 找连通区域
    contours, _ = cv2.findContours(nail_candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        # 过滤太小或太扁的
        if area < (w * h * 0.005):
            continue
        if area > (w * h * 0.3):
            continue
        # 指甲长宽比：通常 0.4 - 2.5
        ratio = hh / max(ww, 1)
        if ratio < 0.3 or ratio > 3.0:
            continue
        # 指甲的"实心度"不能太低（不是细长条）
        cnt_area = cv2.contourArea(c)
        if cnt_area / max(area, 1) < 0.4:
            continue
        candidates.append((x, y, ww, hh, area))

    # 按面积排序，取前 max_nails 个
    candidates.sort(key=lambda c: -c[4])
    candidates = candidates[:max_nails]

    # 排序时让"上方"的在前（手指甲通常在图像上方）
    # 简单起见：按 y 排序
    candidates.sort(key=lambda c: c[1])

    return [(x, y, ww, hh) for (x, y, ww, hh, _) in candidates]


# ========== 渲染 ==========
def render_tryon(hand_bgr: np.ndarray, style_bgr: np.ndarray) -> np.ndarray:
    """
    主渲染函数：
    1. 检测手图中的指甲候选区域
    2. 把款式图透视变换到每个指甲
    3. 多频段融合
    """
    output = hand_bgr.copy()
    nails = detect_nails(hand_bgr, max_nails=5)
    if not nails:
        # 找不到指甲：返回原图 + 文字提示（这里先静默返回原图）
        return output

    # 准备款式图
    style_h, style_w = style_bgr.shape[:2]

    for (x, y, w, h) in nails:
        # 目标区域中心 + 4 个角点
        cx, cy = x + w / 2, y + h / 2
        # 矩形 4 角（加一点内缩，避免溢出）
        pad = 0.05
        x0 = x + w * pad
        y0 = y + h * pad
        x1 = x + w * (1 - pad)
        y1 = y + h * (1 - pad)
        dst_pts = np.array([
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ], dtype=np.float32)

        # 源图 4 角
        src_pts = np.array([
            [0, 0],
            [style_w, 0],
            [style_w, style_h],
            [0, style_h],
        ], dtype=np.float32)

        # 透视变换
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(style_bgr, M, (hand_bgr.shape[1], hand_bgr.shape[0]))

        # 椭圆 mask（指甲形状更自然）
        mask = np.zeros((hand_bgr.shape[0], hand_bgr.shape[1]), dtype=np.float32)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(w / 2 * 0.95), int(h / 2 * 0.95)),
                    0, 0, 360, 1.0, -1)
        # 高斯模糊 mask 让边缘过渡自然
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        mask_3 = np.stack([mask] * 3, axis=-1)

        # 多频段融合：低频用 blend，高频用原图 + warped 边缘细节
        # 简化版：直接 alpha blend
        output = (warped * mask_3 + output * (1 - mask_3)).astype(np.uint8)

    return output


# ========== 调试可视化 ==========
def debug_visualize(hand_bgr: np.ndarray, nails: list) -> np.ndarray:
    """把检测到的指甲画出来"""
    vis = hand_bgr.copy()
    for (x, y, w, h) in nails:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return vis
