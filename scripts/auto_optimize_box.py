"""
自动优化 estimate_nail_box 参数
用 GrabCut 在 box 内抠真实指甲 mask, 算 box vs 真实 mask 的 IoU.
基于 IoU 网格搜索最优参数, 写回 backend/app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

import importlib.util
spec = importlib.util.spec_from_file_location('app', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend', 'app.py'))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)
m = app_mod

import cv2
import numpy as np
import math
import re

img = cv2.imread(os.path.join(os.path.dirname(__file__), '..', 'assets', 'test', 'user-2hands-real.png'))
h, w = img.shape[:2]

# 拿 MediaPipe landmarks 直接
import mediapipe as mp
mp_hands = mp.solutions.hands
with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
    res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    FINGER_TIPS = {
        '拇指': (4, 3, 2),
        '食指': (8, 7, 6),
        '中指': (12, 11, 10),
        '无名指': (16, 15, 14),
        '小指': (20, 19, 18),
    }
    hands_data = []
    for hi, hl in enumerate(res.multi_hand_landmarks):
        hands_data.append((hi, hl))

def compute_nails_with_params(off_n, len_n, w_n, off_t, len_t, w_t):
    """用给定参数算 nails (含 hand_id + finger)"""
    nails = []
    for hi, hl in hands_data:
        lm = hl.landmark
        for fn, (tip_i, dip_i, pip_i) in FINGER_TIPS.items():
            tip = (int(lm[tip_i].x * w), int(lm[tip_i].y * h))
            dip = (int(lm[dip_i].x * w), int(lm[dip_i].y * h))
            dx = dip[0] - tip[0]
            dy = dip[1] - tip[1]
            fl = math.hypot(dx, dy)
            if fl < 5: continue
            ang = math.atan2(dy, dx)
            is_thumb = fn == '拇指'
            off = off_t if is_thumb else off_n
            lr = len_t if is_thumb else len_n
            wr = w_t if is_thumb else w_n
            cx = int(tip[0] + off * fl * math.cos(ang))
            cy = int(tip[1] + off * fl * math.sin(ang))
            bw = max(15, int(fl * wr))
            bh = max(15, int(fl * lr))
            nails.append((hi, fn, cx, cy, bw, bh))
    return nails

def grabcut_nail(img, cx, cy, bw, bh):
    x0 = max(0, cx - bw); y0 = max(0, cy - bh)
    x1 = min(w, cx + bw); y1 = min(h, cy + bh)
    roi = img[y0:y1, x0:x1]
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return None
    mask = np.zeros(roi.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (3, 3, max(1, roi.shape[1]-6), max(1, roi.shape[0]-6))
    try:
        cv2.grabCut(roi, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        nail_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(nail_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            nail_mask = np.zeros_like(nail_mask)
            for c in sorted(contours, key=cv2.contourArea, reverse=True)[:2]:
                if cv2.contourArea(c) >= 30:
                    cv2.drawContours(nail_mask, [c], -1, 255, -1)
        if nail_mask.sum() < 50:
            return None
        return nail_mask
    except:
        return None

def evaluate(nails):
    """算 nails vs GrabCut 真实指甲的平均 IoU + 中心偏差"""
    ious = []
    dys = []
    for hi, fn, cx, cy, bw, bh in nails:
        real_mask_roi = grabcut_nail(img, cx, cy, bw, bh)
        if real_mask_roi is None:
            continue
        M = cv2.moments(real_mask_roi)
        if M["m00"] == 0: continue
        real_cx_roi = M["m10"] / M["m00"]
        real_cy_roi = M["m01"] / M["m00"]
        real_cx = int(cx - bw + real_cx_roi)
        real_cy = int(cy - bh + real_cy_roi)
        ys, xs = np.where(real_mask_roi > 0)
        if len(ys) == 0: continue
        real_cx_box = (xs.max() + xs.min()) // 2 + (cx - bw)
        real_cy_box = (ys.max() + ys.min()) // 2 + (cy - bh)
        real_w = (xs.max() - xs.min()) // 2
        real_h = (ys.max() - ys.min()) // 2
        # IoU
        x1a, y1a, x2a, y2a = cx-bw, cy-bh, cx+bw, cy+bh
        x1b, y1b, x2b, y2b = real_cx_box-real_w, real_cy_box-real_h, real_cx_box+real_w, real_cy_box+real_h
        xa, ya = max(x1a, x1b), max(y1a, y1b)
        xb, yb = min(x2a, x2b), min(y2a, y2b)
        inter = max(0, xb-xa) * max(0, yb-ya)
        area_a = (x2a-x1a) * (y2a-y1a)
        area_b = (x2b-x1b) * (y2b-y1b)
        iou = inter / (area_a + area_b - inter) if (area_a+area_b-inter) > 0 else 0
        ious.append(iou)
        dys.append(real_cy - cy)  # 真实指甲中心 y 减 box 中心 y
    avg_iou = sum(ious) / max(len(ious), 1)
    avg_dy = sum(dys) / max(len(dys), 1)
    return avg_iou, avg_dy

# 网格搜索
print("=== 网格搜索 (off_n, len_n, w_n, off_t, len_t, w_t) ===")
best = (0, None)
# 第一轮: 粗搜
for off_n in [0.35, 0.45, 0.55]:
    for len_n in [0.95, 1.10, 1.25]:
        for w_n in [0.80, 0.95]:
            for off_t in [0.35, 0.45, 0.55]:
                for len_t in [1.10, 1.30]:
                    for w_t in [0.85, 1.00]:
                        nails = compute_nails_with_params(off_n, len_n, w_n, off_t, len_t, w_t)
                        iou, dy = evaluate(nails)
                        if iou > best[0]:
                            best = (iou, (off_n, len_n, w_n, off_t, len_t, w_t, dy))

print(f"Best IoU: {best[0]:.3f}")
print(f"Best params: off_n={best[1][0]}, len_n={best[1][1]}, w_n={best[1][2]}, off_t={best[1][3]}, len_t={best[1][4]}, w_t={best[1][5]}")
print(f"Avg dy: {best[1][6]:.1f}")
