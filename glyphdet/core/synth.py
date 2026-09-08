"""全合成训练数据生成器：掩码条件化字形检测（取件码检测）。

用法:
    python core/synth.py --config experiments/mvp_baseline/config.yaml \
        [--split both|train|val] [--n N] [--seed S] [--workers W]

产出: <out_dir>/{train,val}/{scene,mask}/NNNNNN.png + labels.jsonl
合成顺序: 程序化背景 -> 背景弹性形变 -> 文本贴片(记录四角) -> 全局透视(重投影四角)
          -> 光度增强(不动几何)。
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent  # glyphdet 项目根目录

# ---------------------------------------------------------------- 码生成
_DIGITS = "0123456789"
_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 去掉易混淆的 I/O


def _digits(rng, lo, hi):
    """随机 lo~hi 位数字串。"""
    n = int(rng.integers(lo, hi + 1))
    return "".join(rng.choice(list(_DIGITS), size=n))


def _pickup(rng):
    """标准取件码: \\d{1,3}-\\d{1,2}-\\d{3,5}。"""
    return f"{_digits(rng, 1, 3)}-{_digits(rng, 1, 2)}-{_digits(rng, 3, 5)}"


def gen_code(rng):
    """目标码(query)格式分布:
    60% 标准取件码 / 15% 纯数字4-8位 / 10% 1-2字母+数字 /
    10% 换分隔符(空格/无/点号) / 5% 更长串(10-14字符)。"""
    k = rng.random()
    if k < 0.60:
        return _pickup(rng)
    if k < 0.75:
        return _digits(rng, 4, 8)
    if k < 0.85:
        nl, nd = int(rng.integers(1, 3)), int(rng.integers(2, 6))
        letters = "".join(rng.choice(list(_LETTERS), size=nl))
        return letters + _digits(rng, nd, nd) if rng.random() < 0.7 else _digits(rng, nd, nd) + letters
    if k < 0.95:
        sep = str(rng.choice([" ", "", "."]))
        return sep.join([_digits(rng, 1, 3), _digits(rng, 1, 2), _digits(rng, 3, 5)])
    n = int(rng.integers(10, 15))
    s = "".join(rng.choice(list(_LETTERS + _DIGITS), size=n))
    if rng.random() < 0.3:
        i = int(rng.integers(2, n - 1))
        s = s[:i] + "-" + s[i:]
    return s


def gen_distractor(rng):
    """干扰串: 30% 用"看起来像取件码的格式", 其余随机内容。"""
    if rng.random() < 0.3:
        return _pickup(rng)
    k = rng.random()
    if k < 0.30:
        return _digits(rng, 3, 9)
    if k < 0.55:
        n = int(rng.integers(3, 9))
        return "".join(rng.choice(list(_LETTERS + _DIGITS), size=n))
    if k < 0.75:
        sep = str(rng.choice(["-", " ", ".", "/"]))
        return sep.join([_digits(rng, 2, 4), _digits(rng, 2, 6)])
    n = int(rng.integers(2, 7))
    return "".join(rng.choice(list(_LETTERS), size=n))


def hard_negative(rng, s):
    """与 query 仅一个字符不同: 替换/增删一位, 或连字符位置移动一位。"""
    n = len(s)
    cands = ["sub", "ins"] + (["del"] if n >= 4 else []) + (["hyphen"] if "-" in s else [])
    op = cands[int(rng.integers(len(cands)))]
    if op == "sub":
        i = int(rng.integers(n))
        ch = s[i]
        if ch.isdigit():
            pool = [c for c in _DIGITS if c != ch]
        elif ch.isalpha():
            pool = [c for c in _LETTERS if c != ch]
        else:  # 分隔符替换为另一种分隔符
            pool = [c for c in "- ." if c != ch]
        return s[:i] + pool[int(rng.integers(len(pool)))] + s[i + 1:]
    if op == "ins":
        i = int(rng.integers(n + 1))
        return s[:i] + _DIGITS[int(rng.integers(len(_DIGITS)))] + s[i:]
    if op == "del":
        i = int(rng.integers(n))
        return s[:i] + s[i + 1:]
    # hyphen: 摘一个连字符, 移位一格重新插入(避开与原位置/其他连字符相邻)
    hy = [i for i, c in enumerate(s) if c == "-"]
    i = hy[int(rng.integers(len(hy)))]
    s2 = s[:i] + s[i + 1:]
    js = [j for j in (i - 1, i + 1) if 1 <= j <= len(s2) - 1
          and s2[j - 1] != "-" and s2[j] != "-"]
    if not js:
        return hard_negative(rng, s2)  # 退化时换一种扰动
    j = js[int(rng.integers(len(js)))]
    return s2[:j] + "-" + s2[j:]


# ---------------------------------------------------------------- 字体与渲染
_FONT_CACHE = {}


def _font(path, px):
    """按 (路径, 像素) 缓存 truetype 字体(进程内)。"""
    key = (path, int(px))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(px))
    return _FONT_CACHE[key]


def render_mask(text, font_path, H, W):
    """mask 渲染: 固定雅黑, 整串渲染 -> 等比缩放到高 H -> 贴到 HxW 白底,
    左对齐、垂直居中; 过宽则缩窄适配(允许轻微压扁)。黑字(0)白底(255)。"""
    font = _font(font_path, 96)
    bb = font.getbbox(text)
    w, h = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
    img = Image.new("L", (w, h), 255)
    ImageDraw.Draw(img).text((-bb[0], -bb[1]), text, font=font, fill=0)
    nw = max(1, round(w * H / h))
    if nw > W:
        nw = W  # 超宽: 压扁到 W
    img = img.resize((nw, H), Image.LANCZOS)
    canvas = Image.new("L", (W, H), 255)
    canvas.paste(img, (0, (H - H) // 2))  # 左对齐, 垂直居中(高已为 H)
    return np.array(canvas)


# ---------------------------------------------------------------- 背景
_BG_FAMILIES = [  # (RGB 基色, 抖动幅度)
    ((154, 123, 90), 28),   # 纸箱棕 #9a7b5a
    ((228, 224, 216), 12),  # 面单白
    ((150, 150, 152), 26),  # 货架灰
    ((64, 62, 66), 14),     # 暗色柜机
    ((122, 138, 154), 20),  # 冷灰蓝
    ((186, 160, 110), 22),  # 浅木色
]


def gen_background(rng, S):
    """程序化背景: 随机底色(含随机色相) + 光照梯度 + 模糊噪声
    + 色块拼贴/条纹/圆点/棋盘(稀奇古怪的真实背景) + 偶发隔板线/胶带条。"""
    if rng.random() < 0.2:  # 20% 完全随机色相底色(彩色货架/塑料膜/广告纸)
        base_rgb = rng.uniform(40, 215, 3).astype(np.float32)
        jit = 18.0
    else:
        base_rgb, jit = _BG_FAMILIES[int(rng.integers(len(_BG_FAMILIES)))]
        base_rgb = np.array(base_rgb, np.float32)
    img = np.ones((S, S, 3), np.float32) * (base_rgb + rng.uniform(-jit, jit, 3))
    # 光照梯度: 线性或径向
    d = float(rng.uniform(15, 60))
    if rng.random() < 0.5:
        ang = float(rng.uniform(0, math.pi))
        x = np.linspace(-1, 1, S, dtype=np.float32)
        grad = math.cos(ang) * x[None, :] + math.sin(ang) * x[:, None]
        img += (grad * d)[..., None]
    else:
        cx, cy = rng.uniform(0.2, 0.8, 2) * S
        yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (S * 0.7)
        img += ((0.6 - r) * d)[..., None]
    # 柏林感噪声: 随机噪声大核模糊
    sigma = float(rng.uniform(4, 14))
    noise = cv2.GaussianBlur(rng.normal(0, sigma, (S, S)).astype(np.float32), (0, 0), float(rng.uniform(6, 18)))
    img += noise[..., None] * float(rng.uniform(0.6, 1.4))
    # 色块拼贴: 随机矩形半透明色块(货架贴条/包装/广告纸残片)
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(3, 12))):
            bw = int(rng.uniform(0.05, 0.5) * S)
            bh = int(rng.uniform(0.03, 0.3) * S)
            x0, y0 = int(rng.uniform(-0.1 * S, S)), int(rng.uniform(-0.1 * S, S))
            x1, y1 = max(0, x0), max(0, y0)
            x2, y2 = min(S, x0 + bw), min(S, y0 + bh)
            if x2 <= x1 or y2 <= y1:
                continue
            col = rng.uniform(30, 230, 3).astype(np.float32)
            a = float(rng.uniform(0.15, 0.7))
            img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * (1 - a) + col * a
    # 规则分布纹理: 旋转条纹 / 圆点阵 / 棋盘格
    r = rng.random()
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    if r < 0.18:  # 条纹
        ang = float(rng.uniform(0, math.pi))
        period = float(rng.uniform(12, 60))
        amp = float(rng.uniform(0.08, 0.3))
        pat = 1 + amp * np.sin(2 * math.pi * (xx * math.cos(ang) + yy * math.sin(ang)) / period)
        img *= pat[..., None]
    elif r < 0.3:  # 圆点阵
        step = int(rng.uniform(24, 72))
        rad = max(2, int(step * rng.uniform(0.15, 0.4)))
        col = rng.uniform(30, 230, 3).astype(np.float32)
        a = float(rng.uniform(0.2, 0.6))
        for cy in range(step // 2, S, step):
            for cx in range(step // 2, S, step):
                m = np.zeros((S, S), np.float32)
                cv2.circle(m, (cx, cy), rad, 1.0, -1)
                img = img * (1 - a * m[..., None]) + col * (a * m[..., None])
    elif r < 0.4:  # 棋盘格
        cell = int(rng.uniform(30, 90))
        c1, c2 = rng.uniform(40, 220, 3).astype(np.float32), rng.uniform(40, 220, 3).astype(np.float32)
        a = float(rng.uniform(0.15, 0.5))
        checker = ((np.floor(xx / cell) + np.floor(yy / cell)) % 2)[..., None]
        img = img * (1 - a * checker) + c2 * (a * checker)
    # 偶尔画仿货架隔板横线
    if rng.random() < 0.3:
        for _ in range(int(rng.integers(1, 4))):
            y = int(rng.uniform(0, S))
            t = int(rng.integers(2, 7))
            shade = float(rng.uniform(-40, 40))
            cv2.rectangle(img, (0, y - t // 2), (S, y + t // 2),
                          (float(img[0, 0, 0]) + shade,) * 3, -1)
    # 偶尔画纸箱胶带条纹
    if rng.random() < 0.25:
        bw = int(rng.integers(24, 65))
        tape = np.array((190, 170, 130), np.float32) + rng.uniform(-15, 15, 3)
        a = float(rng.uniform(0.25, 0.45))
        if rng.random() < 0.5:
            x0 = int(rng.uniform(0, S - bw))
            img[:, x0:x0 + bw] = img[:, x0:x0 + bw] * (1 - a) + tape * a
        else:
            y0 = int(rng.uniform(0, S - bw))
            img[y0:y0 + bw] = img[y0:y0 + bw] * (1 - a) + tape * a
    return np.clip(img, 0, 255).astype(np.uint8)


def elastic(rng, img):
    """弹性形变(只作用于背景): 高斯随机场位移 + remap, alpha 小。"""
    S = img.shape[0]
    alpha = float(rng.uniform(3, 7))
    sigma = float(rng.uniform(20, 40))
    dx = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.normal(0, 1, (S, S)).astype(np.float32), (0, 0), sigma)
    dx *= alpha / (np.abs(dx).max() + 1e-6)
    dy *= alpha / (np.abs(dy).max() + 1e-6)
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


# ---------------------------------------------------------------- 文本贴片
_DARK_INK = [(30, 30, 30), (20, 20, 60), (60, 20, 20), (20, 50, 30), (45, 45, 45), (15, 30, 60)]
_LIGHT_INK = [(240, 240, 240), (235, 235, 220), (250, 245, 235)]


def render_text_patch(rng, text, font_path, ink=None):
    """PIL 渲染文本 RGBA patch:
    字号随机 16~44px 高; 80% 深字浅底 / 20% 浅字深底; 字距抖动 ±15%;
    60% 在文字下垫浅色圆角矩形标签卡(可带淡边框/阴影; 浅字时卡片为深色)。
    ink 显式指定墨色(低对比训练用); None 时按分布随机。
    返回 (patch, 文本紧致矩形 (ox, oy, tw, th))。"""
    th_target = int(rng.integers(16, 45))
    f0 = _font(font_path, 64)
    bb = f0.getbbox(text)
    px = max(8, int(round(64 * th_target / max(1, bb[3] - bb[1]))))  # 按字高反推字号
    font = _font(font_path, px)
    ascent, descent = font.getmetrics()
    advs = [font.getlength(c) for c in text]
    extra = float(rng.uniform(-0.15, 0.15)) * (sum(advs) / len(advs))  # 字距抖动
    w_text = max(8, int(math.ceil(sum(advs) + extra * (len(text) - 1))) + 2)
    layer = Image.new("RGBA", (w_text, ascent + descent), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    dark = ink is None and rng.random() < 0.8
    if ink is None:
        ink = (_DARK_INK if dark else _LIGHT_INK)[int(rng.integers(3 if not dark else len(_DARK_INK)))]
    dark = dark or (ink is not None and sum(ink) < 384)  # 供卡片配色判断
    x = 1.0
    for c, a in zip(text, advs):  # 逐字排版以控制字距
        d.text((x, 0), c, font=font, fill=ink + (255,))
        x += a + extra
    layer = layer.crop(layer.getbbox())  # 紧致裁剪到墨迹
    tw, th = layer.size
    with_card = rng.random() < 0.6
    pad = 3
    cp = int(float(rng.uniform(0.15, 0.5)) * th) if with_card else 0  # 卡片内边距
    W, H = tw + 2 * (pad + cp), th + 2 * (pad + cp)
    patch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ox = oy = pad + cp
    if with_card:
        rect = [pad, pad, pad + tw + 2 * cp - 1, pad + th + 2 * cp - 1]
        radius = max(2, int(0.25 * th))
        if dark:  # 深字浅底
            card = tuple(int(rng.uniform(215, 250)) for _ in range(3))
        else:     # 浅字深底
            card = tuple(int(rng.uniform(35, 75)) for _ in range(3))
        if rng.random() < 0.5:  # 阴影
            sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(sh).rounded_rectangle(
                [rect[0] + 2, rect[1] + 3, rect[2] + 2, rect[3] + 3], radius, fill=(0, 0, 0, 90))
            patch = Image.alpha_composite(patch, sh.filter(ImageFilter.GaussianBlur(3)))
        dd = ImageDraw.Draw(patch)
        outline = tuple(max(0, c - int(rng.uniform(40, 90))) for c in card) if rng.random() < 0.5 else None
        dd.rounded_rectangle(rect, radius, fill=card + (255,),
                             outline=outline + (255,) if outline else None,
                             width=int(rng.integers(1, 3)) if outline else 1)
    patch.alpha_composite(layer, (ox, oy))
    return patch, (ox, oy, tw, th)


def warp_patch(rng, patch, text_rect):
    """整 patch 随机旋转 ±12° + 错切 ±0.15 (cv2.warpAffine, 透明边界)。
    返回 (warped RGBA ndarray, 变换画布坐标系下文本四角 4x2)。"""
    w, h = patch.size
    ang = math.radians(float(rng.uniform(-12, 12)))
    shear = float(rng.uniform(-0.15, 0.15))
    ca, sa = math.cos(ang), math.sin(ang)
    A = np.array([[ca, -sa], [sa, ca]]) @ np.array([[1.0, shear], [0.0, 1.0]])  # 旋转·错切
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float64) @ A.T
    mn = corners.min(0)
    out_w = int(math.ceil(corners[:, 0].max() - mn[0])) + 1
    out_h = int(math.ceil(corners[:, 1].max() - mn[1])) + 1
    M = np.hstack([A, -mn[:, None]]).astype(np.float32)
    warped = cv2.warpAffine(np.array(patch), M, (out_w, out_h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ox, oy, tw, th = text_rect
    tc = np.array([[ox, oy], [ox + tw, oy], [ox + tw, oy + th], [ox, oy + th]], np.float64) @ A.T - mn
    return warped, tc


def paste_rgba(scene, patch, px, py):
    """RGBA patch 以 alpha 混合贴到 RGB scene 的 (px, py)(允许部分出图)。"""
    S = scene.shape[0]
    h, w = patch.shape[:2]
    x0, y0, x1, y1 = max(0, px), max(0, py), min(S, px + w), min(S, py + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = patch[y0 - py:y1 - py, x0 - px:x1 - px]
    a = sub[..., 3:4].astype(np.float32) / 255.0
    region = scene[y0:y1, x0:x1].astype(np.float32)
    scene[y0:y1, x0:x1] = (sub[..., :3].astype(np.float32) * a + region * (1 - a)).astype(np.uint8)


def _aabb(corners):
    """四角 -> 轴对齐包围盒 [x1, y1, x2, y2]。"""
    return [float(corners[:, 0].min()), float(corners[:, 1].min()),
            float(corners[:, 0].max()), float(corners[:, 1].max())]


def _cover_ratio(big, small):
    """small 被 big 覆盖的面积比例(用于避免文本完全遮死已有 target)。"""
    x1, y1 = max(big[0], small[0]), max(big[1], small[1])
    x2, y2 = min(big[2], small[2]), min(big[3], small[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    sa = max(1.0, (small[2] - small[0]) * (small[3] - small[1]))
    return inter / sa


# ---------------------------------------------------------------- 几何/光度
def perspective(rng, img, corners_list):
    """全局透视变换(角点扰动 ≤8% 边长), 同矩阵重投影所有文本四角。"""
    S = img.shape[0]
    m = 0.08 * S
    src = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
    dst = src + rng.uniform(-m, m, (4, 2)).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out = [cv2.perspectiveTransform(c.reshape(1, 4, 2).astype(np.float32), M).reshape(4, 2)
           for c in corners_list]
    return warped, out


def _motion_kernel(ksize, angle_deg):
    """运动模糊核: 水平线核旋转任意角度。"""
    k = np.zeros((ksize, ksize), np.float32)
    k[ksize // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize / 2 - 0.5, ksize / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    return k / (k.sum() + 1e-6)


def photometric(rng, img, target_boxes):
    """光度增强(不动几何): 模糊/噪声/亮度对比度gamma/遮挡块/低分辨率模拟。

    铁律: target 必须保持人眼可读——模糊核只轻度, 遮挡块与 target 零相交(含 margin),
    下采样不低于 0.45 倍。target 完全不可读的监督信号是白搭(用户裁定)。"""
    S = img.shape[0]
    if rng.random() < 0.4:  # 轻度高斯模糊或运动模糊(人眼必须仍可读)
        if rng.random() < 0.5:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            ksize = int(rng.integers(3, 8))  # 运动模糊 ≤7px
            img = cv2.filter2D(img, -1, _motion_kernel(ksize, float(rng.uniform(0, 180))))
    if rng.random() < 0.4:  # 高斯噪声
        sig = float(rng.uniform(3, 12))
        img = np.clip(img.astype(np.float32) + rng.normal(0, sig, img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:  # 亮度/对比度/gamma 抖动
        alpha, beta = float(rng.uniform(0.8, 1.25)), float(rng.uniform(-25, 25))
        gamma = float(rng.uniform(0.7, 1.4))
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, lut)
    if rng.random() < 0.3:  # 随机遮挡块 1~3 个(与 target 零相交, margin 4px; 可挡干扰串)
        m = 4
        for _ in range(int(rng.integers(1, 4))):
            for _try in range(8):
                bw, bh = int(rng.uniform(0.06, 0.25) * S), int(rng.uniform(0.06, 0.25) * S)
                x0, y0 = int(rng.uniform(0, S - bw)), int(rng.uniform(0, S - bh))
                hit = any(
                    x0 < b[2] + m and x0 + bw > b[0] - m
                    and y0 < b[3] + m and y0 + bh > b[1] - m
                    for b in target_boxes
                )
                if not hit:
                    break
            else:
                continue  # 8 次都找不到不碰 target 的位置, 放弃这个块
            col = tuple(int(rng.uniform(40, 220)) for _ in range(3))
            img[y0:y0 + bh, x0:x0 + bw] = col
    if rng.random() < 0.4:  # 下采样到 0.45~0.85 倍再放大(模拟远拍低分辨率)
        s = float(rng.uniform(0.45, 0.85))
        small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        img = cv2.resize(small, (S, S), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------- 样本组装
def build_sample(rng, cfg, scene_fonts):
    """合成一张样本; target 全部丢失时返回 None(交由上层重 roll)。"""
    S = cfg["scene_size"]
    query = gen_code(rng)
    # 文本清单: 1~2 个 target(同一 query 多处出现) + 0~3 干扰(可选 hard negative)
    max_texts = cfg["max_texts_per_scene"]
    n_target = min(int(rng.integers(cfg["target_count_range"][0], cfg["target_count_range"][1] + 1)),
                   max_texts)
    hard = rng.random() < cfg["hard_neg_prob"]
    n_distr = int(rng.integers(cfg["distractor_count_range"][0], cfg["distractor_count_range"][1] + 1))
    n_distr = min(n_distr, max(0, max_texts - n_target))
    if hard and n_distr == 0 and n_target < max_texts:
        n_distr = 1  # hard negative 至少占一个干扰位
    texts = [[query, 1] for _ in range(n_target)]
    for i in range(n_distr):
        s = hard_negative(rng, query) if (hard and i == 0) else gen_distractor(rng)
        for _ in range(4):  # 干扰串不得与 query 相同
            if s != query:
                break
            s = gen_distractor(rng)
        if s != query:
            texts.append([s, 0])
    # 1) 背景  2) 弹性形变(仅背景)
    scene = elastic(rng, gen_background(rng, S))
    # 3) 文本贴片(target 严禁被遮挡: 任何新文本覆盖已有 target 面积 >15% 即重试位置;
    #    15% 概率使用与背景接近的低对比墨色——真实低可读场景, 但人眼仍可辨)
    placed = []  # (文本四角 scene 坐标 4x2, is_target)
    boxes = []   # 已贴文本的轴对齐包围盒
    for text, is_t in texts:
        fp = scene_fonts[int(rng.integers(len(scene_fonts)))]
        low_contrast = rng.random() < 0.15
        patch, rect = render_text_patch(rng, text, fp)
        warped, tc = warp_patch(rng, patch, rect)
        ph, pw = warped.shape[:2]
        ok = False
        px = py = 0
        bb = None
        for _ in range(10):
            px = int(rng.uniform(0.08 * S, 0.92 * S) - pw / 2)
            py = int(rng.uniform(0.08 * S, 0.92 * S) - ph / 2)
            bb = _aabb(tc + [px, py])
            if not any(t == 1 and _cover_ratio(bb, ob) > 0.15 for ob, t in boxes):
                ok = True
                break
        if not ok:
            continue  # 位置重试失败, 放弃该文本
        if low_contrast:  # 取贴入位置的背景均值, 墨色 = 背景 ± [25,85] 亮度差
            x1, y1 = max(0, int(bb[0])), max(0, int(bb[1]))
            x2, y2 = min(S, int(bb[2]) + 1), min(S, int(bb[3]) + 1)
            if x2 > x1 and y2 > y1:
                mean = scene[y1:y2, x1:x2].reshape(-1, 3).mean(0)
                sign = -1.0 if mean.mean() > 128 else 1.0
                ink = tuple(int(v) for v in np.clip(mean + sign * rng.uniform(25, 85, 3), 0, 255))
                patch, rect = render_text_patch(rng, text, fp, ink=ink)
                warped, tc = warp_patch(rng, patch, rect)
                bb = _aabb(tc + [px, py])
        paste_rgba(scene, warped, px, py)
        placed.append((tc + [px, py], is_t))
        boxes.append((bb, is_t))
    if not any(t == 1 for _, t in placed):
        return None
    # 4) 全局透视 + 重投影四角 -> 包围盒 clip; 过小/出图则丢弃该标注
    scene, tcs = perspective(rng, scene, [c for c, _ in placed])
    out_boxes, out_flags = [], []
    for corners, (_, is_t) in zip(tcs, placed):
        x1, y1, x2, y2 = _aabb(corners)
        x1, y1 = min(max(x1, 0.0), float(S)), min(max(y1, 0.0), float(S))
        x2, y2 = min(max(x2, 0.0), float(S)), min(max(y2, 0.0), float(S))
        if (x2 - x1) * (y2 - y1) < 64.0:  # 面积 <64px² 或完全出图
            continue
        out_boxes.append([x1, y1, x2, y2])
        out_flags.append(is_t)
    if not any(f == 1 for f in out_flags):
        return None  # target 全丢 -> 上层重 roll
    # 5) 光度增强(遮挡块与 target 零相交; 模糊/低采样保持人眼可读)
    scene = photometric(rng, scene, [b for b, f in zip(out_boxes, out_flags) if f == 1])
    # mask: 整串 query 标准渲染
    mask = render_mask(query, cfg["mask_font"], cfg["mask_height"], cfg["mask_width"])
    return query, mask, scene, out_boxes, out_flags


def gen_one(task):
    """多进程 worker: 生成单张样本并写盘, 返回 labels.jsonl 记录。"""
    rng = np.random.default_rng([task["seed"], task["split_tag"], task["idx"]])
    result = None
    for _ in range(8):  # target 全丢时重 roll(同一 rng 继续, 保持确定性)
        result = build_sample(rng, task["cfg"], task["fonts"])
        if result is not None:
            break
    if result is None:
        raise RuntimeError(f"样本 {task['idx']} 连续重 roll 失败")
    query, mask_img, scene_img, boxes, flags = result
    name = f"{task['idx']:06d}"
    out = Path(task["out_dir"])
    # 写文件用 imencode + tofile(规避中文路径问题); scene 内部为 RGB, 转 BGR 保存
    cv2.imencode(".png", cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR))[1].tofile(str(out / "scene" / f"{name}.png"))
    cv2.imencode(".png", mask_img)[1].tofile(str(out / "mask" / f"{name}.png"))
    return {
        "scene": f"scene/{name}.png",
        "mask": f"mask/{name}.png",
        "boxes": [[round(float(v), 1) for v in b] for b in boxes],
        "is_target": [int(f) for f in flags],
        "query": query,
    }


def main():
    ap = argparse.ArgumentParser(description="掩码条件化字形检测 - 全合成数据生成")
    ap.add_argument("--config", required=True, help="yaml 配置路径")
    ap.add_argument("--split", choices=["both", "train", "val"], default="both")
    ap.add_argument("--n", type=int, default=None, help="覆盖 config 中的数量")
    ap.add_argument("--seed", type=int, default=None, help="覆盖 config 中的 seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    dcfg = cfg_all["data"]
    seed = args.seed if args.seed is not None else int(cfg_all.get("seed", 0))
    out_root = Path(dcfg["out_dir"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    cfg = {k: dcfg[k] for k in ("scene_size", "mask_height", "mask_width", "mask_font",
                                "max_texts_per_scene", "hard_neg_prob",
                                "target_count_range", "distractor_count_range")}
    splits = []
    if args.split in ("both", "train"):  # 训练池(不含 val held-out 字体)
        splits.append(("train", 0, dcfg["train_size"], list(dcfg["scene_fonts"])))
    if args.split in ("both", "val"):    # val 只用 held-out 字体, 检验字体泛化
        splits.append(("val", 1, dcfg["val_size"], list(dcfg["val_held_out_fonts"])))

    for name, tag, size, fonts in splits:
        n = args.n if args.n is not None else int(size)
        sdir = out_root / name
        (sdir / "scene").mkdir(parents=True, exist_ok=True)
        (sdir / "mask").mkdir(parents=True, exist_ok=True)
        tasks = [dict(idx=i, seed=seed, split_tag=tag, cfg=cfg, fonts=fonts, out_dir=str(sdir))
                 for i in range(n)]
        t0 = time.perf_counter()
        recs = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for rec in tqdm(ex.map(gen_one, tasks), total=n, desc=f"synth-{name}"):
                recs.append(rec)
        dt = time.perf_counter() - t0
        with open(sdir / "labels.jsonl", "w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{name}] {n} 张完成, 总耗时 {dt:.1f}s, 平均每张 {dt / max(1, n) * 1000:.0f}ms -> {sdir}")


if __name__ == "__main__":
    main()
