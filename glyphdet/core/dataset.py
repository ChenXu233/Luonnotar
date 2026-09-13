"""PyTorch Dataset v4：读取 synth v4 合成数据集（旋转框标注），worker 内完成分配与在线 mask 渲染。

样本（labels.jsonl 每行）：
  {"scene": "scene/000001.png", "query": "24-6-1234",
   "boxes": [[cx,cy,w,h,θ],...], "is_target": [1,0,...]}   # θ=阅读方向角(弧度,图像坐标顺时针正)

v4 关键变化：
- mask 不落盘：query/变异 mask 用 synth.render_mask_rgba 在线渲染（透明墨迹 float [0,1]，
  墨迹=1），变长宽；训练期每次调用随机字体/字重/字宽抖动 = 免费增广。
- 提案是 query 无关的"文本检测"：所有标注框（target+hard-neg+干扰）都是 textness 正样本。
- targets 每级 (9,H,W)：[textness(1), reg_ltrb(4, 框体系/stride), sincos(2), pos(1), weight(1)]。
- 额外产/bg 随机框（kind=-1）：不进分配，只给 matcher 当负样本对。
- box_kind: 1=target 0=场景非 target -1=bg 随机；matcher pair 标签 = (query×target)=1 其余 0。

cfg 为 None 时返回评估四元组 (scene, query, boxes, is_target)（mask 由调用方按需渲染）。
"""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from glyphdet.core.synth import hard_negative, render_mask_rgba

MAX_BOXES = 8
# v2 按字高分配：字高窄带 16~64px 每级一个倍频程；宽度交给大核宽兜底。
LEVEL_RANGES_H = {4: (0, 28), 8: (28, 56), 16: (56, 1e9)}
CENTER_SHRINK = 0.25


def level_for_box(h, w, reg_max):
    """字高定基准级；回归超界则上浮到能装下的最粗级（0.75w 护栏，见 v2.1 批注）。"""
    lvls = sorted(LEVEL_RANGES_H)
    i = next(
        k for k, s in enumerate(lvls) if LEVEL_RANGES_H[s][0] <= h < LEVEL_RANGES_H[s][1]
    )
    while i < len(lvls) - 1 and 0.75 * w > (reg_max - 2) * lvls[i]:
        i += 1
    return lvls[i]


def imread_chw(path, gray=False):
    """cv2 读图（中文路径安全），返回 CHW float [0,1]。"""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    if img is None:
        raise FileNotFoundError(path)
    if not gray:
        img = img[:, :, ::-1]  # BGR→RGB
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img[None] if gray else img.transpose(2, 0, 1))


def build_targets_rbox(boxes5, strides, in_size, reg_max):
    """boxes5: (N,5) numpy (cx,cy,w,h,θ)；返回 list of (9,H,W) numpy。
    所有框都是 textness 正样本；冲突取面积最小者。"""
    outs = []
    for s in strides:
        hw = in_size // s
        px = (np.arange(hw, dtype=np.float32) + 0.5) * s
        gx, gy = np.meshgrid(px, px)
        text = np.zeros((hw, hw), np.float32)
        weight = np.ones((hw, hw), np.float32)
        reg = np.zeros((4, hw, hw), np.float32)
        ang = np.zeros((2, hw, hw), np.float32)
        pos = np.zeros((hw, hw), np.float32)
        assigned = np.full((hw, hw), 1e18, np.float32)
        for cx, cy, w, h, th in boxes5:
            if w <= 1 or h <= 1:
                continue
            c, sn = math.cos(th), math.sin(th)
            dx, dy = gx - cx, gy - cy
            u = c * dx + sn * dy   # 框体系坐标（阅读方向为 u 轴）
            v = -sn * dx + c * dy
            l, r = w / 2 + u, w / 2 - u
            t, b = h / 2 + v, h / 2 - v
            center = (np.abs(u) < CENTER_SHRINK * w) & (np.abs(v) < CENTER_SHRINK * h)
            lvl = np.full((hw, hw), s == level_for_box(h, w, reg_max), bool)
            area = w * h
            cand = center & lvl & (area < assigned)
            if not cand.any():
                continue
            pos[cand] = 1.0
            text[cand] = 1.0
            reg[0][cand] = l[cand] / s
            reg[1][cand] = t[cand] / s
            reg[2][cand] = r[cand] / s
            reg[3][cand] = b[cand] / s
            ang[0][cand] = sn
            ang[1][cand] = c
            assigned[cand] = area
        reg = np.clip(reg, 0, reg_max - 1.01)
        outs.append(
            np.concatenate(
                [text[None], reg, ang, pos[None], weight[None]], axis=0
            ).astype(np.float32)
        )
    return outs


def _rand_bg_boxes(rng, boxes5, S, n):
    """随机背景框（不与任何标注框的 AABB 相交）：matcher 的"野纹理"负样本。"""
    aabbs = []
    for cx, cy, w, h, th in boxes5:
        c, sn = abs(math.cos(th)), abs(math.sin(th))
        aw, ah = w * c + h * sn, w * sn + h * c
        aabbs.append((cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2))
    out = []
    for _ in range(n):
        for _try in range(6):
            h = float(rng.uniform(16, 56))
            w = h * float(rng.uniform(2.0, 9.0))
            th = float(rng.uniform(0, 2 * math.pi))
            cx = float(rng.uniform(0.05 * S, 0.95 * S))
            cy = float(rng.uniform(0.05 * S, 0.95 * S))
            c, sn = abs(math.cos(th)), abs(math.sin(th))
            aw, ah = w * c + h * sn, w * sn + h * c
            x1, y1, x2, y2 = cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2
            hit = any(
                min(x2, b[2]) - max(x1, b[0]) > 0 and min(y2, b[3]) - max(y1, b[1]) > 0
                for b in aabbs
            )
            if not hit:
                out.append([cx, cy, w, h, th])
                break
    return out


class GlyphDataset(Dataset):
    def __init__(self, root, cfg: dict | None = None):
        """cfg 为 None 时返回评估四元组；否则返回训练组（在线 mask + 预分配 targets）。"""
        self.root = Path(root)
        self.cfg = cfg
        with open(self.root / "labels.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def _render(self, text, rng):
        d = self.cfg["data"]
        fonts = d.get("mask_fonts") or [d["mask_font"]]
        img = render_mask_rgba(text, d["mask_font"], d["mask_height"],
                               d["mask_max_width"], rng=rng, font_pool=fonts)
        return torch.from_numpy(img)[None]  # (1,H,Wn) float32

    def __getitem__(self, i: int):
        row = self.rows[i]
        scene = imread_chw(self.root / row["scene"])
        boxes = np.zeros((MAX_BOXES, 5), np.float32)
        kinds = np.zeros(MAX_BOXES, np.float32)  # 1=target 0=非target -1=bg
        n = min(len(row["boxes"]), MAX_BOXES)
        if n:
            boxes[:n] = np.asarray(row["boxes"][:n], np.float32)
            kinds[:n] = np.asarray(row["is_target"][:n], np.float32)
        if self.cfg is None:
            return (scene, row["query"], torch.from_numpy(boxes),
                    torch.from_numpy((kinds > 0.5).astype(np.float32)))
        rng = np.random.default_rng()
        mq = self._render(row["query"], rng)
        tc = self.cfg["train"]
        muts = []
        for _ in range(int(tc.get("mut_per_scene", 2))):
            m = row["query"]
            for _try in range(4):
                m = hard_negative(rng, row["query"])
                if m != row["query"]:
                    break
            if m != row["query"]:
                muts.append(self._render(m, rng))
        m = self.cfg["model"]
        gt = boxes[:n][kinds[:n] >= 0]  # 分配只用真实标注（bg 框不算文本）
        targets = build_targets_rbox(gt, m["strides"], m["in_size"], m["reg_max"])
        bg = _rand_bg_boxes(rng, boxes[:n], m["in_size"], int(tc.get("bg_neg_boxes", 2)))
        if bg and n + len(bg) <= MAX_BOXES:
            boxes[n:n + len(bg)] = np.asarray(bg, np.float32)
            kinds[n:n + len(bg)] = -1.0
        return (scene, mq, muts, [torch.from_numpy(t) for t in targets],
                torch.from_numpy(boxes), torch.from_numpy(kinds))


def collate_v4(batch):
    """变长 mask padding 批处理：query/mut 分别 pad 到批内最大宽（8 的倍数，透明=0）。"""
    def pad_masks(ms):
        wmax = max(m.shape[-1] for m in ms)
        wmax = (wmax + 7) // 8 * 8
        out = torch.zeros(len(ms), 1, ms[0].shape[1], wmax)
        for i, m in enumerate(ms):
            out[i, :, :, : m.shape[-1]] = m
        return out

    scenes = torch.stack([b[0] for b in batch])
    mq = pad_masks([b[1] for b in batch])
    mm = pad_masks([m for b in batch for m in b[2]]) if batch[0][2] else torch.zeros(0)
    targets = [torch.stack([b[3][lv] for b in batch]) for lv in range(len(batch[0][3]))]
    boxes = torch.stack([b[4] for b in batch])
    kinds = torch.stack([b[5] for b in batch])
    return scenes, mq, mm, targets, boxes, kinds


if __name__ == "__main__":
    import sys
    import yaml

    cfg = yaml.safe_load(open(sys.argv[2], encoding="utf-8")) if len(sys.argv) > 2 else None
    ds = GlyphDataset(sys.argv[1] if len(sys.argv) > 1 else "datasets/mvp6/train", cfg)
    print(f"样本数: {len(ds)}")
    item = ds[0]
    if cfg is None:
        s, q, b, t = item
        print(f"scene {tuple(s.shape)} query {q!r} boxes {b.shape} targets {int(t.sum())}")
    else:
        s, mq, mm, tg, b, k = item
        n_pos = sum(int(t[7].sum()) for t in tg)
        print(f"scene {tuple(s.shape)} mask {tuple(mq.shape)} muts {len(mm)} "
              f"boxes {b.shape} pos点数 {n_pos}")
        batch = collate_v4([ds[0], ds[1]])
        print(f"collate: scene {tuple(batch[0].shape)} mq {tuple(batch[1].shape)} "
              f"mm {tuple(batch[2].shape)} targets {tuple(batch[3][0].shape)}")
