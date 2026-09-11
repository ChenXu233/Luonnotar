"""PyTorch Dataset：读取 core/synth.py 产出的合成数据集，并在 worker 内完成正负分配。

样本（labels.jsonl 每行）：
  {"scene": "scene/000001.png", "mask": "mask/000001.png",
   "boxes": [[x1,y1,x2,y2],...], "is_target": [1,0,...], "query": "24-6-1234"}

__getitem__ 返回 (scene, mask, targets)：
  scene (3,416,416) / mask (1,64,256) float [0,1]；
  targets = 3 级 list，每级 (8,H,W) float32：
    [match(1), reg_ltrb(4, stride 格单位), centerness(1), pos_mask(1), neg_weight(1)]
分配规则（FCOS 式）：中心采样（框向心收缩 25%）+ 层级回归范围，冲突取面积最小者；
is_target=0 的干扰/hard-neg 框中心区域作为加权负样本。
分配放 dataset worker 里做（多进程并行），训练热路径零 Python 循环开销。
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MAX_BOXES = 8
# 每级回归范围（像素，max(l,t,r,b)），FCOS 惯例
LEVEL_RANGES = {8: (0, 64), 16: (64, 128), 32: (128, 1e9)}
# v2 按字高分配：字高窄带 16~64px 每级一个倍频程；宽度交给大核宽兜底。
# v1 的 maxd 分配在极端宽高比下 maxd≈半宽，小字长串会被错配到粗级别
LEVEL_RANGES_H = {4: (0, 28), 8: (28, 56), 16: (56, 1e9)}
CENTER_SHRINK = 0.25


def level_for_box(h, w, reg_max):
    """字高定基准级；回归超界则上浮到能装下的最粗级。
    v2.1 护栏：ltrb 目标 clip 在 reg_max-1.01 格；中心采样带内格点最远距左边
    0.75w（CENTER_SHRINK=0.25），故按 0.75w 判定。旧规则 68% 目标被钳，
    recall@0.5 存在结构性天花板。"""
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


def build_targets(boxes, is_target, strides, in_size, reg_max, hard_neg_w,
                  assign="maxd"):
    """boxes: (N,4) numpy xyxy；返回 list of (8,H,W) numpy。
    assign: "maxd"(v1, FCOS 按最大边分配) / "height"(v2, 按字高分配)。"""
    outs = []
    for s in strides:
        hw = in_size // s
        px = (np.arange(hw, dtype=np.float32) + 0.5) * s  # 格中心 x
        gx, gy = np.meshgrid(px, px)  # (H,W)
        match = np.zeros((hw, hw), np.float32)
        weight = np.ones((hw, hw), np.float32)
        reg = np.zeros((4, hw, hw), np.float32)
        cen = np.zeros((hw, hw), np.float32)
        pos = np.zeros((hw, hw), np.float32)
        assigned_area = np.full((hw, hw), 1e18, np.float32)
        lo, hi = (LEVEL_RANGES_H if assign == "height" else LEVEL_RANGES)[s]
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            l = gx - x1
            t = gy - y1
            r = x2 - gx
            b = y2 - gy
            maxd = np.maximum(np.maximum(l, r), np.maximum(t, b))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hbw, hbh = (x2 - x1) * CENTER_SHRINK, (y2 - y1) * CENTER_SHRINK
            in_center = (np.abs(gx - cx) < hbw) & (np.abs(gy - cy) < hbh)
            if assign == "height":  # 字高定级 + 宽度护栏（标量），广播到网格
                in_range = np.full(
                    (hw, hw), s == level_for_box(y2 - y1, x2 - x1, reg_max), bool
                )
            else:
                in_range = (maxd >= lo) & (maxd < hi)
            if is_target[j] > 0.5:
                cand = in_center & in_range & (area < assigned_area)
                if not cand.any():
                    continue
                pos[cand] = 1.0
                match[cand] = 1.0
                reg[0][cand] = l[cand] / s
                reg[1][cand] = t[cand] / s
                reg[2][cand] = r[cand] / s
                reg[3][cand] = b[cand] / s
                assigned_area[cand] = area
                lr_min = np.minimum(l, r)
                lr_max = np.maximum(l, r)
                tb_min = np.minimum(t, b)
                tb_max = np.maximum(t, b)
                cen[cand] = np.sqrt(
                    np.clip(lr_min / np.maximum(lr_max, 1e-6), 0, 1)
                    * np.clip(tb_min / np.maximum(tb_max, 1e-6), 0, 1)
                )[cand]
            else:
                neg = in_center & in_range & (pos < 0.5)
                weight[neg] = hard_neg_w
        reg = np.clip(reg, 0, reg_max - 1.01)
        outs.append(
            np.concatenate(
                [match[None], reg, cen[None], pos[None], weight[None]], axis=0
            ).astype(np.float32)
        )
    return outs


class GlyphDataset(Dataset):
    def __init__(self, root, cfg: dict | None = None):
        """cfg 为 None 时返回裸 boxes（评估用）；否则返回预分配 targets（训练用）。"""
        self.root = Path(root)
        self.cfg = cfg
        with open(self.root / "labels.jsonl", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        scene = imread_chw(self.root / row["scene"])
        mask = imread_chw(self.root / row["mask"], gray=True)
        boxes = np.zeros((MAX_BOXES, 4), np.float32)
        is_target = np.zeros(MAX_BOXES, np.float32)
        n = min(len(row["boxes"]), MAX_BOXES)
        if n:
            boxes[:n] = np.asarray(row["boxes"][:n], np.float32)
            is_target[:n] = np.asarray(row["is_target"][:n], np.float32)
        if self.cfg is None:
            return scene, mask, torch.from_numpy(boxes), torch.from_numpy(is_target)
        m = self.cfg["model"]
        targets = build_targets(
            boxes, is_target, m["strides"], m["in_size"], m["reg_max"],
            self.cfg["train"]["loss"]["hard_neg_center_weight"],
            m.get("assign", "maxd"),
        )
        return scene, mask, [torch.from_numpy(t) for t in targets]


if __name__ == "__main__":
    import sys

    ds = GlyphDataset(sys.argv[1] if len(sys.argv) > 1 else "datasets/mvp/train")
    print(f"样本数: {len(ds)}")
    s, m, b, t = ds[0]
    print(f"scene {tuple(s.shape)} mask {tuple(m.shape)} boxes {b.shape} targets {int(t.sum())}")
