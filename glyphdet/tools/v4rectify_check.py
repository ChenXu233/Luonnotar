"""rectify 几何自检：用 MatcherV4.strips 的同一套网格数学直接采原始场景 RGB，
验证"框→网格"映射抽出来的条带是否就是正立文本（不经模型，纯几何）。

用法: pdm run python tools/v4rectify_check.py --data datasets/mvp6mini/val --n 8
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from glyphdet.core.dataset import imread_chw

SH, SW, FS = 8, 128, 4  # 与 MatcherV4 一致（条带 8 行、128 列、P2 stride 4）


def rectify_raw(img, boxes, out_h=64):
    """img (3,S,S) float tensor；boxes (n,5)；返回 (n, 3, out_h, SW*out_h//8) 上采样条带。"""
    S = img.shape[-1]
    n = len(boxes)
    ys, xs = torch.meshgrid(torch.arange(SH), torch.arange(SW), indexing="ij")
    ys, xs = ys.float()[None], xs.float()[None]
    sp = (boxes[:, 3] / SH).view(n, 1, 1)
    ou = (xs + 0.5 - SW / 2) * sp
    ov = (ys + 0.5 - SH / 2) * sp
    c = torch.cos(boxes[:, 4]).view(n, 1, 1)
    s = torch.sin(boxes[:, 4]).view(n, 1, 1)
    px = boxes[:, 0].view(n, 1, 1) + c * ou - s * ov
    py = boxes[:, 1].view(n, 1, 1) + s * ou + c * ov
    gx = px / S * 2 - 1  # 直接采原图：stride=1
    gy = py / S * 2 - 1
    grid = torch.stack([gx, gy], -1)
    strips = F.grid_sample(img[None].expand(n, -1, -1, -1), grid,
                           mode="bilinear", padding_mode="zeros", align_corners=False)
    return F.interpolate(strips, scale_factor=out_h // SH, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/mvp6mini/val")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default="runs/_v4rectify")
    args = ap.parse_args()
    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(root / "labels.jsonl", encoding="utf-8")]
    for idx in range(min(args.n, len(rows))):
        row = rows[idx]
        img = imread_chw(root / row["scene"])
        boxes = torch.tensor([b for b, t in zip(row["boxes"], row["is_target"]) if t == 1])
        if len(boxes) == 0:
            continue
        strips = rectify_raw(img, boxes)
        parts = []
        for j in range(len(boxes)):
            st = np.ascontiguousarray(
                (strips[j].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
            cv2.putText(st, row["query"], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            parts.append(st)
        cv2.imencode(".png", np.vstack(parts))[1].tofile(str(out / f"{idx:04d}.png"))
    print(f"输出: {out}（每张图的所有 target 条带，左上黄字=真 query）")


if __name__ == "__main__":
    main()
