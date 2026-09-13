"""v4 数据目检：旋转框画法校验 + 在线 mask 渲染检查。

每张图：rbox 多边形（绿=target，红=干扰/hard-neg）+ 阅读方向刻度线 + query；
下方拼该 query 的在线渲染 mask（棋盘底显透明区）。输出到 runs/_v4peek/。
另打印角度分布与框尺寸统计，供 sanity check。

用法: pdm run python tools/v4peek.py --data datasets/mvp6peek/train --n 12
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from glyphdet.core.synth import render_mask_rgba

FONTS = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"]


def rbox_corners(b):
    cx, cy, w, h, th = b
    c, s = math.cos(th), math.sin(th)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([cx, cy])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/mvp6peek/train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="runs/_v4peek")
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(root / "labels.jsonl", encoding="utf-8")]

    angs, ws, hs = [], [], []
    for idx, row in enumerate(rows[: args.n]):
        img = cv2.imdecode(np.fromfile(str(root / row["scene"]), np.uint8),
                           cv2.IMREAD_COLOR)
        for b, t in zip(row["boxes"], row["is_target"]):
            pts = rbox_corners(b).astype(np.int32)
            col = (0, 210, 0) if t else (0, 0, 220)
            cv2.polylines(img, [pts], True, col, 2 if t else 1)
            cx, cy, w, h, th = b
            tip = (int(cx + math.cos(th) * w * 0.6), int(cy + math.sin(th) * w * 0.6))
            cv2.arrowedLine(img, (int(cx), int(cy)), tip, (255, 160, 0), 1, tipLength=0.15)
            angs.append(math.degrees(th) % 360)
            ws.append(w)
            hs.append(h)
        rng = np.random.default_rng(idx)
        mask = render_mask_rgba(row["query"], FONTS[0], 64, 768, rng=rng, font_pool=FONTS)
        mh, mw = mask.shape
        checker = ((np.indices((mh, mw)).sum(0) // 6) % 2 * 40 + 140).astype(np.float32)
        mvis = (checker * (1 - mask) + 255 * mask).astype(np.uint8)
        scale = max(1, round(img.shape[1] / mw))
        mvis = cv2.resize(mvis, (mw * scale, mh * scale), interpolation=cv2.INTER_NEAREST)
        mvis = cv2.cvtColor(mvis, cv2.COLOR_GRAY2BGR)
        canvas = np.zeros((img.shape[0] + mvis.shape[0], img.shape[1], 3), np.uint8)
        canvas[: img.shape[0]] = img
        mw_fit = min(mvis.shape[1], img.shape[1])
        canvas[img.shape[0]:, :mw_fit] = mvis[:, :mw_fit]
        cv2.putText(canvas, row["query"], (6, img.shape[0] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imencode(".png", canvas)[1].tofile(str(out / f"{idx:04d}.png"))

    angs, ws, hs = np.array(angs), np.array(ws), np.array(hs)
    print(f"框数 {len(angs)}  角度 min/p25/p50/p75/max = "
          f"{np.percentile(angs, [0, 25, 50, 75, 100]).round(0)}")
    print(f"w min/p50/max = {np.percentile(ws, [0, 50, 100]).round(0)}  "
          f"h min/p50/max = {np.percentile(hs, [0, 50, 100]).round(0)}")
    hist, _ = np.histogram(angs, bins=8, range=(0, 360))
    print(f"角度直方图(45°一格): {hist}")
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
