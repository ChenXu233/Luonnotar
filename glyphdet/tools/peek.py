"""抽检可视化: 每行一个样本, 左边 mask 图 + query 文本, 右边场景画框
(target 绿色粗框, 干扰红色细框), 拼成网格存一张大图。

用法:
    python tools/peek.py --dir datasets/mvp/train --n 16 --out /tmp/peek.png [--cols C]
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def imread(p, flag=cv2.IMREAD_COLOR):
    """cv2.imdecode + np.fromfile, 规避中文路径问题。"""
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flag)


def imwrite(p, img):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".png", img)[1].tofile(str(p))


def main():
    ap = argparse.ArgumentParser(description="合成数据抽检可视化")
    ap.add_argument("--dir", required=True, help="split 目录(含 labels.jsonl)")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=None, help="网格列数, 默认自动(每列最多8行)")
    args = ap.parse_args()

    d = Path(args.dir)
    with open(d / "labels.jsonl", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f][: args.n]
    rows = []
    for rec in recs:
        scene = imread(d / rec["scene"])
        for box, t in zip(rec["boxes"], rec["is_target"]):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            if t:  # target: 绿色粗框
                cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 200, 0), 2)
            else:  # 干扰: 红色细框
                cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 0, 230), 1)
        mask_gray = imread(d / rec["mask"], cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"无法读取 mask: {d / rec['mask']}")
        mask = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
        h = scene.shape[0]
        panel = np.full((h, max(280, mask.shape[1] + 24), 3), 255, np.uint8)  # 左侧面板: mask + query
        mh, mw = mask.shape[:2]
        panel[8:8 + mh, 12:12 + mw] = mask
        cv2.putText(panel, "Q:" + rec["query"], (12, 8 + mh + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        rows.append(np.hstack([panel, np.full((h, 6, 3), 200, np.uint8), scene]))

    n = len(rows)
    cols = args.cols or max(1, math.ceil(n / 8))
    rpp = math.ceil(n / cols)  # 每列行数
    rh, rw = rows[0].shape[:2]
    canvas = np.full((rpp * (rh + 8) + 8, cols * (rw + 8) + 8, 3), 245, np.uint8)
    for i, r in enumerate(rows):
        c, rr = divmod(i, rpp)
        y, x = 8 + rr * (rh + 8), 8 + c * (rw + 8)
        canvas[y:y + rh, x:x + rw] = r
    imwrite(args.out, canvas)
    print(f"peek -> {args.out} ({n} 样本, {cols} 列)")


if __name__ == "__main__":
    main()
