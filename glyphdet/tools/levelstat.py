"""级别归因：val 上每个被检出的 target 到底是哪一级（stride）检出的，按字高分桶。

回答两个问题：v1 的 P5（stride32）是否真是死重；v2 的 P2 是否真接管了小字。
口径：各级别单独 decode（score_thr=0.3），target 按 IoU≥0.5 归因到最佳级别。

  PYTHONPATH=仓库根 pdm run python tools/levelstat.py --config ... --weights ... --scan 300
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.decode import decode_outputs
from glyphdet.core.eval import iou_matrix
from glyphdet.core.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--scan", type=int, default=300)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, ec = cfg["model"], cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    strides = m["strides"]

    val = GlyphDataset(Path(cfg["data"]["out_dir"]) / "val")
    buckets = [("S<28", lambda h: h < 28), ("M28-56", lambda h: 28 <= h < 56),
               ("L56+", lambda h: h >= 56)]
    # hit[level][bucket]，tot[bucket]，以及桶内目标平均宽/宽高中位数
    hit = [[0] * len(buckets) for _ in strides]
    tot = [0] * len(buckets)
    widths = [[] for _ in buckets]
    n_pred_lvl = [0] * len(strides)
    for idx in range(min(len(val), args.scan)):
        scene, mask, boxes, is_target = val[idx]
        with torch.no_grad():
            outs = model(scene[None].to(device), mask[None].to(device))
        per_lvl = [decode_outputs([outs[i]], [strides[i]], m["reg_max"],
                                  score_thr=0.3, nms_iou=ec["nms_iou"])
                   for i in range(len(strides))]
        for i in range(len(strides)):
            n_pred_lvl[i] += len(per_lvl[i][0])
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[tgt > 0.5]
        gt_t = gt_t[gt_t.sum(1) > 0]
        for b in gt_t:
            h, w = b[3] - b[1], b[2] - b[0]
            bi = next(k for k, (_, f) in enumerate(buckets) if f(h))
            tot[bi] += 1
            widths[bi].append(w)
            ious = []
            for i in range(len(strides)):
                pb = per_lvl[i][0]
                ious.append(iou_matrix(pb, b[None]).max() if len(pb) else 0.0)
            if max(ious) >= 0.5:
                hit[int(np.argmax(ious))][bi] += 1

    print(f"weights={args.weights}  strides={strides}")
    print("命中矩阵（行=级别, 列=字高桶）:")
    header = "        " + "".join(f"{name:>10}" for name, _ in buckets) + "     检出数"
    print(header)
    for i, s in enumerate(strides):
        row = "".join(f"{hit[i][k]:>10}" for k in range(len(buckets)))
        print(f"  s{s:<4}  {row}  {n_pred_lvl[i]:>8}")
    print("桶内 target 总数 / 平均宽度:")
    for k, (name, _) in enumerate(buckets):
        mw = float(np.mean(widths[k])) if widths[k] else 0
        print(f"  {name:>8}: n={tot[k]:>4}  avg_w={mw:6.1f}px")


if __name__ == "__main__":
    main()
