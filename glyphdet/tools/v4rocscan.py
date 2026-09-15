# -*- coding: utf-8 -*-
"""v4 操作点扫描：pos/mut 分数分布 → 给定 fp 目标的阈值与对应 recall。

recall(thr) = top1 分 ≥thr 且该框 IoU≥0.5 命中 target 的图占比（真能取到件）
fp(thr)     = 变异查询 max 分 ≥thr 的图占比（会错认）
auroc 只评排序；本工具回答"App 该把阈值定在哪"。

用法: pdm run python tools/v4rocscan.py --weights runs/mvp_v4/ep25.pt [--n 300]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.decode import decode_rprops
from glyphdet.core.eval import rbox_iou
from glyphdet.core.model import build_model, pair_grid
from glyphdet.core.synth import hard_negative, render_mask_rgba


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/mvp6/val")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    m, ec, d = cfg["model"], cfg["eval"], cfg["data"]
    model = build_model(cfg).eval()
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    fonts = d.get("mask_fonts") or [d["mask_font"]]

    def render(text, seed):
        rng = np.random.default_rng(seed)
        img = render_mask_rgba(text, d["mask_font"], d["mask_height"],
                               d["mask_max_width"], rng=rng, font_pool=fonts)
        t = torch.from_numpy(img)[None, None]
        w8 = (t.shape[-1] + 7) // 8 * 8
        return torch.nn.functional.pad(t, (0, w8 - t.shape[-1])).to(device)

    def pipeline(feats, text, seed, nbox):
        mq = render(text, seed)
        tpl = model.mask_enc(mq)
        ink = model.matcher.ink_cols(mq)
        if nbox == 0:
            return np.zeros(0, np.float32)
        bt = torch.from_numpy(rb).to(device)
        strips = model.matcher.strips(feats[0], bt)
        pairs = pair_grid(1, len(bt), device)
        span = 8.0 * bt[:, 2] / bt[:, 3].clamp(min=1.0)
        return torch.sigmoid(model.matcher.score(
            strips, tpl, ink, pairs, span=span[pairs[1]])).cpu().numpy()

    ds = GlyphDataset(args.data)
    idxs = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)
    pos_score, pos_hit, mut_score = [], [], []
    for idx in idxs:
        scene, query, boxes, is_target = ds[int(idx)]
        feats = model.extract(scene[None].to(device))
        outs = model.prop_maps(feats)
        rb, _ = decode_rprops(outs, m["strides"], m["reg_max"],
                              ec["prop_threshold"], ec["nms_iou"])
        boxes_np = boxes.numpy()
        gt_t = boxes_np[(is_target.numpy() > 0.5) & (boxes_np[:, 2] > 1)]

        ps = pipeline(feats, query, int(idx), len(rb))
        if len(ps):
            top = int(np.argmax(ps))
            pos_score.append(float(ps[top]))
            hit = bool(len(gt_t) and rbox_iou(rb[top:top + 1], gt_t)[0].max() >= 0.5)
        else:
            pos_score.append(0.0)
            hit = False
        pos_hit.append(hit)

        mut = hard_negative(np.random.default_rng(int(idx) + 99991), query)
        if mut == query:
            continue
        ms = pipeline(feats, mut, int(idx) + 555, len(rb))
        mut_score.append(float(ms.max()) if len(ms) else 0.0)

    pos_score = np.array(pos_score)
    pos_hit = np.array(pos_hit)
    mut_score = np.array(mut_score)
    print(f"n={len(idxs)}（变异 {len(mut_score)}） 权重 {args.weights}")
    print(f"{'fp目标':>7} {'阈值':>7} {'recall':>8} {'fp实测':>8}")
    for fp_t in (0.10, 0.05, 0.02, 0.01, 0.005):
        if len(mut_score) * fp_t < 1:
            continue
        thr = float(np.quantile(mut_score, 1 - fp_t))
        rec = float(((pos_score >= thr) & pos_hit).mean())
        fp_r = float((mut_score >= thr).mean())
        print(f"{fp_t:7.1%} {thr:7.3f} {rec:8.3f} {fp_r:8.3f}")
    # 全分布参考点
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        rec = float(((pos_score >= thr) & pos_hit).mean())
        fp_r = float((mut_score >= thr).mean())
        print(f"  固定thr {thr:.2f} -> recall {rec:.3f} fp {fp_r:.3f}")
    out = Path(args.weights).parent / "v4rocscan.json"
    out.write_text(json.dumps({
        "pos_score": pos_score.tolist(), "pos_hit": pos_hit.tolist(),
        "mut_score": mut_score.tolist()}, ensure_ascii=False), encoding="utf-8")
    print(f"分布写入 {out}")


if __name__ == "__main__":
    main()
