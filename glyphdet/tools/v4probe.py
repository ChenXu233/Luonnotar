"""v4 go/no-go 探针：mutation 分离度（口径同 v3probe，旋转框 + 两段式版）。

对 val 子集每张图：
  正例：query mask → 提案 decode → matcher 分（记录 top1 分与是否旋转 IoU≥0.5 命中 target）
  变异：query 经 hard_negative 一字扰动 → 同管线（记录全场最高 matcher 分）
判定：mutation-fp（变异查询仍出 ≥0.5 框的比例）与分数 AUROC。
门：mutation_fp ≤2%（v3 在此 0.80 翻车）；分离 AUROC 应 →1。

用法: pdm run python tools/v4probe.py --weights runs/mvp_v4/last.pt [--n 150] [--data datasets/mvp6/val]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.decode import decode_rprops
from glyphdet.core.eval import auroc, rbox_iou
from glyphdet.core.model import build_model, pair_grid
from glyphdet.core.synth import hard_negative, render_mask_rgba


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/mvp6/val")
    ap.add_argument("--n", type=int, default=150)
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

    ds = GlyphDataset(args.data)
    idxs = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)
    pos_top, mut_max = [], []
    n_hit = mut_fp = 0
    examples = []
    for idx in idxs:
        scene, query, boxes, is_target = ds[int(idx)]
        feats = model.extract(scene[None].to(device))
        outs = model.prop_maps(feats)
        rb, _ = decode_rprops(outs, m["strides"], m["reg_max"],
                              ec["prop_threshold"], ec["nms_iou"])
        boxes_np = boxes.numpy()
        gt_t = boxes_np[(is_target.numpy() > 0.5) & (boxes_np[:, 2] > 1)]

        def pipeline(text, seed):
            mq = render(text, seed)
            tpl = model.mask_enc(mq)
            ink = model.matcher.ink_cols(mq)
            if len(rb) == 0:
                return np.zeros(0, np.float32)
            bt = torch.from_numpy(rb).to(device)
            strips = model.matcher.strips(feats[0], bt)
            pairs = pair_grid(1, len(rb), device)
            span = 8.0 * bt[:, 2] / bt[:, 3].clamp(min=1.0)
            return torch.sigmoid(model.matcher.score(
                strips, tpl, ink, pairs, span=span[pairs[1]])).cpu().numpy()

        ps = pipeline(query, int(idx))
        top = int(np.argmax(ps)) if len(ps) else -1
        pos_top.append(float(ps.max()) if len(ps) else 0.0)
        hit = False
        if top >= 0 and len(gt_t):
            hit = bool(rbox_iou(rb[top : top + 1], gt_t)[0].max() >= 0.5)
        n_hit += int(hit)

        mut = hard_negative(np.random.default_rng(int(idx) + 99991), query)
        if mut == query:
            continue
        ms = pipeline(mut, int(idx) + 555)
        mm = float(ms.max()) if len(ms) else 0.0
        mut_max.append(mm)
        if mm >= 0.5:
            mut_fp += 1
            if len(examples) < 10:
                examples.append({"img": ds.rows[int(idx)]["scene"],
                                 "query": query, "mut": mut, "score": round(mm, 3)})

    pos_top = np.array(pos_top)
    mut_max = np.array(mut_max)
    report = {
        "weights": args.weights,
        "n_img": len(idxs),
        "pos_top1_mean": float(pos_top.mean()),
        "pos_hit_iou05": n_hit / max(len(idxs), 1),
        "mut_max_mean": float(mut_max.mean()),
        "mut_max_p90": float(np.percentile(mut_max, 90)) if len(mut_max) else None,
        "mutation_fp": mut_fp / max(len(mut_max), 1),
        "separation_auroc": auroc(pos_top, mut_max),
        "fp_examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(args.weights).parent / "v4probe_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
