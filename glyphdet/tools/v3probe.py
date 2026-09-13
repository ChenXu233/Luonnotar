"""v3 toy 验证：单字变异分离度探针（全量训练的 go/no-go 门）。

对 val 子集每张图跑两次：
  正例：图自带的 query mask（应检出，记录 top1 分数与是否命中 target）
  变异：query 经 hard_negative 一字扰动后重渲 mask（应无 ≥0.5 检出，记录全场最高分）
判定：mutation-fp（变异查询仍出 ≥0.5 框的比例）与分数 AUROC。
v2.2 在这项上的病灶表现：近邻串分数 mean 0.72 / max 0.998——v3 的 min 聚合就是治它。

用法: pdm run python tools/v3probe.py --weights runs/mvp_v3_mini/last.pt [--n 150] [--data datasets/mvp5mini/val]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from glyphdet.core.dataset import GlyphDataset, imread_chw
from glyphdet.core.decode import decode_outputs
from glyphdet.core.eval import auroc, iou_matrix
from glyphdet.core.model import build_model
from glyphdet.core.synth import hard_negative, render_mask_slots


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="datasets/mvp5/val")
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    m = cfg["model"]
    model = build_model(cfg).eval()
    model.load_state_dict(ckpt["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dcfg = cfg["data"]
    fonts = dcfg.get("mask_fonts") or [dcfg["mask_font"]]
    rng = np.random.default_rng(7)
    ds = GlyphDataset(args.data)
    idxs = np.linspace(0, len(ds) - 1, min(args.n, len(ds))).astype(int)

    pos_top, mut_max = [], []  # 正例 top1 分 / 变异全场最高分
    n_hit = n_img = 0
    mut_fp = 0
    examples = []
    for idx in idxs:
        scene, mask, boxes, is_target = ds[int(idx)]
        query = ds.rows[int(idx)]["query"]
        scene_b = scene[None].to(device)

        outs = model(scene_b, mask[None].to(device))
        pb, ps = decode_outputs(outs, m["strides"], m["reg_max"])
        boxes_np = boxes.numpy()
        gt = boxes_np[(is_target.numpy() > 0.5) & (boxes_np.sum(1) > 0)]
        hit = False
        if len(pb) and len(gt):
            top = int(np.argmax(ps))
            hit = bool(iou_matrix(pb[top : top + 1], gt)[0].max() >= 0.5)
        pos_top.append(float(ps.max()) if len(ps) else 0.0)
        n_hit += int(hit)
        n_img += 1

        for _ in range(4):
            mut = hard_negative(rng, query)
            if mut != query:
                break
        if mut == query:
            continue
        mimg = render_mask_slots(mut, dcfg["mask_font"], 64, 384, rng=rng,
                                 font_pool=fonts)
        mt = torch.from_numpy(mimg.astype(np.float32) / 255.0)[None, None]
        outs = model(scene_b, mt.to(device))
        pb, ps = decode_outputs(outs, m["strides"], m["reg_max"])
        mm = float(ps.max()) if len(ps) else 0.0
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
        "n_img": n_img,
        "pos_top1_mean": float(pos_top.mean()),
        "pos_hit_iou05": n_hit / max(n_img, 1),
        "mut_max_mean": float(mut_max.mean()),
        "mut_max_p90": float(np.percentile(mut_max, 90)),
        "mutation_fp": mut_fp / max(len(mut_max), 1),
        "separation_auroc": auroc(pos_top, mut_max),
        "fp_examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = Path(args.weights).parent / "v3probe_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")


if __name__ == "__main__":
    main()
