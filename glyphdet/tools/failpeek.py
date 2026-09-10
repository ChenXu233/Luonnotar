"""失败案例检视：找出 val 中漏检的样本，并排显示 mask（上）+ 场景带框（下）。

  pdm run python tools/failpeek.py --config ... --weights ... --n 12 --out /tmp/fail.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.decode import decode_outputs
from glyphdet.core.eval import iou_matrix
from glyphdet.core.model import GlyphDet, build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="/tmp/failpeek.png")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, ec = cfg["model"], cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    val = GlyphDataset(Path(cfg["data"]["out_dir"]) / "val")
    fails = []
    for idx in range(len(val)):
        scene, mask, boxes, is_target = val[idx]  # type: ignore[misc]
        with torch.no_grad():
            outs = model(scene[None].to(device), mask[None].to(device))
        pb, ps = decode_outputs(outs, m["strides"], m["reg_max"],
                                score_thr=0.3, nms_iou=ec["nms_iou"])
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[tgt > 0.5]
        gt_t = gt_t[gt_t.sum(1) > 0]
        iou_t = iou_matrix(pb, gt_t) if len(pb) else np.zeros((0, len(gt_t)))
        missed = [j for j in range(len(gt_t))
                  if not len(pb) or not (iou_t[:, j] >= 0.3).any()]
        if missed:
            fails.append((idx, missed, pb, ps))
        if len(fails) >= args.n:
            break

    tiles = []
    for idx, missed, pb, ps in fails:
        row = val.rows[idx]
        scene_bgr = cv2.imdecode(
            np.fromfile(str(val.root / row["scene"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if scene_bgr is None:
            print(f"[warn] skip idx={idx}: 场景读取失败 {val.root / row['scene']}")
            continue
        mask_g = cv2.imdecode(
            np.fromfile(str(val.root / row["mask"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask_g is None:
            print(f"[warn] skip idx={idx}: mask 读取失败 {val.root / row['mask']}")
            continue
        # mask 画布放大到场景同宽，白底黑字，写 query
        mask3 = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
        mh, mw = mask3.shape[:2]
        scale = 416 / mw
        mask3 = cv2.resize(mask3, (416, int(mh * scale)), interpolation=cv2.INTER_NEAREST)
        cv2.putText(mask3, f"Q:{row['query']}  (miss {len(missed)})", (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for j, b in enumerate(val.rows[idx]["boxes"]):
            t = val.rows[idx]["is_target"][j]
            col = (0, 200, 0) if t else (0, 0, 220)
            if t and j in missed:
                col = (0, 255, 255)  # 漏检的 target 用黄色
            cv2.rectangle(scene_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        for b, s in zip(pb, ps):
            cv2.rectangle(scene_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 1)
            cv2.putText(scene_bgr, f"{s:.2f}", (int(b[0]), max(int(b[1]) - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 160, 0), 1)
        tiles.append(np.vstack([mask3, scene_bgr]))
    # 简单两列拼图
    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        if len(pair) == 1:
            pair = pair + [np.zeros_like(pair[0])]
        rows.append(np.hstack(pair))
    grid = np.vstack(rows)
    cv2.imencode(".png", grid)[1].tofile(args.out)
    print(f"漏检样本 {len(fails)}（val 前 {len(val)} 张内）-> {args.out}")


if __name__ == "__main__":
    main()
