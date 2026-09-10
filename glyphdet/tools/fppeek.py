"""误检案例检视：找出 val 中模型误检（fp）的框，判定它框住的到底是什么。

  pdm run python tools/fppeek.py --config ... --weights ... --n 16 --scan 300

fp 口径与 eval.py 一致：预测框与所有 target GT 的 IoU<0.3。
每个 fp 再与干扰串 GT（is_target=0）比对：IoU≥0.3 记为 distr（干扰串型误检），
否则记为 bg（背景型误检）——这个分类直接决定下一轮数据配方方向。
扫描前 scan 张，按分数降序取 top n 出图：mask（左上）+ fp 区域放大特写（右上）+ 场景（下）。
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

CROP = 208  # 特写面板边长


def make_tile(val, idx, fp_box, fp_score, tag, pb, ps):
    row = val.rows[idx]
    scene_bgr = cv2.imdecode(
        np.fromfile(str(val.root / row["scene"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if scene_bgr is None:
        raise FileNotFoundError(f"场景读取失败: {val.root / row['scene']}")
    mask_g = cv2.imdecode(
        np.fromfile(str(val.root / row["mask"]), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask_g is None:
        raise FileNotFoundError(f"mask 读取失败: {val.root / row['mask']}")
    # mask 面板：白底黑字缩到 208 宽，padding 到 CROP 高
    mask3 = cv2.cvtColor(mask_g, cv2.COLOR_GRAY2BGR)
    mh, mw = mask3.shape[:2]
    mask3 = cv2.resize(mask3, (CROP, int(mh * CROP / mw)), interpolation=cv2.INTER_NEAREST)
    panel_m = np.full((CROP, CROP, 3), 255, np.uint8)
    panel_m[: mask3.shape[0]] = mask3
    cv2.putText(panel_m, f"Q:{row['query']}", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    # fp 区域特写：带 8px 边距裁出，放大到 CROP 见方
    h, w = scene_bgr.shape[:2]
    x1 = max(int(fp_box[0]) - 8, 0)
    y1 = max(int(fp_box[1]) - 8, 0)
    x2 = min(int(fp_box[2]) + 8, w)
    y2 = min(int(fp_box[3]) + 8, h)
    crop = scene_bgr[y1:y2, x1:x2]
    panel_c = cv2.resize(crop, (CROP, CROP), interpolation=cv2.INTER_LINEAR) \
        if crop.size else np.zeros((CROP, CROP, 3), np.uint8)
    tag_col = (0, 0, 255) if tag == "distr" else (255, 0, 255)
    cv2.putText(panel_c, f"{tag} {fp_score:.2f}", (4, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_col, 2)
    # 场景：target 绿 / 干扰串暗红 / 普通预测蓝 / 当前 fp 亮红粗框
    for j, b in enumerate(row["boxes"]):
        t = row["is_target"][j]
        col = (0, 200, 0) if t else (0, 0, 160)
        cv2.rectangle(scene_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2 if t else 1)
    for b, s in zip(pb, ps):
        cv2.rectangle(scene_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 1)
    cv2.rectangle(scene_bgr, (int(fp_box[0]), int(fp_box[1])),
                  (int(fp_box[2]), int(fp_box[3])), tag_col, 2)
    cv2.putText(scene_bgr, f"{tag} {fp_score:.2f}",
                (int(fp_box[0]), max(int(fp_box[1]) - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_col, 2)
    return np.vstack([np.hstack([panel_m, panel_c]), scene_bgr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--scan", type=int, default=300)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent
                                         / "docs" / "fpreview" / "fppeek.png"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, ec = cfg["model"], cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    val = GlyphDataset(Path(cfg["data"]["out_dir"]) / "val")
    fps: list[tuple[float, int, "np.ndarray", str]] = []  # (score, idx, box, tag)
    # 缓存 idx -> 该样本的 (pb, ps)，避免画图时重复 forward/NMS（非确定性会让 fp_box 错位）
    pred_cache: dict[int, tuple["np.ndarray", "np.ndarray"]] = {}
    n_pred = 0
    n_scan = min(len(val), args.scan)
    for idx in range(n_scan):
        scene, mask, boxes, is_target = val[idx]  # type: ignore[misc]
        with torch.no_grad():
            outs = model(scene[None].to(device), mask[None].to(device))
        pb, ps = decode_outputs(outs, m["strides"], m["reg_max"],
                                score_thr=0.3, nms_iou=ec["nms_iou"])
        pred_cache[idx] = (pb, ps)
        if not len(pb):
            continue
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[tgt > 0.5]
        gt_n = boxes_np[tgt <= 0.5]
        gt_t = gt_t[gt_t.sum(1) > 0]
        gt_n = gt_n[gt_n.sum(1) > 0]
        iou_t = iou_matrix(pb, gt_t)
        wrong = (iou_t.max(1) < 0.3) if len(gt_t) else np.ones(len(pb), bool)
        iou_n = iou_matrix(pb, gt_n) if len(gt_n) else np.zeros((len(pb), 0))
        n_pred += len(pb)
        for i in np.where(wrong)[0]:
            tag = "distr" if iou_n.shape[1] and iou_n[i].max() >= 0.3 else "bg"
            fps.append((float(ps[i]), idx, pb[i], tag))
    fps.sort(key=lambda t: -t[0])

    n_d = sum(1 for f in fps if f[3] == "distr")
    print(f"scan {n_scan} 张: 预测框 {n_pred}, fp {len(fps)} "
          f"(distr {n_d} / bg {len(fps) - n_d})")
    if fps:
        sc = np.array([f[0] for f in fps])
        print(f"fp 分数: mean {sc.mean():.3f} p50 {np.median(sc):.3f} max {sc.max():.3f}")

    tiles = [make_tile(val, idx, b, s, tag, *pred_cache[idx])
             for s, idx, b, tag in fps[: args.n]]
    rows = []
    for i in range(0, len(tiles), 2):
        pair = tiles[i:i + 2]
        if len(pair) == 1:
            pair = pair + [np.zeros_like(pair[0])]
        rows.append(np.hstack(pair))
    if rows:
        grid = np.vstack(rows)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", grid)[1].tofile(str(out))
        print(f"top {len(tiles)} fp -> {out}")


if __name__ == "__main__":
    main()
