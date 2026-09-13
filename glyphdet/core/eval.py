"""评估：检出率 / 误检率 / 一位之差可分性（go/no-go ①②）。

  pdm run python core/eval.py --config experiments/mvp_baseline/config.yaml --weights runs/mvp_baseline/last.pt

指标口径：
- 检出率：GT target 框找到 IoU≥0.5 且分数超阈的预测框的比例
- 误检率：预测框不命中任何 target GT（IoU<0.3）的比例
- 一位之差可分性：target 框区域最高分 vs hard-neg 框区域最高分 的 AUROC
- PC 侧单帧耗时：CPU / CUDA 各测一遍（端侧参考，真机以部署后实测为准）
产物：runs/<name>/eval_report.json + runs/<name>/vis/*.png（前 12 张可视化）
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from glyphdet.core.dataset import GlyphDataset, imread_chw
from glyphdet.core.decode import decode_outputs, has_cen
from glyphdet.core.model import GlyphDet, build_model, pair_grid


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) xyxy → (N,M) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None] - inter + 1e-7)


def auroc(pos, neg):
    """秩统计 AUROC。pos/neg: 一维分数数组。"""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    equal = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * equal) / (len(pos) * len(neg)))


def box_max_score(score_map, box):
    """在框区域内取 decode 前的最高最终分（sigmoid 后）。score_map: (H,W) numpy。"""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, score_map.shape[1]), min(y2, score_map.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(score_map[y1:y2, x1:x2].max())


@torch.no_grad()
def quick_eval(model, val_ds, cfg, device, n=300):
    """训练中期轻量评估：val 等距子集上的 auroc/recall@0.5/top1/fp。
    与 main() 同口径（同一 decode 与分数图语义），供 train 周期性调用快速判定。"""
    m, ec = cfg["model"], cfg["eval"]
    idxs = np.linspace(0, len(val_ds) - 1, min(n, len(val_ds))).astype(int)
    was_training = model.training
    model.eval()
    n_hit = n_target = n_pred = n_wrong = n_top1 = n_img_t = 0
    tgt_scores, neg_scores = [], []
    for idx in idxs:
        scene, mask, boxes, is_target = val_ds[int(idx)]
        outs = model(scene[None].to(device), mask[None].to(device))
        pb, ps = decode_outputs(
            outs, m["strides"], m["reg_max"],
            score_thr=ec["score_threshold"], nms_iou=ec["nms_iou"],
        )
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[(tgt > 0.5) & (boxes_np.sum(1) > 0)]
        gt_n = boxes_np[(tgt <= 0.5) & (boxes_np.sum(1) > 0)]
        n_target += len(gt_t)
        n_img_t += int(len(gt_t) > 0)
        if len(pb):
            iou_t = iou_matrix(pb, gt_t)
            for j in range(len(gt_t)):
                if (iou_t[:, j] >= ec["iou_threshold"]).any():
                    n_hit += 1
            wrong = (iou_t.max(1) < 0.3) if len(gt_t) else np.ones(len(pb), bool)
            n_pred += len(pb)
            n_wrong += int(wrong.sum())
            top = int(np.argmax(ps))
            if len(gt_t) and iou_t[top].max() >= ec["iou_threshold"]:
                n_top1 += 1
        maps = []
        for out in outs:
            full = torch.sigmoid(out[0, 0])
            if has_cen(out.shape[1], m["reg_max"]):
                full = full * torch.sigmoid(out[0, -1])
            maps.append(torch.nn.functional.interpolate(
                full[None, None], size=scene.shape[-2:], mode="bilinear")[0, 0])
        s_map = torch.stack(maps).max(0).values.cpu().numpy()
        for b in gt_t:
            tgt_scores.append(box_max_score(s_map, b))
        for b in gt_n:
            neg_scores.append(box_max_score(s_map, b))
    if was_training:
        model.train()
    return {
        "auroc": auroc(tgt_scores, neg_scores),
        "recall": n_hit / max(n_target, 1),
        "top1": n_top1 / max(n_img_t, 1),
        "fp": n_wrong / max(n_pred, 1) if n_pred else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n-vis", type=int, default=12)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    m, ec = cfg["model"], cfg["eval"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    val = GlyphDataset(Path(cfg["data"]["out_dir"]) / "val")
    n_hit = n_target = 0
    n_hit_iou03 = 0
    n_pred = n_pred_wrong = 0
    n_top1_hit = 0  # 每张图最高分预测框命中 target(IoU≥0.5) 的图数（App 只高亮最优匹配）
    tgt_scores, neg_scores = [], []  # target 框内最高分 vs 干扰/hard-neg 框内最高分
    thr_sweep = {t: [0, 0, 0] for t in (0.3, 0.4, 0.5, 0.6)}  # thr -> [hit, pred, wrong]
    vis_dir = Path(args.weights).parent / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    t_infer = 0.0
    for idx in range(len(val)):
        scene, mask, boxes, is_target = val[idx]
        with torch.no_grad():
            t0 = time.perf_counter()
            outs = model(scene[None].to(device), mask[None].to(device))
            t_infer += time.perf_counter() - t0
        pb, ps = decode_outputs(
            outs, m["strides"], m["reg_max"],
            score_thr=ec["score_threshold"], nms_iou=ec["nms_iou"],
        )
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[tgt > 0.5]
        gt_n = boxes_np[tgt <= 0.5]
        gt_t = gt_t[gt_t.sum(1) > 0]
        gt_n = gt_n[gt_n.sum(1) > 0]
        # 检出 / 误检（主阈值）
        n_target += len(gt_t)
        if len(pb):
            iou_t = iou_matrix(pb, gt_t)
            for j in range(len(gt_t)):
                if (iou_t[:, j] >= ec["iou_threshold"]).any():
                    n_hit += 1
                if (iou_t[:, j] >= 0.3).any():  # 宽 IoU：分离"没检出"与"框不准"
                    n_hit_iou03 += 1
            wrong = (iou_t.max(1) < 0.3) if len(gt_t) else np.ones(len(pb), bool)
            n_pred += len(pb)
            n_pred_wrong += int(wrong.sum())
            # top-1：最高分框是否命中任一 target
            top = int(np.argmax(ps))
            if len(gt_t) and iou_t[top].max() >= ec["iou_threshold"]:
                n_top1_hit += 1
        # 阈值扫描（同一批 outs 用不同阈值重 decode，仅统计）
        for t in thr_sweep:
            pb2, _ = decode_outputs(
                outs, m["strides"], m["reg_max"], score_thr=t, nms_iou=ec["nms_iou"],
            )
            if not len(pb2):
                continue
            iou2 = iou_matrix(pb2, gt_t)
            thr_sweep[t][0] += int(sum((iou2[:, j] >= ec["iou_threshold"]).any() for j in range(len(gt_t))))
            wrong2 = (iou2.max(1) < 0.3) if len(gt_t) else np.ones(len(pb2), bool)
            thr_sweep[t][1] += len(pb2)
            thr_sweep[t][2] += int(wrong2.sum())
        # 一位之差可分性：框内最高分（三级 sigmoid 分数图插值回原图后取逐点 max）
        maps = []
        for out in outs:
            full = torch.sigmoid(out[0, 0])
            if has_cen(out.shape[1], m["reg_max"]):
                full = full * torch.sigmoid(out[0, -1])
            maps.append(
                torch.nn.functional.interpolate(
                    full[None, None], size=scene.shape[-2:], mode="bilinear"
                )[0, 0]
            )
        s_map = torch.stack(maps).max(0).values.cpu().numpy()
        for b in gt_t:
            tgt_scores.append(box_max_score(s_map, b))
        for b in gt_n:
            neg_scores.append(box_max_score(s_map, b))
        # 可视化
        if idx < args.n_vis:
            img = (scene.permute(1, 2, 0).numpy()[:, :, ::-1] * 255).astype(np.uint8).copy()
            for b in gt_t:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 2)
            for b in gt_n:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 220), 1)
            for b, s in zip(pb, ps):
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 160, 0), 2)
                cv2.putText(img, f"{s:.2f}", (int(b[0]), max(int(b[1]) - 3, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1)
            cv2.imencode(".png", img)[1].tofile(str(vis_dir / f"{idx:04d}.png"))

    recall = n_hit / max(n_target, 1)
    fp_rate = n_pred_wrong / max(n_pred, 1)
    report = {
        "weights": args.weights,
        "n_val": len(val),
        "recall@thr%.2f_iou%.2f" % (ec["score_threshold"], ec["iou_threshold"]): round(recall, 4),
        "recall_iou0.3": round(n_hit_iou03 / max(n_target, 1), 4),
        "pred_fp_rate": round(fp_rate, 4),
        "top1_acc": round(n_top1_hit / max(len(val), 1), 4),
        "one_char_auroc": round(auroc(tgt_scores, neg_scores), 4),
        "tgt_score_mean": round(float(np.mean(tgt_scores)), 4) if tgt_scores else None,
        "neg_score_mean": round(float(np.mean(neg_scores)), 4) if neg_scores else None,
        "thr_sweep": {
            str(t): {
                "recall": round(v[0] / max(n_target, 1), 4),
                "fp_rate": round(v[2] / max(v[1], 1), 4),
            }
            for t, v in thr_sweep.items()
        },
        "infer_ms_gpu_avg": round(t_infer / max(len(val), 1) * 1000, 2),
    }
    # CPU 耗时（端侧参考下限）
    model.cpu()
    s, mk, _, _ = val[0]
    with torch.no_grad():
        for _ in range(3):
            model(s[None], mk[None])
        t0 = time.perf_counter()
        for _ in range(10):
            model(s[None], mk[None])
        report["infer_ms_cpu_pc"] = round((time.perf_counter() - t0) / 10 * 1000, 2)

    out = Path(args.weights).parent / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"可视化: {vis_dir}")


if __name__ == "__main__":
    main()


# ============================================================ v4：旋转框评估
def rbox_iou(a, b):
    """a (N,5), b (M,5) (cx,cy,w,h,θrad) → (N,M) 旋转 IoU。
    cv2 RotatedRect 的角度与 synth 同约定（图像坐标 y 向下、顺时针正），单位度。"""
    import math

    def rr(box):
        cx, cy, w, h, th = [float(v) for v in box]
        return ((cx, cy), (max(w, 1e-3), max(h, 1e-3)), math.degrees(th) % 360.0)

    out = np.zeros((len(a), len(b)), np.float32)
    for i in range(len(a)):
        for j in range(len(b)):
            rv, pts = cv2.rotatedRectangleIntersection(rr(a[i]), rr(b[j]))
            if rv == cv2.INTERSECT_NONE:
                continue
            if rv == cv2.INTERSECT_FULL:
                inter = min(a[i][2] * a[i][3], b[j][2] * b[j][3])
            else:
                inter = cv2.contourArea(pts)
            out[i, j] = inter / (a[i][2] * a[i][3] + b[j][2] * b[j][3] - inter + 1e-7)
    return out


@torch.no_grad()
def quick_eval_v4(model, val_ds, cfg, device, n=300):
    """v4 训练中期轻量评估：auroc/recall@0.5(旋转IoU)/top1/fp。
    auroc = query@target框 分 vs [query@非target框 + 变异@target框] 分（pair BCE 的直接口径）。
    eval mask 用 idx 定种子渲染（跨 epoch 可比）。"""
    from glyphdet.core.decode import decode_rprops
    from glyphdet.core.synth import hard_negative, render_mask_rgba

    m, ec, d = cfg["model"], cfg["eval"], cfg["data"]
    fonts = d.get("mask_fonts") or [d["mask_font"]]
    idxs = np.linspace(0, len(val_ds) - 1, min(n, len(val_ds))).astype(int)
    was_training = model.training
    model.eval()

    def render(text, seed):
        rng = np.random.default_rng(seed)
        img = render_mask_rgba(text, d["mask_font"], d["mask_height"],
                               d["mask_max_width"], rng=rng, font_pool=fonts)
        t = torch.from_numpy(img)[None, None]
        w8 = (t.shape[-1] + 7) // 8 * 8
        return torch.nn.functional.pad(t, (0, w8 - t.shape[-1])).to(device)

    n_hit = n_target = n_pred = n_wrong = n_top1 = n_img_t = 0
    tgt_scores, neg_scores = [], []
    for idx in idxs:
        scene, query, boxes, is_target = val_ds[int(idx)]
        feats = model.extract(scene[None].to(device))
        outs = model.prop_maps(feats)
        rb, _ = decode_rprops(outs, m["strides"], m["reg_max"],
                              ec["prop_threshold"], ec["nms_iou"])
        mq = render(query, int(idx))
        tpl = model.mask_enc(mq)
        ink = model.matcher.ink_cols(mq)

        def mscore(boxes_np, tpl_=tpl, ink_=ink):
            if len(boxes_np) == 0:
                return np.zeros(0, np.float32)
            bt = torch.from_numpy(boxes_np).to(device)
            strips = model.matcher.strips(feats[0], bt, jitter=False)
            pairs = pair_grid(tpl_.shape[0], len(bt), device)
            span = 8.0 * bt[:, 2] / bt[:, 3].clamp(min=1.0)
            sig = torch.sigmoid(model.matcher.score(strips, tpl_, ink_, pairs,
                                                    span=span[pairs[1]]))
            return sig.cpu().numpy()

        ps = mscore(rb)
        boxes_np, tgt = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[(tgt > 0.5) & (boxes_np[:, 2] > 1)]
        gt_n = boxes_np[(tgt <= 0.5) & (boxes_np[:, 2] > 1)]
        n_target += len(gt_t)
        n_img_t += int(len(gt_t) > 0)
        fin = ps >= ec["score_threshold"]
        rb_f = rb[fin]
        if len(rb_f):
            iou_t = rbox_iou(rb_f, gt_t) if len(gt_t) else np.zeros((len(rb_f), 0))
            for j in range(len(gt_t)):
                if (iou_t[:, j] >= ec["iou_threshold"]).any():
                    n_hit += 1
            wrong = (iou_t.max(1) < 0.3) if len(gt_t) else np.ones(len(rb_f), bool)
            n_pred += len(rb_f)
            n_wrong += int(wrong.sum())
            top = int(np.argmax(ps[fin]))
            if len(gt_t) and iou_t[top].max() >= ec["iou_threshold"]:
                n_top1 += 1
        # auroc 口径：GT 框上的直接 pair 分
        tgt_scores += list(mscore(gt_t))
        neg_scores += list(mscore(gt_n))
        mut = hard_negative(np.random.default_rng(int(idx) + 99991), query)
        if mut != query and len(gt_t):
            mqm = render(mut, int(idx) + 555)
            neg_scores += list(mscore(gt_t, model.mask_enc(mqm),
                                      model.matcher.ink_cols(mqm)))
    if was_training:
        model.train()
    return {
        "auroc": auroc(tgt_scores, neg_scores),
        "recall": n_hit / max(n_target, 1),
        "top1": n_top1 / max(n_img_t, 1),
        "fp": n_wrong / max(n_pred, 1) if n_pred else 0.0,
    }
