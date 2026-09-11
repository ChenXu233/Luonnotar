"""centerness 死信判定（toy 验证）：

对已训练的 v1 终局模型（带 cen 通道），在同一批 val 图上对比三种 decode 分数图：
  1. with   : sigmoid(score) * sigmoid(cen_pred)   —— 训练时设计的用法
  2. without: sigmoid(score)                       —— 砍掉 cen
  3. oracle : sigmoid(score) * 真·centerness(GT 几何) —— cen 概念的天花板

若 with≈without 且 oracle 也不更好 → cen 概念在这里就是死的，移除合理。
若 oracle 明显更好而 with 没用 → 概念有救但没学会，v3 可考虑重设计。
另报：cen 预测值与 GT centerness 的 Pearson 相关（学没学到定位）。
"""

import sys
import yaml
import numpy as np
import torch
import torch.nn.functional as F

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.model import build_model


def box_max(s_map, b):
    x1, y1, x2, y2 = [int(v) for v in b]
    return float(s_map[max(y1, 0):max(y2, y1 + 1), max(x1, 0):max(x2, x1 + 1)].max())


def auroc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


def oracle_cen_map(h, w, gt_t):
    """(H,W) 真 centerness：多 target 取逐点 max，框外为 1（不干扰 fp 侧）。"""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    m = np.ones((h, w), np.float32)
    for x1, y1, x2, y2 in gt_t:
        l, r = xs - x1, x2 - xs
        t, bb = ys - y1, y2 - ys
        inside = (l > 0) & (r > 0) & (t > 0) & (bb > 0)
        cen = np.sqrt(
            np.clip(np.minimum(l, r) / np.maximum(l, r), 0, 1)
            * np.clip(np.minimum(t, bb) / np.maximum(t, bb), 0, 1)
        )
        m = np.where(inside, np.maximum(m * 0, cen), m)  # 框内用 cen，框外保持 1
    return m


def main(cfg_path, ckpt, n=100):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    net = build_model(cfg)
    sd = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(sd.get("model", sd) if isinstance(sd, dict) else sd)
    net.eval()
    m, strides = cfg["model"], cfg["model"]["strides"]
    ds = GlyphDataset(f"{cfg['data']['out_dir']}/val")  # 裸 boxes，与该 run 官方 eval 同分布
    tw, two, to_, nw, nwo, nor = [], [], [], [], [], []
    cen_pred_all, cen_gt_all = [], []
    top1_change = 0
    used = 0
    for idx in range(min(n, len(ds))):
        scene, mask, boxes, is_target = ds[idx]
        boxes_np, tgt_np = boxes.numpy(), is_target.numpy()
        gt_t = boxes_np[(tgt_np > 0.5) & (boxes_np.sum(1) > 0)]
        gt_n = boxes_np[(tgt_np <= 0.5) & (boxes_np.sum(1) > 0)]
        if not len(gt_t):
            continue
        used += 1
        with torch.no_grad():
            outs = net(scene[None], mask[None])
        maps = {k: [] for k in ("with", "without")}
        for out, s in zip(outs, strides):
            sc = torch.sigmoid(out[0, 0])
            ce = torch.sigmoid(out[0, -1])
            for k, v in (("with", sc * ce), ("without", sc)):
                maps[k].append(F.interpolate(v[None, None], size=scene.shape[-2:],
                                             mode="bilinear")[0, 0].numpy())
            # cen 学习质量：GT 框中心带内的预测 vs 真 cen（在该级网格上）
            for x1, y1, x2, y2 in gt_t:
                H, W = out.shape[-2:]
                cx, cy = (x1 + x2) / 2 / s, (y1 + y2) / 2 / s
                r = max(1, int((x2 - x1) / s * 0.2))
                x1g = max(0, min(W - 1, int(cx - r))); x2g = max(0, min(W, int(cx + r) + 1))
                y1g = max(0, min(H - 1, int(cy - r))); y2g = max(0, min(H, int(cy + r) + 1))
                patch = ce[y1g:y2g, x1g:x2g]
                ys, xs = torch.meshgrid(torch.arange(y1g, y2g), torch.arange(x1g, x2g),
                                        indexing="ij")
                px, py = (xs.float() + 0.5) * s, (ys.float() + 0.5) * s
                l, rr = px - x1, x2 - px
                t, bb = py - y1, y2 - py
                gtcen = torch.sqrt(
                    (torch.minimum(l, rr) / torch.maximum(l, rr)).clamp(0, 1)
                    * (torch.minimum(t, bb) / torch.maximum(t, bb)).clamp(0, 1)
                )
                cen_pred_all.append(patch.flatten().numpy())
                cen_gt_all.append(gtcen.flatten().numpy())
        sw = np.stack(maps["with"]).max(0)
        so = np.stack(maps["without"]).max(0)
        sox = so * oracle_cen_map(*so.shape, gt_t)
        for b in gt_t:
            tw.append(box_max(sw, b)); two.append(box_max(so, b)); to_.append(box_max(sox, b))
        for b in gt_n:
            nw.append(box_max(sw, b)); nwo.append(box_max(so, b)); nor.append(box_max(sox, b))
        # top-1 格点是否变化
        if int(np.unravel_index(sw.argmax(), sw.shape)[0]) != \
           int(np.unravel_index(so.argmax(), so.shape)[0]) or \
           int(np.unravel_index(sw.argmax(), sw.shape)[1]) != \
           int(np.unravel_index(so.argmax(), so.shape)[1]):
            top1_change += 1
    cp = np.concatenate(cen_pred_all); cg = np.concatenate(cen_gt_all)
    print(f"图片 {used} 张 | target 框 {len(tw)} | 干扰框 {len(nw)}")
    print(f"AUROC  with cen   = {auroc(tw, nw):.4f}")
    print(f"AUROC  without   = {auroc(two, nwo):.4f}")
    print(f"AUROC  oracle    = {auroc(to_, nor):.4f}   <- cen 概念天花板")
    print(f"top-1 格点变化率  = {top1_change}/{used}")
    print(f"cen 预测 vs GT cen Pearson r = {np.corrcoef(cp, cg)[0,1]:.3f}")
    print(f"cen 预测分布: mean={cp.mean():.3f} std={cp.std():.3f}  (GT cen std={cg.std():.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "experiments/mvp4_hardneg/config.yaml",
         sys.argv[2] if len(sys.argv) > 2 else "runs/mvp_100k_hardneg/best.pt")
