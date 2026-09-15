# -*- coding: utf-8 -*-
"""v4 部署导出：双 ONNX 图 + meta.json + 三级对拍（含 matcher numpy 参考实现）。

  pdm run python tools/v4export.py --weights runs/mvp_v4/ep25.pt

图1 glyphdet_v4_prop.onnx：scene(1,3,416,416) → p2proj(1,64,104,104)
    + out4/out8/out16(1,99,H,W)（matcher.proj 已内联，BackboneV2 已 reparam）
图2 glyphdet_v4_mask.onnx：mask(1,1,64,W)（W 动态，8 倍数）→ tpl(1,64,8,W/8)
meta.json：strides/reg_max/条带几何/尺度集/训练后 mix+cov_w/默认阈值 0.82（fp≈8% 工作点）

对拍：①prop 图 ORT vs torch <1e-3；②mask 图多宽度 <1e-3；
③端到端：numpy 移植 matcher（部署语义参考实现）跑 ORT 特征 vs torch matcher 跑
torch 特征，同一 val 样本同一 query 的提案分数差 <2e-2（logit 域）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.decode import decode_rprops
from glyphdet.core.model import build_model, pair_grid
from glyphdet.core.synth import render_mask_rgba

SCALES = (0.80, 0.89, 1.00, 1.12, 1.25, 1.40)


class PropGraph(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.neck, self.prop = m.backbone, m.neck, m.prop
        self.proj = m.matcher.proj

    def forward(self, scene):
        p2, p3, p4 = self.neck(*self.backbone(scene))
        o4, o8, o16 = self.prop((p2, p3, p4))
        return self.proj(p2), o4, o8, o16


# ---------------------------------------------------------- numpy 参考实现（C++ 移植语义）
def bilinear_strip(feat, cx, cy, w, h, th, sh=8, sw=128, fs=4):
    """feat (C,Hf,Wf) → (C,sh,sw)：v4 strips 语义的 numpy 版（align_corners=False）。"""
    C, Hf, Wf = feat.shape
    sp = h / sh
    xs = (np.arange(sw) + 0.5 - sw / 2) * sp
    ys = (np.arange(sh) + 0.5 - sh / 2) * sp
    c, s = np.cos(th), np.sin(th)
    ou, ov = np.meshgrid(xs, ys)  # (sh,sw)
    px = cx + c * ou - s * ov
    py = cy + s * ou + c * ov
    xf = px / fs  # 连续特征坐标（像素中心=整数）
    yf = py / fs
    x0 = np.floor(xf).astype(int)
    y0 = np.floor(yf).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    wx = (xf - x0).astype(np.float32)
    wy = (yf - y0).astype(np.float32)
    out = np.zeros((C, sh, sw), np.float32)

    def gather(xx, yy):
        valid = (xx >= 0) & (xx < Wf) & (yy >= 0) & (yy < Hf)
        v = np.zeros((C, sh, sw), np.float32)
        xc = np.clip(xx, 0, Wf - 1)
        yc = np.clip(yy, 0, Hf - 1)
        v[:, :, :] = feat[:, yc, xc] * valid[None]
        return v

    v00, v01 = gather(x0, y0), gather(x1, y0)
    v10, v11 = gather(x0, y1), gather(x1, y1)
    top = v00 * (1 - wx)[None] + v01 * wx[None]
    bot = v10 * (1 - wx)[None] + v11 * wx[None]
    return top * (1 - wy)[None] + bot * wy[None]


def center_norm(feat):
    """(C,...) → 沿 C 减均值并 L2 归一。"""
    f = feat - feat.mean(axis=0, keepdims=True)
    n = np.linalg.norm(f, axis=0, keepdims=True)
    return f / np.maximum(n, 1e-12)


def interp_w(feat, W2):
    """(...,W1) 双线性→(...,W2)，align_corners=False 语义（仅 1D 宽方向）。"""
    W1 = feat.shape[-1]
    if W1 == W2:
        return feat.copy()
    xs = (np.arange(W2) + 0.5) * W1 / W2 - 0.5
    x0 = np.clip(np.floor(xs).astype(int), 0, W1 - 1)
    x1 = np.clip(x0 + 1, 0, W1 - 1)
    w = (xs - np.floor(xs)).astype(np.float32)
    return feat[..., x0] * (1 - w) + feat[..., x1] * w


def np_score(S_raw, T_raw, ink, span, mix, cov_w, sh=8, sw=128):
    """单对 (S_raw=(Ct,sh,sw) 条带, T_raw=(Ct,8,Wt) 模板, ink=(Wt,), span=8w/h 框内容列数)
    → match logit。与 MatcherV4.score 逐行对应。"""
    a, b, c0 = mix
    d1, d2 = cov_w
    S = center_norm(S_raw)
    Tc = T_raw - T_raw.mean(axis=0, keepdims=True)
    Ct, _, Wt = T_raw.shape
    Sp = np.pad(S, ((0, 0), (1, 1), (0, 0)))  # 行 ±1 容差
    best_c, best = -1e30, None
    for r in SCALES:
        Wr = max(8, int(round(Wt * r)))
        if Wr > sw:
            continue
        Tr = interp_w(Tc, Wr)
        Tr = Tr / np.maximum(np.linalg.norm(Tr, axis=0, keepdims=True), 1e-12)
        ikr = (interp_w(ink.reshape(1, 1, Wt).astype(np.float32), Wr).reshape(Wr) > 0.5)
        Tk = Tr * ikr[None, None, :]
        ncols = max(float(ikr.sum()), 1.0)
        # 相关：c[dy,dx] = Σ_c,row,col Sp[dy+row, dx+col]*Tk[row,col]，dy∈{0,1,2}
        D = sw - Wr + 1
        c = np.zeros((3, D), np.float32)
        for dy in range(3):
            # (Ct,sh,D,Wr) 滑窗 × 模板核 → 逐位置点积求和
            swin = np.lib.stride_tricks.sliding_window_view(Sp[:, dy:dy + sh, :], Wr, axis=2)
            c[dy] = np.tensordot(swin, Tk, axes=([0, 1, 3], [0, 1, 2]))
        c = c.max(axis=0) / (ncols * sh)
        if c.max() > best_c:
            best_c, best = float(c.max()), (r, Wr, Tk, ikr, c)
    r, Wr, Tk, ikr, c = best
    align = best_c
    dx = int(c.argmax())
    cols = np.clip(dx + np.arange(Wr), 0, sw - 1)
    win = S[:, :, cols]  # (Ct,sh,Wr)
    v = (win * Tk).sum(axis=(0, 1)) / sh  # (Wr,) 逐列余弦
    vi = v[ikr]
    colmin = np.sort(vi)[:4].mean() if len(vi) >= 4 else vi.mean()
    cov = float(ikr.sum()) / max(span, 8.0)
    logit = a * align + b * colmin + c0 - d1 * max(0.85 - cov, 0.0) - d2 * max(cov - 1.25, 0.0)
    return float(np.clip(logit, -32, 32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=None, help="输出目录（默认 weights 同级）")
    ap.add_argument("--thr", type=float, default=0.82, help="默认匹配阈值（fp≈8% 工作点）")
    args = ap.parse_args()

    ck = torch.load(args.weights, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    mcfg = cfg["model"]
    model = build_model(cfg).eval()
    model.load_state_dict(ck["model"])
    model.reparam()
    prop = PropGraph(model).eval()

    out_dir = Path(args.out or Path(args.weights).parent)
    prop_onnx = out_dir / "glyphdet_v4_prop.onnx"
    mask_onnx = out_dir / "glyphdet_v4_mask.onnx"

    scene = torch.rand(1, 3, mcfg["in_size"], mcfg["in_size"])
    torch.onnx.export(
        prop, scene, str(prop_onnx), input_names=["scene"],
        output_names=["p2proj", "out4", "out8", "out16"], opset_version=18)
    mq = torch.rand(1, 1, 64, 384)
    torch.onnx.export(
        model.mask_enc, mq, str(mask_onnx), input_names=["mask"],
        output_names=["tpl"], opset_version=18,
        dynamic_axes={"mask": {3: "w"}, "tpl": {3: "w8"}})

    sd = ck["model"]
    mix = [float(x) for x in sd["matcher.mix"]]
    cov_w = [float(x) for x in sd["matcher.cov_w"]]
    meta = {
        "arch": "v4", "in_size": mcfg["in_size"],
        "strides": mcfg["strides"], "reg_max": mcfg["reg_max"],
        "prop_ch": 1 + 4 * mcfg["reg_max"] + 2,
        "tpl_ch": mcfg.get("tpl_ch", 64),
        "strip_h": 8, "strip_w": 128, "feat_stride": 4,
        "scales": list(SCALES), "mix": mix, "cov_w": cov_w,
        "prop_threshold": cfg["eval"]["prop_threshold"],
        "nms_iou": cfg["eval"]["nms_iou"], "max_prop": 32,
        "match_threshold": args.thr,
        "mask_height": cfg["data"]["mask_height"],
        "mask_max_width": cfg["data"]["mask_max_width"],
        "coverage_hinge": [0.85, 1.25], "worst_k": 4,
    }
    (out_dir / "glyphdet_v4_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"prop {prop_onnx.stat().st_size/1e6:.1f}MB mask {mask_onnx.stat().st_size/1e6:.1f}MB")

    # 对拍① prop
    sess = ort.InferenceSession(str(prop_onnx), providers=["CPUExecutionProvider"])
    o = sess.run(None, {"scene": scene.numpy()})
    with torch.no_grad():
        ref = prop(scene)
    for name, a_, b_ in zip(["p2proj", "out4", "out8", "out16"], ref, o):
        print(f"  prop {name}: maxdiff={np.abs(a_.numpy()-b_).max():.2e}")
    # 对拍② mask 多宽度
    sess2 = ort.InferenceSession(str(mask_onnx), providers=["CPUExecutionProvider"])
    for W in (128, 376, 768):
        x = torch.rand(1, 1, 64, W)
        with torch.no_grad():
            r_ = model.mask_enc(x)
        o_ = sess2.run(None, {"mask": x.numpy()})[0]
        print(f"  mask W={W}: out{tuple(o_.shape)} maxdiff={np.abs(r_.numpy()-o_).max():.2e}")

    # 对拍③ 端到端 matcher（numpy 参考实现 vs torch）
    ds = GlyphDataset("datasets/mvp6/val")
    rng = np.random.default_rng(7)
    worst = 0.0
    for idx in [0, 5, 11]:
        sc_t, query, boxes, is_target = ds[idx]
        with torch.no_grad():
            feats = model.extract(sc_t[None])
            outs_t = model.prop_maps(feats)
            p2proj_t = model.matcher.proj(feats[0])
        o = sess.run(None, {"scene": sc_t[None].numpy()})
        p2proj_o = torch.from_numpy(o[0])
        outs_o = [torch.from_numpy(x) for x in o[1:]]
        rb, _ = decode_rprops(outs_o, mcfg["strides"], mcfg["reg_max"],
                              cfg["eval"]["prop_threshold"], cfg["eval"]["nms_iou"])
        if len(rb) == 0:
            continue
        # torch 侧：torch 特征 + torch matcher
        img = render_mask_rgba(query, cfg["data"]["mask_font"], 64, 768, rng=rng)
        mqt = torch.from_numpy(img)[None, None]
        w8 = (mqt.shape[-1] + 7) // 8 * 8
        mqt = F.pad(mqt, (0, w8 - mqt.shape[-1]))
        with torch.no_grad():
            tpl_t = model.mask_enc(mqt)
            ink_t = model.matcher.ink_cols(mqt)
            bt = torch.from_numpy(rb).float()
            strips_t = model.matcher.strips(feats[0], bt)
            pairs = pair_grid(1, len(bt), "cpu")
            span = 8.0 * bt[:, 2] / bt[:, 3].clamp(min=1.0)
            lg_t = model.matcher.score(strips_t, tpl_t, ink_t, pairs,
                                       span=span[pairs[1]]).numpy()
        # numpy 侧：ONNX 特征 + 参考实现
        tpl_o = sess2.run(None, {"mask": mqt.numpy()})[0][0]  # (Ct,8,Wt)
        ink_o = (img[::8].reshape(-1) if False else None)
        ink_np = model.matcher.ink_cols(mqt).numpy()[0]
        S_np = p2proj_o[0].numpy()
        lg_o = []
        for (cx, cy, w, h, th) in rb:
            strip = bilinear_strip(S_np, cx, cy, w, h, th)
            lg_o.append(np_score(strip, tpl_o, ink_np, 8.0 * w / max(h, 1.0), mix, cov_w))
        lg_o = np.array(lg_o)
        d = np.abs(lg_t - lg_o).max()
        worst = max(worst, float(d))
        print(f"  e2e sample{idx}: {len(rb)} 提案 maxdiff={d:.2e} "
              f"(torch top={lg_t.max():.2f} vs np top={lg_o.max():.2f})")
    print(f"e2e 最差 maxdiff={worst:.2e}（门槛 2e-2）{'PASS' if worst < 2e-2 else 'FAIL'}")


if __name__ == "__main__":
    main()
