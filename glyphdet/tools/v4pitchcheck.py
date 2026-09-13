# -*- coding: utf-8 -*-
"""v4 列距失配量化验证：条带列/字 vs 模板列/字 的比值分布 + 跨字形特征余弦矩阵。

用法: pdm run python tools/v4pitchcheck.py [--weights runs/mvp_v4_mini20/last.pt]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from glyphdet.core.synth import render_mask_rgba  # noqa: E402


def pitch_ratio_check():
    """对每个训练样本的 target 框：条带列/字 = 8w/(h·len)；模板列/字 = Wt/len。比值应≈1。"""
    font_path = "C:/Windows/Fonts/msyh.ttc"
    ratios, scene_p, mask_p = [], [], []
    labels = (ROOT / "datasets" / "mvp6mini" / "train" / "labels.jsonl").read_text(encoding="utf-8")
    for line in labels.splitlines()[:400]:
        meta = json.loads(line)
        txt = meta["query"]
        n = len(txt)
        if n == 0:
            continue
        for box, is_t in zip(meta["boxes"], meta["is_target"]):
            if not is_t:
                continue
            cx, cy, w, h, th = box
            sp = 8.0 * w / (h * n)  # 条带列/字
            m = render_mask_rgba(txt, font_path, 64, 512, rng=None, jitter=False)
            mp_ = m.shape[1] / 8.0 / n  # 模板列/字
            scene_p.append(sp)
            mask_p.append(mp_)
            ratios.append(sp / mp_)
    ratios = np.array(ratios)
    print(f"n={len(ratios)}")
    print(f"条带列/字: mean={np.mean(scene_p):.2f} p5={np.percentile(scene_p,5):.2f} p95={np.percentile(scene_p,95):.2f}")
    print(f"模板列/字: mean={np.mean(mask_p):.2f} p5={np.percentile(mask_p,5):.2f} p95={np.percentile(mask_p,95):.2f}")
    print(f"比值(条带/模板): mean={ratios.mean():.3f} std={ratios.std():.3f} "
          f"p5={np.percentile(ratios,5):.3f} p25={np.percentile(ratios,25):.3f} "
          f"p50={np.percentile(ratios,50):.3f} p75={np.percentile(ratios,75):.3f} p95={np.percentile(ratios,95):.3f}")
    print(f"|比值-1|>10% 占比: {(np.abs(ratios-1)>0.10).mean()*100:.1f}%  "
          f">20%: {(np.abs(ratios-1)>0.20).mean()*100:.1f}%")


def glyph_cosine_check(weights):
    """训练后 mask_enc 的跨字形列余弦矩阵：若异字余弦≈0.8 则特征坍塌。"""
    import torch
    from glyphdet.core.model import build_model
    import yaml
    cfg = yaml.safe_load((ROOT / "experiments" / "mvp7_v4" / "config_mini20.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg)
    sd = torch.load(weights, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd, strict=False)
    model.eval()
    chars = list("0123456789-")
    font_path = "C:/Windows/Fonts/msyh.ttc"
    masks = [render_mask_rgba(c, font_path, 64, 512, rng=None, jitter=False) for c in chars]
    Wmax = max(m.shape[1] for m in masks)
    Wmax = (Wmax + 7) // 8 * 8
    x = np.zeros((len(chars), 1, 64, Wmax), np.float32)
    for i, m in enumerate(masks):
        x[i, 0, :, : m.shape[1]] = m
    with torch.no_grad():
        feat = model.mask_enc(torch.from_numpy(x))  # (N,Ct,8,Wt)
        feat = feat.flatten(2).mean(2)  # (N,Ct) 全列平均（单字无对齐问题）
        featn = torch.nn.functional.normalize(feat, dim=1)
        cos = featn @ featn.T
    cos = cos.numpy()
    off = cos[~np.eye(len(chars), dtype=bool)]
    print(f"异字平均特征余弦: mean={off.mean():.3f} min={off.min():.3f} max={off.max():.3f}")
    print("余弦矩阵(0-9,-):")
    print(np.array2string(cos, precision=2, suppress_small=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "mvp_v4_mini20" / "last.pt"))
    args = ap.parse_args()
    print("=== 列距失配 ===")
    pitch_ratio_check()
    print("\n=== 跨字形特征余弦 ===")
    try:
        glyph_cosine_check(args.weights)
    except Exception as e:
        print(f"跳过（{e}）")
