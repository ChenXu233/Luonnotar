"""matcher 分量解剖：正例 vs 变异在 GT 框上的 align/colmin/logit 原值。
定位"pair BCE 无分离"的根因——是对齐找不到、列余弦没差异、还是混合层问题。

用法: pdm run python tools/v4pairdump.py --weights runs/mvp_v4_mini20/last.pt
"""
import numpy as np
import torch
import torch.nn.functional as F

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.model import build_model
from glyphdet.core.synth import hard_negative, render_mask_rgba

ckpt = torch.load(r"E:\git\Luonnotar\glyphdet\runs\mvp_v4_mini20\last.pt",
                  map_location="cpu", weights_only=False)
cfg = ckpt["cfg"]
d = cfg["data"]
model = build_model(cfg).eval()
model.load_state_dict(ckpt["model"])
device = "cuda"
model = model.to(device)
mt = model.matcher
fonts = d.get("mask_fonts") or [d["mask_font"]]


def render(text, seed):
    rng = np.random.default_rng(seed)
    img = render_mask_rgba(text, d["mask_font"], d["mask_height"], d["mask_max_width"],
                           rng=rng, font_pool=fonts)
    t = torch.from_numpy(img)[None, None]
    w8 = (t.shape[-1] + 7) // 8 * 8
    return F.pad(t, (0, w8 - t.shape[-1])).to(device)


ds = GlyphDataset(r"E:\git\Luonnotar\glyphdet\datasets\mvp6mini\train")
print(f"mix 参数: a={mt.mix[0].item():.3f} b={mt.mix[1].item():.3f} c0={mt.mix[2].item():.3f}")

with torch.no_grad():
    for idx in range(6):
        scene, query, boxes, is_target = ds[idx]
        gt = boxes.numpy()[(is_target.numpy() > 0.5) & (boxes.numpy()[:, 2] > 1)]
        if len(gt) == 0:
            continue
        gt = torch.from_numpy(gt[:1]).to(device)  # 只取第一个 target
        feats = model.extract(scene[None].to(device))
        strips = mt.strips(feats[0], gt)  # (1,Ct,8,128)
        S = F.normalize(strips - strips.mean(1, keepdim=True), dim=1)
        for tag, text, seed in [("正", query, idx), ("变", hard_negative(
                np.random.default_rng(idx + 777), query), idx + 555)]:
            mq = render(text, seed)
            tpl = model.mask_enc(mq)
            ink = mt.ink_cols(mq)
            Tc = tpl - tpl.mean(1, keepdim=True)
            Wt = tpl.shape[3]
            best_c, best = -1e9, None
            for r in mt.scales:
                Wr = max(8, int(round(Wt * r)))
                if Wr > mt.sw:
                    continue  # 模板放大后比条带宽，几何上不可能对齐
                Tr = F.normalize(F.interpolate(Tc, size=(mt.sh, Wr),
                                               mode="bilinear", align_corners=False), dim=1)
                ikr = (F.interpolate(ink.view(1, 1, 1, Wt), size=(1, Wr),
                                     mode="bilinear", align_corners=False)
                       .view(1, Wr) > 0.5).float()
                Tk = Tr * ikr.view(1, 1, 1, Wr)
                Ct = Tk.shape[1]
                Sp = F.pad(S, (0, 0, 1, 1))
                R = F.conv2d(Sp.reshape(1, Ct, mt.sh + 2, mt.sw),
                             Tk.reshape(Ct, 1, mt.sh, Wr), groups=Ct)
                c = R.reshape(Ct, 3, -1).sum(0).max(0).values
                c = c / (ikr.sum().clamp(min=1.0) * mt.sh)
                if c.max() > best_c:
                    best_c, best = c.max(), (r, Wr, Tk, ikr, c)
            r, Wr, Tk, ikr, c = best
            align = torch.logsumexp(c / mt.tau, 0) * mt.tau
            dx = int(c.argmax())
            cols = (torch.arange(Wr, device=device) + dx).clamp(0, mt.sw - 1)
            win = S[0].gather(2, cols.view(1, 1, Wr).expand(Tk.shape[1], mt.sh, Wr))
            v = (win * Tk[0]).sum((0, 1)) / mt.sh
            vi = v[ikr[0] > 0.5]
            colmin = torch.topk(vi, min(4, len(vi)), largest=False).values.mean()  # worst-k
            logit = mt.mix[0] * align + mt.mix[1] * colmin + mt.mix[2]
            print(f"[{tag}] {text!r:16s} r*={r:.2f} Wt={Wt:3d}->{Wr:3d} "
                  f"墨列={int(ikr.sum()):2d} dx*={dx:3d} "
                  f"align={align.item():+.3f} colmin={colmin.item():+.3f} "
                  f"墨列余弦 mean={vi.mean().item():+.3f} min={vi.min().item():+.3f} "
                  f"logit={logit.item():+.2f} p={torch.sigmoid(logit).item():.3f}")
