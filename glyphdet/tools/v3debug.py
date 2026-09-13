"""v3 NaN 定位：前向各节点幅值统计 + fp32/AMP 对照 50 步，找首个 NaN 与量级失控点。

用法: pdm run python tools/v3debug.py
"""

import torch

from glyphdet.core.dataset import GlyphDataset
from glyphdet.core.model import build_model
from glyphdet.core.train import compute_loss
import yaml

cfg = yaml.safe_load(open("experiments/mvp6_v3/config_mini.yaml", encoding="utf-8"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model(cfg).to(device)
model.train()

ds = GlyphDataset("datasets/mvp5mini/train", cfg)
scene, mask, targets = ds[0], ds[1], None
loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
scene, mask, targets = next(iter(loader))
scene, mask = scene.to(device), mask.to(device)
targets = [t.to(device) for t in targets]

# ---- 1) 前向幅值统计（fp32）
with torch.no_grad():
    w, tpl, valid = model.mask_enc(mask)
    print(f"tpl 幅值: min {tpl.min():.3f} max {tpl.max():.3f} absmean {tpl.abs().mean():.3f}")
    print(f"w 幅值: absmean {w.abs().mean():.3f}  valid 每样本: {valid.sum(1).tolist()}")
    feats = model.neck(*model.backbone(scene), w)
    outs = model.head(feats, w, tpl, valid)
    head = model.head
    for i, f in enumerate(feats):
        e = torch.nn.functional.normalize(head.proj(f), dim=1)
        wn = torch.nn.functional.normalize(w, dim=1).view(w.size(0), head.wdim, 1, 1)
        cos_map = (e * wn).sum(1, keepdim=True) * head.alpha + head.beta
        s_feat = head.corr_proj(f)
        print(f"[level {i}] cos_map [{cos_map.min():.1f},{cos_map.max():.1f}] "
              f"s_feat absmean {s_feat.abs().mean():.3f}")
        from glyphdet.core.model import xcorr_slots, V3_SCALES
        B, K = tpl.shape[0], tpl.shape[1]
        for sh in V3_SCALES[i]:
            sw = sh // 2
            t = torch.nn.functional.interpolate(
                tpl.flatten(0, 1), size=(sh, sw), mode="bilinear",
                align_corners=False).view(B, K, -1, sh, sw)
            r = xcorr_slots(s_feat, t)
            print(f"  sh={sh}: xcorr [{r.min():.1f},{r.max():.1f}] absmean {r.abs().mean():.1f}")
            rp = head.pools[str(sh)](r)
            agg = head._align_min(rp, valid, sh)
            print(f"        agg [{agg.min():.1f},{agg.max():.1f}]")
    for s, o in zip([4, 8, 16], outs):
        print(f"stride {s}: score [{o[:, 0].min():.1f},{o[:, 0].max():.1f}] "
              f"reg absmean {o[:, 1:].abs().mean():.3f}")

# ---- 2) fp32 50 步
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
print("\n---- fp32 50 步 ----")
for step in range(50):
    loss, parts = compute_loss(model, scene, mask, targets, cfg)
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    opt.step()
    if step % 5 == 0 or not torch.isfinite(loss):
        print(f"step {step:3d} loss {loss.item():.4f} grad_norm {gn:.1f} match {parts['match']:.1f}")
    if not torch.isfinite(loss):
        print("!!! fp32 NaN at step", step)
        break
else:
    print("fp32 50 步无 NaN")
