"""计时定位：v4 单步 _batch_loss 各段耗时（smoke 卡死排查）。"""
import time

import torch
import yaml
from torch.utils.data import DataLoader

from glyphdet.core.dataset import GlyphDataset, collate_v4
from glyphdet.core.model import build_model
from glyphdet.core.train import compute_loss_v4

cfg = yaml.safe_load(open(r"E:\git\Luonnotar\glyphdet\experiments\mvp7_v4\config_mini.yaml",
                          encoding="utf-8"))
device = "cuda"
model = build_model(cfg).to(device).train()
ds = GlyphDataset(r"E:\git\Luonnotar\glyphdet\datasets\mvp6mini\train", cfg)
loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_v4)

t0 = time.perf_counter()
batch = next(iter(loader))
print(f"取 batch: {time.perf_counter()-t0:.2f}s")
scene, mq, mm, targets, boxes, kinds = [b.to(device) if torch.is_tensor(b) else [t.to(device) for t in b] for b in batch]
print(f"scene {tuple(scene.shape)} mq {tuple(mq.shape)} mm {tuple(mm.shape)}")

with torch.amp.autocast("cuda"):
    for it in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = model.extract(scene)
        outs = model.prop_maps(feats)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        # 只跑 matcher 段（逐样本循环）
        f2 = feats[0]
        for b in range(scene.shape[0]):
            kd, bb = kinds[b], boxes[b]
            sel = (bb[:, 2] > 1) & (bb[:, 3] > 1) & (kd > -1.5)
            gt = bb[sel]
            if len(gt) == 0:
                continue
            mb = [mq[b], mm[b * 2], mm[b * 2 + 1]]
            Wb = max(x.shape[-1] for x in mb)
            masks_b = torch.stack([torch.nn.functional.pad(x, (0, Wb - x.shape[-1])) for x in mb], 0)
            tpl = model.mask_enc(masks_b)
            ink = model.matcher.ink_cols(masks_b)
            strips = model.matcher.strips(f2[b:b + 1], gt, jitter=True)
            logits = model.matcher.score(strips, tpl, ink)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss, parts = compute_loss_v4(model, scene, mq, mm, targets, boxes, kinds, cfg)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        print(f"it{it}: extract+prop {t1-t0:.3f}s matcher循环 {t2-t1:.3f}s "
              f"整loss(含提案) {t3-t2:.3f}s backward {t4-t3:.3f}s 合计 {t4-t0:.3f}s")
        model.zero_grad()
