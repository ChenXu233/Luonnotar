"""v4 模型图连通自检：随机输入前向 + matcher + 反向传播 + 参数量。"""
import torch
import yaml

from glyphdet.core.model import build_model, count_params, pair_grid

cfg = yaml.safe_load(open(r"E:\git\Luonnotar\glyphdet\experiments\mvp7_v4\config.yaml",
                          encoding="utf-8"))
model = build_model(cfg)
print(f"arch={cfg['model']['arch']} 参数量 {count_params(model)/1e6:.2f}M")

scene = torch.rand(2, 3, 416, 416)
mask = torch.rand(2, 1, 64, 232)  # 8 的倍数
feats = model.extract(scene)
for f, s in zip(feats, cfg["model"]["strides"]):
    print(f"feat stride{s}: {tuple(f.shape)} (期望 {416//s}x{416//s})")
outs = model.prop_maps(feats)
for o in outs:
    print(f"prop out: {tuple(o.shape)} (期望 C={1+4*cfg['model']['reg_max']+2})")

tpl = model.mask_enc(mask)
ink = model.matcher.ink_cols(mask)
print(f"tpl {tuple(tpl.shape)} ink {tuple(ink.shape)}")
boxes = torch.tensor([
    [100.0, 120.0, 160.0, 32.0, 0.3],
    [300.0, 200.0, 90.0, 24.0, 1.57],
    [200.0, 350.0, 220.0, 40.0, 3.14],
])
strips = model.matcher.strips(feats[0][0:1], boxes, jitter=True)
print(f"strips {tuple(strips.shape)} (期望 (3,32,8,128))")
logits = model.matcher.score(strips, tpl, ink, pair_grid(2, 3, "cpu"))
print(f"score {tuple(logits.shape)} (期望 (6,))")
loss = sum(o.abs().mean() for o in outs) + logits.abs().mean() + tpl.abs().mean()
loss.backward()
gn = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
print(f"backward OK, 梯度总量 {gn:.3f}")
# 关键断言：无梯度死路
dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
print(f"无梯度参数: {len(dead)} {dead[:5]}")
