# -*- coding: utf-8 -*-
"""aux_column_loss 梯度流最小复现：v≈0 时 loss=0.85 且不动 -> 查 autograd 断点。"""
import torch

from glyphdet.core.model import MatcherV4

torch.manual_seed(0)
mt = MatcherV4(neck_ch=96, tpl_ch=32).cuda()
# 伪造：2 模板（Wt=24）、3 条带、2 正例对
tpl = torch.randn(2, 32, 8, 24, device="cuda", requires_grad=True)
strips = torch.randn(3, 32, 8, 128, device="cuda", requires_grad=True)
ink = torch.zeros(2, 24, device="cuda")
ink[0, :20] = 1.0
ink[1, :18] = 1.0
boxes_used = torch.tensor([[100., 100., 80., 40., 0.1],   # 8w/h=16
                           [200., 100., 120., 30., 0.0],  # 8w/h=32
                           [50., 50., 60., 20., 0.0]], device="cuda")
mi = torch.tensor([0, 1], device="cuda")
si = torch.tensor([0, 1], device="cuda")
loss = mt.aux_column_loss(strips, tpl, ink, (mi, si), boxes_used)
print("loss =", loss.item())
loss.backward()
print("tpl grad:", tpl.grad.abs().sum().item() if tpl.grad is not None else None)
print("strips grad:", strips.grad.abs().sum().item() if strips.grad is not None else None)
for n, p in mt.named_parameters():
    print(f"  param {n}: grad={p.grad.abs().sum().item() if p.grad is not None else None:.6f}")
# 再查 v 的实际值
mt2 = MatcherV4(neck_ch=96, tpl_ch=32).cuda()
with torch.no_grad():
    S = torch.nn.functional.normalize(strips - strips.mean(1, keepdim=True), dim=1)
    Tc = tpl - tpl.mean(1, keepdim=True)
    Tr = torch.nn.functional.interpolate(Tc[mi], size=(8, 16), mode="bilinear", align_corners=False)
    Tr = torch.nn.functional.normalize(Tr, dim=1)
    cols = torch.arange(16, device="cuda").view(1, 16) + (128 - 16) // 2
    win = torch.gather(S[si], 3, cols.view(1, 1, 1, 16).expand(2, 32, 8, 16))
    v = (win * Tr).sum((1, 2)) / 8
    print("v mean/min/max:", v.mean().item(), v.min().item(), v.max().item())
