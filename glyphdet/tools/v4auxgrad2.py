# -*- coding: utf-8 -*-
"""分段定位 aux 梯度消失：逐级 backward 看哪一级把梯度压到 1e-8。"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"

# stage 0: 裸余弦（无 normalize/无中心化/无 interpolate）
a = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
b = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
v = (a * b).sum((1, 2)) / 8
F.relu(0.85 - v).sum().backward()
print("stage0 裸乘:", a.grad.abs().sum().item())

# stage 1: + normalize
a = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
b = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
an = F.normalize(a, dim=1)
bn = F.normalize(b, dim=1)
v = (an * bn).sum((1, 2)) / 8
F.relu(0.85 - v).sum().backward()
print("stage1 normalize:", a.grad.abs().sum().item())

# stage 2: + 中心化
a = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
b = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
an = F.normalize(a - a.mean(1, keepdim=True), dim=1)
bn = F.normalize(b - b.mean(1, keepdim=True), dim=1)
v = (an * bn).sum((1, 2)) / 8
F.relu(0.85 - v).sum().backward()
print("stage2 中心化+normalize:", a.grad.abs().sum().item())

# stage 3: + interpolate（模板 24 -> 16 列）
a = torch.randn(2, 32, 8, 16, device=dev, requires_grad=True)
t = torch.randn(2, 32, 8, 24, device=dev, requires_grad=True)
an = F.normalize(a - a.mean(1, keepdim=True), dim=1)
tc = t - t.mean(1, keepdim=True)
Tr = F.normalize(F.interpolate(tc, size=(8, 16), mode="bilinear", align_corners=False), dim=1)
v = (an * Tr).sum((1, 2)) / 8
F.relu(0.85 - v).sum().backward()
print("stage3 +interpolate: a", a.grad.abs().sum().item(), " t", t.grad.abs().sum().item())

# stage 4: + gather 从宽条带取窗（完整 aux 路径）
s = torch.randn(3, 32, 8, 128, device=dev, requires_grad=True)
t = torch.randn(2, 32, 8, 24, device=dev, requires_grad=True)
S = F.normalize(s - s.mean(1, keepdim=True), dim=1)
tc = t - t.mean(1, keepdim=True)
Tr = F.normalize(F.interpolate(tc, size=(8, 16), mode="bilinear", align_corners=False), dim=1)
cols = torch.arange(16, device=dev).view(1, 16) + 56
win = torch.gather(S[[0, 1]], 3, cols.view(1, 1, 1, 16).expand(2, 32, 8, 16))
v = (win * Tr).sum((1, 2)) / 8
F.relu(0.85 - v).sum().backward()
print("stage4 +gather: s", s.grad.abs().sum().item(), " t", t.grad.abs().sum().item())
