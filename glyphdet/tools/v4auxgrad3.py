# -*- coding: utf-8 -*-
"""精确复刻 aux_column_loss 内部，打印每步形状。"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cuda"
sh, sw = 8, 128
strips = torch.randn(3, 32, 8, 128, device=dev, requires_grad=True)
tpl = torch.randn(2, 32, 8, 24, device=dev, requires_grad=True)
ink = torch.zeros(2, 24, device=dev)
ink[0, :20] = 1.0
ink[1, :18] = 1.0
boxes_used = torch.tensor([[100., 100., 80., 40., 0.1],
                           [200., 100., 120., 30., 0.0],
                           [50., 50., 60., 20., 0.0]], device=dev)
mi = torch.tensor([0, 1], device=dev)
si = torch.tensor([0, 1], device=dev)

S = F.normalize(strips - strips.mean(1, keepdim=True), dim=1)
Tc = tpl - tpl.mean(1, keepdim=True)
Wt = 24
w = boxes_used[si, 2]
h = boxes_used[si, 3].clamp(min=1.0)
Wr = (8.0 * w / h).round().long().clamp(min=8, max=sw)
print("Wr =", Wr.tolist())
for g in Wr.unique():
    sel = Wr == g
    gw = int(g)
    dx = (sw - gw) // 2
    mg = mi[sel]
    Tr = F.interpolate(Tc[mg], size=(sh, gw), mode="bilinear", align_corners=False)
    Tr = F.normalize(Tr, dim=1)
    ikr = (F.interpolate(ink[mg].view(-1, 1, 1, Wt).float(), size=(1, gw),
                         mode="bilinear", align_corners=False).view(-1, gw) > 0.5).float()
    Ps = int(sel.sum())
    cols = torch.arange(gw, device=dev).view(1, gw) + dx
    print(f"g={gw} Ps={Ps} dx={dx} cols shape {tuple(cols.shape)}")
    idx = cols.view(1, 1, 1, gw).expand(Ps, -1, sh, gw)
    print("gather index shape:", tuple(idx.shape))
    Sin = S[si[sel]]
    print("gather input shape:", tuple(Sin.shape))
    win = torch.gather(Sin, 3, idx)
    print("win shape:", tuple(win.shape), " Tr shape:", tuple(Tr.shape))
    v = (win * Tr).sum((1, 2)) / sh
    print("v shape:", tuple(v.shape), "mean", v.mean().item())
    pull = F.relu(0.85 - v) * ikr
    loss = pull.sum() / ikr.sum()
    loss.backward()
    print("loss", loss.item(),
          " strips grad", strips.grad.abs().sum().item() if strips.grad is not None else None,
          " tpl grad", tpl.grad.abs().sum().item() if tpl.grad is not None else None)
    break
