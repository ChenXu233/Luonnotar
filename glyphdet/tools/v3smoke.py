"""v3 图连通冒烟：随机输入前向/反向 + 全白 mask 边界 + 参数量。

用法: pdm run python tools/v3smoke.py
"""

import torch

from glyphdet.core.model import build_model, count_params

cfg = {
    "model": {
        "arch": "v3", "in_size": 416, "mask_size": [64, 384],
        "widths": [24, 48, 96, 160], "neck_ch": 96, "mask_dim": 128,
        "reg_max": 24, "strides": [4, 8, 16], "assign": "height",
    }
}
net = build_model(cfg).eval()
print(f"参数量: {count_params(net)/1e6:.2f}M")

scene = torch.rand(2, 3, 416, 416)
mask = torch.ones(2, 1, 64, 384)
mask[:, :, :, 64:96] = 0.2  # 槽 2 有墨迹
mask[:, :, :, 96:128] = 0.1  # 槽 3 有墨迹
with torch.no_grad():
    outs = net(scene, mask)
for s, o in zip([4, 8, 16], outs):
    print(f"stride {s}: {tuple(o.shape)}")
    assert o.shape[1] == 1 + 4 * 24, o.shape

net.train()
loss = sum(o.sum() for o in net(scene, mask))
loss.backward()
print("backward OK")

net.eval()
with torch.no_grad():
    outs = net(scene, torch.ones(2, 1, 64, 384))
print("all-white mask OK, s4 score range:",
      float(outs[0][:, 0].min()), float(outs[0][:, 0].max()))
