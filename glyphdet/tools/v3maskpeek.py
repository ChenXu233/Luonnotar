"""v3 槽位 mask 渲染自检：渲染几组码到 runs/_maskpeek/ 供目检 + 打印槽位几何断言。"""

from pathlib import Path

import numpy as np
import cv2

from glyphdet.core.synth import render_mask_slots

OUT = Path("runs/_maskpeek")
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
FONT = "C:/Windows/Fonts/msyh.ttc"
POOL = [FONT, "C:/Windows/Fonts/arial.ttf"]

for text in ["24-6-1234", "123-45-6789", "12345678", "8049"]:
    m = render_mask_slots(text, FONT, 64, 384, rng=rng, font_pool=POOL)
    cv2.imencode(".png", m)[1].tofile(str(OUT / f"{text.replace('-', '_')}.png"))
    n = len(text)
    ne = n + (n & 1)
    start = (12 - ne) // 2
    # 断言：每个字的槽内墨迹质心都在该槽中心 ±6px 内
    for j, c in enumerate(text):
        x0, x1 = (start + j) * 32, (start + j + 1) * 32
        slot = m[:, x0:x1]
        ink = slot < 128
        assert ink.any(), f"{text}[{j}]={c} 槽 {start + j} 无墨迹"
        cx = float(np.where(ink)[1].mean()) + x0
        want = (start + j) * 32 + 16
        assert abs(cx - want) <= 6, f"{text}[{j}]={c} 质心 {cx:.1f} 期望 {want}"
    print(f"{text}: n={n} ne={ne} start={start} 槽位断言全过, "
          f"墨迹槽数={sum(1 for k in range(12) if (m[:, k*32:(k+1)*32] < 128).any())}")
print("输出目录:", OUT)
