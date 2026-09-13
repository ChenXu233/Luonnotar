"""rbox_iou 约定单测：cv2 RotatedRect 角度方向 vs synth θ（y 向下顺时针正）。"""
import math
import numpy as np
from glyphdet.core.eval import rbox_iou

# 1) 自身 IoU = 1
a = np.array([[100.0, 100.0, 60.0, 20.0, 0.0]])
print("self:", rbox_iou(a, a)[0, 0], "(期望 1.0)")

# 2) θ=90° 的 (60x20) 应等于 θ=0 的 (20x60)
b1 = np.array([[100.0, 100.0, 60.0, 20.0, math.pi / 2]])
b2 = np.array([[100.0, 100.0, 20.0, 60.0, 0.0]])
print("rot90 swap:", rbox_iou(b1, b2)[0, 0], "(期望 ~1.0)")

# 3) 平移不交 = 0
c = np.array([[300.0, 300.0, 60.0, 20.0, 0.7]])
print("disjoint:", rbox_iou(a, c)[0, 0], "(期望 0.0)")

# 4) θ=45° 自身 = 1；与 θ=0 同中心框 IoU 应 < 1 且 > 0
d = np.array([[100.0, 100.0, 60.0, 20.0, math.pi / 4]])
print("rot45 self:", rbox_iou(d, d)[0, 0], "(期望 1.0)")
print("rot45 vs axis:", round(float(rbox_iou(a, d)[0, 0]), 3), "(期望 0~1 之间)")

# 5) 方向敏感性：θ 与 θ+π 是同一个矩形（阅读方向相反，矩形相同）
e1 = np.array([[100.0, 100.0, 60.0, 20.0, 0.3]])
e2 = np.array([[100.0, 100.0, 60.0, 20.0, 0.3 + math.pi]])
print("theta vs theta+pi:", rbox_iou(e1, e2)[0, 0], "(期望 ~1.0)")

# 6) 几何真值：横向半叠（x 方向错开 30px，框 60x20 轴对齐）IoU = 30*20/(60*20*2-30*20)=1/3
f1 = np.array([[100.0, 100.0, 60.0, 20.0, 0.0]])
f2 = np.array([[130.0, 100.0, 60.0, 20.0, 0.0]])
print("half overlap:", round(float(rbox_iou(f1, f2)[0, 0]), 3), "(期望 0.333)")
