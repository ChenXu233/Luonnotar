"""生成 C++ decode 对拍基准数据（输出到本目录）。

用法（glyphdet 只读，输出写到 vendor/glyph_det/tests/parity/）：
    cd E:/git/Luonnotar/glyphdet
    PYTHONPATH=E:/git/Luonnotar pdm run python ../vendor/glyph_det/tests/parity/make_parity.py

产物：
- ort_scene.npy / ort_mask.npy        固定 seed 随机输入 (1,3,416,416) / (1,1,64,384)
- ort_out4.npy / ort_out8.npy / ort_out16.npy   ORT 前向三级输出 (1,97,H,W)
- ort_expected.json / ort_expected_boxes.npy / ort_expected_scores.npy
    decode.py 对上述输出的解码结果（可能为 0 框，照样记录）
- crafted_out4/8/16.npy               手工构造的三级输出（植入边界 case）
- crafted_expected.json / *_boxes.npy / *_scores.npy
    decode.py 对构造输出的解码结果（覆盖：阈值边界、DFL 期望、跨尺度 NMS、
    max_det=20 截断）
"""

import json
import os

import numpy as np
import onnxruntime as ort
import torch

from glyphdet.core.decode import decode_outputs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ONNX = os.path.join(REPO, "glyphdet", "runs", "mvp_v2_2", "glyphdet.onnx")

REG_MAX = 24
STRIDES = [4, 8, 16]
SHAPES = [(1, 97, 104, 104), (1, 97, 52, 52), (1, 97, 26, 26)]


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def dump(prefix: str, outs: list[np.ndarray], boxes: np.ndarray, scores: np.ndarray) -> None:
    for name, arr in zip(["out4", "out8", "out16"], outs):
        np.save(os.path.join(HERE, f"{prefix}_{name}.npy"), arr.astype(np.float32))
    np.save(os.path.join(HERE, f"{prefix}_expected_boxes.npy"), boxes.astype(np.float32))
    np.save(os.path.join(HERE, f"{prefix}_expected_scores.npy"), scores.astype(np.float32))
    with open(os.path.join(HERE, f"{prefix}_expected.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "reg_max": REG_MAX,
                "strides": STRIDES,
                "score_thr": 0.5,
                "nms_iou": 0.4,
                "max_det": 20,
                "nbox": int(len(boxes)),
                "boxes_xyxy": np.round(boxes, 6).tolist() if len(boxes) else [],
                "scores": np.round(scores, 6).tolist() if len(scores) else [],
            },
            f,
            indent=1,
        )
    print(f"[{prefix}] nbox={len(boxes)}")


def ref_decode(outs: list[np.ndarray]):
    ts = [torch.from_numpy(o) for o in outs]
    boxes, scores = decode_outputs(ts, STRIDES, REG_MAX)
    return boxes.astype(np.float32), scores.astype(np.float32)


def make_ort_case() -> None:
    rng = np.random.default_rng(20260915)
    scene = rng.random((1, 3, 416, 416), dtype=np.float32)
    mask = rng.random((1, 1, 64, 384), dtype=np.float32)
    np.save(os.path.join(HERE, "ort_scene.npy"), scene)
    np.save(os.path.join(HERE, "ort_mask.npy"), mask)

    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]  # out4, out8, out16
    raw = sess.run(names, {"scene": scene, "mask": mask})
    outs = [raw[names.index(n)] for n in ["out4", "out8", "out16"]]
    boxes, scores = ref_decode(outs)
    dump("ort", outs, boxes, scores)


def plant(head: np.ndarray, y: int, x: int, p: float, ltrb: tuple[float, float, float, float]) -> None:
    """在 (1,97,H,W) 的 head 上植入一个格点：score 概率 p + 确定性的 DFL logits。

    DFL 期望必须精确等于 ltrb：对每个 side，把全部概率质量放在 round 后的整数 bin
    上做不到任意小数，所以用相邻两个 bin 的线性组合逼近（softmax 逆运算）。"""
    head[0, 0, y, x] = logit(p)
    for side, val in enumerate(ltrb):
        lo = int(np.floor(val))
        lo = min(max(lo, 0), REG_MAX - 2)
        frac = val - lo
        # 令 softmax 后 E[bin] = lo + frac：p_lo=1-frac, p_lo1=frac（log 域差即可）
        base = side * REG_MAX + 1  # +1: 通道 0 是 score
        head[0, base : base + REG_MAX, y, x] = -20.0
        head[0, base + lo, y, x] = float(np.log(max(1 - frac, 1e-6)))
        head[0, base + lo + 1, y, x] = float(np.log(max(frac, 1e-6)))


def make_crafted_case() -> None:
    outs = [np.full(s, -20.0, dtype=np.float32) for s in SHAPES]
    o4, o8, o16 = outs

    # --- out4 (stride 4, 104x104) ---
    # 孤立高分框：格心 (10.5*4, 20.5*4)=(42,82), ltrb=(4,8,12,16)*? 单位是格数
    plant(o4, 20, 10, 0.90, (2.5, 3.0, 4.5, 5.25))
    # NMS 抑制：同位置略低分，IoU~1 必被吞
    plant(o4, 20, 11, 0.80, (2.5, 3.0, 4.5, 5.25))
    plant(o4, 21, 10, 0.70, (2.5, 3.0, 4.5, 5.25))
    # 阈值边界的精确保留/丢弃见 thresh case（这里 0.501 会被 max_det 挤掉）
    # DFL 小数期望精度：非 .0/.5 的 frac
    plant(o4, 80, 30, 0.65, (1.37, 2.71, 0.42, 3.83))

    # --- out8 (stride 8, 52x52) ---
    plant(o8, 10, 10, 0.75, (2.0, 2.0, 2.0, 2.0))
    # 与 out4 第一框跨尺度重叠（IoU>0.4），应被 0.90 分框抑制
    plant(o8, 10, 5, 0.85, (1.25, 1.5, 2.25, 2.625))

    # --- out16 (stride 16, 26x26) ---
    plant(o16, 13, 13, 0.60, (3.0, 3.0, 3.0, 3.0))

    # max_det 截断：out4 上 25 个互不重叠的 0.60 分小框（8px），加上其余植入
    # 框后总数 > 20 → 必须按分排序后截断到 20
    for i in range(25):
        plant(o4, 4 + (i // 5) * 20, 64 + (i % 5) * 8, 0.60, (1.0, 1.0, 1.0, 1.0))

    # 均匀 DFL 期望路径：26x26 部分区域 0.40 分（<0.5 被阈值过滤，不入选）
    o16[0, 0, :5, :5] = logit(0.40)

    boxes, scores = ref_decode(outs)
    dump("crafted", outs, boxes, scores)


def make_thresh_case() -> None:
    """阈值边界：score > 0.5 严格保留。0.501 入选、0.499 丢弃、恰好 0.5 丢弃。"""
    outs = [np.full(s, -20.0, dtype=np.float32) for s in SHAPES]
    o4 = outs[0]
    plant(o4, 60, 10, 0.501, (1.0, 1.0, 1.0, 1.0))
    plant(o4, 60, 20, 0.499, (1.0, 1.0, 1.0, 1.0))
    plant(o4, 60, 30, 0.500, (1.0, 1.0, 1.0, 1.0))

    boxes, scores = ref_decode(outs)
    assert len(boxes) == 1 and abs(scores[0] - 0.501) < 1e-3, (boxes, scores)
    dump("thresh", outs, boxes, scores)


if __name__ == "__main__":
    make_ort_case()
    make_crafted_case()
    make_thresh_case()
    print("parity data written to", HERE)
