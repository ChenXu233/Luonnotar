"""裸 head 输出 → 框的参考 decode 实现（训练评估与未来 Dart/C++ 移植共用同一语义）。

输出张量布局（每尺度）：(B, 2+4*reg_max, H, W) = [score_logit, reg_dfl(4*reg_max), centerness]。
最终分 = sigmoid(score_logit) * sigmoid(centerness)。
DFL：reg 沿 reg_max softmax 后取期望 → l,t,r,b（单位：stride 格数）。
"""

import numpy as np
import torch
import torchvision


def dfl_expect(reg, reg_max):
    """reg (N, 4*reg_max) → ltrb 期望 (N, 4)。"""
    r = reg.reshape(-1, 4, reg_max).softmax(dim=2)
    bins = torch.arange(reg_max, dtype=reg.dtype, device=reg.device)
    return (r * bins).sum(dim=2)


@torch.no_grad()
def decode_outputs(
    outs, strides, reg_max,
    score_thr: float = 0.5, nms_iou: float = 0.4, max_det: int = 20,
) -> tuple["np.ndarray", "np.ndarray"]:
    """单张图的裸输出 → (boxes xyxy numpy, scores numpy)。输入为 list of (1,C,H,W)。"""
    all_boxes, all_scores = [], []
    for out, s in zip(outs, strides):
        B, C, H, W = out.shape
        score = torch.sigmoid(out[:, 0]) * torch.sigmoid(out[:, -1])  # (B,H,W)
        reg = out[:, 1 : 1 + 4 * reg_max].permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)
        ltrb = dfl_expect(reg, reg_max) * s  # (H*W, 4)，像素单位
        ys, xs = torch.meshgrid(
            torch.arange(H, device=out.device), torch.arange(W, device=out.device),
            indexing="ij",
        )
        px = (xs.reshape(-1).float() + 0.5) * s
        py = (ys.reshape(-1).float() + 0.5) * s
        boxes = torch.stack(
            [px - ltrb[:, 0], py - ltrb[:, 1], px + ltrb[:, 2], py + ltrb[:, 3]], dim=1
        )
        sc = score.reshape(-1)
        keep = sc > score_thr
        if keep.any():
            all_boxes.append(boxes[keep])
            all_scores.append(sc[keep])
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    keep = torchvision.ops.nms(boxes, scores, nms_iou)[:max_det]
    return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy()
