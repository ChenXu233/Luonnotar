"""裸 head 输出 → 框的参考 decode 实现（训练评估与未来 Dart/C++ 移植共用同一语义）。

输出张量布局（每尺度）：(B, C, H, W)，C = 1+4*reg_max 或 2+4*reg_max（后者含 centerness，
v1 遗存；v2.2 起移除——cen 标签被压在 [0.44,0.80] 窄带，模型只学到 0.06 nat，等于常数因子）。
最终分 = sigmoid(score_logit)（有 cen 通道时再乘 sigmoid(centerness)，按通道数自动判别）。
DFL：reg 沿 reg_max softmax 后取期望 → l,t,r,b（单位：stride 格数）。
"""

import numpy as np
import torch
import torchvision


def has_cen(C, reg_max):
    return C == 2 + 4 * reg_max


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
        score = torch.sigmoid(out[:, 0])
        if has_cen(C, reg_max):
            score = score * torch.sigmoid(out[:, -1])  # (B,H,W)
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


@torch.no_grad()
def decode_rprops(outs, strides, reg_max, prop_thr=0.3, nms_iou=0.4, max_prop=32):
    """v4 提案 decode：textness>thr 点 → 旋转框 (n,5) numpy + 提案分 numpy。
    通道布局 [textness, reg_dfl(4*reg_max), sin, cos]；ltrb 在框体系（阅读方向轴）。
    NMS 用 AABB（场景稀疏）；精读由 matcher 分数再筛。"""
    all_rb, all_sc = [], []
    for out, s in zip(outs, strides):
        B, C, H, W = out.shape
        score = torch.sigmoid(out[:, 0])
        reg = out[:, 1 : 1 + 4 * reg_max].permute(0, 2, 3, 1).reshape(-1, 4 * reg_max)
        ltrb = dfl_expect(reg, reg_max) * s
        sincos = out[:, 1 + 4 * reg_max : 3 + 4 * reg_max].permute(0, 2, 3, 1).reshape(-1, 2)
        th = torch.atan2(sincos[:, 0], sincos[:, 1])
        ys, xs = torch.meshgrid(
            torch.arange(H, device=out.device), torch.arange(W, device=out.device),
            indexing="ij",
        )
        px = (xs.reshape(-1).float() + 0.5) * s
        py = (ys.reshape(-1).float() + 0.5) * s
        l, t, r, b = ltrb.unbind(1)
        c, sn = torch.cos(th), torch.sin(th)
        du, dv = (r - l) / 2, (b - t) / 2
        cx = px + c * du - sn * dv
        cy = py + sn * du + c * dv
        rb = torch.stack([cx, cy, l + r, t + b, th], 1)
        sc = score.reshape(-1)
        keep = sc > prop_thr
        if keep.any():
            all_rb.append(rb[keep])
            all_sc.append(sc[keep])
    if not all_rb:
        return np.zeros((0, 5), np.float32), np.zeros(0, np.float32)
    rb = torch.cat(all_rb)
    sc = torch.cat(all_sc)
    aw = rb[:, 2] * rb[:, 4].cos().abs() + rb[:, 3] * rb[:, 4].sin().abs()
    ah = rb[:, 2] * rb[:, 4].sin().abs() + rb[:, 3] * rb[:, 4].cos().abs()
    aabb = torch.stack(
        [rb[:, 0] - aw / 2, rb[:, 1] - ah / 2, rb[:, 0] + aw / 2, rb[:, 1] + ah / 2], 1
    )
    keep = torchvision.ops.nms(aabb, sc, nms_iou)[:max_prop]
    return rb[keep].cpu().numpy(), sc[keep].cpu().numpy()
