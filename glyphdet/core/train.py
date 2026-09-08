"""训练循环：focal(match) + CIoU/DFL(reg) + centerness；正负分配在 dataset worker 内完成。

用法：
  pdm run python core/train.py --config experiments/mvp_baseline/config.yaml
  pdm run python core/train.py --config ... --smoke    # 单 batch 过拟合冒烟
"""

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import GlyphDataset
from decode import dfl_expect
from model import GlyphDet, count_params


def focal_sum(logit, target, weight, alpha=0.25, gamma=2.0):
    """未归一化的 focal 求和；调用方最后统一除以 batch 内正样本总数。"""
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = torch.where(target > 0.5, p, 1 - p)
    w = torch.where(target > 0.5, alpha, 1 - alpha) * (1 - pt).pow(gamma)
    return (ce * w * weight).sum()


def ciou_loss(pred, target):
    """pred/target: (N,4) xyxy。"""
    px1, py1, px2, py2 = pred.unbind(1)
    tx1, ty1, tx2, ty2 = target.unbind(1)
    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * (
        torch.min(py2, ty2) - torch.max(py1, ty1)
    ).clamp(0)
    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ta = (tx2 - tx1).clamp(0) * (ty2 - ty1).clamp(0)
    union = pa + ta - inter + 1e-7
    iou = inter / union
    cw = torch.max(px2, tx2) - torch.min(px1, tx1)
    ch = torch.max(py2, ty2) - torch.min(py1, ty1)
    c2 = cw.pow(2) + ch.pow(2) + 1e-7
    rho2 = ((px1 + px2 - tx1 - tx2).pow(2) + (py1 + py2 - ty1 - ty2).pow(2)) / 4
    v = (4 / math.pi**2) * (
        torch.atan((tx2 - tx1) / (ty2 - ty1).clamp(min=1e-6))
        - torch.atan((px2 - px1) / (py2 - py1).clamp(min=1e-6))
    ).pow(2)
    alpha = v / (1 - iou + v + 1e-7)
    return (1 - iou + rho2 / c2 + alpha * v).mean()


def dfl_loss(reg_logits, target, reg_max):
    """reg_logits: (N,4,reg_max) 未 softmax；target: (N,4) 连续值 ∈ [0, reg_max-1]。"""
    tl = target.floor().long().clamp(0, reg_max - 2)
    tr = tl + 1
    wl = tr.float() - target
    wr = 1 - wl
    logp = F.log_softmax(reg_logits, dim=2)
    loss_l = -logp.gather(2, tl.unsqueeze(2)).squeeze(2) * wl
    loss_r = -logp.gather(2, tr.unsqueeze(2)).squeeze(2) * wr
    return (loss_l + loss_r).mean()


def compute_loss(model, scene, mask, targets, cfg):
    """targets: list of (B,8,H,W)（dataset 预算）。"""
    outs = model(scene, mask)
    m = cfg["model"]
    lw = cfg["train"]["loss"]
    reg_max = m["reg_max"]
    device = scene.device
    match_sum = torch.zeros((), device=device)
    reg_sum = torch.zeros((), device=device)
    cen_sum = torch.zeros((), device=device)
    n_pos_total = 0
    for out, tgt, s in zip(outs, targets, m["strides"]):
        B, _, H, W = out.shape
        match, cen_t = tgt[:, 0], tgt[:, 5]
        pos = tgt[:, 6] > 0.5
        weight = tgt[:, 7]
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        match_sum = match_sum + focal_sum(out[:, 0], match, weight)
        if n_pos:
            preg = out[:, 1 : 1 + 4 * reg_max].reshape(B, 4, reg_max, H, W)
            pred_logits = preg.permute(0, 3, 4, 1, 2)[pos]  # (n,4,reg_max)
            reg_t = tgt[:, 1:5].permute(0, 2, 3, 1)[pos]  # (n,4) stride 格单位
            reg_sum = reg_sum + dfl_loss(pred_logits, reg_t, reg_max) * n_pos
            ltrb = dfl_expect(pred_logits.reshape(n_pos, 4 * reg_max), reg_max) * s
            ys, xs = torch.meshgrid(
                torch.arange(H, device=device), torch.arange(W, device=device),
                indexing="ij",
            )
            cx = ((xs.float() + 0.5) * s).unsqueeze(0).expand(B, -1, -1)
            cy = ((ys.float() + 0.5) * s).unsqueeze(0).expand(B, -1, -1)
            px = cx[pos]
            py = cy[pos]
            pred_box = torch.stack(
                [px - ltrb[:, 0], py - ltrb[:, 1], px + ltrb[:, 2], py + ltrb[:, 3]], 1
            )
            gt_box = torch.stack(
                [px - reg_t[:, 0] * s, py - reg_t[:, 1] * s,
                 px + reg_t[:, 2] * s, py + reg_t[:, 3] * s], 1,
            )
            reg_sum = reg_sum + ciou_loss(pred_box, gt_box) * n_pos
            cen_sum = cen_sum + F.binary_cross_entropy_with_logits(
                out[:, -1][pos], cen_t[pos], reduction="sum"
            )
    n_pos = max(n_pos_total, 1)
    total = (
        lw["match_weight"] * match_sum / n_pos
        + lw["reg_weight"] * reg_sum / n_pos
        + lw["cen_weight"] * cen_sum / n_pos
    )
    parts = dict(
        match=(match_sum / n_pos).item(), reg=(reg_sum / n_pos).item(),
        cen=(cen_sum / n_pos).item(), n_pos=n_pos_total,
    )
    return total, parts


def git_version():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True,
            cwd=Path(__file__).parent,
        ).stdout.strip())
        return f"{commit}{' (dirty)' if dirty else ''}"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="单 batch 过拟合冒烟")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tc = cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GlyphDet(cfg).to(device)
    print(f"设备: {device}，参数量 {count_params(model)/1e6:.2f}M")

    root = Path(cfg["data"]["out_dir"])
    train_ds = GlyphDataset(root / "train", cfg)
    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=0 if args.smoke else tc["workers"], pin_memory=True,
        drop_last=True, persistent_workers=not args.smoke,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    if args.smoke:
        # 单 batch 过拟合：loss 必须显著下降，验证图连通与分配正确
        model.train()
        scene, mask, targets = next(iter(loader))
        scene, mask = scene.to(device), mask.to(device)
        targets = [t.to(device) for t in targets]
        for step in range(200):
            loss, parts = compute_loss(model, scene, mask, targets, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 20 == 0 or step == 199:
                print(f"step {step:3d} loss {loss.item():.4f} {parts}")
        return

    run_dir = Path("runs") / tc["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    (run_dir / "code_version.txt").write_text(git_version() + "\n", encoding="utf-8")
    log_f = open(run_dir / "log.txt", "a", encoding="utf-8")

    total_epochs = tc["epochs"]
    steps_per_epoch = len(loader)
    warmup_steps = tc["warmup_epochs"] * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return tc["lr"] * step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * p))

    step = 0
    t0 = time.time()
    for epoch in range(total_epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for scene, mask, targets in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            scene = scene.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            targets = [t.to(device, non_blocking=True) for t in targets]
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, parts = compute_loss(model, scene, mask, targets, cfg)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            ep_n += 1
            step += 1
        msg = (
            f"epoch {epoch+1}/{total_epochs} loss {ep_loss/max(ep_n,1):.4f} "
            f"match {parts['match']:.4f} reg {parts['reg']:.4f} cen {parts['cen']:.4f} "
            f"n_pos {parts['n_pos']} lr {lr_at(step):.2e} "
            f"显存 {torch.cuda.max_memory_allocated()/2**30:.1f}G "
            f"用时 {time.time()-t0:.0f}s"
        )
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()
        torch.cuda.reset_peak_memory_stats()
        torch.save({"model": model.state_dict(), "cfg": cfg}, run_dir / "last.pt")
    log_f.close()
    print(f"完成。权重: {run_dir}/last.pt")


if __name__ == "__main__":
    main()
