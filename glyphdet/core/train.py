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

from glyphdet.core.dataset import GlyphDataset, collate_v4
from glyphdet.core.decode import dfl_expect, has_cen
from glyphdet.core.eval import quick_eval, quick_eval_v4
from glyphdet.core.model import GlyphDet, build_model, count_params


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
    margin_sum = torch.zeros((), device=device)
    n_pos_total = 0
    n_margin = 0
    for out, tgt, s in zip(outs, targets, m["strides"]):
        B, _, H, W = out.shape
        match, cen_t = tgt[:, 0], tgt[:, 5]
        pos = tgt[:, 6] > 0.5
        weight = tgt[:, 7]
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        match_sum = match_sum + focal_sum(out[:, 0], match, weight)
        # margin 排序损失：hard-neg 中心区(weight>1.5)最高分不得贴近正样本均分
        # ——直接优化 AUROC/top-1 口径的点间差距，focal 只管逐点分类不管排序
        neg_mask = (weight > 1.5) & (match < 0.5)
        for b in range(B):
            pb, nb = pos[b], neg_mask[b]
            if pb.any() and nb.any():
                margin_sum = margin_sum + F.relu(
                    out[b, 0][nb].max() - out[b, 0][pb].mean() + lw["margin"]
                )
                n_margin += 1
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
            if has_cen(out.shape[1], reg_max):  # v2.2 起无 cen 通道，跳过
                cen_sum = cen_sum + F.binary_cross_entropy_with_logits(
                    out[:, -1][pos], cen_t[pos], reduction="sum"
                )
    n_pos = max(n_pos_total, 1)
    total = (
        lw["match_weight"] * match_sum / n_pos
        + lw["reg_weight"] * reg_sum / n_pos
        + lw["cen_weight"] * cen_sum / n_pos
        + lw["margin_weight"] * margin_sum / max(n_margin, 1)
    )
    parts = dict(
        match=(match_sum / n_pos).item(), reg=(reg_sum / n_pos).item(),
        cen=(cen_sum / n_pos).item(), margin=(margin_sum / max(n_margin, 1)).item(),
        n_pos=n_pos_total,
    )
    return total, parts


def compute_loss_v4(model, scene, mq, mm, targets, boxes, kinds, cfg):
    """v4：提案(textness focal + 框体系 DFL/L1 + sincos L1) + matcher pair BCE。
    pair 监督（v3 失败复盘的核心修复）：真串×target框=1，真串×其他框=0，
    变异串×所有框=0——"查询是变异"的方向首次被直接监督，无 margin 饱和问题。"""
    m = cfg["model"]
    lw = cfg["train"]["loss"]
    reg_max = m["reg_max"]
    k_mut = int(cfg["train"].get("mut_per_scene", 2))
    feats = model.extract(scene)
    outs = model.prop_maps(feats)
    device = scene.device
    text_sum = torch.zeros((), device=device)
    reg_sum = torch.zeros((), device=device)
    ang_sum = torch.zeros((), device=device)
    n_pos_total = 0
    for out, tgt, s in zip(outs, targets, m["strides"]):
        B, _, H, W = out.shape
        pos = tgt[:, 7] > 0.5
        n_pos = int(pos.sum())
        n_pos_total += n_pos
        text_sum = text_sum + focal_sum(out[:, 0], tgt[:, 0], tgt[:, 8])
        if n_pos:
            preg = out[:, 1 : 1 + 4 * reg_max].reshape(B, 4, reg_max, H, W)
            pred_logits = preg.permute(0, 3, 4, 1, 2)[pos]
            reg_t = tgt[:, 1:5].permute(0, 2, 3, 1)[pos]
            reg_sum = reg_sum + dfl_loss(pred_logits, reg_t, reg_max) * n_pos
            ltrb = dfl_expect(pred_logits.reshape(n_pos, 4 * reg_max), reg_max)
            reg_sum = reg_sum + F.smooth_l1_loss(ltrb, reg_t, reduction="sum")
            pang = out[:, 1 + 4 * reg_max : 3 + 4 * reg_max].permute(0, 2, 3, 1)[pos]
            ang_sum = ang_sum + F.smooth_l1_loss(
                pang, tgt[:, 5:7].permute(0, 2, 3, 1)[pos], reduction="sum"
            )
    # matcher pair BCE：全 batch 汇总后三次调用（mask_enc/strips/score 各一次）。
    # pair 列表 = 逐样本 (query+变异 mask) × (全部标注/bg 框)，标签 真串×target框=1 其余 0。
    f2 = feats[0]
    B = scene.shape[0]
    Wg = max(mq.shape[-1], mm.shape[-1] if mm.numel() else 0)
    all_masks, all_boxes, box_sample = [], [], []
    pair_mi, pair_si, pair_y = [], [], []
    for b in range(B):
        kd, bb = kinds[b], boxes[b]
        sel = (bb[:, 2] > 1) & (bb[:, 3] > 1) & (kd > -1.5)
        if int(sel.sum()) == 0:
            continue
        gt, kd = bb[sel], kd[sel]
        mb = [mq[b]]
        for i in range(k_mut):
            j = b * k_mut + i
            if mm.numel() and j < len(mm):
                mb.append(mm[j])
        masks_b = torch.stack([F.pad(x, (0, Wg - x.shape[-1])) for x in mb], 0)
        base_m = sum(len(t) for t in all_masks)
        base_s = sum(len(t) for t in all_boxes)
        all_masks.append(masks_b)
        all_boxes.append(gt)
        box_sample += [b] * len(gt)
        for mi_ in range(len(masks_b)):
            for si_ in range(len(gt)):
                pair_mi.append(base_m + mi_)
                pair_si.append(base_s + si_)
                pair_y.append(1.0 if (mi_ == 0 and float(kd[si_]) > 0.5) else 0.0)
    pair_sum = torch.zeros((), device=device)
    aux_sum = torch.zeros((), device=device)
    n_pos_pairs = 0
    n_pairs = len(pair_y)
    if n_pairs:
        masks_all = torch.cat(all_masks, 0)
        boxes_all = torch.cat(all_boxes, 0)
        tpl = model.mask_enc(masks_all)
        ink = model.matcher.ink_cols(masks_all)
        strips, boxes_used = model.matcher.strips(
            f2, boxes_all, batch_idx=torch.tensor(box_sample, device=device),
            jitter=model.training, return_boxes=True,
        )
        pair_si_t = torch.tensor(pair_si, device=device)
        span = 8.0 * boxes_used[:, 2] / boxes_used[:, 3].clamp(min=1.0)  # 内容列数 8w/h
        logits = model.matcher.score(
            strips, tpl, ink,
            (torch.tensor(pair_mi, device=device), pair_si_t),
            span=span[pair_si_t],
        )
        labels = torch.tensor(pair_y, device=device)
        # 正负 ~1:8：pos_weight 补偿，否则负例梯度质量 8× 把正例压在 p~0.6（v4.2 平衡态）
        pw = torch.tensor(float(lw.get("pair_pos_weight", 1.0)), device=device)
        pair_sum = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pw, reduction="sum")
        # v4.3 密集列监督：正例对（真串×target 框）GT 几何对齐处逐墨列上拉
        pos_t = labels > 0.5
        n_pos_pairs = int(pos_t.sum())
        if n_pos_pairs and lw.get("aux_weight", 1.0) > 0:
            aux_sum = model.matcher.aux_column_loss(
                strips, tpl, ink,
                (torch.tensor(pair_mi, device=device)[pos_t], pair_si_t[pos_t]),
                boxes_used,
            ) * n_pos_pairs
    n_pos = max(n_pos_total, 1)
    total = (
        lw["text_weight"] * text_sum / n_pos
        + lw["reg_weight"] * reg_sum / n_pos
        + lw["ang_weight"] * ang_sum / n_pos
        + lw["pair_weight"] * pair_sum / max(n_pairs, 1)
        + lw.get("aux_weight", 1.0) * aux_sum / max(n_pos_pairs, 1)
    )
    parts = dict(
        text=(text_sum / n_pos).item(), reg=(reg_sum / n_pos).item(),
        ang=(ang_sum / n_pos).item(), pair=(pair_sum / max(n_pairs, 1)).item(),
        aux=(aux_sum / max(n_pos_pairs, 1)).item(),
        n_pos=n_pos_total,
    )
    return total, parts


def _batch_loss(model, batch, cfg, device):
    """按 arch 分发 batch 解包与损失（v4 六元组 vs 旧三元组）。"""
    if cfg["model"].get("arch") == "v4":
        scene, mq, mm, targets, boxes, kinds = batch
        return compute_loss_v4(
            model, scene.to(device), mq.to(device), mm.to(device),
            [t.to(device) for t in targets], boxes.to(device), kinds.to(device), cfg,
        )
    scene, mask, targets = batch
    return compute_loss(
        model, scene.to(device), mask.to(device),
        [t.to(device) for t in targets], cfg,
    )


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
    ap.add_argument("--resume", action="store_true",
                    help="从 runs/<run_name>/last.pt 续训（含优化器/scaler/epoch/最优 auroc）")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    tc = cfg["train"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg).to(device)
    print(f"设备: {device}，参数量 {count_params(model)/1e6:.2f}M")

    root = Path(cfg["data"]["out_dir"])
    train_ds = GlyphDataset(root / "train", cfg)
    loader = DataLoader(
        train_ds, batch_size=tc["batch_size"], shuffle=True,
        num_workers=0 if args.smoke else tc["workers"], pin_memory=True,
        drop_last=True, persistent_workers=not args.smoke,
        collate_fn=collate_v4 if cfg["model"].get("arch") == "v4" else None,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    if args.smoke:
        # 单 batch 过拟合：loss 必须显著下降，验证图连通与分配正确
        model.train()
        batch = next(iter(loader))
        for step in range(200):
            loss, parts = _batch_loss(model, batch, cfg, device)
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

    # 周期性轻量 eval（Lightning check_val_every_n_epoch / ultralytics fitness 同款惯例）：
    # 每 eval_every epoch 在 val 等距子集上出 auroc/recall/top1/fp，auroc 最优存 best.pt
    eval_every = tc.get("eval_every", 0)
    val_ds = GlyphDataset(root / "val") if eval_every else None
    best_auroc = -1.0
    start_epoch = 0
    if args.resume:
        ckpt_p = run_dir / "last.pt"
        if ckpt_p.exists():
            ck = torch.load(ckpt_p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"])
            if "opt" in ck:  # 旧格式 last.pt（无优化器态）则只恢复权重
                opt.load_state_dict(ck["opt"])
                scaler.load_state_dict(ck["scaler"])
            start_epoch = int(ck.get("epoch", 0))
            best_auroc = float(ck.get("best_auroc", -1.0))
            print(f"resume: {ckpt_p} 从 epoch {start_epoch}/{tc['epochs']} 续训")
        else:
            print(f"resume: {ckpt_p} 不存在，从头训")

    total_epochs = tc["epochs"]
    steps_per_epoch = len(loader)
    warmup_steps = tc["warmup_epochs"] * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return tc["lr"] * step / max(warmup_steps, 1)
        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * p))

    step = start_epoch * steps_per_epoch  # 续训时 lr 调度按全局 step 对齐
    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for batch in loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, parts = _batch_loss(model, batch, cfg, device)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item()
            ep_n += 1
            step += 1
        if cfg["model"].get("arch") == "v4":
            pdetail = (f"text {parts['text']:.4f} reg {parts['reg']:.4f} "
                       f"ang {parts['ang']:.4f} pair {parts['pair']:.4f}")
        else:
            pdetail = (f"match {parts['match']:.4f} reg {parts['reg']:.4f} "
                       f"cen {parts['cen']:.4f} margin {parts['margin']:.4f}")
        msg = (
            f"epoch {epoch+1}/{total_epochs} loss {ep_loss/max(ep_n,1):.4f} "
            f"{pdetail} n_pos {parts['n_pos']} lr {lr_at(step):.2e} "
            f"显存 {torch.cuda.max_memory_allocated()/2**30:.1f}G "
            f"用时 {time.time()-t0:.0f}s"
        )
        print(msg)
        log_f.write(msg + "\n")
        log_f.flush()
        torch.cuda.reset_peak_memory_stats()
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch + 1,
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "best_auroc": best_auroc}, run_dir / "last.pt")
        if val_ds is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == total_epochs):
            qe = quick_eval_v4 if cfg["model"].get("arch") == "v4" else quick_eval
            r = qe(model, val_ds, cfg, device, tc.get("eval_n", 300))
            msg = (
                f"  [eval@{epoch+1}] auroc {r['auroc']:.4f} recall {r['recall']:.4f} "
                f"top1 {r['top1']:.4f} fp {r['fp']:.4f}"
            )
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()
            # 每 eval 一个断点（mini20 被杀全丢的教训；模型小，断点 ~8MB 无负担）
            torch.save({"model": model.state_dict(), "cfg": cfg,
                        "epoch": epoch + 1, "auroc": r["auroc"]},
                       run_dir / f"ep{epoch+1}.pt")
            if r["auroc"] > best_auroc:
                best_auroc = r["auroc"]
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "epoch": epoch + 1, "auroc": best_auroc}, run_dir / "best.pt")
    log_f.close()
    print(f"完成。权重: {run_dir}/last.pt")


if __name__ == "__main__":
    main()
