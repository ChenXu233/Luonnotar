"""掩码条件化字形检测模型（路线 A MVP）。

输入：场景图 (B,3,416,416) + 目标码整串 mask 图 (B,1,64,256)。
输出：3 个尺度的裸卷积结果 (B, 2+4*reg_max, H, W)，通道布局 [score_logit, reg_dfl(4*reg_max), centerness]。
decode（DFL→框、sigmoid、阈值、NMS）在部署侧，训练侧在 loss 里。

设计依据（docs/design.md）：
- anchor-free + decoupled head（YOLOv8 式）：取件码宽高比极端，anchor 先验不可用
- 匹配分支 = 与 mask 全局向量的余弦打分（YOLO-World 式 s = α·cos(e,w) + β，α/β 可学习）
- FiLM 调制 neck（mask 风格/尺度的全局条件化，代价近零）
- 全卷积 + 固定输入 shape，无 attention/动态 shape/GridSample —— ONNX 与 ncnn 双友好
- mask 分支与场景骨干权重不共享（OS2D 消融：解开绑定更好；两域输入分布本就不同）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """两层 3x3 残差块。"""

    def __init__(self, c):
        super().__init__()
        self.c1 = ConvBNAct(c, c)
        self.c2 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        return F.silu(x + self.c2(self.c1(x)))


class SPPF(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c // 2
        self.cv1 = ConvBNAct(c, h, 1)
        self.cv2 = ConvBNAct(h * 4, c, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)

    def forward(self, x):
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([y, y1, y2, self.pool(y2)], 1))


class Backbone(nn.Module):
    """nano 级全卷积骨干。输出 stride 8/16/32 三级特征。"""

    def __init__(self, widths=(16, 32, 64, 128, 192), depths=(1, 1, 2, 2, 1)):
        super().__init__()
        assert len(widths) == 5 and len(depths) == 5
        self.stem = nn.Sequential(
            ConvBNAct(3, widths[0], 3, 2), BasicBlock(widths[0])  # stride 2
        )
        stages = []
        for i in range(1, 5):
            layers = [ConvBNAct(widths[i - 1], widths[i], 3, 2)]
            layers += [BasicBlock(widths[i]) for _ in range(depths[i])]
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)  # stride 4/8/16/32
        self.sppf = SPPF(widths[-1])

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, st in enumerate(self.stages):
            x = st(x)
            if i == 3:
                x = self.sppf(x)
            outs.append(x)
        return outs[1], outs[2], outs[3]  # P3(s8), P4(s16), P5(s32)


class MaskEncoder(nn.Module):
    """mask 图 (B,1,64,256) → (全局向量 w, 模板特征 T)。

    - w (B,mask_dim)：FiLM 调制 + 余弦粗打分
    - T (B,tpl_ch,8,32)：细粒度字形模式，用于与场景特征做多尺度互相关（DW-XCorr 思路），
      解决全局池化丢字形序列细节的问题（一位之差/子串混淆）
    """

    def __init__(self, mask_dim=128, base=32, tpl_ch=48):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, base, 3, 2),        # 32×128
            BasicBlock(base),
            ConvBNAct(base, base * 2, 3, 2),  # 16×64
            BasicBlock(base * 2),
            ConvBNAct(base * 2, base * 4, 3, 2),  # 8×32
            BasicBlock(base * 4),
        )
        self.tpl = nn.Conv2d(base * 4, tpl_ch, 1)          # (B,48,8,32)
        self.head = nn.Sequential(
            ConvBNAct(base * 4, mask_dim, 3, 2),  # 4×16
        )
        self.out_dim = mask_dim
        self.tpl_ch = tpl_ch

    def forward(self, m):
        f = self.stem(m)
        t = self.tpl(f)
        w = F.adaptive_avg_pool2d(self.head(f), 1).flatten(1)
        return w, t


class FiLM(nn.Module):
    """mask 向量 → 每通道 γ/β 仿射调制。零初始化 → 初始为恒等变换。"""

    def __init__(self, wdim, c):
        super().__init__()
        self.fc = nn.Linear(wdim, 2 * c)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, feat, w):
        gb = self.fc(w).view(w.size(0), 2, feat.size(1), 1, 1)
        return feat * (1.0 + gb[:, 0]) + gb[:, 1]


class Neck(nn.Module):
    """简化自顶向下 FPN + 每级 FiLM 条件化。"""

    def __init__(self, in_chs=(64, 128, 192), c=96, wdim=128):
        super().__init__()
        self.lat3 = nn.Conv2d(in_chs[0], c, 1, bias=False)
        self.lat4 = nn.Conv2d(in_chs[1], c, 1, bias=False)
        self.lat5 = nn.Conv2d(in_chs[2], c, 1, bias=False)
        self.film3, self.film4, self.film5 = FiLM(wdim, c), FiLM(wdim, c), FiLM(wdim, c)
        self.fuse3, self.fuse4, self.fuse5 = BasicBlock(c), BasicBlock(c), BasicBlock(c)

    def forward(self, p3, p4, p5, w):
        x5 = self.film5(self.lat5(p5), w)
        x4 = self.film4(self.lat4(p4) + F.interpolate(x5, scale_factor=2, mode="nearest"), w)
        x3 = self.film3(self.lat3(p3) + F.interpolate(x4, scale_factor=2, mode="nearest"), w)
        return self.fuse3(x3), self.fuse4(x4), self.fuse5(x5)


def xcorr_same(S, T):
    """逐通道互相关（SiamRPN++ DW-XCorr 思路）：T 当动态 depthwise 核在 S 上滑窗。

    S (B,C,H,W)，T (B,C,h,w) → (B,H,W)（same padding，中心裁剪）。
    用 grouped-conv 改写以兼容训练期 batch>1；部署 B=1 时即标准 depthwise 动态核卷积，
    ONNX Conv 节点权重可作图输入（ORT 原生支持）。
    """
    B, C, H, W = S.shape
    h, w = T.shape[2], T.shape[3]
    R = F.conv2d(
        S.reshape(1, B * C, H, W), T.reshape(B * C, 1, h, w),
        groups=B * C, padding=(h // 2, w // 2),
    )  # (1, B*C, H+pad, W+pad)
    R = R.reshape(B, C, R.shape[2], R.shape[3]).sum(1)  # (B,H',W')
    # 中心裁剪回 (H,W)（奇数核时 conv 输出恰好多 1 像素）
    y0 = (R.shape[1] - H) // 2
    x0 = (R.shape[2] - W) // 2
    return R[:, y0 : y0 + H, x0 : x0 + W]


class Head(nn.Module):
    """每级 anchor-free 解耦头：匹配分支（余弦粗分 + 模板互相关细分融合）+ 回归分支。

    细匹配只在 P3（stride 8）做：mask 模板 T 缩放到 4 个字高等比尺度与 P3 特征互相关，
    与余弦分图 concat 后 1×1 卷积融合出最终 match logit。P4/P5 仅用余弦分。
    输出通道布局 [score_logit(1), reg_dfl(4*reg_max), centerness(1)]。
    """

    # 模板缩放目标 (h, w)：模板原生 8×48（mask 384 宽），覆盖场景字高 16-48px
    TPL_SCALES = ((2, 12), (3, 18), (4, 24), (6, 36))

    def __init__(self, c=96, wdim=128, reg_max=8, tpl_ch=48):
        super().__init__()
        self.reg_max = reg_max
        self.wdim = wdim
        self.proj = nn.Conv2d(c, wdim, 1)  # 三级共享：同一 mask 语义空间
        self.corr_proj = nn.Conv2d(c, tpl_ch, 1)  # P3 相关特征投影
        self.fuse = nn.Conv2d(1 + len(self.TPL_SCALES), 1, 1)  # cos + 4 尺度相关 → logit
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(-3.0))
        self.reg_convs = nn.ModuleList(ConvBNAct(c, c) for _ in range(3))
        self.reg_out = nn.ModuleList(
            nn.Conv2d(c, 4 * reg_max + 1, 1) for _ in range(3)
        )

    def forward(self, feats, w, tpl):
        wn = F.normalize(w, dim=1).view(w.size(0), self.wdim, 1, 1)
        outs = []
        for i, f in enumerate(feats):
            e = F.normalize(self.proj(f), dim=1)
            cos_map = (e * wn).sum(1, keepdim=True) * self.alpha + self.beta
            if i == 0:
                s_feat = self.corr_proj(f)
                rs = [cos_map]
                for sh, sw in self.TPL_SCALES:
                    t = F.interpolate(tpl, size=(sh, sw), mode="bilinear", align_corners=False)
                    rs.append(xcorr_same(s_feat, t).unsqueeze(1))
                score = self.fuse(torch.cat(rs, 1))
            else:
                score = cos_map
            r = self.reg_out[i](self.reg_convs[i](f))
            outs.append(torch.cat([score, r], 1))
        return outs


class GlyphDet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        widths = tuple(m["widths"])
        tpl_ch = m.get("tpl_ch", 48)
        self.backbone = Backbone(widths=widths)
        self.mask_enc = MaskEncoder(mask_dim=m["mask_dim"], tpl_ch=tpl_ch)
        self.neck = Neck(
            in_chs=(widths[2], widths[3], widths[4]), c=m["neck_ch"], wdim=m["mask_dim"]
        )
        self.head = Head(
            c=m["neck_ch"], wdim=m["mask_dim"], reg_max=m["reg_max"], tpl_ch=tpl_ch
        )
        self.reg_max = m["reg_max"]
        self.strides = m["strides"]

    def forward(self, scene, mask):
        w, tpl = self.mask_enc(mask)
        p3, p4, p5 = self.backbone(scene)
        return self.head(self.neck(p3, p4, p5, w), w, tpl)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ============================================================ v2：像素流重构
# 三个像素死亡点修复：① P5(stride32) 死重砍掉，匹配下沉到 P2(stride4)；
# ② 模板核不再整体压扁——P2 核 (4,24)/(6,36) 盖 16~24px 小字，P3 核
# (4,24)/(6,36)/(8,48) 盖 32~64px 大字，逐字符列数翻倍；
# ③ RepVGG 训练期多分支、导出期融合回单 3×3（零成本增容量）+ SE 注意力。
# 字高窄带 16~64px 专配：每级覆盖一个倍频程，宽长串由大核宽兜底。


class SE(nn.Module):
    """Squeeze-Excitation 通道注意力（参数 ~2c²/r，几乎免费）。"""

    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(4, c // r), 1)
        self.fc2 = nn.Conv2d(max(4, c // r), c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class RepVGGBlock(nn.Module):
    """RepVGG 块：训练期 3×3+1×1+identity 三分支，fuse() 后等价单 3×3 卷积。"""

    def __init__(self, c):
        super().__init__()
        self.b3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.b1 = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, bias=False), nn.BatchNorm2d(c))
        self.bid = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.silu(self.b3(x) + self.b1(x) + self.bid(x))

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return w * scale[:, None, None, None], bn.bias - bn.running_mean * scale

    def fuse(self):
        """返回等价的 nn.Sequential(conv3x3, SiLU)（部署形态）。"""
        w3, b3 = self._fuse_conv_bn(self.b3[0], self.b3[1])
        w1, b1 = self._fuse_conv_bn(self.b1[0], self.b1[1])
        # identity 分支 = 每通道 delta 核的 1×1 卷积
        c = self.b3[0].weight.shape[0]
        wid = torch.zeros(c, c, 1, 1, device=w3.device, dtype=w3.dtype)
        wid[torch.arange(c), torch.arange(c), 0, 0] = 1.0
        std = (self.bid.running_var + self.bid.eps).sqrt()
        scale = self.bid.weight / std
        bid = self.bid.bias - self.bid.running_mean * scale
        w = w3 + F.pad(w1, (1, 1, 1, 1)) + F.pad(wid * scale[:, None, None, None], (1, 1, 1, 1))
        b = b3 + b1 + bid
        conv = nn.Conv2d(c, c, 3, 1, 1, bias=True)
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return nn.Sequential(conv, nn.SiLU())


class BackboneV2(nn.Module):
    """v2 骨干：stride 2/4/8/16 四级（砍掉 stride32 死重），RepVGG 块 + SE 收尾。"""

    def __init__(self, widths=(24, 48, 96, 160), depths=(1, 2, 2, 2)):
        super().__init__()
        assert len(widths) == 4 and len(depths) == 4
        self.stem = nn.Sequential(
            ConvBNAct(3, widths[0], 3, 2), RepVGGBlock(widths[0])  # stride 2
        )
        stages = []
        for i in range(1, 4):
            layers = [ConvBNAct(widths[i - 1], widths[i], 3, 2)]
            layers += [RepVGGBlock(widths[i]) for _ in range(depths[i])]
            layers.append(SE(widths[i]))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)  # stride 4/8/16

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for st in self.stages:
            x = st(x)
            outs.append(x)
        return outs  # P2(s4), P3(s8), P4(s16)


class NeckV2(nn.Module):
    """P2/P3/P4 自顶向下 FPN + 每级 FiLM 条件化。"""

    def __init__(self, in_chs=(48, 96, 160), c=96, wdim=128):
        super().__init__()
        self.lat2 = nn.Conv2d(in_chs[0], c, 1, bias=False)
        self.lat3 = nn.Conv2d(in_chs[1], c, 1, bias=False)
        self.lat4 = nn.Conv2d(in_chs[2], c, 1, bias=False)
        self.film2, self.film3, self.film4 = FiLM(wdim, c), FiLM(wdim, c), FiLM(wdim, c)
        self.fuse2, self.fuse3, self.fuse4 = BasicBlock(c), BasicBlock(c), BasicBlock(c)

    def forward(self, p2, p3, p4, w):
        x4 = self.film4(self.lat4(p4), w)
        x3 = self.film3(self.lat3(p3) + F.interpolate(x4, scale_factor=2, mode="nearest"), w)
        x2 = self.film2(self.lat2(p2) + F.interpolate(x3, scale_factor=2, mode="nearest"), w)
        return self.fuse2(x2), self.fuse3(x3), self.fuse4(x4)


class HeadV2(nn.Module):
    """v2 头：P2/P3 两级做模板互相关（P2 盖小字、P3 盖大字），P4 仅余弦粗分。

    模板核不再整串压扁：P2 (4,24)/(6,36) → 字高 16/24px；P3 (4,24)/(6,36)/(8,48)
    → 字高 32/48/64px。10 字符长串在 (6,36) 下每字符 3.6 列（v1 (2,12) 仅 1.2 列）。
    输出通道布局同 v1：[score_logit(1), reg_dfl(4*reg_max), centerness(1)]。"""

    XCORR_SCALES = {0: ((4, 24), (6, 36)), 1: ((4, 24), (6, 36), (8, 48))}

    def __init__(self, c=96, wdim=128, reg_max=8, tpl_ch=48):
        super().__init__()
        self.reg_max = reg_max
        self.wdim = wdim
        self.proj = nn.Conv2d(c, wdim, 1)
        self.corr_proj = nn.Conv2d(c, tpl_ch, 1)
        self.fuse = nn.ModuleDict(  # 每级：cos + k 个相关图 → logit
            {str(i): nn.Conv2d(1 + len(k), 1, 1) for i, k in self.XCORR_SCALES.items()}
        )
        self.fuse_p4 = nn.Conv2d(1, 1, 1)
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(-3.0))
        self.reg_convs = nn.ModuleList(ConvBNAct(c, c) for _ in range(3))
        self.reg_out = nn.ModuleList(
            nn.Conv2d(c, 4 * reg_max + 1, 1) for _ in range(3)
        )

    def forward(self, feats, w, tpl):
        wn = F.normalize(w, dim=1).view(w.size(0), self.wdim, 1, 1)
        outs = []
        for i, f in enumerate(feats):
            e = F.normalize(self.proj(f), dim=1)
            cos_map = (e * wn).sum(1, keepdim=True) * self.alpha + self.beta
            if i in self.XCORR_SCALES:
                s_feat = self.corr_proj(f)
                rs = [cos_map]
                for sh, sw in self.XCORR_SCALES[i]:
                    t = F.interpolate(tpl, size=(sh, sw), mode="bilinear", align_corners=False)
                    rs.append(xcorr_same(s_feat, t).unsqueeze(1))
                score = self.fuse[str(i)](torch.cat(rs, 1))
            else:
                score = self.fuse_p4(cos_map)
            r = self.reg_out[i](self.reg_convs[i](f))
            outs.append(torch.cat([score, r], 1))
        return outs


class GlyphDetV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        widths = tuple(m["widths"])
        tpl_ch = m.get("tpl_ch", 48)
        self.backbone = BackboneV2(widths=widths)
        self.mask_enc = MaskEncoder(mask_dim=m["mask_dim"], tpl_ch=tpl_ch)
        self.neck = NeckV2(
            in_chs=(widths[1], widths[2], widths[3]), c=m["neck_ch"], wdim=m["mask_dim"]
        )
        self.head = HeadV2(
            c=m["neck_ch"], wdim=m["mask_dim"], reg_max=m["reg_max"], tpl_ch=tpl_ch
        )
        self.reg_max = m["reg_max"]
        self.strides = m["strides"]

    def forward(self, scene, mask):
        w, tpl = self.mask_enc(mask)
        p2, p3, p4 = self.backbone(scene)
        return self.head(self.neck(p2, p3, p4, w), w, tpl)

    def reparam(self):
        """导出前调用：所有 RepVGGBlock 原地替换为融合后的部署形态。"""
        for name, mod in list(self.named_modules()):
            if isinstance(mod, RepVGGBlock):
                parent = self
                *path, attr = name.split(".")
                for p in path:
                    parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
                if attr.isdigit():
                    parent[int(attr)] = mod.fuse()
                else:
                    setattr(parent, attr, mod.fuse())


def build_model(cfg):
    """按 cfg["model"]["arch"] 构造模型（缺省 v1，向后兼容）。"""
    if cfg["model"].get("arch") == "v2":
        return GlyphDetV2(cfg)
    return GlyphDet(cfg)


if __name__ == "__main__":
    # 自检：随机输入前向 + 参数量 + ONNX 导出试跑
    cfg = {
        "model": {
            "widths": [16, 32, 64, 128, 192],
            "neck_ch": 96,
            "mask_dim": 128,
            "reg_max": 8,
            "strides": [8, 16, 32],
        }
    }
    net = GlyphDet(cfg).eval()
    n = count_params(net)
    print(f"参数量: {n/1e6:.2f}M（fp16 约 {n*2/1e6:.1f}MB）")
    scene = torch.rand(1, 3, 416, 416)
    mask = torch.rand(1, 1, 64, 384)
    with torch.no_grad():
        outs = net(scene, mask)
    for s, o in zip(cfg["model"]["strides"], outs):
        print(f"stride {s}: 输出 {tuple(o.shape)}（期望 H=W={416//s}）")
    import tempfile
    from pathlib import Path

    out = str(Path(tempfile.gettempdir()) / "glyphdet_smoke.onnx")
    torch.onnx.export(
        net, (scene, mask), out,
        input_names=["scene", "mask"],
        output_names=[f"out{s}" for s in cfg["model"]["strides"]],
        opset_version=18,
    )
    print(f"ONNX 导出成功: {out}")
