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
    """v2 头：三级全配模板互相关（P2 盖小字、P3 盖中字、P4 盖大字/长串）。

    模板核不再整串压扁：P2 (4,24)/(6,36)；P3 (4,24)/(6,36)/(8,48)；P4 (4,24)/(6,36)。
    10 字符长串在 (6,36) 下每字符 3.6 列（v1 (2,12) 仅 1.2 列）。
    输出通道布局（v2.2 起移除 centerness）：[score_logit(1), reg_dfl(4*reg_max)]。"""

    # 三级全配 xcorr——levelstat 实锤 v1 的 86% 检出发生在仅余弦级别（近邻混淆病根），
    # v2 曾留 s16 仅余弦的同构缺陷（大字/长串占 target 32%），v2.1 补齐
    XCORR_SCALES = {
        0: ((4, 24), (6, 36)),
        1: ((4, 24), (6, 36), (8, 48)),
        2: ((4, 24), (6, 36)),
    }

    def __init__(self, c=96, wdim=128, reg_max=8, tpl_ch=48):
        super().__init__()
        self.reg_max = reg_max
        self.wdim = wdim
        self.proj = nn.Conv2d(c, wdim, 1)
        self.corr_proj = nn.Conv2d(c, tpl_ch, 1)
        self.fuse = nn.ModuleDict(  # 每级：cos + k 个相关图 → logit
            {str(i): nn.Conv2d(1 + len(k), 1, 1) for i, k in self.XCORR_SCALES.items()}
        )
        if 2 not in self.XCORR_SCALES:
            self.fuse_p4 = nn.Conv2d(
                1, 1, 1
            )  # 末级仅余弦时的占位融合（v2.1 起末级也有 xcorr）
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(-3.0))
        self.reg_convs = nn.ModuleList(ConvBNAct(c, c) for _ in range(3))
        self.reg_out = nn.ModuleList(
            nn.Conv2d(c, 4 * reg_max, 1)
            for _ in range(3)  # v2.2 起无 centerness
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
    arch = cfg["model"].get("arch")
    if arch == "v4":
        return GlyphDetV4(cfg)
    if arch == "v3":
        return GlyphDetV3(cfg)
    if arch == "v2":
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


# ============================================================ v3：逐字 min 聚合判别头
# 治 v2.2 的"不看全体"病根（fppeek：fp 97% 是单字符变异近邻，分数 mean 0.72）：
# 整串模板相关求和 → 逐字槽位模板分别相关、取最小——任何一个字对不上分就死。
#
# mask 契约（画布仍 64×384，渲染端逐字入槽，见 synth.render_mask_slots）：
#   12 槽 × (64×32)；整串居中；奇数串尾补空槽（偶数化）；槽有效性 = 墨迹均值在线判定。
# 对齐设计（全静态，pnnx/ncnn 零风险，训练与部署同一张图）：
#   槽距/字高 = 0.5；特征尺度 sh∈{4,8} → 字距恒为偶数列（2/4），
#   偏移表 off_j=(j-5.5)·sh/2 对任意串长奇偶都是整数常量；
#   奇数串偶数化带来的半槽系统偏差由容差 maxpool 恰好吸收（sh4: ±1列=半字距，
#   sh8: ±2列=半字距），纵向池宽兼吃 ±12° 旋转的极端槽纵向散布（±0.6 字高）。
# 分配/回归/损失/输出布局与 v2.2 完全一致：[score(1), reg_dfl(4*reg_max)]。

V3_SLOTS = 12
V3_SLOT_W = 32
# 每尺度的 (maxpool核, 字距列数)：sh=4 → (7,3) 池、d=2；sh=8 → (11,5) 池、d=4
V3_SCALES = {0: (4, 8), 1: (4, 8), 2: (4, 8)}
V3_POOL = {4: (7, 3), 8: (11, 5)}


class CharSlotEncoder(nn.Module):
    """逐字槽位编码器：mask (B,1,64,K*slot_w) → 逐字模板 (B,K,Cc,8,4) + 全局向量 w。

    槽切片 = 最后一维静态 reshape（槽在渲染端左对齐排布）；w = 有效槽嵌入的
    masked 均值，供 FiLM 与余弦粗分沿用。
    """

    def __init__(self, mask_dim=128, tpl_ch=48, slots=V3_SLOTS, slot_w=V3_SLOT_W):
        super().__init__()
        self.slots = slots
        self.slot_w = slot_w
        base = 24
        self.stem = nn.Sequential(
            ConvBNAct(1, base, 3, 2),            # 32×16
            BasicBlock(base),
            ConvBNAct(base, base * 2, 3, 2),     # 16×8
            BasicBlock(base * 2),
            ConvBNAct(base * 2, tpl_ch, 3, 2),   # 8×4
            BasicBlock(tpl_ch),
        )
        self.tpl = nn.Conv2d(tpl_ch, tpl_ch, 1)
        self.head = ConvBNAct(tpl_ch, mask_dim, 3, 2)  # 4×2

    def forward(self, m):
        B, _, H, W = m.shape
        slots = m.view(B, 1, H, self.slots, self.slot_w)
        slots = slots.permute(0, 3, 1, 2, 4).reshape(B * self.slots, 1, H, self.slot_w)
        # 槽有效性：有墨迹（均值明显低于白底 1.0）为有效
        valid = (slots.flatten(1).mean(1).view(B, self.slots) < 0.98).float()
        f = self.stem(slots)
        t = self.tpl(f)                                   # (B*K, Cc, 8, 4)
        e = F.adaptive_avg_pool2d(self.head(f), 1).flatten(1)  # (B*K, D)
        t = t.view(B, self.slots, *t.shape[1:])
        e = e.view(B, self.slots, -1)
        denom = valid.sum(1, keepdim=True).clamp(min=1.0)
        w = (e * valid.unsqueeze(-1)).sum(1) / denom
        return w, t, valid


def xcorr_slots(S, T):
    """逐槽逐通道互相关：S (B,C,H,W)，T (B,K,C,h,w) → (B,K,H,W)。

    xcorr_same 的 K 扩展：S 沿通道复制 K 份，T 展平为 B*K*C 个 depthwise 核，
    一次 grouped-conv 算完再按通道求和。部署 B=1 时即 576 组深度卷积。
    """
    B, C, H, W = S.shape
    K, h, w = T.shape[1], T.shape[3], T.shape[4]
    Si = S.repeat(1, K, 1, 1).reshape(1, B * K * C, H, W)
    Tk = T.reshape(B * K * C, 1, h, w)
    R = F.conv2d(Si, Tk, groups=B * K * C, padding=(h // 2, w // 2))
    R = R.reshape(B, K, C, R.shape[2], R.shape[3]).sum(2)  # (B,K,H',W')
    y0 = (R.shape[2] - H) // 2
    x0 = (R.shape[3] - W) // 2
    return R[:, :, y0 : y0 + H, x0 : x0 + W]


class HeadV3(nn.Module):
    """v3 头：逐字相关 + min 聚合；回归分支与输出布局同 v2.2。

    每级每尺度：逐字模板缩放后分别 xcorr → (B,K,H,W) → 容差 maxpool →
    按常量偏移表平移（出界读 -1e4，min 自动抑制）→ 有效槽 softmin →
    与 cos_map concat 过 1×1 出 score。softmin 用 logsumexp（数值稳定实现），
    梯度集中在最差匹配字——正是"每个字都必须对"的监督方向。
    """

    def __init__(self, c=96, wdim=128, reg_max=24, tpl_ch=48, slots=V3_SLOTS, tau=1.0):
        super().__init__()
        self.reg_max = reg_max
        self.wdim = wdim
        self.slots = slots
        self.tau = tau
        self.proj = nn.Conv2d(c, wdim, 1)
        self.corr_proj = nn.Conv2d(c, tpl_ch, 1)
        self.fuse = nn.ModuleDict(
            {str(i): nn.Conv2d(1 + len(k), 1, 1) for i, k in V3_SCALES.items()}
        )
        self.alpha = nn.Parameter(torch.tensor(10.0))
        self.beta = nn.Parameter(torch.tensor(-3.0))
        self.pools = nn.ModuleDict(
            {str(sh): nn.MaxPool2d(ph, 1, (ph[0] // 2, ph[1] // 2))
             for sh, ph in V3_POOL.items()}
        )
        self.reg_convs = nn.ModuleList(ConvBNAct(c, c) for _ in range(3))
        self.reg_out = nn.ModuleList(nn.Conv2d(c, 4 * reg_max, 1) for _ in range(3))

    def _align_min(self, r, valid, sh):
        """r (B,K,H,W) → 常量偏移对齐 + masked softmin → (B,1,H,W)。"""
        B, K, H, W = r.shape
        d = sh // 2  # 字距列数（SLOT_RATIO=0.5 × sh）
        cols = []
        for j in range(K):
            o = int(round((j - (K - 1) / 2) * d))
            rj = r[:, j]
            if o > 0:
                rj = F.pad(rj, (0, o), value=-1e4)[..., o : o + W]
            elif o < 0:
                rj = F.pad(rj, (-o, 0), value=-1e4)[..., :W]
            cols.append(rj)
        v = torch.stack(cols, 1)  # (B,K,H,W)，列 x 处 = 槽 j 在 x+off_j 的响应
        v = v + (1.0 - valid).view(B, K, 1, 1) * 1e4  # 无效槽踢出 min
        agg = -self.tau * torch.logsumexp(-v / self.tau, dim=1, keepdim=True)
        # 出界槽读到 -1e4 填充，必须钳位后再进 fuse——否则 -1e4×fuse 权重符号随机
        # 产生 ±5000 级 logit，BCE/梯度直接爆炸（mini run NaN 的实锤根因）
        return agg.clamp(-32.0, 32.0)

    def forward(self, feats, w, tpl, valid):
        wn = F.normalize(w, dim=1).view(w.size(0), self.wdim, 1, 1)
        outs = []
        B, K = tpl.shape[0], tpl.shape[1]
        for i, f in enumerate(feats):
            e = F.normalize(self.proj(f), dim=1)
            cos_map = (e * wn).sum(1, keepdim=True) * self.alpha + self.beta
            s_feat = self.corr_proj(f)
            rs = [cos_map]
            for sh in V3_SCALES[i]:
                sw = sh // 2
                t = F.interpolate(
                    tpl.flatten(0, 1), size=(sh, sw), mode="bilinear",
                    align_corners=False,
                ).view(B, K, -1, sh, sw)
                r = xcorr_slots(s_feat, t)  # (B,K,H,W)
                r = self.pools[str(sh)](r)  # 容差池化（吃半槽偏差+旋转散布）
                rs.append(self._align_min(r, valid, sh))
            score = self.fuse[str(i)](torch.cat(rs, 1))
            r = self.reg_out[i](self.reg_convs[i](f))
            outs.append(torch.cat([score, r], 1))
        return outs


class GlyphDetV3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        widths = tuple(m["widths"])
        tpl_ch = m.get("tpl_ch", 48)
        slots = m.get("slots", V3_SLOTS)
        slot_w = m.get("slot_w", V3_SLOT_W)
        self.backbone = BackboneV2(widths=widths)
        self.mask_enc = CharSlotEncoder(
            mask_dim=m["mask_dim"], tpl_ch=tpl_ch, slots=slots, slot_w=slot_w
        )
        self.neck = NeckV2(
            in_chs=(widths[1], widths[2], widths[3]), c=m["neck_ch"], wdim=m["mask_dim"]
        )
        self.head = HeadV3(
            c=m["neck_ch"], wdim=m["mask_dim"], reg_max=m["reg_max"], tpl_ch=tpl_ch,
            slots=slots, tau=m.get("tau", 1.0),
        )
        self.reg_max = m["reg_max"]
        self.strides = m["strides"]

    def forward(self, scene, mask):
        w, tpl, valid = self.mask_enc(mask)
        p2, p3, p4 = self.backbone(scene)
        return self.head(self.neck(p2, p3, p4, w), w, tpl, valid)

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


# ============================================================ v4：内部两段式（先选框再识别）
# v3 失败复盘（mutation_fp 0.80 / 分离 AUROC 0.594）：监督恒为 mask=真串、场景藏变异，
# 模型从未被训过"查询是变异"的方向；槽位对齐被插入变异+容差池化吸收。
# v4 设计：
#   ① 提案 query 无关：PropHeadV4 出 textness + 框体系 ltrb(DFL) + sincos——
#      "哪里有字符串"与"是不是这串"解耦，所有标注框（含 hard-neg/干扰）都是提案正样本。
#   ② 识别 = rectify + 墨列软 min：按（预测/GT）框几何 grid_sample 规范化条带，
#      模板互相关找全局最优对齐，readout 逐墨列余弦取 softmin——任何一笔墨对不上就死。
#      留白不参与聚合（非墨列踢出 softmin、对齐分按墨列数归一），模型只能学字形。
#   ③ pair BCE 多查询监督：同场景 真串×target框=1，真串×其他框=0，变异×所有框=0，
#      两个方向都被监督，不再需要会饱和的 margin hack。
import math as _math


class ConvGNAct(nn.Module):
    """mask 编码器专用：变长 padding 输入下 BN 统计不稳，用 GroupNorm。"""

    def __init__(self, cin, cout, k=3, s=1, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.gn = nn.GroupNorm(min(groups, cout), cout)

    def forward(self, x):
        return F.silu(self.gn(self.conv(x)))


class MaskEncoderV4(nn.Module):
    """(B,1,64,Wm) → (B,Ct,8,Wm/8) 模板特征（Wm 为 8 的倍数，collate 保证）。"""

    def __init__(self, tpl_ch=32, base=24):
        super().__init__()
        self.stem = nn.Sequential(
            ConvGNAct(1, base, 3, 2),
            ConvGNAct(base, base * 2, 3, 2),
            ConvGNAct(base * 2, tpl_ch, 3, 2),
        )
        self.tpl = nn.Conv2d(tpl_ch, tpl_ch, 1)

    def forward(self, m):
        return self.tpl(self.stem(m))


class NeckPlain(nn.Module):
    """无 mask 条件的 FPN（v4 提案 query 无关，FiLM 随 v3 一起退役）。"""

    def __init__(self, in_chs=(48, 96, 160), c=96):
        super().__init__()
        self.lat2 = nn.Conv2d(in_chs[0], c, 1, bias=False)
        self.lat3 = nn.Conv2d(in_chs[1], c, 1, bias=False)
        self.lat4 = nn.Conv2d(in_chs[2], c, 1, bias=False)
        self.fuse2, self.fuse3, self.fuse4 = BasicBlock(c), BasicBlock(c), BasicBlock(c)

    def forward(self, p2, p3, p4):
        x4 = self.lat4(p4)
        x3 = self.lat3(p3) + F.interpolate(x4, scale_factor=2, mode="nearest")
        x2 = self.lat2(p2) + F.interpolate(x3, scale_factor=2, mode="nearest")
        return self.fuse2(x2), self.fuse3(x3), self.fuse4(x4)


class PropHeadV4(nn.Module):
    """每级提案头：textness(1) + 框体系 ltrb DFL(4*reg_max) + sincos(2)。
    输出通道布局 [textness, reg_dfl, sin, cos]。"""

    def __init__(self, c=96, reg_max=24):
        super().__init__()
        self.reg_max = reg_max
        self.convs = nn.ModuleList(ConvBNAct(c, c) for _ in range(3))
        self.outs = nn.ModuleList(nn.Conv2d(c, 1 + 4 * reg_max + 2, 1) for _ in range(3))

    def forward(self, feats):
        return [self.outs[i](self.convs[i](f)) for i, f in enumerate(feats)]


class MatcherV4(nn.Module):
    """rectify + 墨列软 min 模板匹配（v4 判别核心）。

    strips: 按框几何从 P2 采规范化条带——列/行间距 = h/8 场景像素（各向同性），
      框内容映射为 ~8×(8w/h) 区域，周围留足对齐搜索余量；训练期框几何加抖
      （±6% 中心 / ±20% 宽高 / ±4°）模拟预测框噪声，消 train/test 不一致。
    score: 全局对齐分（墨列归一的互相关 logsumexp）+ 最优对齐处逐墨列余弦 softmin，
      可学习两元混合出 logit。非墨列踢出 softmin——模型不知道"留白"为何物。
    """

    def __init__(self, neck_ch=96, tpl_ch=32, strip_h=8, strip_w=128,
                 feat_stride=4, tau=0.25):
        super().__init__()
        self.proj = nn.Conv2d(neck_ch, tpl_ch, 1)
        self.sh, self.sw = strip_h, strip_w
        self.fs = feat_stride
        self.tau = tau
        self.mix = nn.Parameter(torch.tensor([1.0, 4.0, -2.0]))
        # v4.2 覆盖度 hinge 权重（under/over），见 score 末尾
        self.cov_w = nn.Parameter(torch.tensor([2.0, 2.0]))
        # v4.1 多尺度对齐：条带/模板列距比实测 p5~p95 = 0.82~1.45（pitchcheck），
        # 单平移 dx 对长串必然后半串失配（mini20 零分离根因 A）。
        self.scales = (0.80, 0.89, 1.00, 1.12, 1.25, 1.40)

    def ink_cols(self, mask):
        """mask (m,1,H,Wm) → (m,Wm/8) 墨列 0/1（列最大 ink 池化 >0.3）。"""
        colmax = mask.amax(dim=2).squeeze(1)  # (m,Wm)
        pooled = F.avg_pool1d(colmax, kernel_size=8, stride=8)
        return (pooled > 0.3).float()

    def strips(self, f2, boxes, batch_idx=None, jitter=False, return_boxes=False):
        """f2 (B,C,Hf,Wf)，boxes (n,5) 场景像素 (cx,cy,w,h,θ)，batch_idx (n,) 每条所属样本
        → (n,Ct,sh,sw)。全向量化：一次 grid_sample 采完整批（v4prof：逐框循环曾 0.45s/step）。
        return_boxes=True 时同时返回实际采样用框（jitter 后），供覆盖度 span 对齐。"""
        dev = f2.device
        proj = self.proj(f2)
        Hf, Wf = proj.shape[2], proj.shape[3]
        n = len(boxes)
        if n == 0:
            empty = torch.zeros(0, proj.shape[1], self.sh, self.sw, device=dev)
            return (empty, boxes) if return_boxes else empty
        if batch_idx is None:
            batch_idx = torch.zeros(n, dtype=torch.long, device=dev)
        if jitter:
            boxes = boxes.clone()
            boxes[:, 0] += (torch.rand_like(boxes[:, 0]) - 0.5) * 0.06 * boxes[:, 2]
            boxes[:, 1] += (torch.rand_like(boxes[:, 1]) - 0.5) * 0.06 * boxes[:, 3]
            boxes[:, 2] *= 1.0 + (torch.rand_like(boxes[:, 2]) - 0.5) * 0.2
            boxes[:, 3] *= 1.0 + (torch.rand_like(boxes[:, 3]) - 0.5) * 0.2
            boxes[:, 4] += (torch.rand_like(boxes[:, 4]) - 0.5) * _math.radians(4.0)
        ys, xs = torch.meshgrid(
            torch.arange(self.sh, device=dev), torch.arange(self.sw, device=dev),
            indexing="ij",
        )
        ys, xs = ys.float()[None], xs.float()[None]  # (1,sh,sw)
        sp = (boxes[:, 3] / self.sh).view(n, 1, 1)  # 场景像素/采样格（行列同距，保长宽比）
        ou = (xs + 0.5 - self.sw / 2) * sp
        ov = (ys + 0.5 - self.sh / 2) * sp
        c = torch.cos(boxes[:, 4]).view(n, 1, 1)
        s = torch.sin(boxes[:, 4]).view(n, 1, 1)
        cx = boxes[:, 0].view(n, 1, 1)
        cy = boxes[:, 1].view(n, 1, 1)
        px = cx + c * ou - s * ov
        py = cy + s * ou + c * ov
        gx = (px / self.fs + 0.5) / Wf * 2 - 1
        gy = (py / self.fs + 0.5) / Hf * 2 - 1
        grid = torch.stack([gx, gy], -1)  # (n,sh,sw,2)
        out = F.grid_sample(proj[batch_idx], grid, mode="bilinear",
                            padding_mode="zeros", align_corners=False)
        return (out, boxes) if return_boxes else out

    def score(self, strips, tpl, ink, pairs, span=None):
        """strips (n,Ct,sh,sw)，tpl (M,Ct,8,Wt)，ink (M,Wt)，
        pairs = (mask_idx(P,), strip_idx(P,)) 显式配对，
        span = (P,) 各 pair 条带框的内容列数 8w/h（覆盖度基准，可None跳过）→ match logit。
        v4.1：特征中心化（异字余弦 0.97 坍塌根因 B）+ 多尺度对齐（列距失配根因 A）。
        两遍法省显存：第一遍逐尺度 conv 找最优 (s*,dx*)，第二遍只对各尺度命中的
        pair 子集重取窗口算逐墨列余弦。"""
        mi, si = pairs
        P = mi.shape[0]
        M, Wt = tpl.shape[0], tpl.shape[3]
        Ct = strips.shape[1]
        dev = strips.device
        # 中心化：消公共方向，恢复余弦动态范围
        S = F.normalize(strips - strips.mean(1, keepdim=True), dim=1)
        Tc = tpl - tpl.mean(1, keepdim=True)
        Sp = F.pad(S, (0, 0, 1, 1))[si]  # (P,Ct,sh+2,sw)
        g = P * Ct
        Spf = Sp.reshape(1, g, self.sh + 2, self.sw)

        def scaled_tpl(r):
            Wr = max(8, int(round(Wt * r)))
            if Wr > self.sw:  # 模板放大后比条带还宽：几何上不可能对齐，跳过该尺度
                return None
            Tr = F.interpolate(Tc, size=(self.sh, Wr), mode="bilinear",
                               align_corners=False)
            Tr = F.normalize(Tr, dim=1)
            ikr = (F.interpolate(ink.view(M, 1, 1, Wt).float(), size=(1, Wr),
                                 mode="bilinear", align_corners=False)
                   .view(M, Wr) > 0.5).float()
            return Tr, ikr, Wr

        # 第一遍：逐尺度互相关，合并所有 (scale, dy, dx) 候选
        Dmax = self.sw - 8 + 1
        C = torch.full((P, len(self.scales), Dmax), -1e4, device=dev)
        for k, r in enumerate(self.scales):
            st = scaled_tpl(r)
            if st is None:
                continue  # C[:, k, :] 保持 -1e4，该尺度退出候选
            Tr, ikr, Wr = st
            Tk = (Tr * ikr.view(M, 1, 1, Wr))[mi]  # (P,Ct,sh,Wr)
            R = F.conv2d(Spf, Tk.reshape(g, 1, self.sh, Wr), groups=g)
            Dr = R.shape[3]
            c = R.reshape(P, Ct, 3, Dr).sum(1).max(1).values  # (P,Dr) 行容差取 max
            c = c / (ikr.sum(1).clamp(min=1.0)[mi].view(P, 1) * self.sh)
            C[:, k, :Dr] = c
        flat = C.view(P, -1)
        # v4.2 align 改 hard max：logsumexp 的 τ·log(候选数) 熵膨胀对短模板（删除型
        # 变异，dx 候选更多且子串完美对齐）系统性偏高，反判别；max 候选数不变量。
        align = flat.max(1).values  # (P,)
        best = flat.argmax(1)
        ks, dxs = best // Dmax, best % Dmax

        # 第二遍：最优尺度处逐墨列余弦（pad 到 Wmax 向量化，非墨列 +1e4 踢出聚合）
        Wmax = max(8, int(round(Wt * max(self.scales))))
        v = torch.full((P, Wmax), 1e4, device=dev)
        spanink = torch.zeros(P, device=dev)  # 各 pair 最优尺度下的模板墨列数
        arange_cache = {}
        for k, r in enumerate(self.scales):
            sel = ks == k
            if not sel.any():
                continue
            Tr, ikr, Wr = scaled_tpl(r)
            Tk = (Tr * ikr.view(M, 1, 1, Wr))[mi[sel]]  # (Ps,Ct,sh,Wr)
            Ps = Tk.shape[0]
            if Wr not in arange_cache:
                arange_cache[Wr] = torch.arange(Wr, device=dev).view(1, Wr)
            cols = (dxs[sel].view(Ps, 1) + arange_cache[Wr]).clamp(0, self.sw - 1)
            win = torch.gather(S[si[sel]], 3,
                               cols.view(Ps, 1, 1, Wr).expand(Ps, Ct, self.sh, Wr))
            vv = (win * Tk).sum((1, 2)) / self.sh  # (Ps,Wr) 逐墨列余弦
            vv = vv + (1.0 - ikr[mi[sel]]) * 1e4
            v[sel, :Wr] = vv
            spanink[sel] = ikr[mi[sel]].sum(1)
        # v4.2 worst-k 聚合：softmin 的 -τ·log(N) 熵偏差把 colmin 全拖负且惩罚随串长
        # 递增（变异串更短反而分高，反判别——零分离根因 C）。取最差 4 墨列均值：
        # 变异 1 字≈连续 4-6 坏列直接命中；正例 worst-4 保持高分；无温度无计数偏差。
        kw = min(4, Wmax)
        colmin = torch.topk(v, kw, dim=1, largest=False).values.mean(1)  # (P,)
        a, b, c0 = self.mix
        logit = a * align + b * colmin + c0
        if span is not None:
            # 覆盖度双侧 hinge：删除型变异是子串（逐列匹配天然盲），靠
            # 模板墨列跨幅/框内容列数(8w/h) 偏离 1 来杀（零分离根因 D）。
            # span 须用实际采样框（训练期 jitter 后）。±15%/25% 余量吸收框回归噪声。
            cov = spanink / span.clamp(min=8.0)
            d1, d2 = self.cov_w
            logit = logit - d1 * F.relu(0.85 - cov) - d2 * F.relu(cov - 1.25)
        return logit.clamp(-32.0, 32.0)


def pair_grid(m, n, device):
    """m×n 全配对下标（评估用）：返回 (mask_idx, strip_idx)，行主序 reshape(m,n) 可还原。"""
    mi = torch.arange(m, device=device).view(m, 1).expand(m, n).reshape(-1)
    si = torch.arange(n, device=device).view(1, n).expand(m, n).reshape(-1)
    return mi, si


class GlyphDetV4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        widths = tuple(m["widths"])
        tpl_ch = m.get("tpl_ch", 32)
        self.backbone = BackboneV2(widths=widths)
        self.neck = NeckPlain(in_chs=(widths[1], widths[2], widths[3]), c=m["neck_ch"])
        self.prop = PropHeadV4(c=m["neck_ch"], reg_max=m["reg_max"])
        self.mask_enc = MaskEncoderV4(tpl_ch=tpl_ch)
        self.matcher = MatcherV4(
            neck_ch=m["neck_ch"], tpl_ch=tpl_ch,
            strip_h=m.get("strip_h", 8), strip_w=m.get("strip_w", 128),
        )
        self.reg_max = m["reg_max"]
        self.strides = m["strides"]

    def extract(self, scene):
        return self.neck(*self.backbone(scene))

    def prop_maps(self, feats):
        return self.prop(feats)

    def forward(self, scene, mask=None):
        """提案特征图（v4 提案 query 无关，mask 不进前向；匹配走 matcher.score）。"""
        return self.prop_maps(self.extract(scene))

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
