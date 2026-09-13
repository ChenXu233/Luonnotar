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
