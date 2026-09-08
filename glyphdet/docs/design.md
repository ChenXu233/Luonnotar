# 掩码条件化字形检测器 — 架构与训练/部署设计

> 路线 A 的落地设计。任务定义与实测依据见 `../../docs/AI-report/paddle-ocr-postmortem.md`。
> 本文档随实现演进更新；实验冻结参数在 `experiments/<阶段>/config.yaml`。

## 1. 任务定义

> 输入 = 目标码整串的标准 mask 渲染图（雅黑，不拆字符）+ 场景图 → 输出 = 场景中所有相似字形区域的 box（0..N 个）。

- **检测任务，非识别任务**：模型学字形形变不变性（扭曲/透视/缩放/换字体），永不"读"字。
- 无封闭字符集、无模板特征库、无字符几何装配。场景中其余一切文字都是背景。
- 特化甜区：快递标签无艺术字，印刷字形族稳定。

## 2. 模型结构（MVP）

```
场景图 3×416×416 → nano 全卷积骨干 (16/32/64/128/192, stride 8/16/32) → 简化 FPN neck (96ch)
mask 图 1×64×256 → 4 层轻量 CNN → 全局向量 w(128) ──→ FiLM(γ,β) 零初始化调制 neck 三级特征
检测头（每级 anchor-free 解耦，输出裸卷积结果）：
  match 分支：e = 1×1conv(feat)；score = α·cos(e,w) + β（α/β 可学习，三级共享投影）
  reg 分支：3×3conv → 1×1 出 DFL(reg_max=8)×4边 + centerness
输出：3 级张量 (B, 34, H, W) = [score_logit, reg_dfl(32), centerness]
```

- **decode 在部署侧**（DFL 期望→ltrb→框、sigmoid、阈值、NMS），训练/评估用同一参考实现 `core/decode.py`。
- 正负分配 = FCOS 式：中心采样（框向心收缩 25%）+ 按 max(l,t,r,b) 的层级范围（P3:0-64px，P4:64-128，P5:128+），冲突取面积最小者。
- 损失 = focal(match，hard-neg 中心区负样本 ×3 加权) + 2.0×(CIoU+DFL) + 0.5×centerness BCE，统一按 batch 正样本数归一。
- 参数量 ~2M（fp16 ~4MB），预算 10MB 内余量充足。
- 全卷积 + 固定输入 shape；无 attention / 动态 shape / GridSample。

**已知风险（最大技术风险）**：mask 全局池化丢失字形序列细节，`24-6-1234` vs `24-6-1232` 一位之差的区分全靠 hard-negative 对比训练拉开 → go/no-go ① 验证；不达标则二期上 DW-XCorr 细匹配（模板特征当 depthwise 核滑窗出相关图，ncnn 需改写为 MatMul/broadcast-mul+ReduceSum，导出风险先行验证）。

## 3. 数据合成配方（core/synth.py）

- **mask 侧**：固定雅黑干净渲染，定高 64px 左对齐 pad 到 256 宽，承受零形变。
- **场景侧承受全部形变**：场景字体池 15 款（黑/楷/仿宋/宋体/等线 + arial/cour/consola/impact/verdana/tahoma/georgia/times/trebuc），**禁用雅黑**；验证集用 held-out 字体（comic/msjh）检验字体泛化。
- 几何：逐贴片旋转 ±12°/错切 ±0.15/字距抖动 → 全局透视（角点扰动 ≤8%，同矩阵重投影框四角）。弹性形变只作用背景（保证框精确）。
- 光度：模糊/噪声/亮度对比度 gamma/遮挡块/0.4-0.8× 低分辨率重采样（模拟远拍）。
- 标签卡：60% 文本垫白/浅色圆角卡片 + 淡边框/阴影（快递货架标签先验）。
- **hard negative**：50% 概率场景内含与 query 仅一字符之差（替换/增/删/连字符移位）的干扰串，标注 is_target=0，训练负样本加权。
- 确定性：`default_rng([seed, split, idx])`，同 config 重跑逐字节一致；多进程生成 ~24ms/图。

## 4. go/no-go 门槛

| 编号 | 指标 | 门槛 | 验证方式 |
| :--- | :--- | :--- | :--- |
| ⓪ | 导出链路 | 双输入图导出 + 数值对拍 <1e-3 | `core/export_onnx.py`（随机权重先行） |
| ① | 一位之差可分性 | 框内最高分 target vs hard-neg AUROC ≥0.9 | `core/eval.py` one_char_auroc |
| ② | 检出与误检 | held-out 字体 val：recall@0.5/IoU0.5 ≥95%，预测误检率 ≤1%（远期；MVP 先看趋势） | `core/eval.py` |
| ③ | 端侧单帧 | ≤40ms（25fps），目标 10ms 档 | 二期真机插件实测 |

## 5. 部署方案（runtime 开放，用户已拍板不被 ncnn 绑定）

- **导出首选 ONNX**（opset 17，固定 shape 双输入）。ONNX 对余弦打分/DFL/双输入零障碍。
- **端侧 runtime 候选**（二期插件实测选型）：
  1. **ONNX Runtime Mobile + XNNPACK**（CPU）：构建体积可控，XNNPACK 对小 conv 模型 CPU 性能强，fp16 可转。
  2. **ORT + NNAPI EP**：直通手机 NPU/GPU 代理（复盘否决的是 ncnn-NPU 路径，NNAPI 路径从未实测）。
  3. ncnn（pnnx/onnx2ncnn）：备选对照，本模型全是基础算子，转换风险低。
- **Flutter 插件从零自研**（候选名 `vendor/glyph_det`），不学 fast_paddle_ocr 结构：Dart 侧用 Flutter canvas 渲染目标码 mask（会话级缓存，换码才重传）→ 平台通道传帧 + mask 位图 → 原生侧 runtime 推理 → 裸输出回传 Dart 做 decode/NMS（与 `core/decode.py` 同语义）。复用 app 侧既有成果：相机管理/TLHC 视图/`LuonnotarPerf` 埋点口径。
- 10ms 档三杠杆：mask 特征会话缓存（每帧只剩场景骨干+头）、两级推理（路线 C：256 粗筛 + ROI 精检）、输入 320-416 + fp16/int8。

## 6. 调研依据（关键来源）

- YOLO-World（region-text 对比打分头 s=α·L2Norm(e)·L2Norm(w)ᵀ+β，MaxSigmoid 门控）：[arXiv:2401.17270](https://arxiv.org/abs/2401.17270)
- FiLM（逐通道 γ/β 调制）：[arXiv:1709.07871](https://arxiv.org/pdf/1709.07871)
- SiamRPN++ DW-XCorr（二期细匹配融合机制）：[arXiv:1812.11703](https://arxiv.org/abs/1812.11703)
- YOLOE（视觉 prompt YOLO 先例，prompt 缓存/重参数化）：[ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_YOLOE_Real-Time_Seeing_Anything_ICCV_2025_paper.pdf)
- OS2D（class-agnostic 密集匹配范式；共享权重消融）：[ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123600630.pdf)
- AirDet（泛化需要海量基类 = 随机串×字体笛卡尔积）：[ECCV 2022](https://arxiv.org/pdf/2112.01740.pdf)
- SynthText（合成数据管线配方）：[CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Gupta_Synthetic_Data_for_CVPR_2016_paper.pdf)
- CRAFT（高斯热图监督/字符亲和，备选辅助信号）：[arXiv:1904.01941](https://arxiv.org/abs/1904.01941)
- YOLOv8 anchor-free/decoupled/DFL、YOLOv10 NMS-free（二期可选项）、NanoDet-Plus（1.8MB 手机 97FPS 量级参照）、RepViT（备选骨干）

## 7. 演进预案（被否证时的退路）

- ① 失败 → DW-XCorr 细匹配头（保留宽度维序列特征，不做全局池化）。
- ② 字体泛化失败 → 场景字体池扩到 50+（开源字体包）+ 字重/描边程序扰动。
- 实拍域差过大 → 混入实拍无码背景图做底 + 少量实拍正样本微调。
- 端侧超时 → 输入降 320、int8 量化、路线 C 两级推理兜底。
