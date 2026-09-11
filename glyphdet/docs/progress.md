# 实验进度记录（随实验演进追加，新记录在前）

> 口径：eval 在 held-out 字体 val（comic/msjh）上测；one_char_auroc = target 框内最高分 vs 干扰框内最高分的 AUROC；top1_acc = 每图最高分预测框命中 target(IoU≥0.5) 的图比例（App 只高亮最优匹配，最贴近实际）。
> go/no-go 门槛：AUROC ≥0.9；recall ≥95%；误检 ≤1%；端侧 ≤40ms/帧。

## 实验谱系

| 实验 | 数据 | 模型 | AUROC | recall@0.5 | FP | top1 | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2k/60ep（中断@42） | 2k 旧配方 | 全局向量 cos | 0.613 | 0.433 | 0.41 | — | 欠训+数据量不足 |
| 50k/30ep | 50k 旧配方 | 同上 | 0.752 | 0.697 | 0.28 | — | train loss≈0 但 val 上不去 → 架构上限：子串混淆 F342>F42 |
| 50k/30ep xcorr | 50k 旧配方 | +模板分支多尺度互相关 | 0.848 | 0.754 | 0.21 | 0.77 | 细匹配有效但不够；漏检边缘截断目标；thr 扫描证非阈值问题 |
| mvp_f39_e45 | 50k 39字体池 | 同上 | 0.84 | — | — | — | 字体覆盖假设否证：39 字体+45ep 边际收益≈0，瓶颈不在字体空间 |
| mvp_f39_e45_margin | 同上（毒数据） | +margin 排序损失 | 0.876@epoch18 | 0.47 | 0.14 | 0.60 | 中期被杀读数：margin 直攻排序有效（同数据 +0.03~0.04），假设带入 maskfix |

## 2026-09-09 margin 实验裁决 → maskfix 训练接管

- margin 排序损失版（旧毒数据）epoch 18/45 中期 eval：AUROC 0.876 已超 f39 全程 0.84 —— 排序损失假设成立。但旧数据三大毒 bug 已修，继续在毒数据上训 27 epoch 价值低，**杀掉让 GPU**，margin 损失已内置于 maskfix config（weight 0.5 / margin 2.0 / hard_neg_center 3.0）。
- **mvp_maskfix 训练启动**（runs/mvp_maskfix）：datasets/mvp2 v5 干净数据（mask 384 等比缩放 + target 完整入框 + 墨色极性校正 + 纯字形 alpha 可读性终检），45ep，含全部修正。数据抽检：train 50000 / val 1000 齐全，6 张跨段目检全过。
- **epoch 13/45 中期 eval（val=mvp2/val 干净配方，与旧毒 val 数字不完全同分布）**：**AUROC 0.9241 已破 0.9 门槛**（旧基线全程最好 0.85，margin 中期 0.876）；fp_rate 0.043（旧 0.14~0.21 量级）暴降，neg_score_mean 0.072 vs tgt 0.400 间距拉开；top1 0.485 / recall 0.36 仍低待退火。loss 侧证：match 0.061 vs 旧同配置 0.186，margin 项 0.083 vs 旧 0.581（旧毒数据上 hinge 压不下去——隐身/残串 target 与 hard-neg 不可分）。

## 2026-09-10 100k+mask 抖动全程裁决（epoch 40/40，held-out comic/msjh val 1000）

- **终 eval：AUROC 0.9508 ✓；top1 0.923；recall@0.5 0.9001（门槛 95% 未达）；fp 0.0833（门槛 1% 未达）；GPU 13.52ms / CPU-PC 74.16ms**。对比 maskfix 终局（0.9361/0.907/0.894/0.104）：AUROC +0.015、top1 +0.016、recall +0.006、fp −0.021——**数据量翻倍+mask 抖动收益递减**，泛化平台期特征明显。注意 ep12 中期 0.955 略高于终局 0.9508，后段退火未再涨（轻微过拟合训练字体）。
- recall_iou0.3 == recall@0.5（0.9001）→ 框质量依旧无罪，漏检=判别置信度不足；fp 8.3% 是离门槛最远的指标。
- thr_sweep：阈值 0.3 时 recall 0.9296/fp 0.0911 —— 降阈值能换 recall 但 fp 更差，印证"低阈值+时序积分"路线必须以压 fp 为前提。
- **下一步**：fppeek（tools/fppeek.py，新建）分解 fp 成分（distr 干扰串型 vs bg 背景型）+ failpeek 重跑，给用户人工判定后定配方：对抗负样本（同字体/同卡 hard-neg）vs 类文字背景纹理。

## 2026-09-10 v6（hardneg）终局裁决 + v2 架构 A/B 开火

- **mvp_100k_hardneg 终 eval（ep40，lineage val）**：AUROC 0.9351 / top1 0.885 / recall 0.8612 / fp 0.097 / GPU 13.54ms——**全面低于 maskaug 终局（0.9508/0.923/0.9001/0.0833）**。v6 数据杠杆（同字体同卡 hard-neg 加量）在 v1 架构上收益为负：判别预算被更难的训练负样本耗尽，held-out 字体轴被挤垮。**架构天花板实锤，像素/匹配粒度问题必须架构解**。
- **v2 已实现并冒烟通过**（GlyphDetV2，experiments/mvp5_v2）：① 砍 stride32 死重（P5 占 ~40% 参数却对 16~48px 字高带无贡献），匹配下沉 P2(stride4)——P2 核 (4,24)/(6,36) 盖 16~24px、P3 核 (4,24)/(6,36)/(8,48) 盖 32~64px，10 字符长串每字符 3.6 列（v1 仅 1.2 列）；② RepVGG 训练期三分支、导出 fuse 回单 3×3（fuse 等价 diff ~1e-6）+ SE 注意力；③ 层级分配按字高 {4:(0,28),8:(28,56),16:(56,∞)}（修 v1 maxd 在极端宽高比下把小字长串错配到粗级别的暗伤）。**参数 2.47M→fuse 2.39M vs v1 3.36M（-27%）**。
- **A/B 设计**：同数据 datasets/mvp4（v6 配方，零重合成），v1 基线 0.9351/0.885/0.8612/0.097。run=mvp_v2 40ep 后台训练中。
- 用户判断背书：像素传递（骨干早降采样斩笔画）与 RepVGG 类重参数化一起解决；假设"架构对了之后泛化不再依赖数据堆量"——本轮 A/B 即验证。

## 2026-09-10 fppeek/failpeek 裁决（终局权重，val 前 300 张）

- **fp 成分：39/39 全是 distr 型（框住干扰串），0 个背景型** → 类文字背景纹理路线砍掉，对抗负样本是唯一方向（用户预判正确）。
- **fp 分数高得危险：mean 0.675 / p50 0.696 / max 0.914**——不是低分噪声，是高分误检，降阈值+时序积分路线的前提（时序层杀随机 fp）对这类"空间稳定型 fp"无效，必须在数据/模型层压掉。
- fp 两类实锤模式（top16 特写人工核查）：① **一位之差**：Q=7870 fp 框 7874（0.91）、Q=81449 fp 框 8049（0.91）；② **同数字重分段**：Q=9-77-62674 fp 框 97-7-62674（0.84）、Q=47-95-8384 fp 框 479-5-8384（0.88）——xcorr 模板匹配对字符集合敏感、对逐位身份/分段位置不够敏感。
- failpeek 重跑（failpeek_100k.png）：漏检主导模式=**同场景双 target 实例，一个贴线检出（0.49~0.56）一个漏**——target 分数带与 hard-neg 分数带在 0.5~0.9 区间重叠，fp↓ 与 recall↑ 是同一个杠杆：拉宽两分布间距。
- **v6 配方裁决**：每场景显式注入 1~2 个 hard-neg（一位之差 / 同数字重分段 / 截断少位），hard-neg 与 target 同字体同卡片同墨色极性；背景配方不动；维持 100k+40ep。出图：docs/fpreview/fppeek.png、docs/failreview/failpeek_100k.png。
- **v6 落地**（experiments/mvp4_hardneg）：`hard_negative` 拆 `_one_mutation`，sub 数字 60% 走形近对（_CONFUSE：0/8/6、1/7、3/8/5、5/6、6/9/0、8/9 等），hyphen 移位 1→1~2 格，30% 叠第二次扰动（覆盖"换一位+少一位"）；del 避开两连字符夹位（防 "--"，冒烟实锤过）。`render_text_patch` 加 `card` 参数（卡片决策上提到 build_sample 才可拷贝）；build_sample 预分配字体/卡片：hard-neg 拷贝随机 target 的字体与卡片决策、强制走极性校正（必须可读、禁止低对比）；`hard_neg_count_range: [1,2]` 覆盖旧 hard_neg_prob 伯努利（旧键保留兼容）。冒烟 8 张目检全过：近邻串同字体同卡清晰可见（7874 型、重分段型、二次扰动型齐全）。val 复用 mvp2 保口径（新配方的 val 会更难，不可与谱系对比，故只产 train）。
- **ep29 中期裁决（并发训练测得，infer_ms 失真勿采信）**：AUROC 0.9384 / top1 0.863 / recall 0.8095 / fp 0.0897——headline 全面低于 maskaug 终局，但 recall 历来退火段才爬（maskaug ep12 仅 0.64→终 0.90）。**关键证据是 fppeek：fp 38 个仍全是 distr 型，分数 mean 0.679/p50 0.706/max 0.921 与 maskaug 终局完全持平，失败模式一字不差（一位之差、重分段原图重现）**。train 侧 margin 早已归零 = 训练字体上的近邻判别已解决却不泛化到 held-out 字体。**疑似根因：模板 T 宽度仅 8 token，10~12 字符的串每字符分不到 1 个 token，逐位判别在表示层被平均掉，模型只能靠背训练字体的细节作弊**。若 ep40 终局确认 fp 不动，下一杠杆=架构：模板加宽（8→16/24 token）/ P4 浅层也做 xcorr——不是扩容，是给逐位判别开物理通道。
- 环境备忘：core/ 已重构为包（glyphdet.core.*，__init__.py 定位"稳定引擎层"），eval 等须 `PYTHONPATH=仓库根 python -m glyphdet.core.eval` 方式运行；tools 已同步转换。相对路径（datasets/...）仍以 CWD=glyphdet 解析。

## 2026-09-10 100k+mask 抖动中期裁决（epoch 12/40）

- **中期 eval：AUROC 0.955**（maskfix 同期 0.9241，+0.031）；top1 0.797（同期 0.485）；recall@0.5 0.6445（同期 0.36）；fp 0.055（同期 0.043 持平偏好）。任务更难（mask 字体/字重抖动）但 val 更高 → 泛化真在涨，非过拟合假象。infer_ms 受训评并发争抢影响失真，勿采信。
- 同 epoch loss 对比：match 0.043 vs 0.075、margin 0.019 vs 0.032——更难任务更低损失。

## 2026-09-09 maskfix 全程裁决（epoch 45，held-out comic/msjh val 1000）

- **终 eval：AUROC 0.9361 ✓（门槛 0.9）；top1 0.907；recall@0.5 0.894（门槛 95% 未达）；fp 0.104（门槛 1% 未达）；GPU 13.5ms/CPU 83ms**。train loss≈0（match 0.001）→ 已过拟合，瓶颈=泛化而非容量。
- failpeek + 按目标高度分桶：recall 在 14-20/20-30/30-45/45+px 全平（92/88/93/93%）→ **小字假设否证**，漏检均匀分布，主因是 held-out 字体泛化 + 低对比/模糊复合。
- **下一轮（进行中）**：datasets/mvp3 = 100k 训练数据 + **mask 字体/字重抖动**（msyh/arial/segoe/calibri/tahoma/verdana 随机 + 30% 描边加粗；迫使模板匹配对 mask 字体不变，同时兜住部署侧 App 渲染 mask 的字体域漂）。val 直接复用 mvp2/val 保口径。config：experiments/mvp3_100k/config.yaml（run=mvp_100k_maskaug，40ep，其余同 maskfix）。
- 部署侧备忘：mask 字体在 App 端用 fonttools subset 打包雅黑（仅 0-9/字母/分隔符，几 KB）可完全消除域漂；二期实现时二选一。

## 2026-09-09 failpeek 实锤两个数据 bug（用户指示手动核查失败样本 mask）

1. **mask 压扁**：`render_mask` 宽度定死 256px，9+ 字符长码被单向压扁 1.3~1.6×，而场景目标保持自然比例 → 模板互相关必然失配。漏检重灾区：长码、小字（16-20px）、边缘截断目标、暗底繁忙背景。
   修复：超宽时等比整体缩小（不压扁）；mask 宽度 256→384；TPL_SCALES (2,8),(3,12),(4,16),(6,24) → (2,12),(3,18),(4,24),(6,36) 适配 8×48 模板。
2. **target 边缘截断**：贴片位置按"中心在 [0.08S,0.92S]"采样，长码 patch 必然出图；全局透视 ±8% 角点扰动再把边角目标推出去，截断后的残串仍标注为正样本 → mask（完整串）与可见字形（残串）不匹配，直接教坏排序。用户补充裁定规则：**mask 对应的目标必须完整，无关干扰数字出屏无所谓**（#95 "5-7-829" 被右边界切掉尾字实锤）。
   修复（synth.py）：① target 贴片必须完整入框（0.03S 边距），放不下就逐轮压低字高重渲（th_cap 44→10，8 轮）；② `perspective(..., must_inside=target下标)` 重采样单应矩阵直到 target 四角全在图内，8 次失败退恒等；③ 干扰串维持允许截断（真实负样本）；④ 低对比重渲后重新钳位，入框优先于低对比。
   验证：32 张试产 + peek 网格目检 12 张全绿框完整入框，mask 比例自然（长码整体缩小非压扁）。
   **全量审计**：旧 mvp 75,021 个 target 中 **25,981 个（34.63%）触边截断**；新 mvp2 v5 **0 个触边**（干扰串 24,459 触边，设计内）。旧数据三分之一正样本是残串——0.85 瓶颈主账。
3. **target 渲染不可读**（用户点名 datasets/mvp/train/scene/000149.png 实锤）：target "3-9-7222" 深字贴暗底、无标签卡（正常路径墨色 80% 概率取深、与背景亮度无关），再被光度 gamma/噪声压垮 → 直方图拉伸后仍只剩噪声，人眼不可辨却标注为正样本。估算此类样本占 10-20%（暗底 × 深字 0.8 × 无卡 0.4）。
   根源教训：**构造时的规则挡不住后续环节的破坏，必须在最终像素上验证**。
   修复（synth.py）：① target 按贴入处背景亮度强制墨色极性（亮底深字/暗底亮字），低对比路径差值提到 [35,95]；② 新增可读性终检——贴片时累积每个 target 的字形 alpha 掩码、随透视同矩阵 warp，光度后逐 target 比较字形区与 4px 环带的灰度均值，差 <25 级即重 roll；③ 顺手修干扰串互叠 >60% 重试。
4. **35% 拒绝率调查（用户追问能否从算法上避免）**：插桩分解发现 100% 拒绝来自终检门。两级根因：
   - **终检度量本身错了**：字形 alpha 取自整个 patch 的 alpha 通道，**60% 带卡样本量到的是整张卡片**（alpha 255）——"卡片均值 vs 卡外背景"，亮卡贴亮底 delta≈10 被误判不可读；dump 12 张失败样本目检全清晰可读（d=1~24 的"7-54-5913"、"GH8WBJ0-JTL6"等）。修复：render_text_patch 额外输出**纯字形 alpha**（不含卡片/阴影），warp/paste 全程跟随，终检量"墨迹 vs 墨迹周边"——亮卡深字 delta≈200 如实通过。
   - **真失败的打捞机制**：薄笔画被模糊/低采样抹掉是光度参数的锅，不该连累整样本——终检不过就保留布局换一组光度参数重来（最多 6 组），仍不过才整样重 roll；极性校正重渲不入框也改为压低字高重试 4 轮、绝不退回随机极性。
   结果：拒绝率 **34.4% → 0.8%**（残余全是低对比路径的真不可读，门在诚实工作）；通过 target 的 delta p5=43、中位 104；产出效率回到 ~1.0×。
   教训：dump 失败样本目检比统计指标更快定位——统计说"中灰背景 polarity 失效"，眼睛一眼看出"失败样本全都清晰可读"。

## 进行中

- **mvp_v2 训练 40ep bs16**（后台 bash-5f28sm4n，20:09 首启 bs32 因显存 7.2G/8G 顶到交换区被杀，20:4x 以 bs16 重启，重启后 5.6G 安全水位；lr 未动——bs 32→16 在容忍范围内，保 A/B 配方可比）：A/B 同数据 datasets/mvp4，v1 基线 0.9351/0.885/0.8612/0.097。P2 级使 epoch 比 v1（13.6min）长且 bs 减半，估 25~35min/epoch，40ep ≈ 17~23h。训完：终 eval 对比基线 → fppeek 复看近邻串分数 → 达标则 export_onnx（v2 已内置 reparam 融合）。
- 待用户人工判定：docs/failreview/（maskfix 时代 16 张漏检，未回收判定）、docs/fpreview/fppeek.png 与 docs/fpreview/fppeek_hardneg_ep29.png（fp 特写，已自查全 distr 型）。

## 二期设计共识（用户讨论沉淀，2026-09-10）

**总线：检测非识别、两阶段、时序积分、像素预算只花在刀刃上。**

1. **时序层（不改模型，纯推理期逻辑，住插件层）**：逐帧检测 + IoU 关联成 track + k-of-n 滑窗积分确认（单帧 recall 0.9 → 4 帧中 2 ≈ 0.996，随机 fp 空间一致性被杀）+ 运动门控（IMU/帧差，模糊帧降权）+ 确认后切 ROI 锁定跟踪（裁小图推理 ~5ms，帧率跑满，丢失回全帧）。decode 阈值插件可配（建议 0.15~0.2 低阈值跑高 recall，fp 交给时序层）。自回归/多帧网络均否掉（无序列解码结构、破 40ms 帧预算、训练数据代价大）；空间稳定型高分 fp（近邻串）时序层杀不掉，必须数据/模型层压。
2. **1080p 物理账**：模型入口 416²，训练字高分布 16~44px。整帧 1080p→416 缩放比 0.217，40cm 手持 1cm 字（≈34px@1080p）只剩 7px——整帧缩放必死。正解：相机层数码变焦（SCALER crop，免费）+ 原生分辨率 ROI 裁切；UX 预期管理（太远就提示靠近/自动放大）。判别必须发生在笔画分得开的原生像素上。
3. **两阶段架构（用户提出虚拟类别）**：stage-1 = TextProposal 提案网（摘掉 mask 条件的单类检测器："这里像不像一串字"，类别无关 objectness；~0.2-0.4M 参数、208²~416² 输入、1~3ms NPU，25fps 常驻；KPI=recall@topK，精度交给 stage-2）；stage-2 = GlyphDet 只吃原生分辨率 ROI（416 letterbox），永远在训练分布内。标签白拿：现有合成数据所有文本框 is_target 全置 1 重打包。stage-1 用 CRAFT 字符级 textness 路线（文献证据：textness 泛化比判别泛化容易，但需字符级粒度而非纯纹理）。
4. **stage-1 分辨率可变**（用户坚持）：全卷积+FPN → ONNX 动态输入轴 320/416/608/832 按场景/帧率选档；训练用 YOLO 式随机输入尺寸一套权重通吃；synth 的 stage-1 训练分布字高下探 6~10px（stage-2 维持 16px 下沿=判别物理极限）。
5. **文献锚点**：任务本体=query-by-string word spotting（Wang CVPR'21 联合检测+相似度；PHOC 谱系——mask 模板本质可学习 PHOC）；stage-2 近亲=OS2D/CoAE one-shot 模板匹配检测 + SiamRPN 系 xcorr；stage-1 近亲=CRAFT/EAST/DBNet。DeepSeek 视觉三代（VL2 动态 tiling、OCR/DeepEncoder 先局部后压缩再全局、2026 视觉基元 CSA）核心教训：**分辨率预算只花在刀刃上，空间压缩绝不进判别路径**；一套权重多分辨率原生训练可抄。
6. **骨干参数效率路线**（用户拍板）：RepVGG 重参数化（训练多分支/导出融合零成本）+ SE + 深阶段深度可分离 + 参数向 P2/P3 集中——v2 已落地前两条，深阶段 DW 留作后续。单位参数智能超 ResNet-50 那代设计是行业常态（MobileNetV3 证据），非野心。

## 已确认的修正（按用户裁定）

1. **target 零遮挡**：贴片覆盖 target >15% 即重选；遮挡块与 target 零相交（4px margin）。旧版"半字也算目标"信号已根除。
2. **模糊限轻度**：运动模糊 ≤7px、高斯 ≤5、下采样 ≥0.45。人眼不可读的监督是白搭。
3. **背景多样化**：20% 随机色相、色块拼贴、条纹/圆点/棋盘；15% 文本低对比墨色（背景均值 ±[25,85]）。
4. 分配挪进 dataloader worker（GPU 88%，1.6G 显存，6min/epoch @50k/bs32）。

## 待办/下一步候选

- 若 f39_e45 仍不达门槛：① 100k 数据；② tpl_ch 48→64、P4 也做相关；③ mask 模板加字重抖动增广；④ 实拍小样集建立（手机拍货架/面单 20-30 张）。
- 达标后：导出 ONNX（core/export_onnx.py，对拍已过）→ 二期 Flutter 插件从零自研（vendor/glyph_det，ORT-Mobile/NNAPI/XNNPACK 选型实测，ncnn 备选；Dart 渲染 mask 会话级缓存；decode 用 core/decode.py 同语义移植）。

## 已知坑

- 重 roll 上限要按规模算概率：单次拒绝率 p 时上限 k 的全量失败期望 = N·p^k。8 次上限在 35% 拒绝率 ×50k 下必撞（崩在 #14849），已放宽到 64。
- Bash 工具里用 shell `&` 起后台进程会随 shell 退出被杀——必须 run_in_background=true。
- torch 2.11 ONNX 导出要 opset 18 + onnxscript；导出文件是 .onnx + .onnx.data 外部分离格式。
- pdm 2.26 的 use_uv 配置写法是 `.pdm.toml` 顶层 `use_uv = false`（[python] 节写法会报 NoConfigError）。
- 训练 90 分钟后台超时杀过首轮——长任务 timeout 给足 21600s。

## v2.2（2026-02-25）：监督信号审计与修复（未开训，等 GPU 休息）

- 实锤：v2 字高分配 × reg_max=8 → 68% 目标 ltrb 被 clip（s4 91%/s8 77%/s16 41%），recall@0.5 存在结构天花板；v1 免疫（maxd 边界=reg_max×stride 自洽）。报告：docs/supervision-analysis.md
- 修复：reg_max 8→24；分配加宽度护栏 level_for_box（中心带最远 0.75w 超 (reg_max-2)×s 则上浮）。验证：钳位 0%、上浮 9.2%、reg 实测 max 21.51<22.99、零静默丢目标
- 附带：centerness 实锤零信息（最优常数 BCE 0.670 vs 训练 0.61~0.63）→ HeadV2 砍 cen 通道（97ch），decode/eval 通道数自动判别兼容旧 ckpt
- 冒烟全绿（CPU）：97ch 输出、reparam diff 8e-7、双布局 decode、compute_loss cen=0
- run_name: mvp_v2_2，bs16，reg_max24 + 三级 xcorr + 无 cen —— 待开训
- 病根未修（留 v3）：match 逐格位置先验监督 → 模型只看单字图形不看整串全等（levelstat 86% 余弦检出 + fppeek 近邻串 0.86 同根）；方案 A per-char min 聚合头 / B ROI 判别头

### cen 移除的 toy 验证（tools/cenprobe.py，v1 last.pt + mvp4/val 100 张，CPU）
- with cen 0.9515 / without 0.9481 / oracle cen 0.9320 → 概念天花板为负，移除定案
- cen 预测 Pearson r=0.333（弱信号但非零），仅值 +0.003 AUROC；top-1 格点 94/100 变但指标不动
- 注：v1 官方 eval 数据集是 datasets/mvp4/val（不是 mvp/val），工具默认值曾踩错
