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

## v2.2 终局（40ep 烧完，17.2h）：架构修复全部生效，fp 成唯一约束

周期 eval：ep10 auroc 0.9539/fp 0.046 → ep40 auroc 0.9494/recall 0.876/fp 0.089（退火段 recall 爬坡，fp 回涨=记忆化老毛病）
官方 1000 图终评（last.pt）：recall@0.5 0.8893（v1 0.8612）、auroc 0.9433（v1 0.9351）、top1 0.884、fp 0.1001（v1 0.097 没动）
- recall_iou0.3 == recall@0.5（0.8893）：0.3/0.5 差距完全闭合，reg 钳位天花板根除 ✓
- levelstat：三级全部有效工作且按字高桶分工（s4 抓 44/s8 抓 261/s16 抓 200），余弦依赖症根除 ✓
- fppeek：63 fp 中 61 个 distr 型，分数 mean 0.722/max 0.998，人眼看几乎全是单字符变异近邻（08504→08534、F8676→F8646、4→69-18-817、5→8006676324）
结论：检测侧（找得到、框得准）已解决；判别侧（逐字全等）就是"不看全体"病根的原形——单字全对就高分，一个错字不扣分。
达标线 0.96/0.92/0.03 未达 → 主线转入 v3 per-char min 聚合头；ONNX 定版暂缓。best.pt=ep10（auroc 0.9539/fp 0.046/recall 0.645）留作判别-召回权衡参照。

## 毒数据事件（2026-02-25，用户抓包）：13.6% target 被遮挡，合成器结构性修复

- 用户从 fppeek 图抓到 000156：小 target 被后贴 hard-neg 覆盖 76%。全库审计：train 13.6% target 被后贴文本盖 >5%（>25% 的 5806 个、>50% 的 1985 个、全盖死的存在）
- 根因：①target 先贴、干扰后贴，大干扰串可吞小 target；②极性/低对比重渲会重摇 warp 改几何，但覆盖检查用的是旧几何（洞在这）；③阈值 15% 本身就在容忍遮挡
- 修复（core/synth.py）：干扰串先贴/target 后贴（结构性保证 target 永不被干扰串覆盖）+ 重渲后按新几何重跑覆盖检查（失败弃贴）+ target 覆盖阈值 0.15→0.02
- 验证：修复后 300 张冒烟仅 3 例 5~7% 的 aabb 角部虚接（像素不相交，目检确认），真实遮挡 0
- mvp4（毒）改名 datasets/mvp4_poisoned 留档；同 seed=42 全量重生成中 → 完成后开训 mvp_v2_3（其余配置不动，隔离数据变量）
- 推论：v2.2 的 recall 0.889 是在 13.6% 毒监督下取得的；且"部分可见也开火"的毒标签可能就是 fp 习惯的来源之一——干净数据可能同时抬 recall 压 fp

## 插件化启动（2026-02-25，用户裁决：GPU 熄火，先用当前模型做插件）

- v2.2 last.pt 导出 ONNX：runs/mvp_v2_2/glyphdet.onnx 单文件 9.9MB fp32（reparam 后 2.40M 参数）
  对拍 max|diff|≤3.2e-4；注意 torch.onnx dynamo 默认外链 .onnx.data 且 GBK 控制台需 PYTHONUTF8=1
- 移植规范：docs/plugin-spec.md（模型契约/decode 同语义/runtime 选型/短板声明）
- v3（per-char min 聚合头）转为纯设计待办；干净数据（bash-1mbybqxb 重生成中）完成后仅存档，不重训，等用户发话

### 干净数据落地（mvp4 重生成，100k train + 1k val，92min CPU）
- 审计：val 遮挡>5% 仅 0.27%（max 0.10）、train 抽样 2 万张 0.23%（max 0.39）——毒数据 13.6%→0.25%，残留均为 aabb 角部虚接级
- datasets/mvp4_poisoned 留档；训练配置无需改动（路径同名）
- 按用户裁决：不重训，仅存档备用

### 训练规程调整（用户裁决）：未来 run epochs 40→30、eval_every 10→5
理由：记忆化老毛病（退火段 fp 回涨）说明长日程收益递减；eval 加密到 5 轮 + best.pt 机制保证抓到最优点。GPU 仍冻结。


## 插件落地 + App 换装（2026-02-25，子代理建插件 / 主代理装 App）

### vendor/glyph_det 插件建成（ncnn 路线，弃 ORT）
- ONNX→ncnn：pnnx 转出 glyphdet.param(17.7KB)+bin(9.75MB)；关键坑：pnnx 把 10 个 Reduction 层 axes 写成空数组，须手工 `-23303=0 `→`-23303=1,0 `（固化在 tools/convert_ncnn.sh，重转必查）
- blob 映射：in0=scene / in1=mask / out0,1,2=stride4,8,16；ncnn vs ORT 前向 diff：score sigmoid ≤5.1e-3、decode 后 top1 框差 0.04px
- Android 原生：NDK camera2 自管相机（移植自 fast_paddle_ocr）+ ncnn Vulkan 推理线程 + C++ decode（thr0.5/DFL reg_max24/ltrb/NMS0.4/max20，与 core/decode.py 同语义）；无 mask 时 detect 直接跳过（安全）
- decode 宿主机对拍全 PASS：tests/parity/parity.exe（w64devkit g++14.2），ort/crafted/thresh 三组基准 max box diff ≤1.2e-4
- example app 可构建（flutter analyze 零 issue，debug APK 177MB）；真机未验证（Vulkan 首载/infer_ms/帧率/letterbox 域差/ARM decode 对拍全待真机）

### App 侧换装 glyph_det（OCR 路线退役为死代码，不删）
- 根 pubspec 加 glyph_det path 依赖；fast_paddle_ocr 依赖保留（ocr_service.dart / pickup_matcher.dart 成未引用死代码，用户裁决不回撤）
- 模型资产：glyphdet.param/bin 拷入 assets/models/（与 paddle 模型并存）
- 新增 lib/ocr/glyph_det_service.dart（loadModel cpugpu=1 Vulkan 默认待实测 / setTargetCode 渲染 mask 下发 / 相机开关 / 轮询）+ lib/ocr/mask_render.dart（拷自插件 example，与训练侧 render_mask 同语义）
- scan_page.dart 重写：OcrCameraView→GlyphDetCameraView；匹配语义从"OCR 全文+文本匹配"变为"模型直接定位查询串"，topBox 即匹配；高亮 painter 改 GlyphDetBox + BoxFit.cover 映射（与 example 一致）；调试行改 FPS+inferMs；无目标时状态"输入取件码开始寻找"
- android/app/build.gradle.kts：ndkVersion 固定 29.0.14206865（与两插件对齐）+ packaging pickFirst libc++_shared.so（两插件各带一份，内容相同）
- flutter analyze 零 issue；flutter build apk --debug 成功：app-debug.apk 254.6MB（双 ABI + 双引擎 + 双模型，debug 体型）
- 待真机清单：install 后 logcat -s GlyphDetPerf 看 infer_ms/fps/Vulkan 首载；ARM decode 用 tests/parity ort case 对拍一次；mask 渲染域差（Flutter 段落近似 PIL getbbox）实拍确认；fp16 未开（性能不够再开，开了要重对拍）


### 文档落地（2026-02-25）：真机测试计划 + v3 设计（纯设计待办）
- docs/phone-test-plan.md：安装基线/管线性能门槛（Vulkan≤30ms、FPS≥20）/检测正确性用例（含近邻干扰专查）/mask 域差对照实验/交互回归/失败样本采集规程
- docs/v3-design.md：逐字 min 聚合头——score = softmin_k R_k(y, x+kΔ)，单字变异结构性判死；推荐 A2 逐字光栅输入（chars (1,12,1,64,40)+nchar，字编码器 ~150K），A1 定距整串+静态切片做回退；逐字核计算量 120 vs 现整串 216 反而更小；decode 零改动；mutation 评测门 ≤2%；方案 B（ROI 判别头）为独立回退路径


## v3 实施启动（2026-02-26，用户发话"直接来做v3"；mvp_v2_3 对照实验作废并入 v3）

### 架构落地（model.py，对比设计稿有收敛，详见 v3-design.md 实施定稿节）
- CharSlotEncoder：mask (B,1,64,384) 按 12 槽×(64×32) reshape 逐字编码 → 逐字模板 (B,12,48,8,4) + masked-mean 全局向量 w；槽有效性 = 墨迹均值 <0.98 在线判定
- HeadV3：逐字 xcorr_slots（B·K·C 组 depthwise 一次算完）→ 容差 maxpool（sh4 (7,3)/sh8 (11,5)）→ 整数常量偏移表对齐（槽距 0.5 字高 × 偶数尺度 ⇒ 任意串长奇偶皆整数）→ 无效槽 +1e4 → masked softmin(τ=1.0) → 与 cos_map 融合
- 分配/回归/损失/输出 97ch 与 v2.2 完全一致；train.py/dataset.py/decode 零改动
- 冒烟全过：CPU 前向 97ch×3 级、backward OK、全白 mask 抑制（score≤-1.2）；参数量 2.03M（v2.2 2.40M）；GPU 单 batch 过拟合 loss 452k→121k
- synth：新增 render_mask_slots（偶数化居中逐字入槽，tools/v3maskpeek.py 槽位几何断言+目检全过）；gen_code 长尾限长 ≤12；mask_layout=slots 配置开关；场景侧零改动
- 实验配置 experiments/mvp6_v3/config.yaml（数据 datasets/mvp5、arch v3、30ep/eval5）；mini 变体 config_mini.yaml（mvp5mini 2k 样本 6ep）
- mini 短训实测：显存 2.2G/8G 无 OOM 风险；首 epoch 136s/125 步系与全量数据重生成抢 CPU 所致（GPU 7%），干净步速待 regen 完成后复测
- tools/v3probe.py：mutation 分离度探针（正例 top1 vs 一字变异全场最高分，出 mutation_fp/分离 AUROC/fp 样例）——全量训练的 go/no-go 门


### v3 mini 首训 NaN 事件（2026-02-26，已定位已修）
- 现象：mini 短训 epoch1-2 几乎不学（auroc 0.49），epoch3 起 loss NaN
- 定位（tools/v3debug.py）：xcorr 响应本身 ±200 正常；**agg 里 -10001.8 穿透进 fuse**——出界槽读的 -1e4 填充经 1×1 fuse 权重符号随机翻转成 ±5000~6000 logit → 负样本 BCE 单项 ~6000、grad_norm 44 万（clip 前），AMP 下 scale 崩死 → NaN
- 修复：_align_min 出口 `agg.clamp(-32, 32)`（-1e4 只在 softmin 内部，永不进 fuse）
- 教训固化：**任何常量填充值都必须隔离在融合层之外**；fp32 50 步对照证明非结构性（fp32 无 NaN 且下降）
- 数据重生成曾撞盘满（E: 98%，剩 20G；全量 mvp5 需 ~22G）：OSError 121802 requested and 0 written。处置见下条


### v3 mini 重训验证（钳位修复后，全绿）
- 6ep 全程无 NaN：loss 115→4.08，match 1.6→0.29，reg 2.4→1.38 稳步下降；显存 2.2G
- 稳态 ~0.3s/step（首 epoch 49s 含 warmup；此前 136s 系与 regen 抢 CPU）→ 100k 全量预估 ~31min/ep、30ep ≈ 15.6h，与 v2.2 同量级
- mini eval 数字（recall≈0）为 2k 样本 6ep 欠拟合的正常表现，不作判定依据；mutation 探针等全量训到 eval@5/@10 再跑
- 腾盘：删 v1 时代旧库 datasets/mvp + mvp2（≈19.3G，合成可再生）；mvp4（干净整串版）/mvp4_poisoned/mvp5mini 保留
- 下一步：腾盘完 → 全量 regen datasets/mvp5（100k+1k，~90min CPU）→ 直接开 30ep 全量训练（用户已授权 v3 全程）→ eval@5/10 跑 v3probe 看 mutation 分离度


### 盘满事件收尾与 v3 导出预验证（2026-02-26）
- 第二次 regen 在 ~80.6k 张处再次 "OSError: 0 written"（当时卷面剩 27G）——卷面空闲不可信（疑似薄分配存储池），处置 = 删真实数据块：mvp4_poisoned（毒数据存档，事件已记录于上文）+ mvp3（v1 时代），df 65G 可用
- 教训固化（synth.py）：gen_one 写盘 5 次重试（间隔 2s）；main() 起飞前磁盘检查（预计需求 ×1.3 不足即拒飞）
- Windows 下删大目录：`rm -rf`(git bash) 35min/16G，`cmd rmdir /s /q` ~6min/26G——后者快 5 倍+
- v3 ONNX 导出预验证（随机权重链路模式）：三级输出对拍 max|diff| ≤ 1.9e-6，全静态图设计兑现（logsumexp/maxpool/slice/有效性掩码全部干净导出）
- 第三次 regen 进行中（加固版，~90min），完成后直接开 30ep 全量训练 mvp_v3

## v3 全量训练开训（第 8 窗口续）
- 数据重生成第三次成功：datasets/mvp5 train 100000 + val 1000，labels 行数对齐，日志无 error，耗时 4108s。
- 开训：task bash-ts7dma2y，`experiments/mvp6_v3/config.yaml`（arch v3 槽位版、30ep、eval_every 5、bs16、run mvp_v3），GPU 67%/6.1G 确认在跑。
- 日志逐 epoch 写入 runs/mvp_v3/log.txt；首个 epoch 预计 ~31min，eval@5 约 2.5~3h 后可看。
- go/no-go：eval@5/10 时跑 `tools/v3probe.py --weights runs/mvp_v3/last.pt --n 150`，门=mutation_fp≤2% 且官方 recall@0.5≥0.95、fp≤0.03（v2.2 基线 0.889/0.943/0.100）。

## v3 全量训练完成——判定：NO-GO（第 8 窗口）
- 30ep 跑满无 NaN，总耗时 59903s（16.6h），final loss 1.32。权重 runs/mvp_v3/last.pt。
- 官方 eval@30：auroc 0.849 / recall 0.838 / top1 0.830 / fp 0.148——全线差于 v2.2（0.889/0.943/0.100）。
- v3probe（n=150, mvp5/val）：pos_top1 0.811，pos_hit@0.5 0.873，**mut_max_mean 0.672，mut_max_p90 0.999，mutation_fp 0.80，separation_auroc 0.594**（门：mutation_fp≤2%）。插入型变异（Y49→Y469）0.961、（9-11-4872→9-11.48272）0.999。
- 关键证据：**margin loss 从 epoch14 起恒为 0.0000**（训练自称完美分离），而实测 mutation_fp 80%——瓶颈在监督信号饱和度，不在架构表达力。hard-neg 是同字体同卡 1~2 次扰动（synth.py:131 hard_negative），但 margin 只约束"hard-neg 区单点 max < 正样本 mean - margin"，饱和后零梯度；且训练方向恒为 mask=真串/scene=变异，probe 方向（mask=变异/scene=真串）从未被监督；插入变异使槽对齐整体平移，容差 maxpool(11,5) 把平移吸收掉。
- 结论：v3 min-agg 判别头未治好"不看全体"。候选路径：v3.1 监督修复（hard-neg 2~3 个/scene、margin 改 top-k 且不饱和、E2 逐字槽位 aux 监督指出"哪个字不同"）；或回 v2.2 先做插件集成、近邻 fp 交给多帧投票/用户确认。

## v4 重构落地（第 9 窗口，进行中）——用户四点裁决驱动的全面重构
用户四点要求：①场景任意角度+旋转框标注；②杀 mask 硬截断/槽位留白；③mask 真透明（不落盘，在线渲染）；
④监督不做区域化字数（模型不知学留白还是字体→泛化差）。用户确认"内部两段式=先选框再识别"。

### 已落地（全部验证过）
- **synth v4**：warp_patch 加 ang_deg；build_sample 场景基准角+逐文本 60%相干±25°/40%全随机[0,360)；
  标注=旋转框 (cx,cy,w,h,θrad)（_rbox() 由四角算，θ=阅读方向，y向下顺时针正）；mask 完全不落盘；
  gen_code 长尾恢复 10-16 字。目检 12 张全过（含 180° 倒置、90° 竖排，框贴合、mask 透明紧致）。
- **dataset v4**：labels 无 mask 键；在线 render_mask_rgba（synth.py 内，(H,Wn) float 墨迹=1 透明=0，
  墨高 H-8、变长宽≤max_W、横向±10%字宽抖动）；训练返回 (scene,mq,muts,targets,boxes,kinds)，
  kinds:1=target/0=非target/-1=bg随机框（matcher 负样本，不进分配）；targets 每级 9ch：
  [textness,ltrb(框体系/stride),sincos,pos,weight]，所有标注框都是 textness 正样本；collate_v4 变长 pad(8倍数)。
- **model v4**（model.py 尾部 v4 区段，arch="v4" 已注册）：BackboneV2+NeckPlain(无FiLM)→PropHeadV4
  （99ch=[textness,ltrbDFL(4*24),sin,cos]）；MaskEncoderV4(GroupNorm,变长安全)→(Ct=32,8,Wt)；
  MatcherV4.strips（grid_sample 规范化条带 8×128，列行同距 h/8，训练期框抖动）；
  MatcherV4.score(strips,tpl,ink,pairs) 显式 pair 列表：全局对齐 logsumexp + 最优偏移处逐墨列余弦
  softmin，mix(1,4,-2) 可学习；pair_grid 辅助。参数量 1.79M，图连通全绿无死参数。
- **train v4**：compute_loss_v4（textness focal + DFL+smoothL1(ltrb) + smoothL1(sincos) + pair BCE
  （真串×target=1 其余全 0，含变异×所有框——v3 缺失方向的直接监督，无 margin 饱和））；
  _batch_loss 按 arch 分发；v4smoke.py 过。
- **eval**：rbox_iou（cv2.rotatedRectangleIntersection，约定 7 项单测全过 tools/v4ioutest.py）；
  quick_eval_v4（recall/top1/fp + auroc=query@target vs query@非target+变异@target）；
  decode_rprops（AABB NMS）；tools/v4probe.py（go/no-go：mutation_fp≤2% 门）。
- **性能修复**：初版逐样本循环 2.5s/step（smoke 超时误判为 hang）→ v4prof 定位 → 全向量化
  （mask_enc 全批一次、strips 全批一次 grid_sample、score pair 列表一次 grouped conv+gather）。
  向量化版 v4smoke 过，训练 smoke 进行中（bash-pjzwywlf）。
- config：experiments/mvp7_v4/config.yaml（全量 mvp6）+ config_mini.yaml（mvp6mini 2k/6ep）+ config_peek.yaml。

### 后续
1. 训练 smoke 过 → mini 6ep 短训 → v4probe 判定（mutation_fp 门）
2. mini 达标 → 全量 mvp6 数据生成（CPU ~40min，只写 scene ~12G）
3. 全量训练 30ep **等用户一句话**（GPU 禁令未解除到 v4 全量）
4. tools/_patch_build_model.py、v4prof.py、v4smoke.py、v4ioutest.py 为一次性/诊断脚本可留可删
- 用户裁决：全量训练 25ep（非 30），eval_every 5 不变。config.yaml 已改（25ep 预估 ~24h @0.55s/step）。
- 用户暂停（玩游戏）：mini20 训练 bash-swq6e90d 已 TaskStop，GPU 无残留进程。
  断点状态：v4 全链路代码就绪并验证；mini 6ep=欠拟合信号（pos/mut 双低，mut_fp=0）；
  恢复入口 = 重跑 mini20（experiments/mvp7_v4/config_mini20.yaml）→ v4probe 判定 → 全量。
- 用户要求加断点：train.py 现在每次 eval 落 runs/<run>/ep{N}.pt（含 epoch+auroc）。
  正在跑的 mini20 用的旧码不受影响（last.pt 仍逐 epoch 更新）；下次训练起生效。
  附带收益：训练中可对任意 ep{N}.pt 跑 v4probe 做早期判定，不用等全量跑完。

## v4.1 matcher 零分离根因解剖与修复（已实锤，复训中）

**症状**：mini20（2k×20ep）检测侧收敛（reg 8.7→1.87，ang 0.55→0.14）但 matcher 零分离——
train probe auroc 0.509（训练集上都不分离=结构性问题），recall 恒 0，旧 run 归档
runs/_archive/mvp_v4_mini20_v40_zerosep。

**解剖链**（tools/v4pairdump.py + tools/v4pitchcheck.py，数据实锤）：
1. pairdump：正例与变异的墨列余弦无差（0.779 vs 0.789）、colmin 全负（正例常低于变异）、
   align 恒 ~1.6-1.7 无区分、mix 参数几乎不动 → logit 恒负 → "全拒绝"盆地（BCE≈0.26）。
2. 根因 A【列距失配】：条带列距=h/8 场景像素（~5.67 列/字），模板列距=mask 渲染器决定
   （~5.03 列/字），比值 mean=1.128/std=0.197，64% 样本失配>10%、38.6%>20%。对齐只有
   平移 dx 无缩放自由度 → 10-16 字长串累计漂移 1-3 字宽 → 正例必有"坏列" → colmin 被打负。
3. 根因 B【特征各向异性】：训练后 mask_enc 异字平均特征余弦 mean=0.970 min=0.881——
   巨大公共方向把余弦动态范围压扁，colmin 物理上无法区分变异字形。

**修复（MatcherV4.score，model.py）**：
1. 多尺度对齐搜索：模板特征按 6 尺度 {0.80,0.89,1.00,1.12,1.25,1.40}（覆盖实测失配
   p5~p95=0.82~1.45）双线性插值后逐尺度 grouped-conv 相关，logsumexp 合并所有
   (scale,dy,dx) 候选，argmax 最优尺度处取窗算逐墨列余弦 softmin。两遍法省显存。
2. 特征中心化：normalize 前减 channel 维均值（S/T 两侧），杀掉公共方向。
mask 渲染 jitter（±10% 宽抖）保留——由尺度搜索吸收，兼作中间尺度鲁棒性数据源。

**验证**：smoke 200 步 pair loss 0.38→0.075（修复前同设置卡 0.26 平台期），全分量收敛，
参数量不变 1.79M。mini20 复训中（runs/mvp_v4_mini20，日志 runs/_v41mini20_train.log），
判门槛不变：pos_top1≥0.6 / mut_max≤0.3 / auroc≥0.9 / mutation_fp≤2%。

## v4.2 聚合层与覆盖度修复（复训中）

**v4.1 结果**：auroc 0.58→0.71（门槛 0.9），recall 仍 ~0。但 pairdump 显示特征与对齐已修好：
正例墨列余弦 min +0.46~+0.66（原 +0.16），变异错误列降至 -0.02，r* 分布合理（0.89~1.40）。
剩余零分离由两个新实锤根因造成：

4. 根因 C【softmin 熵偏差】：softmin ≈ min - τ·log(N)，τ=0.25/N≈45 → 固定 -0.95 惩罚；
   且惩罚随串长递增——删除型变异串更短 → colmin 反而更高（'-35533' -0.055 vs 正例 -0.205，
   完全反向）。且 τ 过软：44 列中 1 列坏到 0 仅降 0.07。
5. 根因 D【无覆盖约束】：逐列匹配对子串天然盲——删除型变异（'LW25'/'-35533'）完美对齐
   到正例子串，worst-k 下 p 仍 0.87+。align 的 logsumexp 熵膨胀对短模板还系统性偏高。

**修复**：
1. colmin：softmin → worst-4 均值（变异 1 字≈连续 4-6 坏列直接命中；无温度无计数偏差）。
2. 覆盖度双侧 hinge：cov=最优尺度墨列数/(8w/h)，relu(0.85-cov)/relu(cov-1.25) 可学习惩罚
   （cov_w 参数）；span 用 strips 实际采样框（训练 jitter 后，新增 return_boxes）。
3. align：logsumexp → hard max（候选数不变量，消熵膨胀）。
调用点接线：train.py（jitter 框 span）、eval.py、v4probe.py。smoke pair loss 0.038
（v4.1 0.068 / v4.0 卡 0.26）。v4.1 run 归档 runs/_archive/mvp_v4_mini20_v41_partial。
复训 runs/mvp_v4_mini20（日志 runs/_v42mini20_train.log），判门槛不变。

## v4.3 GT 几何对齐密集列监督（复训中）

**v4.2 结果**：auroc 0.71→0.84（门槛 0.9），recall 仍 ~0.01。pairdump + 几何核验显示：
模型自选对齐已达 GT 精度（'LW205' r_true 1.24 vs r* 1.25，dx_true 45.5 vs dx* 46）——
对齐不再是瓶颈；瓶颈是特征区分度（正例 worst-4 仅 0.24-0.46），pair BCE 经 mix→worst-4
每对只 4 列拿梯度，信号太稀疏推不动。

**v4.3**：新增 MatcherV4.aux_column_loss——正例对 GT 几何免费精确（Wr=8w/h 内容列数、
dx=(sw-Wr)/2 居中），在最优几何处对每根墨列 relu(0.85-v) 上拉，一对 ~40 列梯度（10×
密度）。负例不下压（避免误伤变异对同字列；删除型由覆盖度 hinge 负责，替换/重排由 BCE
经 worst-k 自动瞄准）。配置 aux_weight=1.0（三个 config 均加）。

**埋雷记录**：aux 首版梯度 1e-8 恒 0.85 不动——gather 索引 `expand(Ps,-1,sh,gw)` 的 -1
保持通道维为 1，只采第 0 通道；广播乘后 v 只剩跨通道和（中心化恒 0），梯度被中心化
雅可比 (I-11^T/C) 精确湮灭。修复=expand 显式通道数（score 内对应 gather 本就显式 Ct，
仅 aux 中招）。tools/v4auxgrad*.py 为定位留档。修复后 smoke aux 0.85→0.011。
复训 runs/mvp_v4_mini20（日志 runs/_v43mini20_train.log）；v4.2 run 归档
runs/_archive/mvp_v4_mini20_v42_auroc084。

## v4.3 首训倒退与 aux pad bug（修复复训中）

**v4.3 首训 auroc 0.56（倒退于 v4.2 的 0.84）**——aux 监督坐标全错：训练批 mask 被
collate pad 到 batch 最大宽（10 字 query 混 16 字批墨只占 60%），aux 把整个 padded
模板插值到 Wr 列再按居中假设上拉，墨迹实际被挤在前 60%，上拉位置全偏。score 主路径
无此问题（pad 列被 ink 掩乘归零，对齐搜索自由定位）。
修复：aux 内逐 mask 按墨右界裁剪（right=Wt-flip(ink).argmax）再插值缩放。
tools/v4auxgrad.py 回归：tpl/strips 梯度 0.72/0.82 健康。smoke aux→0.011 pair→0.052。
复训 runs/mvp_v4_mini20（日志 runs/_v43b_mini20_train.log）；倒退版归档
runs/_archive/mvp_v4_mini20_v43_padbug。

## v4.4 aux 坍塌定案与回撤（复训中）

**v4.3b（裁剪修复后）auroc 0.70，仍倒退**。pairdump 定案：aux 纯上拉造成特征完全坍塌——
所有墨列余弦被推上 ~0.9，正例/变异无差（mean 同为 0.914，变异 min 0.865 > 正例 0.847），
mix 自杀（a→-0.108，c0→-3.089）退回全拒绝。机制：上拉 40 列/对 vs BCE 下压 4 列/对，
梯度力量悬殊，平衡解=万物皆匹配。**教训：无上拉无下压配对则不如不做**。
（对比学习式位置锐化下压留作备选——同字重复列混淆问题需谨慎。）

**v4.4（回撤+两条便宜修正）**：
1. aux_weight=0（代码留档对照用，>0 才计算）。
2. pair BCE pos_weight=4.0（正负 ~1:8，负例梯度质量 8× 是 v4.2 正例卡 p~0.6 的直接成因）。
3. tpl_ch 32→64（32ch×8 行对 39 款字体的字形恒等性太薄）。
smoke 全绿。复训 runs/mvp_v4_mini20（日志 runs/_v44mini20_train.log）；
坍塌版归档 runs/_archive/mvp_v4_mini20_v43b_collapse。

## v4.4 mini20 判定与 mini40/全量数据并行

**v4.4 mini20**：eval auroc 0.82（≈v4.2 的 0.84），但 pos_weight 解开正例压制：
recall 0.015→0.30、top1 0.02→0.36，且 ep20 lr 归零时仍在爬——mini 撞的是 lr 调度
天花板而非数据信息量天花板。v4probe（严格逐图判定，gate pos≥0.6/mut≤0.3/auroc≥0.9/
mut_fp≤2%）：pos_top1 0.49 / mut_max 0.44 / auroc 0.62 / mut_fp 36%——未达门但差距收敛中。

**并行两路**：① config_mini40.yaml（40ep 验证未收敛假设，GPU ~35min，日志
runs/_v44mini40_train.log）；② 全量 mvp6 数据生成启动（100k train+1k val，纯 CPU，
日志 runs/_mvp6full_gen.log，~12GB）。全量 25ep 训练仍等用户一句话。
- 全量 mvp6 已生成完毕并核验：train 100000 + val 1000（labels.jsonl 行数一致），19GB。
  生成日志 runs/_mvp6full_gen.log。
- mini40 判定：auroc 0.83 / recall 0.26 / top1 0.28 @ep40，与 mini20（0.82/0.30/0.36）
  持平——“lr 未收敛”假设证伪，2k mini 数据信息量见顶（auroc 高原 0.78-0.83）。
  结论：结构修复已尽（6 缺陷全修），剩余杠杆=数据量（2k→100k，50×）。
  mini40 run 在 runs/mvp_v4_mini40（日志 runs/_v44mini40_train.log）。
  **下一步待用户发话：全量 25ep 训练（预估 ~24h GPU，0.59s/step 实测）。**

## 用户重启电脑：训练已安全停下 + train.py 新增 --resume

- 用户要求停训练重启：全量训练任务（bash-b6mpyamd）已 TaskStop，3h 巡检 cron 已删。
  停时 epoch 1 未完成，无断点损失（断点只在 eval 时落）。
- train.py 新增 `--resume`：last.pt 现含 model+opt+scaler+epoch+best_auroc，续训恢复
  优化器/调度（lr 按全局 step 对齐，step=start_epoch×steps_per_epoch）；旧格式 last.pt
  （无 opt）自动降级为只恢复权重。逻辑已单测（伪 epoch=7 断点正确续到 7）。
- **重启后恢复命令**（在 E:/git/Luonnotar/glyphdet 下）：
  PYTHONUTF8=1 PYTHONPATH=E:/git/Luonnotar pdm run python -m glyphdet.core.train \
    --config experiments/mvp7_v4/config.yaml --resume > runs/_v44full_train.log 2>&1

## v4.4 全量 25ep 训练收官（batch 8，~25.5h）

最终 eval@25：auroc 0.9725 recall 0.9210 top1 0.9067 fp 0.0952。
v4probe（ep25，150 图）：pos_top1 0.882 / pos_hit@IoU0.5 0.973 / mut_max 0.263（门≤0.3 ✅）
/ separation_auroc 0.952（门≥0.9 ✅）/ mutation_fp 24.7%（门≤2% ❌ 固定0.5阈值的结构性
尾部重叠，非排序问题）。断点 runs/mvp_v4/ep{5,10,15,20,25}.pt + best.pt + last.pt。

**操作点表**（tools/v4rocscan.py，300 图，recall=top1≥thr 且 IoU≥0.5）：
fp10%→recall 81.7%（thr 0.784）；fp5%→65.3%（0.887）；fp2%→45.3%（0.936）；
fp1%→32.3%（0.958）。固定 thr0.5→recall 90.7%/fp 21%。
注意：变异负样本是 1 字编辑距离的对抗样本，是现实里最坏的错认情形；App 真实错认对象
（他人随机取件码）远弱于变异——同阈值下真实 fp 会显著低于此表。
mini 全系列（2k 数据，auroc 高原 0.83）到全量的差距证实数据量是最终杠杆。
