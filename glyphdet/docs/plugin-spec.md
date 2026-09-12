# glyph_det 插件识别规范（当前模型 v2.2）

模型交付物：`glyphdet/runs/mvp_v2_2/glyphdet.onnx`（单文件 fp32，9.9MB，2.40M 参数，
已 reparam 为单 3×3 卷积部署形态；ORT 对拍 max|diff| ≤ 3.2e-4）。

## 1. 模型契约

- 输入 `scene`：(1,3,416,416) float32，RGB，[0,1]（相机帧 letterbox 到 416²）
- 输入 `mask`：(1,1,64,384) float32，灰度 [0,1]，**白底黑字**的整串取件码标准渲染
  （与 `core/synth.py: render_mask` 同语义；字体用常规无衬线即可，模型对字体泛化按
  held-out 字体训练）
- 输出三个尺度（无 centerness 通道，v2.2 起）：
  - `out4` (1,97,104,104)、`out8` (1,97,52,52)、`out16` (1,97,26,26)
  - 通道布局 `[score_logit(1), reg_dfl(4×24=96)]`，reg_max=24

## 2. decode 算法（参考实现 `core/decode.py`，移植必须同语义）

每尺度：score = sigmoid(out[0])；保留 score > 0.5 的格点；
reg 96 通道 reshape (4,24)，沿 24 softmax 取期望 → ltrb（格单位）×stride；
框 = 格心 ± l/t/r/b；汇总三级后 torchvision-NMS(IoU 0.4)，最多 20 框。
最终分 = sigmoid(score_logit)。App 语义：只高亮最高分框（top-1）。

## 3. 前后处理

- 前处理：相机帧 → 保持长宽比 resize 进 416²（短边填充黑边），BGR→RGB，/255。
  注意：训练数据是全幅 416² 合成图，letterbox 引入的黑边属于分布外，
  上线前用真实截图回归验证一次（failpeek 工具可复用）。
- mask：输入取件码文本 → 标准字体渲染 64×384 白底黑字 PNG → 灰度 /255。
- 后处理：top-1 框 + 分数；分数 < 0.5 视为未检出（不高亮）。

## 4. 运行时选型（按优先级实测）

1. onnxruntime-mobile + NNAPI EP（吃手机 NPU/DSP）
2. onnxruntime-mobile + XNNPACK EP（CPU 兜底）
3. 备选：onnx2ncnn 转 ncnn（参考级，非必需——不需要学 fast_paddle_ocr）

性能预算：PC GPU 15ms/帧；手机目标单帧 <40ms（异步管线内跑满 25fps 相机）。
体积优化（后续）：fp16 量化 → ~5MB；int8 → ~2.5MB（需回归验证精度）。

## 5. 当前模型的已知短板（写进插件 README，别装不知道）

- fp ≈ 0.10：单字符变异近邻串会误高亮（如 08504 vs 08534，最高 0.998 分）。
  病根=逐格监督不强制整串全等，v3（per-char min 聚合头）解决，见 progress.md。
- recall 0.889：约 1/9 的样本漏检；运动模糊/极端透视下更差。
- 当前模型用于**打通链路**（相机→前处理→推理→decode→高亮），精度迭代等 v3。

## 6. 待做

- [ ] vendor/glyph_det Flutter 插件骨架（FFI/platform channel 选型）
- [ ] C++ 侧 decode 移植（与 decode.py 对拍：同一输出张量 → 框逐位一致）
- [ ] 真机链路：adb 截图回归 + 帧率实测
- [ ] 数据重生成完成后存档（不训练）；v3 设计文档（待办）
