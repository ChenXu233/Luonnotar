# glyph_det

掩码条件化字形检测 Flutter 插件（ncnn + Vulkan 优先，可切 CPU）。
输入一段取件码文本的查询掩码（mask），在实时相机画面里定位该码所在的
包裹标签并返回框坐标。替代被判失败的 `vendor/fast_paddle_ocr` 路线。

模型：`glyphdet/runs/mvp_v2_2/glyphdet.onnx`（单文件 fp32，9.9MB，
已 reparam 为单 3×3 卷积部署形态），经 pnnx 转为 ncnn param+bin。

## 架构

```
Dart (example app)
  │  取件码文本 ──renderQueryMask()──► 64×384 灰度 mask (Uint8List)
  │  loadModel / setQueryMask / openCamera / pollResults (MethodChannel "glyph_det")
  ▼
Kotlin  GlyphDetPlugin ── JNI ── GlyphDetNcnn
  ▼
C++ (libglyphdet.so)
  NdkCameraWindow (camera2 NDK, 720p NV21 → ROI crop/rotate → RGB)
    ├─ proc 线程：on_image_render → 喂推理线程 + 渲染 ANativeWindow 预览
    └─ infer 线程：letterbox 416² RGB [0,1] ─┐
                                             ├─ ncnn Net (Vulkan, cpugpu=1)
                       mask /255 ────────────┘   in0=scene in1=mask
                                             out0/1/2 (97ch, stride 4/8/16)
                                             decode.cpp (DFL+NMS) → 框缓存
  pollResults ◄── JSON {"w","h","fps","infer_ms","nbox","results":[...]}
```

相机管线（ndkcamera.cpp/h）移植自 fast_paddle_ocr 的 ncnn 官方示例风格：
camera2 NDK 自管相机、加速计定向、ROI 裁剪适配窗口宽高比、
ANativeWindow 直接上屏（Flutter 侧 TLHC PlatformView 叠加高亮层）。

## Dart API（`package:glyph_det/glyph_det.dart`）

| 方法 | 说明 |
| --- | --- |
| `loadModel(paramPath, binPath, {cpugpu=1})` | 加载 ncnn 模型文件。`cpugpu=1` Vulkan（默认；首次加载编译着色器要数秒，原生侧在后台线程执行），`0` CPU |
| `setQueryMask(Uint8List gray)` | 64×384 灰度字节（白底黑字），长度必须 24576。渲染工具见 `example/lib/mask_render.dart` |
| `openCamera(facing)` / `closeCamera()` | `facing`: 0=前置 1=后置 |
| `toggleFlash()` | 切换补光灯 |
| `pollResults()` → `GlyphDetFrame` | `{"w","h","fps","infer_ms","nbox","results":[{"score","cx","cy","w","h","angle"}]}`，坐标系=相机帧，按分数降序，最多 20 框；`frame.topBox` 即最高分框 |
| `GlyphDetCameraView` | 原生相机预览 widget（`glyph_det_camera_view`） |

模型契约：输入 `scene` (1,3,416,416) RGB [0,1]、`mask` (1,1,64,384) 灰度
[0,1] 白底黑字；输出 `out4/out8/out16` (1,97,H,W)，通道布局
`[score_logit(1), reg_dfl(4×24)]`（reg_max=24，无 centerness）。

## ncnn 转换（onnx → param+bin）

pnnx 不随 ncnn release 分发；ncnn 预编译包里也没有 onnx2ncnn。
步骤（详见 `tools/convert_ncnn.sh`）：

```bash
pip install pnnx   # 独立 venv
bash vendor/glyph_det/tools/convert_ncnn.sh
```

转换结果（pnnx 20260526）：208 层全部为标准 ncnn 层
（Convolution / Swish / BinaryOp / Reduction / Crop / ExpandDims / Interp /
Normalize / Concat / ConvolutionDepthWise / Reshape / Slice / InnerProduct），
**无 unsupported 层**。

**必须的转换后修复（已写进脚本，重转模型时别漏）**：pnnx 对动态深度卷积
（mask 特征 reshape 成卷积权重）形状推断报 `fallback batch axis 233`，
并把下游 10 个 `torch.sum(dim=1)` 的 ncnn Reduction 层 axes 写成空数组
（`-23303=0`），使这些层退化为 no-op——未修复时整网前向在 Concat 处崩溃
（ncnn python 实测段错误）。修复：这些层全部是按通道求和，批维去掉后
axes 恒为 `{0}`，即 `-23303=0 ` → `-23303=1,0 `。

输入/输出 blob 名映射（pnnx 重命名）：`in0=scene`、`in1=mask`；
`out0=out4(stride 4)`、`out1=out8(stride 8)`、`out2=out16(stride 16)`。

转换数值验证（ncnn python on Windows, CPU）：同输入下三级输出 vs ORT
max|diff| ≤ 8.5e-2（logit 域，winograd/插值实现噪声），score 通道
sigmoid 差 ≤ 5.1e-3；经 decode.py 解码 20/20 框一一对应，top-1 框差 0.04px。

## decode 对拍（C++ vs `glyphdet/core/decode.py`）

`android/src/main/jni/decode.cpp` 与 Android JNI 共用同一份实现，语义逐行
对齐 decode.py：sigmoid(score) > 0.5 严格保留、DFL softmax 期望
（reg_max=24）、ltrb×stride、格心±l/t/r/b、三级汇总、贪心 NMS
（分数降序，IoU > 0.4 抑制）、最多 20 框。

基准数据生成（固定 seed 随机 scene/mask → ORT 前向 → decode.py 解码；
另加手工构造覆盖阈值边界/DFL 小数期望/跨尺度 NMS/max_det 截断）：

```bash
cd glyphdet
PYTHONPATH=E:/git/Luonnotar pdm run python ../vendor/glyph_det/tests/parity/make_parity.py
```

宿主机对拍（w64devkit g++，单文件无第三方依赖）：

```bash
cd vendor/glyph_det/tests/parity
g++ -O2 -std=c++11 -I../../android/src/main/jni \
    parity_main.cpp ../../android/src/main/jni/decode.cpp -o parity.exe
./parity.exe
```

当前结果（容差 1e-3）：

```
[ort]     boxes got=20 expect=20 max|box diff|=0.000122 max|score diff|=0 -> PASS
[crafted] boxes got=20 expect=20 max|box diff|=3.81e-06 max|score diff|=0 -> PASS
[thresh]  boxes got=1  expect=1  max|box diff|=3.81e-06 max|score diff|=0 -> PASS
PARITY: PASS
```

## example 应用

`example/`：输入取件码 → `renderQueryMask()` 用 Flutter 画布渲染 64×384
白底黑字 mask（与 `core/synth.py: render_mask` 同语义：整串渲染、等比缩放、
左对齐垂直居中、超长等比缩小不压扁）→ setQueryMask → 相机预览 +
最高分框高亮（其余框淡显）+ FPS / infer_ms / nbox 显示。

```bash
cd vendor/glyph_det/example
flutter pub get
flutter build apk --debug   # 需要 Android SDK + NDK 29.0.14206865
```

`flutter analyze`：插件 + example 无 issue。

## 已知短板（照 `glyphdet/docs/plugin-spec.md` 第 5 节，别装不知道）

- **fp ≈ 0.10**：单字符变异近邻串会误高亮（如 08504 vs 08534，最高 0.998 分）。
  病根=逐格监督不强制整串全等，v3（per-char min 聚合头）解决。
- **recall 0.889**：约 1/9 的样本漏检；运动模糊/极端透视下更差。
- 当前模型用于**打通链路**（相机→前处理→推理→decode→高亮），精度迭代等 v3。
- 训练数据是全幅 416² 合成图，letterbox 黑边属分布外输入，上线前需用真实
  截图回归验证一次（glyphdet 的 failpeek 工具可复用）。

## 遗留问题 / 待办

- 未做真机验证（当前无 adb 设备）：Vulkan 首次加载时长、infer_ms、
  帧率、以及"相机帧 letterbox 黑边"的分布外影响都需真机回归
  （`adb logcat -s GlyphDetPerf` 看每秒 fps/infer/boxes）。
- ncnn 侧 fp16 未开（`use_fp16_arithmetic=false`，先保精度）；性能不够再开
  fp16 或做 ncnn2int8 量化，需重新对拍。
- decode 对拍在宿主机（g++）通过；真机 ARM 路径（NEON/Vulkan 下载回的
  fp32 输出）尚未逐位对拍，建议首台设备上跑一次 ort case 比对。
- example 的 mask 渲染用 Flutter 段落 ink bbox 近似 PIL `getbbox`，
  与训练字体渲染存在域差（模型按 held-out 字体训练，预期可泛化，
  但真机效果需回归确认）。
- 插件仅 Android（arm64-v8a / armeabi-v7a），无 Windows/iOS 端。
