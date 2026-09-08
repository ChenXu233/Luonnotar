# fast_paddle_ocr — Luonnotar fork

Vendored from https://github.com/Saifulkamil/Fast-Paddle-OCR-with-NCNN (main @ 19045ba, CC0).

## Fork modifications

1. **`getOcrResults()` API**（新增，取件码高亮的核心依赖）
   - `android/src/main/jni/ppocrv5ncnn.cpp`
     - 新增 `decode_object_text()` 辅助函数（复用 char_filter 规则，与 rec 线程/ocrFromImage 解码一致）
     - `MyNdkCamera` 新增 `latest_results` / `latest_results_lock` / `latest_frame_w/h` 成员，在 `on_image_render()` 的 IoU 合并之后快照
     - 新增 `MyNdkCamera::get_results_json()`：序列化为 `{"w","h","results":[{"text","prob","rprob","cx","cy","w","h","angle"}]}`，坐标基于渲染 rgb 帧空间（竖屏时与预览仅差均匀缩放）
     - 新增 JNI 导出 `Java_com_iweka_ocr_PPOCRv5Ncnn_getOcrResults`
   - `android/src/main/kotlin/com/iweka/ocr/PPOCRv5Ncnn.kt`：`external fun getOcrResults(): String`
   - `android/src/main/kotlin/com/iweka/ocr/OcrPlugin.kt`：`"getOcrResults"` channel 分支
   - `lib/ocr_result.dart`（新增）：`OcrResultBox` / `OcrFrame` 模型 + JSON 解析
   - `lib/ocr_platform_interface.dart`、`lib/ocr_method_channel.dart`、`lib/ocr.dart`：透出新 API（`Ocr.getOcrResults()` 返回解析好的 `OcrFrame`）

2. **连字符过滤**——无需 fork 修改：上游 main 分支已提供 `setCharFilter()`，App 启动时调用 `setCharFilter('0123456789-')` 即可。

3. **预览帧原生绘制清理 + FPS 外露**
   - `ppocrv5ncnn.cpp`：移除实时模式下的 `PPOCRv5::draw()` 调用（原生检测框/文字不再画到预览，高亮由 Flutter overlay 绘制）；`draw_fps()` 改为 `update_render_fps()`，仅计算渲染帧率滑动平均存入 `g_render_fps`，经 `get_results_json()` 的 `fps` 字段透出（不再在画面上绘制 FPS 文字）
   - `lib/ocr_result.dart`：`OcrFrame` 新增 `fps` 字段

4. **前置摄像头去镜像**
   - `ndkcamera.cpp`：前置相机（facing==0）帧在旋转后追加 `cv::flip(rgb, rgb, 1)` 水平翻转（`NdkCamera::on_image` 与 `NdkCameraWindow::on_image` 的 ROI 帧/全帧捕获三处），修复前置预览文字镜像导致 OCR 无法识别的问题；代价是前置预览不再是自拍镜像视角
   - 相机朝向约定不变：0=前置 1=后置，App 默认后置

5. **全速 OCR 管线（去防抖）+ 性能埋点**
   - `ppocrv5ncnn.cpp`：删除 `det_thread_loop` 的 `sleep(10)` 与 `rec_thread_loop` 的 `sleep(300)`——上游的"防抖"节流导致移动时文字结果延迟数秒；顺带移除从未使用的 `OCR_THROTTLE_MS` 死代码
   - 新增 EMA 性能计数 `g_det_ms` / `g_rec_ms` / `g_rec_boxes`，经 `get_results_json()` 透出（`det_ms`/`rec_ms`/`nbox` 字段），渲染线程约每秒输出一条 `LuonnotarPerf` logcat
   - `lib/ocr_result.dart`：`OcrFrame` 新增 `fps`/`detMs`/`recMs`/`recBoxes`

6. **平台视图改 TLHC**：`lib/ocr_camera_view.dart` 由默认 VirtualDisplay 的 `AndroidView` 改为 `PlatformViewLink + initSurfaceAndroidView`，消除 25fps SurfaceView 在 VD 下的二次合成卡顿，App 默认后置

## 注意

- `getOcrText()` / `getOcrResults()` 经 `NewStringUTF` 返回，非 ASCII 字符（无过滤时的中文）可能不符合 MUTF-8；使用时务必设置 char filter。
- 跟踪上游更新时，上述带 "Luonnotar fork" 注释的代码块即为全部差异点。
