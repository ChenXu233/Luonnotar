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

## 注意

- `getOcrText()` / `getOcrResults()` 经 `NewStringUTF` 返回，非 ASCII 字符（无过滤时的中文）可能不符合 MUTF-8；使用时务必设置 char filter。
- 跟踪上游更新时，上述带 "Luonnotar fork" 注释的代码块即为全部差异点。
