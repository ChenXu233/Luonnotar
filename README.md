# 努昂诺塔/Luonnotar

快递取件视觉导航助手：手机相机实时识别快递货架上的取件码（如 `23-5-1234`），在画面中高亮目标位置。设计方案见 [docs/AI-report/report.md](docs/AI-report/report.md)。

## 技术栈

- Flutter（Android MVP，minSdk 24，仅 ARM 真机 arm64-v8a/armeabi-v7a）
- OCR：`fast_paddle_ocr`（PP-OCRv5 mobile + NCNN），vendored fork 在 [vendor/fast_paddle_ocr](vendor/fast_paddle_ocr)，改动说明见 [vendor/README.md](vendor/README.md)

## 构建

```bash
# 需要 Android SDK + NDK 29.0.14206865 + CMake 3.31.5（插件原生构建钉死的版本）
flutter pub get
flutter build apk --debug   # 或连接真机后 flutter run
```

## 结构

- `lib/pages/scan_page.dart` — 扫描主页（相机预览 + 目标高亮 overlay + 玻璃拟态 UI）
- `lib/matching/pickup_matcher.dart` — 取件码匹配引擎（邻近框合并 + 多级匹配，含单测）
- `lib/ocr/ocr_service.dart` — OCR 模型初始化与轮询封装
- `lib/history/history_store.dart` — 取件码历史（shared_preferences）
- `lib/theme/frost_theme.dart` — 霜月主题（扁平 + 轻玻璃拟态，色板取自 logo）
- `assets/models/` — PP-OCRv5 NCNN 模型（首次运行拷贝到文档目录）
- `assets/images/` — logo SVG
