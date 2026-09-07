<p align="center">
  <img src="docs/assert/logo.svg" alt="Luonnotar Logo" width="180">
</p>

<h1 align="center">Luonnotar · 努昂诺塔</h1>

<p align="center">
  快递取件视觉导航助手 —— 相机实时识别货架上的取件码，在画面中直接高亮你的包裹位置
</p>

<p align="center">
  <a href="https://flutter.dev"><img src="https://img.shields.io/badge/Flutter-3.12+-02569B?logo=flutter&logoColor=white" alt="Flutter"></a>
  <a href="https://developer.android.com"><img src="https://img.shields.io/badge/Android-7.0%2B%20(ARM)-3DDC84?logo=android&logoColor=white" alt="Android"></a>
  <a href="https://github.com/PaddlePaddle/PaddleOCR"><img src="https://img.shields.io/badge/OCR-PP--OCRv5%20%2B%20NCNN-orange" alt="OCR"></a>
</p>

---

## 这是什么

取快递的最后一米是个视觉问题：驿站货架上堆满包裹，取件码（如 `23-5-1234`）印在小小的标签上，逐件翻找费时费力。Luonnotar 把"找快递"变成"看一眼"：

1. 输入你的取件码；
2. 举起手机扫过货架；
3. 匹配的包裹标签会在实时画面中**高亮框出**。

完全离线、纯用户端，不依赖驿站的任何设施（如智能灯条），菜鸟、中通、圆通、极兔……任何有文字编码的货架都能用。设计方案与产品分析见 [docs/AI-report/report.md](docs/AI-report/report.md)。

## 名字由来

> 努昂诺塔（Luonnotar）是游戏《[原神](https://zh.moegirl.tw/%E5%8A%AA%E6%98%82%E8%AF%BA%E5%A1%94)》中一只最特殊的月灵：月神哥伦比娅进入月之门后，在逆流的时间里用自己的一缕灵魂创造了它，让它**指引旅行者找到藏身在银月之庭的自己**。
>
> 这个名字本身则来自芬兰史诗《卡勒瓦拉》——自然女神 Luonnotar，传说中世界由她膝上的蛋碎裂而诞生。

一只为迷途者指路的月灵，如今引导你在快递山里找到自己的包裹 —— 这就是本项目的全部私心。 app 的"霜月"主题配色也取自 Logo 中的月色。

## 功能特性

- **实时相机流 OCR**：基于 PP-OCRv5 mobile + NCNN，离线毫秒级识别，预览 60 FPS 流畅不卡（识别线程限流 ~3 FPS 控制发热，检测框 15–20 FPS 响应）
- **目标高亮标注**：匹配到的取件码在实时画面中框选高亮，横屏移动时由 IoU 追踪器锁定
- **多级匹配引擎**：规范化比对、同行邻近框自动拼接（`23-5` 与 `1234` 分成两框也能匹配）、精确 / 部分 / 模糊（编辑距离 ≤ 1）/ 尾号兜底四级命中
- **字符白名单**：取件码仅含数字与连字符，`setCharFilter('0123456789-')` 既防连字符被吞又提升准确率
- **取件历史**：自动保存最近 50 条取件码，去重置顶，支持标记"已取件"
- **手电筒**：昏暗货架环境一键补光
- **霜月主题**：扁平 + 轻玻璃拟态 UI，竖屏锁定保证坐标映射稳定

## 技术栈

| 维度 | 方案 |
| :--- | :--- |
| UI 框架 | Flutter（Android MVP，minSdk 24，ARM 真机 arm64-v8a / armeabi-v7a） |
| OCR 引擎 | [`fast_paddle_ocr`](vendor/fast_paddle_ocr)（PP-OCRv5 + NCNN，vendored fork） |
| 状态/存储 | shared_preferences（历史记录）、path_provider（模型落盘） |
| 权限 | permission_handler |

OCR 插件以 **Git 子模块**形式引用[自己的 fork](https://github.com/ChenXu233/Fast-Paddle-OCR-with-NCNN)，新增了 `getOcrResults()` API（取件码高亮的核心依赖），改动明细见 [vendor/README.md](vendor/README.md)。

## 项目结构

```
lib/
├── main.dart                  # 入口：竖屏锁定 + 霜月主题
├── pages/scan_page.dart       # 扫描主页（相机预览 + 高亮 overlay）
├── pages/history_sheet.dart   # 取件历史抽屉
├── ocr/ocr_service.dart       # OCR 模型初始化、相机开关、结果轮询
├── matching/pickup_matcher.dart  # 取件码匹配引擎（含单测）
├── history/history_store.dart # 历史存储（shared_preferences）
└── theme/frost_theme.dart     # 霜月主题
assets/
├── models/                    # PP-OCRv5 NCNN 模型（首次运行拷贝至文档目录）
└── images/                    # Logo SVG
vendor/
└── fast_paddle_ocr/           # OCR 插件（Git 子模块，fork 自上游并自有改造）
```

## 快速开始

**环境要求**：Flutter SDK ^3.12.0 · Android SDK + NDK 29.0.14206865 + CMake 3.31.5（插件原生构建钉死的版本）· Android 7.0+ ARM 真机（模拟器无相机流）

```bash
# 1. 克隆（必须 --recursive，OCR 插件是子模块）
git clone --recursive https://github.com/ChenXu233/Luonnotar.git
cd Luonnotar
# 若克隆时忘了 --recursive：
git submodule update --init

# 2. 安装依赖
flutter pub get

# 3. 连接真机运行（或 flutter build apk --debug）
flutter run
```

首次启动会把 NCNN 模型从 assets 拷贝到应用文档目录，随后完全离线运行。

## 测试

```bash
flutter test   # 匹配引擎单元测试：test/pickup_matcher_test.dart
```

## 子模块与上游同步

- **修改插件**：在 `vendor/fast_paddle_ocr` 内正常 commit 并 push 到 fork，然后回本仓库 `git add vendor/fast_paddle_ocr && git commit` 拨动指针。
- **同步上游**：`cd vendor/fast_paddle_ocr && git fetch upstream && git merge upstream/main`，冲突点即 fork 差异（见 [vendor/README.md](vendor/README.md)），解决后 push 并回主仓库拨指针。

## 路线图

- [x] 实时扫描 + 取件码高亮
- [x] 取件历史记录
- [x] 手电筒补光
- [ ] 识别到目标后语音播报（对老年人友好）
- [ ] 照片模式（先拍后识别，适合库房盘点）
- [ ] 多包裹批量查找
- [ ] 序号预测（根据已识别序列推断未显示目标的位置）
- [ ] iOS 支持（双引擎方案：Apple Vision / ML Kit）

完整规划见 [设计方案报告](docs/AI-report/report.md)。

## 贡献

欢迎 Issue 和 PR。涉及 OCR 插件的改动请提交至 [fork 仓库](https://github.com/ChenXu233/Fast-Paddle-OCR-with-NCNN)，并在 [vendor/README.md](vendor/README.md) 中同步记录差异点。

## 致谢

- [Saifulkamil/Fast-Paddle-OCR-with-NCNN](https://github.com/Saifulkamil/Fast-Paddle-OCR-with-NCNN)（CC0）—— 本项目的 OCR 引擎基础
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 与 [NCNN](https://github.com/Tencent/ncnn) —— 端侧推理能力
- 米哈游《原神》—— 名字与灵感来源

## License

主仓库许可证待定（TBD）。`vendor/fast_paddle_ocr` 遵循其上游的 CC0 协议，fork 改动部分同样以 CC0 发布。
