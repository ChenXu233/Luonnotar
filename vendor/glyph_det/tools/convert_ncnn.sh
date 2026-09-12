#!/usr/bin/env bash
# glyphdet.onnx -> ncnn (param+bin) 转换 + 强制修复脚本
#
# 用法（Git Bash）：
#   pip install pnnx          # 建议装在独立 venv，pnnx 不随 ncnn release 分发
#   bash vendor/glyph_det/tools/convert_ncnn.sh
#
# 产物：vendor/glyph_det/example/assets/models/glyphdet.{param,bin}
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
ONNX="$REPO/glyphdet/runs/mvp_v2_2/glyphdet.onnx"
OUT_DIR="$REPO/vendor/glyph_det/example/assets/models"
JNI_DIR="$REPO/vendor/glyph_det/android/src/main/jni"

# Android 预编译 SDK（ncnn + opencv-mobile）被 .gitignore 排除，
# CMake 构建时需要它们在 jni 目录里；首次构建先跑 fetch 脚本。
mkdir -p "$OUT_DIR"
if [ ! -d "$JNI_DIR/ncnn-20260113-android-vulkan" ] || [ ! -d "$JNI_DIR/opencv-mobile-4.13.0-android" ]; then
    bash "$REPO/vendor/glyph_det/tools/fetch_android_deps.sh"
fi

pnnx "$ONNX" \
    pnnxparam="$OUT_DIR/glyphdet.pnnx.param" \
    pnnxbin="$OUT_DIR/glyphdet.pnnx.bin" \
    ncnnparam="$OUT_DIR/glyphdet.param" \
    ncnnbin="$OUT_DIR/glyphdet.bin" \
    fp16=0

# --- 必须的手工修复 -------------------------------------------------------
# pnnx 20260526 对动态深度卷积（mask 特征 reshape 成卷积权重）做形状推断时
# 会报 "fallback batch axis 233 for operand ..."，并把下游所有
# torch.sum(dim=1)（按通道求和）的 ncnn Reduction 层 axes 写成空数组
# (-23303=0)，使这些层变成 no-op：整网前向会在后面的 Concat 处崩溃
# （实测 ncnn python on Windows 段错误）。
# 本模型的 10 个受影响 Reduction 全部是 torch.sum(dim=1, keepdim=?)：
# 批维被 ncnn 去掉后，通道轴恒为 axis 0，统一补回 axes={0}：
sed -i 's/-23303=0 /-23303=1,0 /g' "$OUT_DIR/glyphdet.param"
# 注意：若模型结构变更后重新转换，必须先确认所有 "-23303=0 " 仍只属于
# 按通道求和的 Reduction 层（对照 glyphdet.pnnx.param 里的 torch.sum dim=1），
# 不能盲目套用本 sed。

# pnnx 中间产物（仅供排错对照，不进 APK）
rm -f "$OUT_DIR/glyphdet.pnnx.param" "$OUT_DIR/glyphdet.pnnx.bin"

# 同步一份到主 app assets（pubspec assets/models 路径）
APP_ASSETS="$REPO/assets/models"
mkdir -p "$APP_ASSETS"
cp -f "$OUT_DIR/glyphdet.param" "$APP_ASSETS/glyphdet.param"
cp -f "$OUT_DIR/glyphdet.bin"  "$APP_ASSETS/glyphdet.bin"

echo "converted -> $OUT_DIR/glyphdet.param / glyphdet.bin"
echo "synced   -> $APP_ASSETS/glyphdet.{param,bin}"
