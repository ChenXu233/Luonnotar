#!/usr/bin/env bash
# 拉 ncnn + opencv-mobile Android 预编译 SDK 到 jni 目录
#
# 用法（Git Bash / Linux）：
#   bash vendor/glyph_det/tools/fetch_android_deps.sh
#
# 锚定到固定版本以保证可复现构建；如需升级，同时改：
#   - NCNN_VERSION / NCNN_ASSET
#   - OPENCV_MOBILE_TAG / OPENCV_MOBILE_ASSET
# 然后清掉旧目录再跑本脚本。
#
# 依赖：curl + unzip（Git Bash / Linux 自带）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JNI_DIR="$SCRIPT_DIR/../android/src/main/jni"
cd "$JNI_DIR"

NCNN_VERSION="20260113"
NCNN_ASSET="ncnn-${NCNN_VERSION}-android-vulkan.zip"
NCNN_DIR="ncnn-${NCNN_VERSION}-android-vulkan"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_VERSION}/${NCNN_ASSET}"

OPENCV_MOBILE_TAG="v35"
OPENCV_MOBILE_ASSET="opencv-mobile-4.13.0-android.zip"
OPENCV_MOBILE_DIR="opencv-mobile-4.13.0-android"
OPENCV_MOBILE_URL="https://github.com/nihui/opencv-mobile/releases/download/${OPENCV_MOBILE_TAG}/${OPENCV_MOBILE_ASSET}"

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || { echo "缺少依赖：$1"; exit 1; }
}
need_cmd curl
need_cmd unzip

fetch_one() {
    local url="$1" asset="$2" marker="$3" dir="$4"
    if [ -e "$marker" ]; then
        echo "[skip] $dir 已存在（$marker 在位）"
        return 0
    fi
    echo "[fetch] $dir"
    curl -fL --retry 3 --connect-timeout 30 -o "$asset" "$url"
    # ncnn 包解压后目录名带日期后缀，opencv-mobile 固定——用通配展开到期望目录名
    if [ "$dir" = "$NCNN_DIR" ]; then
        unzip -q "$asset"
        # 兜底：万一实际目录名跟期望不同，重命名
        local actual
        actual=$(unzip -Z1 "$asset" 2>/dev/null | awk -F/ '{print $1}' | sort -u | head -1)
        [ -n "$actual" ] && [ "$actual" != "$dir" ] && mv "$actual" "$dir" || true
    else
        unzip -q "$asset"
    fi
    rm -f "$asset"
    [ -e "$marker" ] || { echo "ERROR: 解压后仍缺 $marker"; exit 1; }
    echo "[ok] $dir -> $marker"
}

fetch_one "$NCNN_URL"        "$NCNN_ASSET"        "$NCNN_DIR/${ANDROID_ABI:-arm64-v8a}/lib/cmake/ncnn/ncnnConfig.cmake" "$NCNN_DIR"
fetch_one "$OPENCV_MOBILE_URL" "$OPENCV_MOBILE_ASSET" "$OPENCV_MOBILE_DIR/sdk/native/jni/abi-arm64-v8a" "$OPENCV_MOBILE_DIR"

echo
echo "done. jni 目录就绪："
ls -1 "$JNI_DIR" | grep -E "ncnn-|opencv-mobile-"