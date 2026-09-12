#ifndef GLYPHDET_DECODE_H
#define GLYPHDET_DECODE_H

#include <cstddef>
#include <vector>

// Axis-aligned detection box in input-image pixels (416x416 letterbox space).
struct GlyphBox
{
    float x1, y1, x2, y2;
    float score;
};

// One raw head output, CHW float32.
// Layout mirrors the model contract: C = 1 + 4*reg_max (97), or the legacy
// 2 + 4*reg_max (98) with a trailing centerness channel (auto-detected).
struct GlyphHead
{
    const float* data;      // channel 0 plane start
    int channels;           // 97 (v2.2) or 98 (v1 legacy)
    int height;             // 104 / 52 / 26
    int width;              // 104 / 52 / 26
    int stride;             // 4 / 8 / 16
    size_t channel_stride;  // floats between consecutive channels (w*h when packed)
};

// Reference-semantic decode (core/decode.py):
//   score = sigmoid(out[0]) (* sigmoid(cen) when the legacy channel exists)
//   keep score > score_thr
//   reg (4*reg_max) -> per-side softmax over reg_max bins -> expectation -> ltrb
//   ltrb * stride; box = grid_center +/- l/t/r/b
//   concat all heads -> greedy NMS (score desc, suppress IoU > nms_iou)
//   -> at most max_det boxes, sorted by score descending.
std::vector<GlyphBox> glyphdet_decode(
    const GlyphHead* heads, int head_count,
    int reg_max = 24, float score_thr = 0.5f,
    float nms_iou = 0.4f, int max_det = 20);

#endif // GLYPHDET_DECODE_H
