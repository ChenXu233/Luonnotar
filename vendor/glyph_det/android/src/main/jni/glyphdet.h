#ifndef GLYPHDET_H
#define GLYPHDET_H

#include <vector>

#include <opencv2/core/core.hpp>

#include "net.h"

#include "decode.h"

// Mask-conditioned glyph detector (ncnn + optional Vulkan).
// Model contract: scene (1,3,416,416) RGB [0,1], mask (1,1,64,384) gray [0,1]
// white-bg black-text; outputs (1,97,104,104) / (1,97,52,52) / (1,97,26,26).
class GlyphDet
{
public:
    GlyphDet();
    ~GlyphDet();

    int load(const char* parampath, const char* binpath, bool use_gpu);

    // gray: MASK_W x MASK_H row-major bytes, white background black text.
    void set_mask(const unsigned char* gray);
    bool has_mask() const;

    // rgb: camera frame RGB888. Detected boxes are mapped back into the rgb
    // coordinate space. Returns 0 on success, -1 when model/mask missing.
    int detect(const cv::Mat& rgb, std::vector<GlyphBox>& boxes);

public:
    static const int SCENE_SIZE = 416;
    static const int MASK_W = 384;
    static const int MASK_H = 64;
    static const int REG_MAX = 24;

private:
    ncnn::Net net;
    ncnn::Mat mask_f32; // (w=384, h=64) float [0,1]
    bool mask_ready;
    bool net_loaded;
    mutable ncnn::Mutex mask_lock;
};

#endif // GLYPHDET_H
