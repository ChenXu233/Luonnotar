#include "glyphdet.h"

#include <android/log.h>
#include <algorithm>
#include <math.h>

#include "cpu.h"
#include "mat.h"

GlyphDet::GlyphDet()
{
    mask_ready = false;
    net_loaded = false;
}

GlyphDet::~GlyphDet()
{
    net.clear();
}

int GlyphDet::load(const char* parampath, const char* binpath, bool use_gpu)
{
    net.clear();
    net_loaded = false;

    net.opt.num_threads = ncnn::get_big_cpu_count();
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;

#if NCNN_VULKAN
    net.opt.use_vulkan_compute = use_gpu;
#else
    (void)use_gpu;
#endif

    int ret = net.load_param(parampath);
    if (ret != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GlyphDet", "load_param failed %d: %s", ret, parampath);
        return ret;
    }

    ret = net.load_model(binpath);
    if (ret != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GlyphDet", "load_model failed %d: %s", ret, binpath);
        return ret;
    }

    net_loaded = true;

    __android_log_print(ANDROID_LOG_WARN, "GlyphDet", "loaded %s + %s (vulkan=%d, gpu_count=%d)",
                        parampath, binpath, (int)net.opt.use_vulkan_compute, ncnn::get_gpu_count());

    return 0;
}

void GlyphDet::set_mask(const unsigned char* gray)
{
    ncnn::MutexLockGuard g(mask_lock);

    ncnn::Mat m(MASK_W, MASK_H, (size_t)4u, 1);
    float* ptr = (float*)m.data;
    for (int i = 0; i < MASK_W * MASK_H; i++)
    {
        ptr[i] = gray[i] / 255.f;
    }
    mask_f32 = m;
    mask_ready = true;
}

bool GlyphDet::has_mask() const
{
    ncnn::MutexLockGuard g(mask_lock);
    return mask_ready;
}

int GlyphDet::detect(const cv::Mat& rgb, std::vector<GlyphBox>& boxes)
{
    boxes.clear();

    ncnn::Mat mask;
    {
        ncnn::MutexLockGuard g(mask_lock);
        if (!net_loaded || !mask_ready)
            return -1;
        mask = mask_f32;
    }

    const int img_w = rgb.cols;
    const int img_h = rgb.rows;

    // letterbox into 416x416 keeping aspect, black (0) padding
    const float scale = std::min((float)SCENE_SIZE / img_w, (float)SCENE_SIZE / img_h);
    const int resized_w = std::max(1, (int)roundf(img_w * scale));
    const int resized_h = std::max(1, (int)roundf(img_h * scale));

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, img_w, img_h, resized_w, resized_h);

    const int pad_left = (SCENE_SIZE - resized_w) / 2;
    const int pad_top = (SCENE_SIZE - resized_h) / 2;

    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, pad_top, SCENE_SIZE - resized_h - pad_top,
                           pad_left, SCENE_SIZE - resized_w - pad_left,
                           ncnn::BORDER_CONSTANT, 0.f);

    // [0,255] -> [0,1]
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("in0", in_pad);
    ex.input("in1", mask);

    ncnn::Mat out4, out8, out16;
    ex.extract("out0", out4); // (97, 104, 104) stride 4
    ex.extract("out1", out8); // (97, 52, 52) stride 8
    ex.extract("out2", out16); // (97, 26, 26) stride 16

    if (out4.empty() || out8.empty() || out16.empty())
    {
        __android_log_print(ANDROID_LOG_ERROR, "GlyphDet", "extract failed: %d %d %d",
                            (int)out4.empty(), (int)out8.empty(), (int)out16.empty());
        return -1;
    }

    const int strides[3] = {4, 8, 16};
    const ncnn::Mat* outs[3] = {&out4, &out8, &out16};

    GlyphHead heads[3];
    for (int i = 0; i < 3; i++)
    {
        heads[i].data = (const float*)outs[i]->data;
        heads[i].channels = outs[i]->c;
        heads[i].height = outs[i]->h;
        heads[i].width = outs[i]->w;
        heads[i].stride = strides[i];
        heads[i].channel_stride = outs[i]->cstep;
    }

    std::vector<GlyphBox> lb_boxes = glyphdet_decode(heads, 3, REG_MAX, 0.5f, 0.4f, 20);

    // letterbox space -> camera frame space
    for (size_t i = 0; i < lb_boxes.size(); i++)
    {
        GlyphBox b = lb_boxes[i];
        b.x1 = (b.x1 - pad_left) / scale;
        b.y1 = (b.y1 - pad_top) / scale;
        b.x2 = (b.x2 - pad_left) / scale;
        b.y2 = (b.y2 - pad_top) / scale;

        b.x1 = std::max(0.f, std::min(b.x1, (float)img_w));
        b.y1 = std::max(0.f, std::min(b.y1, (float)img_h));
        b.x2 = std::max(0.f, std::min(b.x2, (float)img_w));
        b.y2 = std::max(0.f, std::min(b.y2, (float)img_h));

        if (b.x2 - b.x1 > 1.f && b.y2 - b.y1 > 1.f)
            boxes.push_back(b);
    }

    return 0;
}
