#include "decode.h"

#include <algorithm>
#include <cmath>

static inline float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

// DFL expectation for one grid cell: reg[4*reg_max] -> ltrb[4] (grid units).
static void dfl_expect(const float* reg, size_t channel_stride, int reg_max, float* ltrb)
{
    float prob[32]; // reg_max <= 32
    for (int side = 0; side < 4; side++)
    {
        const float* logits = reg + (size_t)side * reg_max * channel_stride;

        // softmax over reg_max bins (max-subtracted, same as torch.softmax)
        float max_logit = logits[0];
        for (int i = 1; i < reg_max; i++)
            max_logit = std::max(max_logit, logits[i * channel_stride]);

        float sum = 0.f;
        for (int i = 0; i < reg_max; i++)
        {
            prob[i] = expf(logits[i * channel_stride] - max_logit);
            sum += prob[i];
        }

        // expectation over bins 0..reg_max-1
        float expect = 0.f;
        for (int i = 0; i < reg_max; i++)
            expect += (prob[i] / sum) * (float)i;

        ltrb[side] = expect;
    }
}

static inline float box_area(const GlyphBox& b)
{
    return (b.x2 - b.x1) * (b.y2 - b.y1);
}

static inline float box_iou(const GlyphBox& a, const GlyphBox& b)
{
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    if (x2 <= x1 || y2 <= y1)
        return 0.f;
    float inter = (x2 - x1) * (y2 - y1);
    return inter / (box_area(a) + box_area(b) - inter);
}

std::vector<GlyphBox> glyphdet_decode(
    const GlyphHead* heads, int head_count,
    int reg_max, float score_thr,
    float nms_iou, int max_det)
{
    std::vector<GlyphBox> all;

    for (int hi = 0; hi < head_count; hi++)
    {
        const GlyphHead& head = heads[hi];
        const int C = head.channels;
        const int H = head.height;
        const int W = head.width;
        const int s = head.stride;
        const size_t cs = head.channel_stride;

        const float* score_ch = head.data;
        const float* reg_ch = head.data + cs;
        const bool has_cen = (C == 2 + 4 * reg_max);
        const float* cen_ch = has_cen ? head.data + (size_t)(C - 1) * cs : 0;

        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                const size_t off = (size_t)y * W + x;

                float score = sigmoidf(score_ch[off]);
                if (has_cen)
                    score *= sigmoidf(cen_ch[off]);
                if (!(score > score_thr))
                    continue;

                float ltrb[4];
                dfl_expect(reg_ch + off, cs, reg_max, ltrb);

                const float px = (x + 0.5f) * s;
                const float py = (y + 0.5f) * s;

                GlyphBox b;
                b.x1 = px - ltrb[0] * s;
                b.y1 = py - ltrb[1] * s;
                b.x2 = px + ltrb[2] * s;
                b.y2 = py + ltrb[3] * s;
                b.score = score;
                all.push_back(b);
            }
        }
    }

    // torchvision.ops.nms semantics: descending score order, greedy keep,
    // suppress any later box with IoU > threshold.
    std::vector<int> order(all.size());
    for (size_t i = 0; i < all.size(); i++)
        order[i] = (int)i;
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return all[a].score > all[b].score; });

    std::vector<GlyphBox> kept;
    std::vector<char> suppressed(all.size(), 0);
    for (size_t oi = 0; oi < order.size() && (int)kept.size() < max_det; oi++)
    {
        const int i = order[oi];
        if (suppressed[i])
            continue;
        kept.push_back(all[i]);
        for (size_t oj = oi + 1; oj < order.size(); oj++)
        {
            const int j = order[oj];
            if (!suppressed[j] && box_iou(all[i], all[j]) > nms_iou)
                suppressed[j] = 1;
        }
    }

    return kept;
}
