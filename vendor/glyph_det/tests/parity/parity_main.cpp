// Host-side decode parity check: reads the npy dumps produced by
// make_parity.py, runs the same decode.cpp that ships in the Android JNI
// library, and compares boxes/scores against the decode.py reference
// (tolerance 1e-3).
//
// Build (Git Bash, w64devkit g++):
//   g++ -O2 -std=c++11 -I../../android/src/main/jni \
//       parity_main.cpp ../../android/src/main/jni/decode.cpp -o parity.exe
// Run:
//   ./parity.exe            # checks all cases (ort / crafted / thresh)
//   ./parity.exe crafted    # checks one case

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

#include "decode.h"

// ------------------------------------------------ minimal npy reader (<f4, C-order)
static bool read_npy_f32(const char* path, std::vector<float>& data, std::vector<int>& shape)
{
    FILE* f = fopen(path, "rb");
    if (!f)
    {
        fprintf(stderr, "cannot open %s\n", path);
        return false;
    }

    unsigned char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0)
    {
        fprintf(stderr, "%s: bad magic\n", path);
        fclose(f);
        return false;
    }

    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2)
    {
        fclose(f);
        return false;
    }

    unsigned int header_len = 0;
    if (ver[0] == 1)
    {
        unsigned char b[2];
        if (fread(b, 1, 2, f) != 2) { fclose(f); return false; }
        header_len = b[0] | (b[1] << 8);
    }
    else
    {
        unsigned char b[4];
        if (fread(b, 1, 4, f) != 4) { fclose(f); return false; }
        header_len = b[0] | (b[1] << 8) | (b[2] << 16) | ((unsigned int)b[3] << 24);
    }

    std::string header(header_len, '\0');
    if (fread(&header[0], 1, header_len, f) != header_len)
    {
        fclose(f);
        return false;
    }

    if (header.find("'descr': '<f4'") == std::string::npos &&
        header.find("\"descr\": \"<f4\"") == std::string::npos)
    {
        fprintf(stderr, "%s: not <f4: %s\n", path, header.c_str());
        fclose(f);
        return false;
    }
    if (header.find("'fortran_order': True") != std::string::npos)
    {
        fprintf(stderr, "%s: fortran order unsupported\n", path);
        fclose(f);
        return false;
    }

    size_t sp = header.find("'shape': (");
    if (sp == std::string::npos)
    {
        sp = header.find("\"shape\": (");
    }
    if (sp == std::string::npos)
    {
        fprintf(stderr, "%s: no shape in header\n", path);
        fclose(f);
        return false;
    }
    sp = header.find('(', sp);
    size_t ep = header.find(')', sp);
    std::string dims = header.substr(sp + 1, ep - sp - 1);

    shape.clear();
    size_t pos = 0;
    while (pos < dims.size())
    {
        while (pos < dims.size() && (dims[pos] == ' ' || dims[pos] == ',')) pos++;
        if (pos >= dims.size()) break;
        char* end = 0;
        long d = strtol(dims.c_str() + pos, &end, 10);
        if (end == dims.c_str() + pos) break;
        shape.push_back((int)d);
        pos = (size_t)(end - dims.c_str());
    }

    size_t count = 1;
    for (size_t i = 0; i < shape.size(); i++) count *= (size_t)shape[i];

    data.resize(count);
    size_t got = fread(data.data(), sizeof(float), count, f);
    fclose(f);
    if (got != count)
    {
        fprintf(stderr, "%s: short read %zu/%zu\n", path, got, count);
        return false;
    }
    return true;
}

// ------------------------------------------------ one case
static int run_case(const std::string& dir, const std::string& prefix)
{
    const int strides[3] = {4, 8, 16};
    const char* names[3] = {"out4", "out8", "out16"};

    GlyphHead heads[3];
    std::vector<std::vector<float> > blobs(3);
    for (int i = 0; i < 3; i++)
    {
        std::vector<int> shape;
        std::string path = dir + "/" + prefix + "_" + names[i] + ".npy";
        if (!read_npy_f32(path.c_str(), blobs[i], shape))
            return 1;
        if (shape.size() != 4 || shape[0] != 1)
        {
            fprintf(stderr, "%s: expect shape (1,C,H,W)\n", path.c_str());
            return 1;
        }
        heads[i].data = blobs[i].data();
        heads[i].channels = shape[1];
        heads[i].height = shape[2];
        heads[i].width = shape[3];
        heads[i].stride = strides[i];
        heads[i].channel_stride = (size_t)shape[2] * shape[3];
    }

    std::vector<int> eshape;
    std::vector<float> eboxes_flat, escores;
    if (!read_npy_f32((dir + "/" + prefix + "_expected_boxes.npy").c_str(), eboxes_flat, eshape))
        return 1;
    if (!read_npy_f32((dir + "/" + prefix + "_expected_scores.npy").c_str(), escores, eshape))
        return 1;

    std::vector<GlyphBox> got = glyphdet_decode(heads, 3, 24, 0.5f, 0.4f, 20);

    const size_t nexp = escores.size();
    double max_box_diff = 0.0;
    double max_score_diff = 0.0;
    int bad = 0;

    if (got.size() != nexp)
    {
        fprintf(stderr, "[%s] box count mismatch: got %zu expect %zu\n",
                prefix.c_str(), got.size(), nexp);
        bad = 1;
    }

    const size_t n = got.size() < nexp ? got.size() : nexp;
    for (size_t i = 0; i < n; i++)
    {
        const float* eb = &eboxes_flat[i * 4];
        double d = fabs(got[i].x1 - eb[0]);
        d = std::max(d, (double)fabs(got[i].y1 - eb[1]));
        d = std::max(d, (double)fabs(got[i].x2 - eb[2]));
        d = std::max(d, (double)fabs(got[i].y2 - eb[3]));
        max_box_diff = std::max(max_box_diff, d);
        double sd = fabs(got[i].score - escores[i]);
        max_score_diff = std::max(max_score_diff, sd);
        if (d > 1e-3 || sd > 1e-3)
        {
            fprintf(stderr,
                    "[%s] box %zu mismatch: got (%.4f %.4f %.4f %.4f) s=%.6f, "
                    "expect (%.4f %.4f %.4f %.4f) s=%.6f\n",
                    prefix.c_str(), i, got[i].x1, got[i].y1, got[i].x2, got[i].y2,
                    got[i].score, eb[0], eb[1], eb[2], eb[3], escores[i]);
            bad = 1;
        }
    }

    printf("[%s] boxes got=%zu expect=%zu max|box diff|=%.3g max|score diff|=%.3g -> %s\n",
           prefix.c_str(), got.size(), nexp, max_box_diff, max_score_diff,
           bad ? "FAIL" : "PASS");
    return bad;
}

int main(int argc, char** argv)
{
    std::string dir = ".";
    if (argc > 1 && argv[1][0] != '\0' && strchr(argv[1], '/') == 0 && strchr(argv[1], '\\') == 0 && strstr(argv[1], ".npy") == 0)
    {
        // single case name
    }

    int bad = 0;
    if (argc > 1)
    {
        bad |= run_case(dir, argv[1]);
    }
    else
    {
        bad |= run_case(dir, "ort");
        bad |= run_case(dir, "crafted");
        bad |= run_case(dir, "thresh");
    }

    if (bad)
    {
        printf("PARITY: FAIL\n");
        return 1;
    }
    printf("PARITY: PASS\n");
    return 0;
}
