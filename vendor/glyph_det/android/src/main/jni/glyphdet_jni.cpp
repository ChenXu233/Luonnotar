#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "glyphdet.h"
#include "ndkcamera.h"

#include <opencv2/core/core.hpp>

#include <atomic>

// pipeline perf counters (EMA-smoothed), exposed via pollResults + periodic
// logcat for on-device profiling
static std::atomic<float> g_infer_ms(0.f);
static std::atomic<float> g_render_fps(0.f);

static void update_render_fps()
{
    static double t0 = 0.f;
    static float fps_history[10] = {0.f};

    double t1 = ncnn::get_current_time();
    if (t0 == 0.f)
    {
        t0 = t1;
        return;
    }

    float fps = 1000.f / (t1 - t0);
    t0 = t1;

    for (int i = 9; i >= 1; i--)
    {
        fps_history[i] = fps_history[i - 1];
    }
    fps_history[0] = fps;

    if (fps_history[9] == 0.f)
    {
        return;
    }

    float avg_fps = 0.f;
    for (int i = 0; i < 10; i++)
    {
        avg_fps += fps_history[i];
    }
    g_render_fps = avg_fps / 10.f;
}

static GlyphDet* g_glyphdet = 0;
static ncnn::Mutex g_model_lock;

class GlyphDetCamera;
static void* onInferProcess(void* args);

class GlyphDetCamera : public NdkCameraWindow
{
public:
    GlyphDetCamera();
    virtual ~GlyphDetCamera();

    virtual void on_image_render(cv::Mat& rgb) const;

    void infer_thread_loop();

    // serialize latest decoded results as JSON for the Dart-side poll
    std::string get_results_json() const;

private:
    mutable cv::Mat infer_latest_rgb;
    mutable ncnn::Mutex infer_lock;
    mutable ncnn::ConditionVariable infer_condition;
    bool infer_exiting;
    ncnn::Thread* infer_thread;

    // latest decoded boxes snapshot + the frame size they refer to
    mutable std::vector<GlyphBox> latest_boxes;
    mutable ncnn::Mutex latest_boxes_lock;
    mutable int latest_frame_w;
    mutable int latest_frame_h;
};

static void* onInferProcess(void* args)
{
    GlyphDetCamera* self = (GlyphDetCamera*)args;
    self->infer_thread_loop();
    return 0;
}

GlyphDetCamera::GlyphDetCamera()
{
    latest_frame_w = 0;
    latest_frame_h = 0;
    infer_exiting = false;
    infer_thread = new ncnn::Thread(onInferProcess, (void*)this);
}

GlyphDetCamera::~GlyphDetCamera()
{
    infer_exiting = true;
    infer_condition.signal();
    infer_thread->join();
    delete infer_thread;
}

std::string GlyphDetCamera::get_results_json() const
{
    std::vector<GlyphBox> boxes;
    int frame_w = 0;
    int frame_h = 0;
    {
        ncnn::MutexLockGuard g(latest_boxes_lock);
        boxes = latest_boxes;
        frame_w = latest_frame_w;
        frame_h = latest_frame_h;
    }

    char header[160];
    snprintf(header, sizeof(header),
             "{\"w\":%d,\"h\":%d,\"fps\":%.1f,\"infer_ms\":%.1f,\"nbox\":%d,\"results\":[",
             frame_w, frame_h, (double)g_render_fps.load(),
             (double)g_infer_ms.load(), (int)boxes.size());
    std::string json = header;
    for (size_t i = 0; i < boxes.size(); i++)
    {
        const GlyphBox& b = boxes[i];
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "%s{\"score\":%.4f,\"cx\":%.2f,\"cy\":%.2f,\"w\":%.2f,\"h\":%.2f,\"angle\":0.0}",
                 i == 0 ? "" : ",",
                 b.score,
                 (b.x1 + b.x2) * 0.5, (b.y1 + b.y2) * 0.5,
                 b.x2 - b.x1, b.y2 - b.y1);
        json += buf;
    }
    json += "]}";
    return json;
}

void GlyphDetCamera::infer_thread_loop()
{
    while (!infer_exiting)
    {
        cv::Mat rgb;
        {
            ncnn::MutexLockGuard g(infer_lock);
            while (infer_latest_rgb.empty() && !infer_exiting)
            {
                infer_condition.wait(infer_lock);
            }
            if (infer_exiting) break;

            rgb = infer_latest_rgb;
            infer_latest_rgb = cv::Mat();
        }

        if (rgb.empty()) continue;

        std::vector<GlyphBox> boxes;
        {
            ncnn::MutexLockGuard g(g_model_lock);
            if (g_glyphdet && g_glyphdet->has_mask())
            {
                double t0 = ncnn::get_current_time();

                g_glyphdet->detect(rgb, boxes);

                float dt = (float)(ncnn::get_current_time() - t0);
                g_infer_ms = g_infer_ms * 0.8f + dt * 0.2f;
            }
        }

        {
            ncnn::MutexLockGuard g(latest_boxes_lock);
            latest_boxes = boxes;
        }

        // back-to-back at its own pace: detection keeps up with camera motion
    }
}

void GlyphDetCamera::on_image_render(cv::Mat& rgb) const
{
    // feed latest frame to the inference thread when it is idle
    {
        ncnn::MutexLockGuard g(infer_lock);
        if (infer_latest_rgb.empty())
        {
            infer_latest_rgb = rgb.clone();
            infer_condition.signal();
        }
    }

    // frame size snapshot for the Dart-side coordinate space
    {
        ncnn::MutexLockGuard g(latest_boxes_lock);
        latest_frame_w = rgb.cols;
        latest_frame_h = rgb.rows;
    }

    // no native boxes on the preview — the Flutter overlay draws the highlight

    update_render_fps();

    // perf snapshot to logcat roughly once per second (adb logcat -s GlyphDetPerf)
    static int perf_log_counter = 0;
    if (++perf_log_counter >= 25)
    {
        perf_log_counter = 0;
        int nbox = 0;
        {
            ncnn::MutexLockGuard g(latest_boxes_lock);
            nbox = (int)latest_boxes.size();
        }
        __android_log_print(ANDROID_LOG_WARN, "GlyphDetPerf",
            "fps=%.1f infer=%.1fms boxes=%d frame=%dx%d",
            (double)g_render_fps.load(), (double)g_infer_ms.load(),
            nbox, rgb.cols, rgb.rows);
    }
}

static GlyphDetCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new GlyphDetCamera;

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(g_model_lock);

        delete g_glyphdet;
        g_glyphdet = 0;
    }

    ncnn::destroy_gpu_instance();

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(String paramPath, String binPath, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_loadModel(JNIEnv* env, jobject thiz, jstring paramPath, jstring binPath, jint cpugpu)
{
    if (cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    const char* parampath = env->GetStringUTFChars(paramPath, nullptr);
    const char* binpath = env->GetStringUTFChars(binPath, nullptr);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel from files: %s %s", parampath, binpath);

    const bool use_gpu = (int)cpugpu == 1 && ncnn::get_gpu_count() > 0;

    int ret = 0;
    {
        ncnn::MutexLockGuard g(g_model_lock);

        delete g_glyphdet;
        g_glyphdet = 0;

        g_glyphdet = new GlyphDet;
        ret = g_glyphdet->load(parampath, binpath, use_gpu);
    }

    env->ReleaseStringUTFChars(paramPath, parampath);
    env->ReleaseStringUTFChars(binPath, binpath);

    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native boolean setQueryMask(byte[] mask);
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_setQueryMask(JNIEnv* env, jobject thiz, jbyteArray mask)
{
    const jsize len = env->GetArrayLength(mask);
    if (len != GlyphDet::MASK_W * GlyphDet::MASK_H)
    {
        __android_log_print(ANDROID_LOG_ERROR, "GlyphDet", "setQueryMask: bad length %d, expect %d",
                            (int)len, (int)(GlyphDet::MASK_W * GlyphDet::MASK_H));
        return JNI_FALSE;
    }

    jbyte* data = env->GetByteArrayElements(mask, nullptr);

    ncnn::MutexLockGuard g(g_model_lock);
    if (g_glyphdet)
    {
        g_glyphdet->set_mask((const unsigned char*)data);
    }

    env->ReleaseByteArrayElements(mask, data, JNI_ABORT);

    return g_glyphdet ? JNI_TRUE : JNI_FALSE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

// public native boolean clearOutputWindow();
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_clearOutputWindow(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "clearOutputWindow");

    if (g_camera)
    {
        g_camera->clear_window();
    }

    return JNI_TRUE;
}

// public native boolean toggleFlash();
JNIEXPORT jboolean JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_toggleFlash(JNIEnv* env, jobject thiz)
{
    if (!g_camera)
        return JNI_FALSE;

    int ret = g_camera->toggle_flash();
    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

// public native String pollResults();
// Returns JSON {"w","h","fps","infer_ms","nbox","results":[{"score","cx","cy","w","h","angle"}]}
// Coordinates are in the rendered rgb frame space (portrait: maps to preview with uniform scale).
JNIEXPORT jstring JNICALL Java_com_luonnotar_glyphdet_GlyphDetNcnn_pollResults(JNIEnv* env, jobject thiz)
{
    if (!g_camera)
        return env->NewStringUTF("{\"w\":0,\"h\":0,\"fps\":0,\"infer_ms\":0,\"nbox\":0,\"results\":[]}");

    std::string json = g_camera->get_results_json();
    return env->NewStringUTF(json.c_str());
}

}
