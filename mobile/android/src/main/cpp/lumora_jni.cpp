#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <cstring>
#include <cstdint>

#define TAG "LumoraJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// External C-FFI functions exported by Rust lumora-ffi
extern "C" {
    void* lumora_image_create(uint32_t width, uint32_t height, const uint8_t* rgba_data);
    void lumora_image_destroy(void* image);
    uint32_t lumora_image_get_width(const void* image);
    uint32_t lumora_image_get_height(const void* image);
    const uint8_t* lumora_image_get_data(const void* image);
    int32_t lumora_apply_exposure(void* image, float ev);
    int32_t lumora_apply_contrast(void* image, float factor);
    int32_t lumora_apply_saturation(void* image, float factor);
    int32_t lumora_add_upoint(void* image, float x, float y, float radius, float brightness, float contrast, float saturation);
    int32_t lumora_clear_upoints(void* image);
    int32_t lumora_apply_ai_enhancement(void* image, int32_t op_type, float strength);
    int32_t lumora_render_preview(void* image, uint8_t* out_rgba, uint32_t max_width, uint32_t max_height);
}

extern "C" JNIEXPORT jlong JNICALL
Java_art_lumora_mobile_MainActivity_nativeInitEngine(
    JNIEnv* env,
    jobject /* this */,
    jobject bitmap) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        LOGE("Failed to get bitmap info");
        return 0;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        LOGE("Failed to lock bitmap pixels");
        return 0;
    }

    void* handle = lumora_image_create(info.width, info.height, static_cast<const uint8_t*>(pixels));
    AndroidBitmap_unlockPixels(env, bitmap);

    LOGI("Lumora engine initialized: %ux%u", info.width, info.height);
    return reinterpret_cast<jlong>(handle);
}

extern "C" JNIEXPORT void JNICALL
Java_art_lumora_mobile_MainActivity_nativeDestroyEngine(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong handle) {
    if (handle != 0) {
        lumora_image_destroy(reinterpret_cast<void*>(handle));
        LOGI("Lumora engine session destroyed");
    }
}

extern "C" JNIEXPORT jint JNICALL
Java_art_lumora_mobile_MainActivity_nativeApplyExposure(
    JNIEnv* /* env */,
    jobject /* this */,
    jlong handle,
    jfloat ev) {
    if (handle == 0) return -1;
    return lumora_apply_exposure(reinterpret_cast<void*>(handle), ev);
}

extern "C" JNIEXPORT jint JNICALL
Java_art_lumora_mobile_MainActivity_nativeRenderPreview(
    JNIEnv* env,
    jobject /* this */,
    jlong handle,
    jobject out_bitmap,
    jint max_w,
    jint max_h) {
    if (handle == 0) return -1;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, out_bitmap, &pixels) < 0) {
        return -2;
    }

    int32_t res = lumora_render_preview(
        reinterpret_cast<void*>(handle),
        static_cast<uint8_t*>(pixels),
        static_cast<uint32_t>(max_w),
        static_cast<uint32_t>(max_h)
    );

    AndroidBitmap_unlockPixels(env, out_bitmap);
    return res;
}
