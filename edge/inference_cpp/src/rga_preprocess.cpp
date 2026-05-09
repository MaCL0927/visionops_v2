#include "visionops/rga_preprocess.hpp"

#include <algorithm>
#include <cctype>
#include <dlfcn.h>
#include <fstream>
#include <sstream>
#include <string>

#if VISIONOPS_WITH_RGA
#if __has_include("im2d.hpp")
#include "im2d.hpp"
#elif __has_include("rga/im2d.hpp")
#include "rga/im2d.hpp"
#else
#error "VISIONOPS_WITH_RGA=1 but im2d.hpp was not found"
#endif

#if __has_include("RgaUtils.h")
#include "RgaUtils.h"
#elif __has_include("rga/RgaUtils.h")
#include "rga/RgaUtils.h"
#endif
#endif

namespace visionops {

namespace {

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static bool can_dlopen_librga() {
#if VISIONOPS_WITH_RGA
    const char* candidates[] = {
        "librga.so",
        "/usr/lib/librga.so",
        "/usr/lib/aarch64-linux-gnu/librga.so",
        "/usr/local/lib/librga.so",
        "/usr/local/lib/aarch64-linux-gnu/librga.so",
        "/opt/visionops/lib/librga.so"
    };
    for (const char* p : candidates) {
        void* h = dlopen(p, RTLD_LAZY | RTLD_LOCAL);
        if (h) {
            dlclose(h);
            return true;
        }
    }
    return file_exists("/usr/lib/librga.so") ||
           file_exists("/usr/lib/aarch64-linux-gnu/librga.so") ||
           file_exists("/usr/local/lib/librga.so") ||
           file_exists("/usr/local/lib/aarch64-linux-gnu/librga.so") ||
           file_exists("/opt/visionops/lib/librga.so");
#else
    return false;
#endif
}

#if VISIONOPS_WITH_RGA
static std::string im_status_to_string(IM_STATUS status) {
#ifdef IM_STATUS_NOERROR
    if (status == IM_STATUS_NOERROR) return "IM_STATUS_NOERROR";
#endif
#ifdef IM_STATUS_SUCCESS
    if (status == IM_STATUS_SUCCESS) return "IM_STATUS_SUCCESS";
#endif
#ifdef IM_STATUS_NOT_SUPPORTED
    if (status == IM_STATUS_NOT_SUPPORTED) return "IM_STATUS_NOT_SUPPORTED";
#endif
#ifdef IM_STATUS_OUT_OF_MEMORY
    if (status == IM_STATUS_OUT_OF_MEMORY) return "IM_STATUS_OUT_OF_MEMORY";
#endif
#ifdef IM_STATUS_INVALID_PARAM
    if (status == IM_STATUS_INVALID_PARAM) return "IM_STATUS_INVALID_PARAM";
#endif
    std::ostringstream os;
    os << static_cast<int>(status);
    return os.str();
}
#endif

static bool validate_src_and_dst(
    const cv::Mat& src_bgr,
    int dst_w,
    int dst_h,
    std::string* error_message
) {
    if (src_bgr.empty()) {
        if (error_message) *error_message = "empty source image";
        return false;
    }
    if (src_bgr.type() != CV_8UC3) {
        if (error_message) *error_message = "RGA path expects CV_8UC3 BGR source";
        return false;
    }
    if (dst_w <= 0 || dst_h <= 0) {
        if (error_message) *error_message = "invalid destination size";
        return false;
    }
    return true;
}

}  // namespace

std::string normalize_preprocess_backend(std::string backend) {
    std::transform(backend.begin(), backend.end(), backend.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (backend.empty()) return "auto";
    if (backend == "cpu" || backend == "rga" || backend == "auto") return backend;
    return "invalid";
}

std::string normalize_rga_mode(std::string mode) {
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode.empty()) return "resize_color";
    if (mode == "off" || mode == "resize_color" || mode == "resize-only" || mode == "resize_only") {
        return mode == "resize-only" ? "resize_only" : mode;
    }
    return "invalid";
}

PreprocessBackendStatus init_preprocess_backend(
    const std::string& requested_backend,
    const std::string& requested_rga_mode
) {
    PreprocessBackendStatus s;
    s.requested_backend = normalize_preprocess_backend(requested_backend);
    s.requested_rga_mode = normalize_rga_mode(requested_rga_mode);
#if VISIONOPS_WITH_RGA
    s.rga_compiled = true;
#else
    s.rga_compiled = false;
#endif
    s.rga_runtime_available = can_dlopen_librga();
    s.rga_available = s.rga_compiled && s.rga_runtime_available;

    if (s.requested_backend == "invalid") {
        s.requested_backend = "auto";
    }
    if (s.requested_rga_mode == "invalid") {
        s.requested_rga_mode = "resize_color";
    }

    if (s.requested_backend == "cpu" || s.requested_rga_mode == "off") {
        s.active_backend = "cpu";
        s.active_rga_mode = "off";
        s.rga_enabled_for_preprocess = false;
        s.reason = (s.requested_backend == "cpu")
            ? "preprocess backend forced to cpu"
            : "RGA experiment mode is off; using CPU preprocessing";
        return s;
    }

    if (s.requested_backend == "auto" || s.requested_backend == "rga") {
        if (s.rga_available) {
            s.active_backend = "rga";
            s.active_rga_mode = s.requested_rga_mode;
            s.rga_enabled_for_preprocess = true;
            s.reason = (s.active_rga_mode == "resize_only")
                ? "RGA is available; using RGA resize-only with CPU padding and CPU BGR->RGB"
                : "RGA is available; using RGA resize + BGR->RGB with CPU padding";
        } else if (s.rga_compiled) {
            s.active_backend = "cpu";
            s.active_rga_mode = "off";
            s.rga_enabled_for_preprocess = false;
            s.reason = "built with RGA support but librga is not loadable at runtime; fallback to cpu";
        } else {
            s.active_backend = "cpu";
            s.active_rga_mode = "off";
            s.rga_enabled_for_preprocess = false;
            s.reason = "built without RGA support; fallback to cpu";
        }
        return s;
    }

    s.active_backend = "cpu";
    s.active_rga_mode = "off";
    s.rga_enabled_for_preprocess = false;
    s.reason = "invalid preprocess settings; fallback to cpu";
    return s;
}

bool rga_resize_bgr_to_rgb(
    const cv::Mat& src_bgr,
    cv::Mat& dst_rgb,
    int dst_w,
    int dst_h,
    std::string* error_message
) {
#if VISIONOPS_WITH_RGA
    if (!validate_src_and_dst(src_bgr, dst_w, dst_h, error_message)) return false;

    cv::Mat src_contig = src_bgr.isContinuous() ? src_bgr : src_bgr.clone();
    dst_rgb.create(dst_h, dst_w, CV_8UC3);

    rga_buffer_t src = wrapbuffer_virtualaddr(
        static_cast<void*>(src_contig.data),
        src_contig.cols,
        src_contig.rows,
        RK_FORMAT_BGR_888
    );
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        static_cast<void*>(dst_rgb.data),
        dst_w,
        dst_h,
        RK_FORMAT_RGB_888
    );

    IM_STATUS ret = imresize(src, dst);
    if (ret != IM_STATUS_SUCCESS) {
        if (error_message) *error_message = "RGA imresize BGR->RGB failed: " + im_status_to_string(ret);
        return false;
    }
    return true;
#else
    (void)src_bgr;
    (void)dst_rgb;
    (void)dst_w;
    (void)dst_h;
    if (error_message) *error_message = "binary was built without RGA support";
    return false;
#endif
}

bool rga_resize_bgr_to_bgr(
    const cv::Mat& src_bgr,
    cv::Mat& dst_bgr,
    int dst_w,
    int dst_h,
    std::string* error_message
) {
#if VISIONOPS_WITH_RGA
    if (!validate_src_and_dst(src_bgr, dst_w, dst_h, error_message)) return false;

    cv::Mat src_contig = src_bgr.isContinuous() ? src_bgr : src_bgr.clone();
    dst_bgr.create(dst_h, dst_w, CV_8UC3);

    rga_buffer_t src = wrapbuffer_virtualaddr(
        static_cast<void*>(src_contig.data),
        src_contig.cols,
        src_contig.rows,
        RK_FORMAT_BGR_888
    );
    rga_buffer_t dst = wrapbuffer_virtualaddr(
        static_cast<void*>(dst_bgr.data),
        dst_w,
        dst_h,
        RK_FORMAT_BGR_888
    );

    IM_STATUS ret = imresize(src, dst);
    if (ret != IM_STATUS_SUCCESS) {
        if (error_message) *error_message = "RGA imresize BGR->BGR failed: " + im_status_to_string(ret);
        return false;
    }
    return true;
#else
    (void)src_bgr;
    (void)dst_bgr;
    (void)dst_w;
    (void)dst_h;
    if (error_message) *error_message = "binary was built without RGA support";
    return false;
#endif
}

}  // namespace visionops
