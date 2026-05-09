#pragma once

#include <string>

#include <opencv2/opencv.hpp>

namespace visionops {

struct PreprocessBackendStatus {
    std::string requested_backend = "auto";   // cpu | rga | auto
    std::string requested_rga_mode = "resize_color"; // off | resize_color | resize_only
    std::string active_backend = "cpu";       // cpu | rga
    std::string active_rga_mode = "off";      // off | resize_color | resize_only
    std::string reason = "cpu preprocessing active";
    bool rga_compiled = false;                 // built with librga/header detected by CMake
    bool rga_runtime_available = false;        // librga appears loadable / present on target runtime
    bool rga_available = false;                // compiled && runtime_available
    bool rga_enabled_for_preprocess = false;   // true when the active path is allowed to call RGA
};

std::string normalize_preprocess_backend(std::string backend);
std::string normalize_rga_mode(std::string mode);
PreprocessBackendStatus init_preprocess_backend(
    const std::string& requested_backend,
    const std::string& requested_rga_mode = "resize_color"
);

// v0.4.2.1 experiment mode: resize_color.
// RGA performs resize and BGR->RGB conversion. CPU performs final letterbox padding.
bool rga_resize_bgr_to_rgb(
    const cv::Mat& src_bgr,
    cv::Mat& dst_rgb,
    int dst_w,
    int dst_h,
    std::string* error_message = nullptr
);

// v0.4.2.1 experiment mode: resize_only.
// RGA performs resize while keeping BGR format. CPU performs letterbox padding and BGR->RGB conversion.
bool rga_resize_bgr_to_bgr(
    const cv::Mat& src_bgr,
    cv::Mat& dst_bgr,
    int dst_w,
    int dst_h,
    std::string* error_message = nullptr
);

}  // namespace visionops
