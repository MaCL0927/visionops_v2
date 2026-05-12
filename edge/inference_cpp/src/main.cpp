#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <mutex>
#include <memory>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "visionops/stream_backend.hpp"
#include "visionops/rga_preprocess.hpp"

extern "C" {
#include "rknn_api.h"
}


static void configure_opencv_ffmpeg_quiet_logging(bool quiet) {
    if (!quiet) return;

    // v0.8.3.4: force quiet OpenCV/FFmpeg logging.
    // Some boards may already export OPENCV_FFMPEG_DEBUG=1 or
    // OPENCV_FFMPEG_LOGLEVEL=56, which produces very verbose trace logs such as
    // "tcp_read_packet". Override them here so manual shell environment does not
    // pollute the production service.
    setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
    setenv("OPENCV_VIDEOIO_DEBUG", "0", 1);
    setenv("OPENCV_FFMPEG_LOGLEVEL", "16", 1);  // AV_LOG_ERROR
    unsetenv("OPENCV_FFMPEG_DEBUG");

    try {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    } catch (...) {
        // Ignore logging setup failures. This should never block inference.
    }
}

struct Args {
    std::string model = "/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.rknn";
    std::string class_names_file = "/opt/visionops/models/rk3588-001_CUST-000_det_20260427_105932.yaml";
    std::string task = "detection";
    std::string pipeline_config = "";  // v0.8.4: roi_classification pipeline.yaml
    std::string host = "0.0.0.0";
    std::string npu_core = "auto";
    int port = 18080;
    int input_h = 640;
    int input_w = 640;
    int num_classes = 80;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float mask_threshold = 0.5f;
    int topk = 5;
    int max_det = 100;
    std::string output_mode = "float";  // v0.1.6: float | quant

    // v0.2/v0.3: RTSP stream worker sidecar mode.
    std::string camera_source = "";
    // v0.7.1: USB/OpenCV camera options. Existing stream_backend remains the
    // abstraction layer; these fields configure /dev/videoX UVC cameras.
    std::string camera_type = "auto";      // auto | rtsp | usb
    int camera_width = 0;                  // 0 means backend default
    int camera_height = 0;                 // 0 means backend default
    int camera_fps = 0;                    // 0 means backend/default; for USB usually 10/15
    int camera_buffer_size = 1;
    std::string camera_fourcc = "";        // YUYV | MJPG | empty
    int stream_fps = 10;          // legacy alias, kept for compatibility; equals detect_fps by default.
    int camera_read_fps = 10;     // v0.3: throttle RTSP read/decode rate on main stream.
    int detect_fps = 10;          // v0.3: target inference FPS.
    int snapshot_fps = 1;         // v0.3: how often to refresh cached snapshot/annotated caches.
    int snapshot_jpeg_quality = 80;
    // v0.8.3.4: decouple visual cache generation from realtime inference.
    // Disable these to measure RTSP + inference CPU without frame clone / overlay overhead.
    bool enable_snapshot = true;
    bool enable_annotated = true;
    bool stream_auto_start = false;

    // v0.8.3.4: RTSP robustness / log control. Keep TCP by default for main stream.
    std::string rtsp_transport = "tcp";   // tcp | udp
    int rtsp_timeout_ms = 5000;            // maps to FFmpeg stimeout, microseconds internally
    bool quiet_ffmpeg_log = true;          // reduce FFmpeg/OpenCV decode warning noise

    // v0.8.3.4: selectable stream backend.
    // opencv: low-risk OpenCV/FFmpeg path, kept as the default fallback.
    // gst-mpp: OpenCV + GStreamer pipeline using Rockchip mppvideodec.
    std::string stream_backend = "opencv"; // opencv | gst-mpp
    std::string stream_codec = "h264";     // h264 | h265
    int gst_latency_ms = 100;              // rtspsrc latency for gst-mpp backend

    // v0.8.3.4: preprocessing backend switch with RGA experiment modes.
    // preprocess_backend: cpu | rga | auto
    // rga_mode:
    //   off          -> force CPU preprocessing for fair baseline comparison
    //   resize_color -> RGA resize + BGR->RGB, CPU padding
    //   resize_only  -> RGA resize BGR->BGR, CPU padding + CPU BGR->RGB
    std::string preprocess_backend = "auto";
    std::string rga_mode = "resize_color";
    visionops::PreprocessBackendStatus preprocess_status;
};

struct Tensor {
    std::vector<int> dims;
    std::vector<float> data;
};

struct PreprocessMeta {
    int orig_w = 0;
    int orig_h = 0;
    int input_w = 0;
    int input_h = 0;
    float ratio = 1.0f;
    float pad_x = 0.0f;
    float pad_y = 0.0f;
};

struct Detection {
    int class_id = -1;
    std::string class_name;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
};

struct ClassificationItem {
    int class_id = -1;
    std::string class_name;
    float confidence = 0.0f;
    float logit = 0.0f;
};

struct ObbDetection {
    int class_id = -1;
    std::string class_name;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float w = 0.0f;
    float h = 0.0f;
    float angle = 0.0f;  // radians
    std::vector<cv::Point2f> points;
};

struct SegmentationDetection {
    Detection det;
    float mask_area = 0.0f;
    int mask_width = 0;
    int mask_height = 0;
    std::vector<std::vector<cv::Point2f>> segments;
};

struct RoiInfo {
    bool valid = false;
    std::string status = "";
    std::string final_decision = "";
    float final_confidence = 0.0f;
    Detection selected_detection;
    bool has_selected_detection = false;
    std::vector<Detection> detector_predictions;
    std::vector<ClassificationItem> classifier_topk;
    std::vector<std::vector<int>> detector_output_shapes;
    std::string detector_model;
    std::string detector_meta_path;
    int detector_num_classes = 0;
    float detector_conf_threshold = 0.0f;
    float detector_nms_threshold = 0.0f;
    std::vector<std::string> detector_class_names;
    std::string detector_select_policy;
    int detector_target_class_id = -1;
    std::string detector_target_class_name;
    std::vector<float> roi_bbox;
    std::vector<float> base_bbox;
    std::string roi_mode = "full_box";
    std::string pipeline_roi_mode = "full_box";
    float padding_ratio = 0.0f;
    std::map<std::string, float> relative_box;
    std::string class_key;
    std::string matched_class_key;
    std::string source;
    double detector_ms = 0.0;
    double crop_ms = 0.0;
    double classifier_ms = 0.0;
};

struct InferenceResult {
    std::string task = "detection";
    std::string message;
    std::vector<Detection> detections;
    std::vector<ClassificationItem> topk;
    std::vector<ObbDetection> obbs;
    std::vector<SegmentationDetection> segmentations;
    RoiInfo roi;
};

struct RknnTiming {
    double inputs_set_ms = 0.0;
    double run_ms = 0.0;
    double outputs_get_ms = 0.0;
    double outputs_copy_ms = 0.0;
    double outputs_release_ms = 0.0;
    double total_ms = 0.0;
};

struct PreprocessDetailTiming {
    std::string backend = "cpu";
    std::string rga_mode = "off";
    std::string fallback_reason;
    int orig_w = 0;
    int orig_h = 0;
    int resized_w = 0;
    int resized_h = 0;
    double meta_calc_ms = 0.0;
    double cpu_resize_ms = 0.0;
    double rga_resize_color_ms = 0.0;
    double rga_resize_only_ms = 0.0;
    double cpu_canvas_alloc_ms = 0.0;
    double cpu_padding_copy_ms = 0.0;
    double cpu_cvtcolor_ms = 0.0;
    double continuity_ms = 0.0;
    double total_ms = 0.0;
};

struct StreamDetailTiming {
    double capture_read_ms = 0.0;
    double snapshot_clone_ms = 0.0;
    double annotated_draw_ms = 0.0;
    double state_update_ms = 0.0;
    double loop_total_ms = 0.0;
};

struct InferTiming {
    double request_read_ms = 0.0;
    double body_extract_ms = 0.0;
    double image_decode_ms = 0.0;
    double preprocess_ms = 0.0;
    std::string preprocess_backend = "cpu";
    PreprocessDetailTiming preprocess_detail;
    StreamDetailTiming stream_detail;
    RknnTiming rknn;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
};

static std::atomic<uint64_t> g_total{0};
static std::atomic<uint64_t> g_errors{0};
static std::mutex g_latency_mutex;
static std::vector<double> g_latencies;

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static std::string json_escape(const std::string& s) {
    std::ostringstream os;
    for (unsigned char c : s) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                } else {
                    os << c;
                }
        }
    }
    return os.str();
}

static std::vector<unsigned char> read_binary_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("cannot open file: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(ifs)), {});
}

static std::string normalize_task_name(std::string task) {
    std::transform(task.begin(), task.end(), task.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    if (task == "detect" || task == "yolo_detection" || task == "object_detection") return "detection";
    if (task == "cls" || task == "classify" || task == "image_classification") return "classification";
    if (task == "obb" || task == "oriented_detection" || task == "oriented_bbox_detection" ||
        task == "rotated_detection" || task == "rotated_bbox_detection" || task == "yolo_obb" ||
        task == "yolov8_obb") return "obb_detection";
    if (task == "seg" || task == "segment" || task == "instance_segmentation" ||
        task == "yolo_seg" || task == "yolov8_seg" || task == "mask_segmentation") return "segmentation";
    if (task == "roi" || task == "roi_cls" || task == "roi_classification" ||
        task == "two_stage_classification" || task == "pipeline_roi_classification" ||
        task == "roi_detection_classification") return "roi_classification";
    return task;
}

static bool is_classification_task(const Args& args) {
    return normalize_task_name(args.task) == "classification";
}

static std::vector<std::string> load_class_names_simple_yaml(const std::string& path, int num_classes) {
    std::vector<std::string> names;
    std::ifstream ifs(path);
    if (!ifs) {
        for (int i = 0; i < num_classes; ++i) names.push_back(std::to_string(i));
        return names;
    }

    std::string line;
    bool in_class_names = false;
    while (std::getline(ifs, line)) {
        std::string t = line;
        t.erase(0, t.find_first_not_of(" \t\r\n"));
        t.erase(t.find_last_not_of(" \t\r\n") + 1);

        if (t.rfind("class_names:", 0) == 0 || t.rfind("names:", 0) == 0) {
            in_class_names = true;
            continue;
        }
        if (in_class_names) {
            if (t.empty() || t[0] == '#') continue;
            std::string name;
            if (t[0] == '-') {
                name = t.substr(1);
            } else {
                // Support YOLO-style mapping under names:, for example:
                // names:
                //   0: ok
                //   1: ng
                auto colon = t.find(':');
                if (colon == std::string::npos) continue;
                std::string key = t.substr(0, colon);
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                bool numeric_key = !key.empty() && std::all_of(key.begin(), key.end(), [](unsigned char c) { return std::isdigit(c); });
                if (!numeric_key) {
                    in_class_names = false;
                    continue;
                }
                name = t.substr(colon + 1);
            }
            name.erase(0, name.find_first_not_of(" \t\"'"));
            name.erase(name.find_last_not_of(" \t\"'\r\n") + 1);
            if (!name.empty()) names.push_back(name);
        }
    }

    if (names.empty()) {
        for (int i = 0; i < num_classes; ++i) names.push_back(std::to_string(i));
    }
    // Some generated YAML files may contain both class_names and names sections.
    // This lightweight parser scans the whole file, so keep only the configured
    // number of classes to avoid duplicated health/topk labels such as
    // ["tube", "tube"] or ["ng", "ok", "ng", "ok"].
    if (num_classes > 0 && (int)names.size() > num_classes) {
        names.resize((size_t)num_classes);
    }
    return names;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static std::vector<float> softmax_vec(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size(), 0.0f);
    if (logits.empty()) return probs;
    float m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - m);
        sum += probs[i];
    }
    if (sum <= 0.0) return probs;
    for (float& v : probs) v = (float)(v / sum);
    return probs;
}

static std::vector<float> softmax16(const float* p) {
    float m = p[0];
    for (int i = 1; i < 16; ++i) m = std::max(m, p[i]);
    float sum = 0.0f;
    std::vector<float> out(16);
    for (int i = 0; i < 16; ++i) {
        out[i] = std::exp(p[i] - m);
        sum += out[i];
    }
    for (float& v : out) v /= (sum + 1e-12f);
    return out;
}

static float dfl_expectation(const std::vector<float>& logits, int side, int hw, int idx) {
    // v0.1.5: no heap allocation. DFL only runs for candidates above threshold.
    float vals[16];
    float max_v = -std::numeric_limits<float>::infinity();
    for (int k = 0; k < 16; ++k) {
        int c = side * 16 + k;
        float v = logits[c * hw + idx];
        vals[k] = v;
        if (v > max_v) max_v = v;
    }

    float sum = 0.0f;
    float weighted = 0.0f;
    for (int k = 0; k < 16; ++k) {
        float e = std::exp(vals[k] - max_v);
        sum += e;
        weighted += e * (float)k;
    }
    return weighted / (sum + 1e-12f);
}


static void fill_letterbox_meta(const cv::Mat& bgr, int input_w, int input_h, PreprocessMeta& meta, int& new_w, int& new_h) {
    meta.orig_w = bgr.cols;
    meta.orig_h = bgr.rows;
    meta.input_w = input_w;
    meta.input_h = input_h;

    float r = std::min(input_w / (float)bgr.cols, input_h / (float)bgr.rows);
    new_w = std::max(1, (int)std::round(bgr.cols * r));
    new_h = std::max(1, (int)std::round(bgr.rows * r));
    meta.ratio = r;
    meta.pad_x = (input_w - new_w) / 2.0f;
    meta.pad_y = (input_h - new_h) / 2.0f;
}

static bool validate_letterbox_roi(const PreprocessMeta& meta, int new_w, int new_h, int& left, int& top, std::string* error_message) {
    left = std::round(meta.pad_x - 0.1f);
    top = std::round(meta.pad_y - 0.1f);
    if (left < 0 || top < 0 || left + new_w > meta.input_w || top + new_h > meta.input_h) {
        if (error_message) *error_message = "invalid letterbox ROI";
        return false;
    }
    return true;
}

static cv::Mat letterbox_rgb_uint8_cpu_timed(
    const cv::Mat& bgr,
    int input_w,
    int input_h,
    PreprocessMeta& meta,
    PreprocessDetailTiming* detail = nullptr
) {
    double total0 = now_ms();
    double t0 = now_ms();
    int new_w = 0, new_h = 0;
    fill_letterbox_meta(bgr, input_w, input_h, meta, new_w, new_h);
    double t1 = now_ms();

    if (detail) {
        *detail = PreprocessDetailTiming();
        detail->backend = "cpu";
        detail->rga_mode = "off";
        detail->orig_w = bgr.cols;
        detail->orig_h = bgr.rows;
        detail->resized_w = new_w;
        detail->resized_h = new_h;
        detail->meta_calc_ms = t1 - t0;
    }

    t0 = now_ms();
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    t1 = now_ms();
    if (detail) detail->cpu_resize_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    t1 = now_ms();
    if (detail) detail->cpu_canvas_alloc_ms = t1 - t0;

    t0 = now_ms();
    int left = 0, top = 0;
    std::string roi_error;
    if (!validate_letterbox_roi(meta, new_w, new_h, left, top, &roi_error)) {
        if (detail) detail->fallback_reason = roi_error;
        return cv::Mat();
    }
    resized.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    t1 = now_ms();
    if (detail) detail->cpu_padding_copy_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    t1 = now_ms();
    if (detail) detail->cpu_cvtcolor_ms = t1 - t0;

    t0 = now_ms();
    if (!rgb.isContinuous()) rgb = rgb.clone();
    t1 = now_ms();
    if (detail) {
        detail->continuity_ms = t1 - t0;
        detail->total_ms = now_ms() - total0;
    }
    return rgb;
}

static cv::Mat letterbox_rgb_uint8_rga_resize_color_timed(
    const cv::Mat& bgr,
    int input_w,
    int input_h,
    PreprocessMeta& meta,
    PreprocessDetailTiming* detail,
    std::string* error_message = nullptr
) {
    double total0 = now_ms();
    double t0 = now_ms();
    int new_w = 0, new_h = 0;
    fill_letterbox_meta(bgr, input_w, input_h, meta, new_w, new_h);
    double t1 = now_ms();

    if (detail) {
        *detail = PreprocessDetailTiming();
        detail->backend = "rga";
        detail->rga_mode = "resize_color";
        detail->orig_w = bgr.cols;
        detail->orig_h = bgr.rows;
        detail->resized_w = new_w;
        detail->resized_h = new_h;
        detail->meta_calc_ms = t1 - t0;
    }

    t0 = now_ms();
    cv::Mat resized_rgb;
    std::string rga_error;
    if (!visionops::rga_resize_bgr_to_rgb(bgr, resized_rgb, new_w, new_h, &rga_error)) {
        if (error_message) *error_message = rga_error;
        if (detail) {
            detail->rga_resize_color_ms = now_ms() - t0;
            detail->fallback_reason = rga_error;
            detail->total_ms = now_ms() - total0;
        }
        return cv::Mat();
    }
    t1 = now_ms();
    if (detail) detail->rga_resize_color_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat canvas(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    t1 = now_ms();
    if (detail) detail->cpu_canvas_alloc_ms = t1 - t0;

    t0 = now_ms();
    int left = 0, top = 0;
    std::string roi_error;
    if (!validate_letterbox_roi(meta, new_w, new_h, left, top, &roi_error)) {
        if (error_message) *error_message = roi_error;
        if (detail) detail->fallback_reason = roi_error;
        return cv::Mat();
    }
    resized_rgb.copyTo(canvas(cv::Rect(left, top, new_w, new_h)));
    t1 = now_ms();
    if (detail) detail->cpu_padding_copy_ms = t1 - t0;

    t0 = now_ms();
    if (!canvas.isContinuous()) canvas = canvas.clone();
    t1 = now_ms();
    if (detail) {
        detail->continuity_ms = t1 - t0;
        detail->total_ms = now_ms() - total0;
    }
    return canvas;
}

static cv::Mat letterbox_rgb_uint8_rga_resize_only_timed(
    const cv::Mat& bgr,
    int input_w,
    int input_h,
    PreprocessMeta& meta,
    PreprocessDetailTiming* detail,
    std::string* error_message = nullptr
) {
    double total0 = now_ms();
    double t0 = now_ms();
    int new_w = 0, new_h = 0;
    fill_letterbox_meta(bgr, input_w, input_h, meta, new_w, new_h);
    double t1 = now_ms();

    if (detail) {
        *detail = PreprocessDetailTiming();
        detail->backend = "rga";
        detail->rga_mode = "resize_only";
        detail->orig_w = bgr.cols;
        detail->orig_h = bgr.rows;
        detail->resized_w = new_w;
        detail->resized_h = new_h;
        detail->meta_calc_ms = t1 - t0;
    }

    t0 = now_ms();
    cv::Mat resized_bgr;
    std::string rga_error;
    if (!visionops::rga_resize_bgr_to_bgr(bgr, resized_bgr, new_w, new_h, &rga_error)) {
        if (error_message) *error_message = rga_error;
        if (detail) {
            detail->rga_resize_only_ms = now_ms() - t0;
            detail->fallback_reason = rga_error;
            detail->total_ms = now_ms() - total0;
        }
        return cv::Mat();
    }
    t1 = now_ms();
    if (detail) detail->rga_resize_only_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat canvas_bgr(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    t1 = now_ms();
    if (detail) detail->cpu_canvas_alloc_ms = t1 - t0;

    t0 = now_ms();
    int left = 0, top = 0;
    std::string roi_error;
    if (!validate_letterbox_roi(meta, new_w, new_h, left, top, &roi_error)) {
        if (error_message) *error_message = roi_error;
        if (detail) detail->fallback_reason = roi_error;
        return cv::Mat();
    }
    resized_bgr.copyTo(canvas_bgr(cv::Rect(left, top, new_w, new_h)));
    t1 = now_ms();
    if (detail) detail->cpu_padding_copy_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat rgb;
    cv::cvtColor(canvas_bgr, rgb, cv::COLOR_BGR2RGB);
    t1 = now_ms();
    if (detail) detail->cpu_cvtcolor_ms = t1 - t0;

    t0 = now_ms();
    if (!rgb.isContinuous()) rgb = rgb.clone();
    t1 = now_ms();
    if (detail) {
        detail->continuity_ms = t1 - t0;
        detail->total_ms = now_ms() - total0;
    }
    return rgb;
}

static cv::Mat classification_rgb_uint8_cpu_timed(
    const cv::Mat& bgr,
    int input_w,
    int input_h,
    PreprocessMeta& meta,
    PreprocessDetailTiming* detail = nullptr
) {
    double total0 = now_ms();
    double t0 = now_ms();

    meta.orig_w = bgr.cols;
    meta.orig_h = bgr.rows;
    meta.input_w = input_w;
    meta.input_h = input_h;
    meta.ratio = 1.0f;
    meta.pad_x = 0.0f;
    meta.pad_y = 0.0f;

    if (detail) {
        *detail = PreprocessDetailTiming();
        detail->backend = "cpu_classification_resize";
        detail->rga_mode = "off";
        detail->orig_w = bgr.cols;
        detail->orig_h = bgr.rows;
        detail->resized_w = input_w;
        detail->resized_h = input_h;
    }

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(input_w, input_h), 0, 0, cv::INTER_LINEAR);
    double t1 = now_ms();
    if (detail) detail->cpu_resize_ms = t1 - t0;

    t0 = now_ms();
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    t1 = now_ms();
    if (detail) detail->cpu_cvtcolor_ms = t1 - t0;

    t0 = now_ms();
    if (!rgb.isContinuous()) rgb = rgb.clone();
    t1 = now_ms();
    if (detail) {
        detail->continuity_ms = t1 - t0;
        detail->total_ms = now_ms() - total0;
    }
    return rgb;
}

static cv::Mat preprocess_rgb_uint8(
    const cv::Mat& bgr,
    const Args& args,
    PreprocessMeta& meta,
    std::string& actual_backend,
    std::string* fallback_reason = nullptr,
    PreprocessDetailTiming* detail = nullptr
) {
    actual_backend = "cpu";
    if (fallback_reason) fallback_reason->clear();
    if (detail) *detail = PreprocessDetailTiming();

    // v0.8.3.4: classification uses direct resize + BGR->RGB.
    // Do not use detection letterbox/RGA padding here; classification models
    // usually expect a fixed full-image crop such as 224x224.
    if (is_classification_task(args)) {
        cv::Mat rgb = classification_rgb_uint8_cpu_timed(bgr, args.input_w, args.input_h, meta, detail);
        actual_backend = "cpu_classification_resize";
        if (detail) {
            detail->backend = actual_backend;
            detail->rga_mode = "off";
        }
        return rgb;
    }

    if (args.preprocess_status.rga_enabled_for_preprocess &&
        args.preprocess_status.active_backend == "rga") {
        std::string rga_error;
        cv::Mat rgb;
        if (args.preprocess_status.active_rga_mode == "resize_only") {
            rgb = letterbox_rgb_uint8_rga_resize_only_timed(
                bgr, args.input_w, args.input_h, meta, detail, &rga_error
            );
        } else {
            rgb = letterbox_rgb_uint8_rga_resize_color_timed(
                bgr, args.input_w, args.input_h, meta, detail, &rga_error
            );
        }
        if (!rgb.empty()) {
            actual_backend = (args.preprocess_status.active_rga_mode == "resize_only")
                ? "rga_resize_only"
                : "rga_resize_color";
            if (detail) detail->backend = actual_backend;
            return rgb;
        }

        // Keep the service robust. A runtime RGA failure should not break the
        // already validated CPU path; it should only be visible in timing/debug.
        actual_backend = "cpu_fallback";
        if (fallback_reason) *fallback_reason = rga_error.empty() ? "RGA preprocess failed" : rga_error;
        PreprocessDetailTiming failed_rga_detail;
        if (detail) failed_rga_detail = *detail;
        cv::Mat cpu_rgb = letterbox_rgb_uint8_cpu_timed(bgr, args.input_w, args.input_h, meta, detail);
        if (detail) {
            detail->backend = "cpu_fallback";
            detail->rga_mode = args.preprocess_status.active_rga_mode;
            detail->fallback_reason = rga_error.empty() ? "RGA preprocess failed" : rga_error;
            // Preserve the failed RGA call cost in the matching field, then add CPU fallback costs.
            if (failed_rga_detail.rga_resize_color_ms > 0.0) {
                detail->rga_resize_color_ms = failed_rga_detail.rga_resize_color_ms;
            }
            if (failed_rga_detail.rga_resize_only_ms > 0.0) {
                detail->rga_resize_only_ms = failed_rga_detail.rga_resize_only_ms;
            }
        }
        return cpu_rgb;
    }

    cv::Mat rgb = letterbox_rgb_uint8_cpu_timed(bgr, args.input_w, args.input_h, meta, detail);
    actual_backend = "cpu";
    if (detail) {
        detail->backend = "cpu";
        detail->rga_mode = "off";
    }
    return rgb;
}

static void map_box_to_original(Detection& d, const PreprocessMeta& m) {
    auto mapx = [&](float x) {
        x = (x - m.pad_x) / m.ratio;
        return std::min(std::max(x, 0.0f), (float)(m.orig_w - 1));
    };
    auto mapy = [&](float y) {
        y = (y - m.pad_y) / m.ratio;
        return std::min(std::max(y, 0.0f), (float)(m.orig_h - 1));
    };
    d.x1 = mapx(d.x1);
    d.x2 = mapx(d.x2);
    d.y1 = mapy(d.y1);
    d.y2 = mapy(d.y2);
}

static float iou_xyxy(const Detection& a, const Detection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, x2 - x1);
    float ih = std::max(0.0f, y2 - y1);
    float inter = iw * ih;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<Detection> nms(const std::vector<Detection>& dets, float iou_thresh, int max_det) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<Detection> keep;
    std::vector<char> removed(dets.size(), 0);

    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        if ((int)keep.size() >= max_det) break;

        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id && iou_xyxy(dets[i], dets[j]) > iou_thresh) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}


static float iou_xyxy_obb(const ObbDetection& a, const ObbDetection& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, x2 - x1);
    float ih = std::max(0.0f, y2 - y1);
    float inter = iw * ih;
    float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-6f);
}

static std::vector<ObbDetection> nms_obb_bbox(const std::vector<ObbDetection>& dets, float iou_thresh, int max_det) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return dets[a].score > dets[b].score;
    });

    std::vector<ObbDetection> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        if ((int)keep.size() >= max_det) break;

        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id && iou_xyxy_obb(dets[i], dets[j]) > iou_thresh) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}

static float normalize_angle_radian(float angle) {
    if (std::fabs(angle) > 2.0f * (float)CV_PI) {
        return angle * (float)CV_PI / 180.0f;
    }
    return angle;
}

static std::vector<cv::Point2f> xywhr_to_points(float cx, float cy, float w, float h, float angle) {
    angle = normalize_angle_radian(angle);
    const float cos_a = std::cos(angle);
    const float sin_a = std::sin(angle);
    const float dx = w * 0.5f;
    const float dy = h * 0.5f;
    std::vector<cv::Point2f> corners = {
        cv::Point2f(-dx, -dy),
        cv::Point2f( dx, -dy),
        cv::Point2f( dx,  dy),
        cv::Point2f(-dx,  dy),
    };
    for (auto& p : corners) {
        float x = p.x;
        float y = p.y;
        p.x = x * cos_a - y * sin_a + cx;
        p.y = x * sin_a + y * cos_a + cy;
    }
    return corners;
}

static void clip_points_to_image(std::vector<cv::Point2f>& points, int w, int h) {
    for (auto& p : points) {
        p.x = std::min(std::max(p.x, 0.0f), (float)(w - 1));
        p.y = std::min(std::max(p.y, 0.0f), (float)(h - 1));
    }
}

static bool finalize_obb_detection(
    ObbDetection& d,
    float cx_input,
    float cy_input,
    float bw_input,
    float bh_input,
    float angle,
    const PreprocessMeta& meta
) {
    if (bw_input <= 2.0f || bh_input <= 2.0f) return false;
    d.cx = (cx_input - meta.pad_x) / std::max(meta.ratio, 1e-6f);
    d.cy = (cy_input - meta.pad_y) / std::max(meta.ratio, 1e-6f);
    d.w = bw_input / std::max(meta.ratio, 1e-6f);
    d.h = bh_input / std::max(meta.ratio, 1e-6f);
    d.angle = normalize_angle_radian(angle);
    if (d.w <= 2.0f || d.h <= 2.0f) return false;

    d.points = xywhr_to_points(d.cx, d.cy, d.w, d.h, d.angle);
    clip_points_to_image(d.points, meta.orig_w, meta.orig_h);

    float min_x = d.points[0].x, max_x = d.points[0].x;
    float min_y = d.points[0].y, max_y = d.points[0].y;
    for (const auto& pnt : d.points) {
        min_x = std::min(min_x, pnt.x);
        max_x = std::max(max_x, pnt.x);
        min_y = std::min(min_y, pnt.y);
        max_y = std::max(max_y, pnt.y);
    }
    d.x1 = std::min(std::max(min_x, 0.0f), (float)(meta.orig_w - 1));
    d.y1 = std::min(std::max(min_y, 0.0f), (float)(meta.orig_h - 1));
    d.x2 = std::min(std::max(max_x, 0.0f), (float)(meta.orig_w - 1));
    d.y2 = std::min(std::max(max_y, 0.0f), (float)(meta.orig_h - 1));
    return d.x2 > d.x1 + 2.0f && d.y2 > d.y1 + 2.0f;
}



static std::string tensor_type_name(rknn_tensor_type type) {
    switch (type) {
        case RKNN_TENSOR_FLOAT32: return "float32";
        case RKNN_TENSOR_INT8: return "int8";
        case RKNN_TENSOR_UINT8: return "uint8";
        case RKNN_TENSOR_INT16: return "int16";
        case RKNN_TENSOR_UINT16: return "uint16";
        case RKNN_TENSOR_INT32: return "int32";
        case RKNN_TENSOR_UINT32: return "uint32";
        default: return "unknown(" + std::to_string((int)type) + ")";
    }
}


static float float16_to_float32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = h & 0x03FFu;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            // Normalize subnormal half.
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03FFu;
            uint32_t exp32 = (exp + (127 - 15)) << 23;
            uint32_t mant32 = mant << 13;
            f = sign | exp32 | mant32;
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        uint32_t exp32 = (exp + (127 - 15)) << 23;
        uint32_t mant32 = mant << 13;
        f = sign | exp32 | mant32;
    }

    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

static std::vector<float> copy_output_as_float(
    const rknn_output& out,
    const rknn_tensor_attr& attr,
    bool runtime_already_float
) {
    if (!out.buf) {
        throw std::runtime_error("RKNN output buffer is null");
    }

    std::vector<float> data;

    if (runtime_already_float || attr.type == RKNN_TENSOR_FLOAT32) {
        size_t n = out.size / sizeof(float);
        const float* fp = static_cast<const float*>(out.buf);
        data.assign(fp, fp + n);
        return data;
    }

    const int n = (int)attr.n_elems;
    if (n <= 0) {
        throw std::runtime_error("invalid RKNN output n_elems for quant mode");
    }

    data.resize(n);
    const float scale = attr.scale;
    const int zp = attr.zp;

    switch (attr.type) {
        case RKNN_TENSOR_FLOAT16: {
            const uint16_t* hp = static_cast<const uint16_t*>(out.buf);
            size_t n_half = out.size / sizeof(uint16_t);
            data.resize(n_half);
            for (size_t i = 0; i < n_half; ++i) data[i] = float16_to_float32(hp[i]);
            break;
        }
        case RKNN_TENSOR_INT8: {
            const int8_t* q = static_cast<const int8_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int)q[i] - zp) * scale;
            break;
        }
        case RKNN_TENSOR_UINT8: {
            const uint8_t* q = static_cast<const uint8_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int)q[i] - zp) * scale;
            break;
        }
        case RKNN_TENSOR_INT16: {
            const int16_t* q = static_cast<const int16_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int)q[i] - zp) * scale;
            break;
        }
        case RKNN_TENSOR_UINT16: {
            const uint16_t* q = static_cast<const uint16_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int)q[i] - zp) * scale;
            break;
        }
        case RKNN_TENSOR_INT32: {
            const int32_t* q = static_cast<const int32_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int)q[i] - zp) * scale;
            break;
        }
        case RKNN_TENSOR_UINT32: {
            const uint32_t* q = static_cast<const uint32_t*>(out.buf);
            for (int i = 0; i < n; ++i) data[i] = ((int64_t)q[i] - zp) * scale;
            break;
        }
        default:
            throw std::runtime_error(
                "unsupported RKNN output tensor type in quant mode: " + tensor_type_name(attr.type)
            );
    }
    return data;
}

class RknnRunner {
public:
    explicit RknnRunner(const Args& args) : args_(args) {
        model_data_ = read_binary_file(args_.model);
        int ret = rknn_init(&ctx_, model_data_.data(), model_data_.size(), 0, nullptr);
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_init failed, ret=" + std::to_string(ret));

#ifdef RKNN_NPU_CORE_AUTO
        if (args_.npu_core == "auto") {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_AUTO);
        } else if (args_.npu_core == "0") {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0);
        } else if (args_.npu_core == "1") {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_1);
        } else if (args_.npu_core == "2") {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_2);
        } else if (args_.npu_core == "0_1_2" || args_.npu_core == "all") {
            rknn_set_core_mask(ctx_, RKNN_NPU_CORE_0_1_2);
        }
#endif

        memset(&io_num_, 0, sizeof(io_num_));
        ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
        if (ret != RKNN_SUCC) throw std::runtime_error("RKNN_QUERY_IN_OUT_NUM failed, ret=" + std::to_string(ret));

        input_attrs_.resize(io_num_.n_input);
        for (uint32_t i = 0; i < io_num_.n_input; ++i) {
            memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
            input_attrs_[i].index = i;
            rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
        }

        output_attrs_.resize(io_num_.n_output);
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
            output_attrs_[i].index = i;
            rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
        }

        std::cout << "[INFO] RKNN loaded: " << args_.model << "\n";
        std::cout << "[INFO] inputs=" << io_num_.n_input << " outputs=" << io_num_.n_output << "\n";
        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            std::cout << "[INFO] output[" << i << "] dims:";
            for (uint32_t j = 0; j < output_attrs_[i].n_dims; ++j) std::cout << " " << output_attrs_[i].dims[j];
            std::cout << " elems=" << output_attrs_[i].n_elems
                      << " type=" << tensor_type_name(output_attrs_[i].type)
                      << " scale=" << output_attrs_[i].scale
                      << " zp=" << output_attrs_[i].zp
                      << "\n";
        }
    }

    ~RknnRunner() {
        if (ctx_) rknn_destroy(ctx_);
    }

    std::vector<Tensor> infer(const cv::Mat& rgb, RknnTiming* timing = nullptr) {
        double t_total0 = now_ms();
        std::lock_guard<std::mutex> guard(mu_);

        rknn_input input;
        memset(&input, 0, sizeof(input));
        input.index = 0;
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        input.size = rgb.total() * rgb.elemSize();
        input.buf = (void*)rgb.data;

        double t0 = now_ms();
        int ret = rknn_inputs_set(ctx_, 1, &input);
        double t1 = now_ms();
        if (timing) timing->inputs_set_ms = t1 - t0;
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed, ret=" + std::to_string(ret));

        t0 = now_ms();
        ret = rknn_run(ctx_, nullptr);
        t1 = now_ms();
        if (timing) timing->run_ms = t1 - t0;
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_run failed, ret=" + std::to_string(ret));

        const bool want_float = (args_.output_mode == "float");
        std::vector<rknn_output> outputs(io_num_.n_output);
        for (auto& out : outputs) {
            memset(&out, 0, sizeof(out));
            out.want_float = want_float ? 1 : 0;
        }

        t0 = now_ms();
        ret = rknn_outputs_get(ctx_, io_num_.n_output, outputs.data(), nullptr);
        t1 = now_ms();
        if (timing) timing->outputs_get_ms = t1 - t0;
        if (ret != RKNN_SUCC) throw std::runtime_error("rknn_outputs_get failed, ret=" + std::to_string(ret));

        t0 = now_ms();
        std::vector<Tensor> tensors;
        tensors.reserve(io_num_.n_output);

        for (uint32_t i = 0; i < io_num_.n_output; ++i) {
            Tensor t;
            for (uint32_t j = 0; j < output_attrs_[i].n_dims; ++j) {
                t.dims.push_back((int)output_attrs_[i].dims[j]);
            }
            t.data = copy_output_as_float(outputs[i], output_attrs_[i], want_float);
            tensors.push_back(std::move(t));
        }

        t1 = now_ms();
        if (timing) timing->outputs_copy_ms = t1 - t0;

        t0 = now_ms();
        rknn_outputs_release(ctx_, io_num_.n_output, outputs.data());
        t1 = now_ms();
        if (timing) {
            timing->outputs_release_ms = t1 - t0;
            timing->total_ms = t1 - t_total0;
        }
        return tensors;
    }

private:
    Args args_;
    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<unsigned char> model_data_;
    std::mutex mu_;
};

static std::vector<Detection> decode_single_yolo_output(
    const Tensor& t,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    std::vector<Detection> dets;
    int nc = args.num_classes;
    int C = 0, N = 0;
    bool channel_first = true;

    if (t.dims.size() == 3) {
        int d1 = t.dims[1], d2 = t.dims[2];
        if (d1 >= 4 + nc && d1 < d2) {
            C = d1; N = d2; channel_first = true;
        } else {
            N = d1; C = d2; channel_first = false;
        }
    } else if (t.dims.size() == 2) {
        int d0 = t.dims[0], d1 = t.dims[1];
        if (d0 >= 4 + nc && d0 < d1) {
            C = d0; N = d1; channel_first = true;
        } else {
            N = d0; C = d1; channel_first = false;
        }
    } else {
        return dets;
    }

    if (C < 4 + nc) return dets;

    auto at = [&](int i, int c) -> float {
        if (channel_first) return t.data[c * N + i];
        return t.data[i * C + c];
    };

    for (int i = 0; i < N; ++i) {
        float cx = at(i, 0);
        float cy = at(i, 1);
        float w = at(i, 2);
        float h = at(i, 3);

        int best_cls = -1;
        float best_score = -1.0f;
        for (int c = 0; c < nc; ++c) {
            float s = at(i, 4 + c);
            if (s < 0.0f || s > 1.0f) s = sigmoid(s);
            if (s > best_score) {
                best_score = s;
                best_cls = c;
            }
        }
        if (best_score < args.conf_threshold) continue;

        Detection d;
        d.class_id = best_cls;
        d.class_name = (best_cls >= 0 && best_cls < (int)class_names.size()) ? class_names[best_cls] : std::to_string(best_cls);
        d.score = best_score;
        d.x1 = cx - w * 0.5f;
        d.y1 = cy - h * 0.5f;
        d.x2 = cx + w * 0.5f;
        d.y2 = cy + h * 0.5f;
        map_box_to_original(d, meta);
        if (d.x2 > d.x1 && d.y2 > d.y1) dets.push_back(d);
    }

    return nms(dets, args.nms_threshold, args.max_det);
}

static bool is_dfl_head(const Tensor& t, int nc) {
    if (t.dims.size() != 4) return false;
    // Typical RKNN output: [1, 64 + nc, H, W]
    return t.dims[0] == 1 && t.dims[1] == 64 + nc && t.dims[2] > 0 && t.dims[3] > 0;
}

static std::vector<Detection> decode_yolov8_dfl_outputs(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    int nc = args.num_classes;
    std::vector<const Tensor*> heads;
    for (const auto& t : outputs) {
        if (is_dfl_head(t, nc)) heads.push_back(&t);
    }
    std::sort(heads.begin(), heads.end(), [](const Tensor* a, const Tensor* b) {
        return a->dims[2] > b->dims[2]; // 80,40,20
    });

    std::vector<Detection> dets;
    if (heads.empty()) return dets;

    for (const Tensor* tp : heads) {
        const Tensor& t = *tp;
        int C = t.dims[1];
        int H = t.dims[2];
        int W = t.dims[3];
        int HW = H * W;
        float stride_x = args.input_w / (float)W;
        float stride_y = args.input_h / (float)H;
        float stride = (stride_x + stride_y) * 0.5f;

        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;

                int best_cls = -1;
                float best_score = -1.0f;
                for (int c = 0; c < nc; ++c) {
                    float s = t.data[(64 + c) * HW + idx];
                    s = sigmoid(s);
                    if (s > best_score) {
                        best_score = s;
                        best_cls = c;
                    }
                }
                if (best_score < args.conf_threshold) continue;

                float l = dfl_expectation(t.data, 0, HW, idx);
                float top = dfl_expectation(t.data, 1, HW, idx);
                float r = dfl_expectation(t.data, 2, HW, idx);
                float b = dfl_expectation(t.data, 3, HW, idx);

                float ax = x + 0.5f;
                float ay = y + 0.5f;

                Detection d;
                d.class_id = best_cls;
                d.class_name = (best_cls >= 0 && best_cls < (int)class_names.size()) ? class_names[best_cls] : std::to_string(best_cls);
                d.score = best_score;
                d.x1 = (ax - l) * stride;
                d.y1 = (ay - top) * stride;
                d.x2 = (ax + r) * stride;
                d.y2 = (ay + b) * stride;
                map_box_to_original(d, meta);
                if (d.x2 > d.x1 && d.y2 > d.y1) dets.push_back(d);
            }
        }
    }

    return nms(dets, args.nms_threshold, args.max_det);
}


// v0.1.1: support YOLOv8 RKNN split-head outputs:
//   [1, 64, H, W]  box DFL
//   [1, nc, H, W]  class logits/probabilities
//   [1, 1,  H, W]  objectness logits/probabilities
static bool is_split_box_head(const Tensor& t) {
    return t.dims.size() == 4 && t.dims[0] == 1 && t.dims[1] == 64 && t.dims[2] > 0 && t.dims[3] > 0;
}

static bool is_split_cls_head(const Tensor& t, int nc) {
    return t.dims.size() == 4 && t.dims[0] == 1 && t.dims[1] == nc && t.dims[2] > 0 && t.dims[3] > 0;
}

static bool is_split_obj_head(const Tensor& t) {
    return t.dims.size() == 4 && t.dims[0] == 1 && t.dims[1] == 1 && t.dims[2] > 0 && t.dims[3] > 0;
}

static bool tensor_need_sigmoid(const Tensor& t) {
    if (t.data.empty()) return false;
    float mn = t.data[0];
    float mx = t.data[0];
    for (float v : t.data) {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    return mn < 0.0f || mx > 1.0f;
}

static float maybe_sigmoid(float v, bool need_sigmoid) {
    return need_sigmoid ? sigmoid(v) : v;
}

static bool find_split_triplet_for_size(
    const std::vector<Tensor>& outputs,
    int nc,
    int h,
    int w,
    const Tensor*& box,
    const Tensor*& cls,
    const Tensor*& obj
) {
    box = nullptr;
    cls = nullptr;
    obj = nullptr;

    std::vector<const Tensor*> one_channel_heads;

    for (const auto& t : outputs) {
        if (t.dims.size() != 4 || t.dims[0] != 1 || t.dims[2] != h || t.dims[3] != w) {
            continue;
        }
        if (is_split_box_head(t)) {
            if (!box) box = &t;
            continue;
        }

        // Important for one-class detectors:
        // Rockchip YOLOv8 split outputs are usually [box, cls, sum] per scale.
        // When num_classes == 1, both cls and sum/objectness are [1,1,H,W].
        // The old else-if logic treated every 1-channel tensor as cls and never
        // assigned obj, so has_split_dfl_outputs() returned false and the model
        // fell through to unsupported single-output decoding.
        if (t.dims[1] == 1) {
            one_channel_heads.push_back(&t);
            continue;
        }

        if (is_split_cls_head(t, nc) && !cls) {
            cls = &t;
        }
    }

    if (nc == 1) {
        if (!cls && !one_channel_heads.empty()) cls = one_channel_heads[0];
        if (!obj && one_channel_heads.size() >= 2) obj = one_channel_heads[1];
    } else {
        if (!obj && !one_channel_heads.empty()) obj = one_channel_heads[0];
    }

    // The third head is a score/sum helper for Rockchip exports and is not used
    // by our Python-aligned confidence calculation. If only box+cls are present,
    // still allow split-head decoding.
    return box != nullptr && cls != nullptr;
}

static bool has_split_dfl_outputs(const std::vector<Tensor>& outputs, int nc) {
    int triplets = 0;
    for (const auto& t : outputs) {
        if (!is_split_box_head(t)) continue;
        const Tensor* box = nullptr;
        const Tensor* cls = nullptr;
        const Tensor* obj = nullptr;
        if (find_split_triplet_for_size(outputs, nc, t.dims[2], t.dims[3], box, cls, obj)) {
            triplets++;
        }
    }
    return triplets > 0;
}

static float logit_threshold(float prob) {
    prob = std::min(std::max(prob, 1e-6f), 1.0f - 1e-6f);
    return std::log(prob / (1.0f - prob));
}

static std::vector<Detection> decode_yolov8_split_dfl_outputs(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    // v0.1.5: optimized split-head postprocess.
    // Main changes:
    // 1) do not touch obj score; Python-aligned confidence uses cls score only.
    // 2) find best raw class score first; sigmoid/clamp only once per anchor.
    // 3) run DFL decode only for anchors passing conf_threshold.
    // 4) DFL expectation avoids heap allocation.
    int nc = args.num_classes;

    std::vector<const Tensor*> box_heads;
    for (const auto& t : outputs) {
        if (is_split_box_head(t)) box_heads.push_back(&t);
    }

    std::sort(box_heads.begin(), box_heads.end(), [](const Tensor* a, const Tensor* b) {
        return a->dims[2] > b->dims[2]; // 80,40,20
    });

    std::vector<Detection> dets;
    dets.reserve(256);

    for (const Tensor* box_head : box_heads) {
        int H = box_head->dims[2];
        int W = box_head->dims[3];
        int HW = H * W;

        const Tensor* box = nullptr;
        const Tensor* cls = nullptr;
        const Tensor* obj = nullptr;
        if (!find_split_triplet_for_size(outputs, nc, H, W, box, cls, obj)) {
            continue;
        }
        (void)obj;

        bool cls_need_sigmoid = tensor_need_sigmoid(*cls);
        const float raw_conf_threshold = cls_need_sigmoid
            ? logit_threshold(args.conf_threshold)
            : args.conf_threshold;

        float stride_x = args.input_w / (float)W;
        float stride_y = args.input_h / (float)H;
        float stride = (stride_x + stride_y) * 0.5f;

        const std::vector<float>& cls_data = cls->data;
        const std::vector<float>& box_data = box->data;

        for (int y = 0; y < H; ++y) {
            const float ay = (float)y + 0.5f;
            for (int x = 0; x < W; ++x) {
                int idx = y * W + x;

                int best_cls = 0;
                float best_raw_score = cls_data[idx];

                // Class-major layout: [1, nc, H, W].
                // Use raw comparison because sigmoid is monotonic.
                for (int c = 1; c < nc; ++c) {
                    float raw_score = cls_data[c * HW + idx];
                    if (raw_score > best_raw_score) {
                        best_raw_score = raw_score;
                        best_cls = c;
                    }
                }

                if (best_raw_score < raw_conf_threshold) continue;

                float best_score = cls_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
                if (best_score < args.conf_threshold) continue;
                if (best_score < 0.0f) best_score = 0.0f;
                if (best_score > 1.0f) best_score = 1.0f;

                float l = dfl_expectation(box_data, 0, HW, idx);
                float top = dfl_expectation(box_data, 1, HW, idx);
                float r = dfl_expectation(box_data, 2, HW, idx);
                float b = dfl_expectation(box_data, 3, HW, idx);

                float ax = (float)x + 0.5f;

                Detection d;
                d.class_id = best_cls;
                d.class_name = (best_cls >= 0 && best_cls < (int)class_names.size())
                    ? class_names[best_cls]
                    : std::to_string(best_cls);
                d.score = best_score;
                d.x1 = (ax - l) * stride;
                d.y1 = (ay - top) * stride;
                d.x2 = (ax + r) * stride;
                d.y2 = (ay + b) * stride;

                map_box_to_original(d, meta);
                if (d.x2 > d.x1 && d.y2 > d.y1) {
                    dets.push_back(d);
                }
            }
        }
    }

    return nms(dets, args.nms_threshold, args.max_det);
}

static std::vector<Detection> postprocess_detection(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    // v0.1.1: split-head RKNN output, e.g.
    // [1,64,H,W] + [1,nc,H,W] + [1,1,H,W] for each scale.
    if (has_split_dfl_outputs(outputs, args.num_classes)) {
        return decode_yolov8_split_dfl_outputs(outputs, args, class_names, meta);
    }

    bool has_dfl = false;
    for (const auto& t : outputs) {
        if (is_dfl_head(t, args.num_classes)) has_dfl = true;
    }
    if (has_dfl) {
        return decode_yolov8_dfl_outputs(outputs, args, class_names, meta);
    }
    if (!outputs.empty()) {
        return decode_single_yolo_output(outputs[0], args, class_names, meta);
    }
    return {};
}

static bool is_rockchip_obb_outputs(const std::vector<Tensor>& outputs, int nc) {
    if (outputs.size() < 4) return false;
    int head_count = 0;
    bool has_angle = false;
    for (const auto& t : outputs) {
        if (t.dims.size() == 4 && t.dims[0] == 1 && t.dims[1] == 64 + nc &&
            (t.dims[2] == 80 || t.dims[2] == 40 || t.dims[2] == 20) && t.dims[3] == t.dims[2]) {
            head_count++;
        }
        if (t.dims.size() >= 3 && t.dims[0] == 1 && t.dims[1] == 1) {
            long long n = 1;
            for (size_t i = 2; i < t.dims.size(); ++i) n *= std::max(1, t.dims[i]);
            if (n == 8400) has_angle = true;
        }
    }
    return head_count >= 3 && has_angle;
}

static const Tensor* find_rockchip_obb_angle_output(const std::vector<Tensor>& outputs) {
    for (const auto& t : outputs) {
        if (t.dims.size() >= 3 && t.dims[0] == 1 && t.dims[1] == 1) {
            long long n = 1;
            for (size_t i = 2; i < t.dims.size(); ++i) n *= std::max(1, t.dims[i]);
            if (n == 8400) return &t;
        }
    }
    return nullptr;
}

static bool rockchip_obb_cls_need_sigmoid(const Tensor& head, int nc) {
    if (head.dims.size() != 4) return true;
    int H = head.dims[2];
    int W = head.dims[3];
    int HW = H * W;
    if (HW <= 0 || nc <= 0) return true;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    int step = std::max(1, HW / 1024);
    for (int c = 0; c < nc; ++c) {
        int base = (64 + c) * HW;
        for (int idx = 0; idx < HW; idx += step) {
            float v = head.data[base + idx];
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    }
    return mn < 0.0f || mx > 1.0f;
}

static std::vector<ObbDetection> decode_rockchip_obb_outputs(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    const int nc = args.num_classes;
    const Tensor* angle_t = find_rockchip_obb_angle_output(outputs);
    if (!angle_t) return {};

    std::vector<const Tensor*> heads;
    for (const auto& t : outputs) {
        if (t.dims.size() == 4 && t.dims[0] == 1 && t.dims[1] == 64 + nc && t.dims[2] > 0 && t.dims[3] > 0) {
            heads.push_back(&t);
        }
    }
    std::sort(heads.begin(), heads.end(), [](const Tensor* a, const Tensor* b) {
        return a->dims[2] > b->dims[2]; // 80,40,20
    });
    if (heads.empty()) return {};

    const bool angle_need_sigmoid = tensor_need_sigmoid(*angle_t);
    std::vector<ObbDetection> candidates;
    candidates.reserve(256);

    int angle_offset = 0;
    for (const Tensor* head : heads) {
        const int H = head->dims[2];
        const int W = head->dims[3];
        const int HW = H * W;
        const float stride_x = args.input_w / (float)W;
        const float stride_y = args.input_h / (float)H;
        const float stride = (stride_x + stride_y) * 0.5f;
        const bool cls_need_sigmoid = rockchip_obb_cls_need_sigmoid(*head, nc);
        const float raw_conf_threshold = cls_need_sigmoid ? logit_threshold(args.conf_threshold) : args.conf_threshold;

        for (int y = 0; y < H; ++y) {
            const float ay = (float)y + 0.5f;
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                const int global_idx = angle_offset + idx;
                if (global_idx < 0 || global_idx >= (int)angle_t->data.size()) continue;

                int best_cls = 0;
                float best_raw_score = head->data[(64 + 0) * HW + idx];
                for (int c = 1; c < nc; ++c) {
                    float raw = head->data[(64 + c) * HW + idx];
                    if (raw > best_raw_score) {
                        best_raw_score = raw;
                        best_cls = c;
                    }
                }
                if (best_raw_score < raw_conf_threshold) continue;

                float score = cls_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
                if (score < args.conf_threshold) continue;
                score = std::min(std::max(score, 0.0f), 1.0f);

                float l = dfl_expectation(head->data, 0, HW, idx);
                float top = dfl_expectation(head->data, 1, HW, idx);
                float r = dfl_expectation(head->data, 2, HW, idx);
                float b = dfl_expectation(head->data, 3, HW, idx);

                float angle_raw = angle_t->data[global_idx];
                float angle_sigmoid = angle_need_sigmoid ? sigmoid(angle_raw) : angle_raw;
                float angle = (angle_sigmoid - 0.25f) * (float)CV_PI;

                float ax = (float)x + 0.5f;
                float ox = (r - l) * 0.5f;
                float oy = (b - top) * 0.5f;
                float cos_a = std::cos(angle);
                float sin_a = std::sin(angle);
                float cx_input = (ox * cos_a - oy * sin_a + ax) * stride;
                float cy_input = (ox * sin_a + oy * cos_a + ay) * stride;
                float bw_input = (l + r) * stride;
                float bh_input = (top + b) * stride;

                ObbDetection d;
                d.class_id = best_cls;
                d.class_name = (best_cls >= 0 && best_cls < (int)class_names.size())
                    ? class_names[best_cls]
                    : std::to_string(best_cls);
                d.score = score;
                if (finalize_obb_detection(d, cx_input, cy_input, bw_input, bh_input, angle, meta)) {
                    candidates.push_back(std::move(d));
                }
            }
        }
        angle_offset += HW;
    }
    return nms_obb_bbox(candidates, args.nms_threshold, args.max_det);
}

static bool normalize_obb_single_output_shape(const Tensor& t, int expected_channels, int& C, int& N, bool& channel_first) {
    if (t.dims.size() == 3) {
        int d1 = t.dims[1], d2 = t.dims[2];
        if (d1 >= expected_channels && d1 < d2) { C = d1; N = d2; channel_first = true; return true; }
        if (d2 >= expected_channels) { N = d1; C = d2; channel_first = false; return true; }
    } else if (t.dims.size() == 2) {
        int d0 = t.dims[0], d1 = t.dims[1];
        if (d0 >= expected_channels && d0 < d1) { C = d0; N = d1; channel_first = true; return true; }
        if (d1 >= expected_channels) { N = d0; C = d1; channel_first = false; return true; }
    }
    return false;
}

static std::vector<ObbDetection> decode_single_obb_output(
    const Tensor& t,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    const int nc = args.num_classes;
    const int expected_channels = 4 + nc + 1;
    int C = 0, N = 0;
    bool channel_first = true;
    if (!normalize_obb_single_output_shape(t, expected_channels, C, N, channel_first)) return {};

    auto at = [&](int i, int c) -> float {
        return channel_first ? t.data[c * N + i] : t.data[i * C + c];
    };

    float cls_min = std::numeric_limits<float>::infinity();
    float cls_max = -std::numeric_limits<float>::infinity();
    int cls_probe = std::min(N, 2048);
    for (int i = 0; i < cls_probe; ++i) {
        for (int c = 0; c < nc; ++c) {
            float v = at(i, 4 + c);
            cls_min = std::min(cls_min, v);
            cls_max = std::max(cls_max, v);
        }
    }
    bool scores_need_sigmoid = cls_min < 0.0f || cls_max > 1.0f;
    float max_box_abs = 0.0f;
    int probe = std::min(N, 2048);
    for (int i = 0; i < probe; ++i) {
        for (int c = 0; c < 4; ++c) max_box_abs = std::max(max_box_abs, std::fabs(at(i, c)));
    }
    bool normalized_box = max_box_abs <= 2.0f;

    std::vector<ObbDetection> candidates;
    candidates.reserve(256);
    for (int i = 0; i < N; ++i) {
        float cx = at(i, 0);
        float cy = at(i, 1);
        float bw = at(i, 2);
        float bh = at(i, 3);
        if (normalized_box) {
            cx *= (float)args.input_w;
            bw *= (float)args.input_w;
            cy *= (float)args.input_h;
            bh *= (float)args.input_h;
        }

        int best_cls = 0;
        float best_raw_score = at(i, 4);
        for (int c = 1; c < nc; ++c) {
            float raw = at(i, 4 + c);
            if (raw > best_raw_score) { best_raw_score = raw; best_cls = c; }
        }
        float score = scores_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
        if (score < args.conf_threshold) continue;
        score = std::min(std::max(score, 0.0f), 1.0f);

        float angle = at(i, 4 + nc);
        ObbDetection d;
        d.class_id = best_cls;
        d.class_name = (best_cls >= 0 && best_cls < (int)class_names.size())
            ? class_names[best_cls]
            : std::to_string(best_cls);
        d.score = score;
        if (finalize_obb_detection(d, cx, cy, bw, bh, angle, meta)) {
            candidates.push_back(std::move(d));
        }
    }
    return nms_obb_bbox(candidates, args.nms_threshold, args.max_det);
}

static std::vector<ObbDetection> postprocess_obb(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta,
    std::string* message = nullptr
) {
    if (message) message->clear();
    if (outputs.empty()) {
        if (message) *message = "empty OBB outputs";
        return {};
    }
    if (is_rockchip_obb_outputs(outputs, args.num_classes)) {
        return decode_rockchip_obb_outputs(outputs, args, class_names, meta);
    }
    std::vector<ObbDetection> out = decode_single_obb_output(outputs[0], args, class_names, meta);
    if (out.empty() && message) {
        *message = "no OBB predictions or unsupported OBB output format";
    }
    return out;
}


struct SegCandidate {
    Detection det;
    float ix1 = 0.0f;
    float iy1 = 0.0f;
    float ix2 = 0.0f;
    float iy2 = 0.0f;
    std::vector<float> coeffs;
};

struct ProtoView {
    const Tensor* tensor = nullptr;
    int mask_dim = 0;
    int h = 0;
    int w = 0;
    bool channel_first = true;
};

static bool get_proto_view(const Tensor& t, int mask_dim, ProtoView& view) {
    view = ProtoView();
    view.tensor = &t;
    view.mask_dim = mask_dim;
    if (t.dims.size() == 4 && t.dims[0] == 1) {
        if (t.dims[1] == mask_dim && t.dims[2] > 0 && t.dims[3] > 0) {
            view.h = t.dims[2];
            view.w = t.dims[3];
            view.channel_first = true;
            return true;
        }
        if (t.dims[3] == mask_dim && t.dims[1] > 0 && t.dims[2] > 0) {
            view.h = t.dims[1];
            view.w = t.dims[2];
            view.channel_first = false;
            return true;
        }
    }
    if (t.dims.size() == 3) {
        if (t.dims[0] == mask_dim && t.dims[1] > 0 && t.dims[2] > 0) {
            view.h = t.dims[1];
            view.w = t.dims[2];
            view.channel_first = true;
            return true;
        }
        if (t.dims[2] == mask_dim && t.dims[0] > 0 && t.dims[1] > 0) {
            view.h = t.dims[0];
            view.w = t.dims[1];
            view.channel_first = false;
            return true;
        }
    }
    return false;
}

static float proto_at(const ProtoView& view, int m, int y, int x) {
    if (!view.tensor) return 0.0f;
    if (view.channel_first) {
        return view.tensor->data[m * view.h * view.w + y * view.w + x];
    }
    return view.tensor->data[y * view.w * view.mask_dim + x * view.mask_dim + m];
}

static float tensor_chw_at(const Tensor& t, int c, int y, int x) {
    const int H = t.dims[2];
    const int W = t.dims[3];
    return t.data[c * H * W + y * W + x];
}

static Detection make_detection_from_input_xyxy(
    int class_id,
    const std::string& class_name,
    float score,
    float ix1,
    float iy1,
    float ix2,
    float iy2,
    const PreprocessMeta& meta
) {
    Detection d;
    d.class_id = class_id;
    d.class_name = class_name;
    d.score = std::min(std::max(score, 0.0f), 1.0f);
    d.x1 = ix1;
    d.y1 = iy1;
    d.x2 = ix2;
    d.y2 = iy2;
    map_box_to_original(d, meta);
    return d;
}

static std::vector<int> nms_seg_candidates(const std::vector<SegCandidate>& candidates, float iou_thresh, int max_det) {
    std::vector<int> order(candidates.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return candidates[a].det.score > candidates[b].det.score;
    });

    std::vector<int> keep;
    std::vector<char> removed(candidates.size(), 0);
    for (size_t oi = 0; oi < order.size(); ++oi) {
        int i = order[oi];
        if (removed[i]) continue;
        keep.push_back(i);
        if ((int)keep.size() >= max_det) break;
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            int j = order[oj];
            if (removed[j]) continue;
            if (candidates[i].det.class_id == candidates[j].det.class_id &&
                iou_xyxy(candidates[i].det, candidates[j].det) > iou_thresh) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}

static bool is_rockchip_seg_outputs(const std::vector<Tensor>& outputs, int nc, int mask_dim = 32) {
    if (outputs.size() < 13) return false;
    const int scales[3] = {80, 40, 20};
    for (int i = 0; i < 3; ++i) {
        int base = i * 4;
        int s = scales[i];
        if (outputs[base].dims.size() != 4 || outputs[base].dims[1] != 64 || outputs[base].dims[2] != s || outputs[base].dims[3] != s) return false;
        if (outputs[base + 1].dims.size() != 4 || outputs[base + 1].dims[1] != nc || outputs[base + 1].dims[2] != s || outputs[base + 1].dims[3] != s) return false;
        if (outputs[base + 3].dims.size() != 4 || outputs[base + 3].dims[1] != mask_dim || outputs[base + 3].dims[2] != s || outputs[base + 3].dims[3] != s) return false;
    }
    ProtoView view;
    return get_proto_view(outputs[12], mask_dim, view);
}

static void clip_input_box(float& x1, float& y1, float& x2, float& y2, int input_w, int input_h) {
    x1 = std::min(std::max(x1, 0.0f), (float)(input_w - 1));
    y1 = std::min(std::max(y1, 0.0f), (float)(input_h - 1));
    x2 = std::min(std::max(x2, 0.0f), (float)(input_w - 1));
    y2 = std::min(std::max(y2, 0.0f), (float)(input_h - 1));
}

static std::vector<SegCandidate> decode_rockchip_seg_candidates(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta,
    int mask_dim
) {
    std::vector<SegCandidate> candidates;
    candidates.reserve(512);
    const int groups[3][3] = {{0, 1, 3}, {4, 5, 7}, {8, 9, 11}};
    for (const auto& g : groups) {
        const Tensor& box = outputs[g[0]];
        const Tensor& cls = outputs[g[1]];
        const Tensor& coeff = outputs[g[2]];
        const int H = box.dims[2];
        const int W = box.dims[3];
        const int HW = H * W;
        const float stride_x = args.input_w / (float)W;
        const float stride_y = args.input_h / (float)H;
        const float stride = (stride_x + stride_y) * 0.5f;
        const bool cls_need_sigmoid = tensor_need_sigmoid(cls);
        const float raw_conf_threshold = cls_need_sigmoid ? logit_threshold(args.conf_threshold) : args.conf_threshold;

        for (int y = 0; y < H; ++y) {
            const float ay = (float)y + 0.5f;
            for (int x = 0; x < W; ++x) {
                const int idx = y * W + x;
                int best_cls = 0;
                float best_raw_score = cls.data[idx];
                for (int c = 1; c < args.num_classes; ++c) {
                    float raw = cls.data[c * HW + idx];
                    if (raw > best_raw_score) {
                        best_raw_score = raw;
                        best_cls = c;
                    }
                }
                if (best_raw_score < raw_conf_threshold) continue;
                float score = cls_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
                if (score < args.conf_threshold) continue;

                float l = dfl_expectation(box.data, 0, HW, idx);
                float top = dfl_expectation(box.data, 1, HW, idx);
                float r = dfl_expectation(box.data, 2, HW, idx);
                float b = dfl_expectation(box.data, 3, HW, idx);
                const float ax = (float)x + 0.5f;
                float ix1 = (ax - l) * stride;
                float iy1 = (ay - top) * stride;
                float ix2 = (ax + r) * stride;
                float iy2 = (ay + b) * stride;
                clip_input_box(ix1, iy1, ix2, iy2, args.input_w, args.input_h);
                if (ix2 <= ix1 + 2.0f || iy2 <= iy1 + 2.0f) continue;

                SegCandidate cand;
                cand.det = make_detection_from_input_xyxy(
                    best_cls,
                    (best_cls >= 0 && best_cls < (int)class_names.size()) ? class_names[best_cls] : std::to_string(best_cls),
                    score,
                    ix1,
                    iy1,
                    ix2,
                    iy2,
                    meta
                );
                if (cand.det.x2 <= cand.det.x1 + 2.0f || cand.det.y2 <= cand.det.y1 + 2.0f) continue;
                cand.ix1 = ix1;
                cand.iy1 = iy1;
                cand.ix2 = ix2;
                cand.iy2 = iy2;
                cand.coeffs.resize(mask_dim);
                for (int m = 0; m < mask_dim; ++m) {
                    cand.coeffs[m] = coeff.data[m * HW + idx];
                }
                candidates.push_back(std::move(cand));
            }
        }
    }
    return candidates;
}

static bool normalize_seg_single_output_shape(const Tensor& t, int min_channels, int& C, int& N, bool& channel_first) {
    if (t.dims.size() == 3) {
        int d1 = t.dims[1], d2 = t.dims[2];
        if (d1 >= min_channels && d1 < d2) { C = d1; N = d2; channel_first = true; return true; }
        if (d2 >= min_channels) { N = d1; C = d2; channel_first = false; return true; }
    } else if (t.dims.size() == 2) {
        int d0 = t.dims[0], d1 = t.dims[1];
        if (d0 >= min_channels && d0 < d1) { C = d0; N = d1; channel_first = true; return true; }
        if (d1 >= min_channels) { N = d0; C = d1; channel_first = false; return true; }
    }
    return false;
}

static std::vector<SegCandidate> decode_single_seg_candidates(
    const Tensor& det,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta,
    int& mask_dim,
    std::string* message
) {
    std::vector<SegCandidate> candidates;
    int C = 0, N = 0;
    bool channel_first = true;
    const int min_channels = 4 + args.num_classes + 1;
    if (!normalize_seg_single_output_shape(det, min_channels, C, N, channel_first)) {
        if (message) *message = "unsupported YOLOv8-seg single output shape";
        return candidates;
    }
    mask_dim = C - 4 - args.num_classes;
    if (mask_dim <= 0) {
        if (message) *message = "invalid mask_dim from YOLOv8-seg output";
        return candidates;
    }
    auto at = [&](int i, int c) -> float {
        return channel_first ? det.data[c * N + i] : det.data[i * C + c];
    };

    float cls_min = std::numeric_limits<float>::infinity();
    float cls_max = -std::numeric_limits<float>::infinity();
    int probe = std::min(N, 2048);
    for (int i = 0; i < probe; ++i) {
        for (int c = 0; c < args.num_classes; ++c) {
            float v = at(i, 4 + c);
            cls_min = std::min(cls_min, v);
            cls_max = std::max(cls_max, v);
        }
    }
    const bool cls_need_sigmoid = cls_min < 0.0f || cls_max > 1.0f;

    float max_box_abs = 0.0f;
    for (int i = 0; i < probe; ++i) {
        for (int c = 0; c < 4; ++c) max_box_abs = std::max(max_box_abs, std::fabs(at(i, c)));
    }
    const bool normalized_box = max_box_abs <= 2.0f;

    for (int i = 0; i < N; ++i) {
        int best_cls = 0;
        float best_raw_score = at(i, 4);
        for (int c = 1; c < args.num_classes; ++c) {
            float raw = at(i, 4 + c);
            if (raw > best_raw_score) { best_raw_score = raw; best_cls = c; }
        }
        float score = cls_need_sigmoid ? sigmoid(best_raw_score) : best_raw_score;
        if (score < args.conf_threshold) continue;

        float cx = at(i, 0);
        float cy = at(i, 1);
        float bw = at(i, 2);
        float bh = at(i, 3);
        if (normalized_box) {
            cx *= (float)args.input_w;
            bw *= (float)args.input_w;
            cy *= (float)args.input_h;
            bh *= (float)args.input_h;
        }
        float ix1 = cx - bw * 0.5f;
        float iy1 = cy - bh * 0.5f;
        float ix2 = cx + bw * 0.5f;
        float iy2 = cy + bh * 0.5f;
        clip_input_box(ix1, iy1, ix2, iy2, args.input_w, args.input_h);
        if (ix2 <= ix1 + 2.0f || iy2 <= iy1 + 2.0f) continue;

        SegCandidate cand;
        cand.det = make_detection_from_input_xyxy(
            best_cls,
            (best_cls >= 0 && best_cls < (int)class_names.size()) ? class_names[best_cls] : std::to_string(best_cls),
            score,
            ix1,
            iy1,
            ix2,
            iy2,
            meta
        );
        if (cand.det.x2 <= cand.det.x1 + 2.0f || cand.det.y2 <= cand.det.y1 + 2.0f) continue;
        cand.ix1 = ix1;
        cand.iy1 = iy1;
        cand.ix2 = ix2;
        cand.iy2 = iy2;
        cand.coeffs.resize(mask_dim);
        for (int m = 0; m < mask_dim; ++m) {
            cand.coeffs[m] = at(i, 4 + args.num_classes + m);
        }
        candidates.push_back(std::move(cand));
    }
    return candidates;
}

static std::vector<std::vector<cv::Point2f>> mask_to_segments_cpp(
    const cv::Mat& binary_mask_u8,
    float& area,
    int max_segments = 3,
    int max_points_per_segment = 200
) {
    area = (float)cv::countNonZero(binary_mask_u8);
    std::vector<std::vector<cv::Point2f>> segments;
    if (area <= 0.0f) return segments;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask_u8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return segments;
    std::sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
        return cv::contourArea(a) > cv::contourArea(b);
    });

    for (int i = 0; i < (int)contours.size() && (int)segments.size() < max_segments; ++i) {
        if (cv::contourArea(contours[i]) < 4.0) continue;
        double epsilon = std::max(1.0, 0.002 * cv::arcLength(contours[i], true));
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, true);
        if (approx.size() < 3) continue;

        std::vector<cv::Point2f> seg;
        if ((int)approx.size() > max_points_per_segment) {
            for (int k = 0; k < max_points_per_segment; ++k) {
                int idx = (int)std::round(k * (approx.size() - 1) / (double)(max_points_per_segment - 1));
                seg.emplace_back((float)approx[idx].x, (float)approx[idx].y);
            }
        } else {
            for (const auto& p : approx) seg.emplace_back((float)p.x, (float)p.y);
        }
        segments.push_back(std::move(seg));
    }
    return segments;
}

static cv::Mat build_mask_original_for_candidate(
    const SegCandidate& cand,
    const ProtoView& proto,
    const Args& args,
    const PreprocessMeta& meta
) {
    cv::Mat mask_proto(proto.h, proto.w, CV_32F);
    for (int y = 0; y < proto.h; ++y) {
        float* row = mask_proto.ptr<float>(y);
        for (int x = 0; x < proto.w; ++x) {
            float v = 0.0f;
            for (int m = 0; m < proto.mask_dim && m < (int)cand.coeffs.size(); ++m) {
                v += cand.coeffs[m] * proto_at(proto, m, y, x);
            }
            row[x] = sigmoid(v);
        }
    }

    cv::Mat mask_input;
    cv::resize(mask_proto, mask_input, cv::Size(args.input_w, args.input_h), 0, 0, cv::INTER_LINEAR);
    cv::Mat cropped = cv::Mat::zeros(args.input_h, args.input_w, CV_32F);
    int x1 = std::max(0, std::min(args.input_w - 1, (int)std::round(cand.ix1)));
    int y1 = std::max(0, std::min(args.input_h - 1, (int)std::round(cand.iy1)));
    int x2 = std::max(0, std::min(args.input_w, (int)std::round(cand.ix2)));
    int y2 = std::max(0, std::min(args.input_h, (int)std::round(cand.iy2)));
    if (x2 > x1 && y2 > y1) {
        mask_input(cv::Rect(x1, y1, x2 - x1, y2 - y1)).copyTo(cropped(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
    }

    int left = std::max(0, (int)std::round(meta.pad_x - 0.1f));
    int top = std::max(0, (int)std::round(meta.pad_y - 0.1f));
    int valid_w = std::min(args.input_w - left, std::max(1, (int)std::round(meta.orig_w * meta.ratio)));
    int valid_h = std::min(args.input_h - top, std::max(1, (int)std::round(meta.orig_h * meta.ratio)));
    if (valid_w <= 0 || valid_h <= 0 || left >= args.input_w || top >= args.input_h) {
        cv::Mat fallback;
        cv::resize(cropped, fallback, cv::Size(meta.orig_w, meta.orig_h), 0, 0, cv::INTER_LINEAR);
        return fallback;
    }
    cv::Rect valid_roi(left, top, valid_w, valid_h);
    cv::Mat valid = cropped(valid_roi);
    cv::Mat mask_orig;
    cv::resize(valid, mask_orig, cv::Size(meta.orig_w, meta.orig_h), 0, 0, cv::INTER_LINEAR);
    return mask_orig;
}

static std::vector<SegmentationDetection> finalize_segmentation_candidates(
    std::vector<SegCandidate>& candidates,
    const ProtoView& proto,
    const Args& args,
    const PreprocessMeta& meta,
    std::string* message
) {
    if (candidates.empty()) {
        if (message && message->empty()) *message = "no segmentation candidates above confidence threshold";
        return {};
    }
    std::sort(candidates.begin(), candidates.end(), [](const SegCandidate& a, const SegCandidate& b) {
        return a.det.score > b.det.score;
    });
    const int max_candidates = 1000;
    if ((int)candidates.size() > max_candidates) candidates.resize(max_candidates);

    std::vector<int> keep = nms_seg_candidates(candidates, args.nms_threshold, args.max_det);
    std::vector<SegmentationDetection> result;
    result.reserve(keep.size());
    for (int idx : keep) {
        const SegCandidate& cand = candidates[idx];
        cv::Mat mask_orig = build_mask_original_for_candidate(cand, proto, args, meta);
        cv::Mat binary;
        cv::threshold(mask_orig, binary, args.mask_threshold, 255.0, cv::THRESH_BINARY);
        binary.convertTo(binary, CV_8U);

        float area = 0.0f;
        auto segments = mask_to_segments_cpp(binary, area);
        SegmentationDetection sd;
        sd.det = cand.det;
        sd.mask_area = area;
        sd.mask_width = meta.orig_w;
        sd.mask_height = meta.orig_h;
        sd.segments = std::move(segments);
        result.push_back(std::move(sd));
    }
    return result;
}

static std::vector<SegmentationDetection> postprocess_segmentation(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta,
    std::string* message = nullptr
) {
    if (message) message->clear();
    if (outputs.size() < 2) {
        if (message) *message = "empty segmentation outputs or missing proto";
        return {};
    }

    int mask_dim = 32;
    std::vector<SegCandidate> candidates;
    ProtoView proto;

    if (is_rockchip_seg_outputs(outputs, args.num_classes, mask_dim)) {
        if (!get_proto_view(outputs[12], mask_dim, proto)) {
            if (message) *message = "invalid Rockchip YOLOv8-seg proto output";
            return {};
        }
        candidates = decode_rockchip_seg_candidates(outputs, args, class_names, meta, mask_dim);
    } else {
        candidates = decode_single_seg_candidates(outputs[0], args, class_names, meta, mask_dim, message);
        if (mask_dim <= 0 || outputs.size() < 2 || !get_proto_view(outputs[1], mask_dim, proto)) {
            if (message && message->empty()) *message = "invalid YOLOv8-seg proto output";
            return {};
        }
    }

    return finalize_segmentation_candidates(candidates, proto, args, meta, message);
}

static std::vector<ClassificationItem> postprocess_classification(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    std::string* message = nullptr
) {
    if (message) message->clear();
    std::vector<ClassificationItem> items;
    if (outputs.empty() || outputs[0].data.empty()) {
        if (message) *message = "empty classification outputs";
        return items;
    }

    const int nc = std::max(1, args.num_classes);
    std::vector<float> logits = outputs[0].data;
    if ((int)logits.size() < nc) {
        if (message) {
            *message = "classification output size smaller than num_classes: output_size=" +
                std::to_string(logits.size()) + ", num_classes=" + std::to_string(nc);
        }
        return items;
    }
    if ((int)logits.size() > nc) {
        logits.resize(nc);
    }

    std::vector<float> probs = softmax_vec(logits);
    std::vector<int> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return probs[a] > probs[b];
    });

    int topk = std::max(1, std::min(args.topk, nc));
    items.reserve(topk);
    for (int i = 0; i < topk; ++i) {
        int cls_id = order[i];
        ClassificationItem item;
        item.class_id = cls_id;
        item.class_name = (cls_id >= 0 && cls_id < (int)class_names.size())
            ? class_names[cls_id]
            : std::to_string(cls_id);
        item.confidence = probs[cls_id];
        item.logit = logits[cls_id];
        items.push_back(item);
    }
    return items;
}

static InferenceResult postprocess_by_task(
    const std::vector<Tensor>& outputs,
    const Args& args,
    const std::vector<std::string>& class_names,
    const PreprocessMeta& meta
) {
    InferenceResult result;
    result.task = normalize_task_name(args.task);

    if (result.task == "classification") {
        result.topk = postprocess_classification(outputs, args, class_names, &result.message);
        return result;
    }

    if (result.task == "detection") {
        result.detections = postprocess_detection(outputs, args, class_names, meta);
        return result;
    }

    if (result.task == "obb_detection") {
        result.obbs = postprocess_obb(outputs, args, class_names, meta, &result.message);
        return result;
    }

    if (result.task == "segmentation") {
        result.segmentations = postprocess_segmentation(outputs, args, class_names, meta, &result.message);
        return result;
    }

    result.message = "C++ postprocess for task '" + result.task + "' is not implemented in v0.8.3.4";
    return result;
}

static std::string detections_to_json(
    const Args& args,
    const std::vector<Detection>& dets,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"ok\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"output_mode\":\"" << json_escape(args.output_mode) << "\",";
    os << "\"preprocess_backend_requested\":\"" << json_escape(args.preprocess_status.requested_backend) << "\",";
    os << "\"preprocess_backend_active\":\"" << json_escape(args.preprocess_status.active_backend) << "\",";
    os << "\"rga_mode_requested\":\"" << json_escape(args.preprocess_status.requested_rga_mode) << "\",";
    os << "\"rga_mode_active\":\"" << json_escape(args.preprocess_status.active_rga_mode) << "\",";
    os << "\"rga_available\":" << (args.preprocess_status.rga_available ? "true" : "false") << ",";
    os << "\"task\":\"" << json_escape(args.task) << "\",";
    os << "\"model\":\"" << json_escape(args.model) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"count\":" << dets.size() << ",";
    if (meta) {
        os << "\"image_width\":" << meta->orig_w << ",";
        os << "\"image_height\":" << meta->orig_h << ",";
        os << "\"input_width\":" << meta->input_w << ",";
        os << "\"input_height\":" << meta->input_h << ",";
        os << "\"letterbox\":{";
        os << "\"ratio\":" << meta->ratio << ",";
        os << "\"pad_x\":" << meta->pad_x << ",";
        os << "\"pad_y\":" << meta->pad_y;
        os << "},";
    }
    os << "\"output_shapes\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < outputs[i].dims.size(); ++j) {
            if (j) os << ",";
            os << outputs[i].dims[j];
        }
        os << "]";
    }
    os << "],";
    if (timing) {
        os << "\"timing\":{";
        os << "\"request_read_ms\":" << timing->request_read_ms << ",";
        os << "\"body_extract_ms\":" << timing->body_extract_ms << ",";
        os << "\"image_decode_ms\":" << timing->image_decode_ms << ",";
        os << "\"preprocess_ms\":" << timing->preprocess_ms << ",";
        os << "\"preprocess_backend\":\"" << json_escape(timing->preprocess_backend) << "\",";
        os << "\"preprocess_detail\":{";
        os << "\"backend\":\"" << json_escape(timing->preprocess_detail.backend) << "\",";
        os << "\"rga_mode\":\"" << json_escape(timing->preprocess_detail.rga_mode) << "\",";
        os << "\"fallback_reason\":\"" << json_escape(timing->preprocess_detail.fallback_reason) << "\",";
        os << "\"orig_w\":" << timing->preprocess_detail.orig_w << ",";
        os << "\"orig_h\":" << timing->preprocess_detail.orig_h << ",";
        os << "\"resized_w\":" << timing->preprocess_detail.resized_w << ",";
        os << "\"resized_h\":" << timing->preprocess_detail.resized_h << ",";
        os << "\"meta_calc_ms\":" << timing->preprocess_detail.meta_calc_ms << ",";
        os << "\"cpu_resize_ms\":" << timing->preprocess_detail.cpu_resize_ms << ",";
        os << "\"rga_resize_color_ms\":" << timing->preprocess_detail.rga_resize_color_ms << ",";
        os << "\"rga_resize_only_ms\":" << timing->preprocess_detail.rga_resize_only_ms << ",";
        os << "\"cpu_canvas_alloc_ms\":" << timing->preprocess_detail.cpu_canvas_alloc_ms << ",";
        os << "\"cpu_padding_copy_ms\":" << timing->preprocess_detail.cpu_padding_copy_ms << ",";
        os << "\"cpu_cvtcolor_ms\":" << timing->preprocess_detail.cpu_cvtcolor_ms << ",";
        os << "\"continuity_ms\":" << timing->preprocess_detail.continuity_ms << ",";
        os << "\"total_ms\":" << timing->preprocess_detail.total_ms;
        os << "},";
        os << "\"stream_detail\":{";
        os << "\"capture_read_ms\":" << timing->stream_detail.capture_read_ms << ",";
        os << "\"snapshot_clone_ms\":" << timing->stream_detail.snapshot_clone_ms << ",";
        os << "\"annotated_draw_ms\":" << timing->stream_detail.annotated_draw_ms << ",";
        os << "\"state_update_ms\":" << timing->stream_detail.state_update_ms << ",";
        os << "\"loop_total_ms\":" << timing->stream_detail.loop_total_ms;
        os << "},";
        os << "\"rknn\":{";
        os << "\"inputs_set_ms\":" << timing->rknn.inputs_set_ms << ",";
        os << "\"run_ms\":" << timing->rknn.run_ms << ",";
        os << "\"outputs_get_ms\":" << timing->rknn.outputs_get_ms << ",";
        os << "\"outputs_copy_ms\":" << timing->rknn.outputs_copy_ms << ",";
        os << "\"outputs_release_ms\":" << timing->rknn.outputs_release_ms << ",";
        os << "\"total_ms\":" << timing->rknn.total_ms;
        os << "},";
        os << "\"postprocess_ms\":" << timing->postprocess_ms << ",";
        os << "\"total_ms\":" << timing->total_ms;
        os << "},";
    }
    os << "\"predictions\":[";
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        if (i) os << ",";
        float cx = (d.x1 + d.x2) * 0.5f;
        float cy = (d.y1 + d.y2) * 0.5f;
        os << "{";
        os << "\"class_id\":" << d.class_id << ",";
        os << "\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(4) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << cx << "," << cy << "],";
        os << "\"center_x\":" << cx << ",";
        os << "\"center_y\":" << cy;
        os << "}";
    }
    os << "]";
    os << "}";
    return os.str();
}

static std::string classification_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"ok\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"output_mode\":\"" << json_escape(args.output_mode) << "\",";
    os << "\"task\":\"classification\",";
    os << "\"model\":\"" << json_escape(args.model) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"num_classes\":" << args.num_classes << ",";
    os << "\"topk_count\":" << result.topk.size() << ",";
    if (meta) {
        os << "\"image_width\":" << meta->orig_w << ",";
        os << "\"image_height\":" << meta->orig_h << ",";
        os << "\"input_width\":" << meta->input_w << ",";
        os << "\"input_height\":" << meta->input_h << ",";
    }
    os << "\"output_shapes\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < outputs[i].dims.size(); ++j) {
            if (j) os << ",";
            os << outputs[i].dims[j];
        }
        os << "]";
    }
    os << "],";
    if (!result.message.empty()) {
        os << "\"message\":\"" << json_escape(result.message) << "\",";
    }
    if (timing) {
        os << "\"timing\":{";
        os << "\"preprocess_ms\":" << timing->preprocess_ms << ",";
        os << "\"preprocess_backend\":\"" << json_escape(timing->preprocess_backend) << "\",";
        os << "\"rknn\":{";
        os << "\"inputs_set_ms\":" << timing->rknn.inputs_set_ms << ",";
        os << "\"run_ms\":" << timing->rknn.run_ms << ",";
        os << "\"outputs_get_ms\":" << timing->rknn.outputs_get_ms << ",";
        os << "\"outputs_copy_ms\":" << timing->rknn.outputs_copy_ms << ",";
        os << "\"outputs_release_ms\":" << timing->rknn.outputs_release_ms << ",";
        os << "\"total_ms\":" << timing->rknn.total_ms;
        os << "},";
        os << "\"postprocess_ms\":" << timing->postprocess_ms << ",";
        os << "\"total_ms\":" << timing->total_ms;
        os << "},";
    }

    os << "\"prediction\":";
    if (result.topk.empty()) {
        os << "null";
    } else {
        const auto& p = result.topk.front();
        os << "{";
        os << "\"class_id\":" << p.class_id << ",";
        os << "\"class_name\":\"" << json_escape(p.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << p.confidence << std::setprecision(3) << ",";
        os << "\"logit\":" << std::setprecision(6) << p.logit << std::setprecision(3);
        os << "}";
    }
    os << ",\"topk\":[";
    for (size_t i = 0; i < result.topk.size(); ++i) {
        const auto& p = result.topk[i];
        if (i) os << ",";
        os << "{";
        os << "\"class_id\":" << p.class_id << ",";
        os << "\"class_name\":\"" << json_escape(p.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << p.confidence << std::setprecision(3) << ",";
        os << "\"logit\":" << std::setprecision(6) << p.logit << std::setprecision(3);
        os << "}";
    }
    os << "]";
    os << "}";
    return os.str();
}

static std::string obb_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"ok\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"output_mode\":\"" << json_escape(args.output_mode) << "\",";
    os << "\"preprocess_backend_requested\":\"" << json_escape(args.preprocess_status.requested_backend) << "\",";
    os << "\"preprocess_backend_active\":\"" << json_escape(args.preprocess_status.active_backend) << "\",";
    os << "\"rga_mode_requested\":\"" << json_escape(args.preprocess_status.requested_rga_mode) << "\",";
    os << "\"rga_mode_active\":\"" << json_escape(args.preprocess_status.active_rga_mode) << "\",";
    os << "\"rga_available\":" << (args.preprocess_status.rga_available ? "true" : "false") << ",";
    os << "\"task\":\"obb_detection\",";
    os << "\"model\":\"" << json_escape(args.model) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"count\":" << result.obbs.size() << ",";
    os << "\"num_classes\":" << args.num_classes << ",";
    os << "\"nms\":{\"type\":\"horizontal_bbox_nms\",\"iou_threshold\":" << args.nms_threshold
       << ",\"note\":\"v0.8.3.4 uses bbox NMS for OBB; rotated NMS can be added later\"},";
    if (meta) {
        os << "\"image_width\":" << meta->orig_w << ",";
        os << "\"image_height\":" << meta->orig_h << ",";
        os << "\"input_width\":" << meta->input_w << ",";
        os << "\"input_height\":" << meta->input_h << ",";
        os << "\"letterbox\":{";
        os << "\"ratio\":" << meta->ratio << ",";
        os << "\"pad_x\":" << meta->pad_x << ",";
        os << "\"pad_y\":" << meta->pad_y;
        os << "},";
    }
    os << "\"output_shapes\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < outputs[i].dims.size(); ++j) {
            if (j) os << ",";
            os << outputs[i].dims[j];
        }
        os << "]";
    }
    os << "],";
    if (!result.message.empty()) os << "\"message\":\"" << json_escape(result.message) << "\",";
    if (timing) {
        os << "\"timing\":{";
        os << "\"request_read_ms\":" << timing->request_read_ms << ",";
        os << "\"body_extract_ms\":" << timing->body_extract_ms << ",";
        os << "\"image_decode_ms\":" << timing->image_decode_ms << ",";
        os << "\"preprocess_ms\":" << timing->preprocess_ms << ",";
        os << "\"preprocess_backend\":\"" << json_escape(timing->preprocess_backend) << "\",";
        os << "\"rknn\":{";
        os << "\"inputs_set_ms\":" << timing->rknn.inputs_set_ms << ",";
        os << "\"run_ms\":" << timing->rknn.run_ms << ",";
        os << "\"outputs_get_ms\":" << timing->rknn.outputs_get_ms << ",";
        os << "\"outputs_copy_ms\":" << timing->rknn.outputs_copy_ms << ",";
        os << "\"outputs_release_ms\":" << timing->rknn.outputs_release_ms << ",";
        os << "\"total_ms\":" << timing->rknn.total_ms;
        os << "},";
        os << "\"postprocess_ms\":" << timing->postprocess_ms << ",";
        os << "\"total_ms\":" << timing->total_ms;
        os << "},";
    }
    os << "\"predictions\":[";
    for (size_t i = 0; i < result.obbs.size(); ++i) {
        const auto& d = result.obbs[i];
        if (i) os << ",";
        os << "{";
        os << "\"class_id\":" << d.class_id << ",";
        os << "\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << d.cx << "," << d.cy << "],";
        os << "\"center_x\":" << d.cx << ",";
        os << "\"center_y\":" << d.cy << ",";
        os << "\"obb\":{";
        os << "\"cx\":" << d.cx << ",";
        os << "\"cy\":" << d.cy << ",";
        os << "\"w\":" << d.w << ",";
        os << "\"h\":" << d.h << ",";
        os << "\"angle\":" << std::setprecision(6) << d.angle << std::setprecision(3) << ",";
        os << "\"angle_unit\":\"radian\",";
        os << "\"points\":[";
        for (size_t pi = 0; pi < d.points.size(); ++pi) {
            if (pi) os << ",";
            os << "[" << d.points[pi].x << "," << d.points[pi].y << "]";
        }
        os << "]}";
        os << "}";
    }
    os << "]}";
    return os.str();
}


static std::string segmentation_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"ok\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"output_mode\":\"" << json_escape(args.output_mode) << "\",";
    os << "\"preprocess_backend_requested\":\"" << json_escape(args.preprocess_status.requested_backend) << "\",";
    os << "\"preprocess_backend_active\":\"" << json_escape(args.preprocess_status.active_backend) << "\",";
    os << "\"rga_mode_requested\":\"" << json_escape(args.preprocess_status.requested_rga_mode) << "\",";
    os << "\"rga_mode_active\":\"" << json_escape(args.preprocess_status.active_rga_mode) << "\",";
    os << "\"rga_available\":" << (args.preprocess_status.rga_available ? "true" : "false") << ",";
    os << "\"task\":\"segmentation\",";
    os << "\"model\":\"" << json_escape(args.model) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"count\":" << result.segmentations.size() << ",";
    os << "\"num_classes\":" << args.num_classes << ",";
    os << "\"mask_threshold\":" << args.mask_threshold << ",";
    os << "\"nms\":{\"type\":\"horizontal_bbox_nms\",\"iou_threshold\":" << args.nms_threshold << "},";
    if (meta) {
        os << "\"image_width\":" << meta->orig_w << ",";
        os << "\"image_height\":" << meta->orig_h << ",";
        os << "\"input_width\":" << meta->input_w << ",";
        os << "\"input_height\":" << meta->input_h << ",";
        os << "\"letterbox\":{";
        os << "\"ratio\":" << meta->ratio << ",";
        os << "\"pad_x\":" << meta->pad_x << ",";
        os << "\"pad_y\":" << meta->pad_y;
        os << "},";
    }
    os << "\"output_shapes\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < outputs[i].dims.size(); ++j) {
            if (j) os << ",";
            os << outputs[i].dims[j];
        }
        os << "]";
    }
    os << "],";
    if (!result.message.empty()) os << "\"message\":\"" << json_escape(result.message) << "\",";
    if (timing) {
        os << "\"timing\":{";
        os << "\"request_read_ms\":" << timing->request_read_ms << ",";
        os << "\"body_extract_ms\":" << timing->body_extract_ms << ",";
        os << "\"image_decode_ms\":" << timing->image_decode_ms << ",";
        os << "\"preprocess_ms\":" << timing->preprocess_ms << ",";
        os << "\"preprocess_backend\":\"" << json_escape(timing->preprocess_backend) << "\",";
        os << "\"rknn\":{";
        os << "\"inputs_set_ms\":" << timing->rknn.inputs_set_ms << ",";
        os << "\"run_ms\":" << timing->rknn.run_ms << ",";
        os << "\"outputs_get_ms\":" << timing->rknn.outputs_get_ms << ",";
        os << "\"outputs_copy_ms\":" << timing->rknn.outputs_copy_ms << ",";
        os << "\"outputs_release_ms\":" << timing->rknn.outputs_release_ms << ",";
        os << "\"total_ms\":" << timing->rknn.total_ms;
        os << "},";
        os << "\"postprocess_ms\":" << timing->postprocess_ms << ",";
        os << "\"total_ms\":" << timing->total_ms;
        os << "},";
    }
    os << "\"mask\":{\"threshold\":" << args.mask_threshold << ",\"return_format\":\"polygon_segments\"},";
    os << "\"predictions\":[";
    for (size_t i = 0; i < result.segmentations.size(); ++i) {
        const auto& s = result.segmentations[i];
        const auto& d = s.det;
        if (i) os << ",";
        const float cx = (d.x1 + d.x2) * 0.5f;
        const float cy = (d.y1 + d.y2) * 0.5f;
        os << "{";
        os << "\"class_id\":" << d.class_id << ",";
        os << "\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << cx << "," << cy << "],";
        os << "\"center_x\":" << cx << ",";
        os << "\"center_y\":" << cy << ",";
        os << "\"mask\":{";
        os << "\"threshold\":" << args.mask_threshold << ",";
        os << "\"area\":" << s.mask_area << ",";
        os << "\"shape\":[" << s.mask_height << "," << s.mask_width << "],";
        os << "\"segments\":[";
        for (size_t si = 0; si < s.segments.size(); ++si) {
            if (si) os << ",";
            os << "[";
            for (size_t pi = 0; pi < s.segments[si].size(); ++pi) {
                if (pi) os << ",";
                os << "[" << s.segments[si][pi].x << "," << s.segments[si][pi].y << "]";
            }
            os << "]";
        }
        os << "],";
        os << "\"polygon\":";
        if (!s.segments.empty()) {
            os << "[";
            for (size_t pi = 0; pi < s.segments[0].size(); ++pi) {
                if (pi) os << ",";
                os << "[" << s.segments[0][pi].x << "," << s.segments[0][pi].y << "]";
            }
            os << "]";
        } else {
            os << "[]";
        }
        os << "}";
        os << "}";
    }
    os << "]}";
    return os.str();
}

// Forward declarations for ROI JSON helpers.
// These helpers are implemented in the ROI pipeline section below, but
// roi_classification_to_json() is defined earlier in this file.
static std::string json_string_array(const std::vector<std::string>& values);
static std::string json_int_shape_array(const std::vector<std::vector<int>>& shapes);

static std::string roi_classification_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"" << json_escape(result.roi.status.empty() ? "ok" : result.roi.status) << "\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.4\",";
    os << "\"task\":\"roi_classification\",";
    os << "\"pipeline_config\":\"" << json_escape(args.pipeline_config) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"final_decision\":\"" << json_escape(result.roi.final_decision) << "\",";
    os << "\"final_label\":\"" << json_escape(result.roi.final_decision) << "\",";
    os << "\"final_confidence\":" << std::setprecision(6) << result.roi.final_confidence << std::setprecision(3) << ",";
    os << "\"count\":" << result.detections.size() << ",";
    if (meta) {
        os << "\"image_width\":" << meta->orig_w << ",";
        os << "\"image_height\":" << meta->orig_h << ",";
        os << "\"input_width\":" << meta->input_w << ",";
        os << "\"input_height\":" << meta->input_h << ",";
    }
    if (!result.message.empty()) os << "\"message\":\"" << json_escape(result.message) << "\",";
    if (timing) {
        os << "\"timing\":{";
        os << "\"preprocess_ms\":" << timing->preprocess_ms << ",";
        os << "\"preprocess_backend\":\"" << json_escape(timing->preprocess_backend) << "\",";
        os << "\"rknn\":{";
        os << "\"inputs_set_ms\":" << timing->rknn.inputs_set_ms << ",";
        os << "\"run_ms\":" << timing->rknn.run_ms << ",";
        os << "\"outputs_get_ms\":" << timing->rknn.outputs_get_ms << ",";
        os << "\"outputs_copy_ms\":" << timing->rknn.outputs_copy_ms << ",";
        os << "\"outputs_release_ms\":" << timing->rknn.outputs_release_ms << ",";
        os << "\"total_ms\":" << timing->rknn.total_ms;
        os << "},";
        os << "\"postprocess_ms\":" << timing->postprocess_ms << ",";
        os << "\"total_ms\":" << timing->total_ms;
        os << "},";
    }
    os << "\"timing_ms\":{";
    os << "\"detector\":" << result.roi.detector_ms << ",";
    os << "\"crop\":" << result.roi.crop_ms << ",";
    os << "\"classifier\":" << result.roi.classifier_ms << ",";
    os << "\"total\":" << latency_ms;
    os << "},";

    os << "\"detector\":{";
    os << "\"count\":" << result.roi.detector_predictions.size() << ",";
    os << "\"config\":{";
    os << "\"model_path\":\"" << json_escape(result.roi.detector_model) << "\",";
    os << "\"meta_path\":\"" << json_escape(result.roi.detector_meta_path) << "\",";
    os << "\"num_classes\":" << result.roi.detector_num_classes << ",";
    os << "\"class_names\":" << json_string_array(result.roi.detector_class_names) << ",";
    os << "\"conf_threshold\":" << result.roi.detector_conf_threshold << ",";
    os << "\"nms_threshold\":" << result.roi.detector_nms_threshold << ",";
    os << "\"select_policy\":\"" << json_escape(result.roi.detector_select_policy) << "\",";
    os << "\"target_class_id\":" << result.roi.detector_target_class_id << ",";
    os << "\"target_class_name\":\"" << json_escape(result.roi.detector_target_class_name) << "\"";
    os << "},";
    os << "\"output_shapes\":" << json_int_shape_array(result.roi.detector_output_shapes) << ",";
    os << "\"selected\":";
    if (!result.roi.has_selected_detection) {
        os << "null";
    } else {
        const auto& d = result.roi.selected_detection;
        float cx = (d.x1+d.x2)*0.5f, cy=(d.y1+d.y2)*0.5f;
        os << "{";
        os << "\"class_id\":" << d.class_id << ",\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << cx << "," << cy << "],\"center_x\":" << cx << ",\"center_y\":" << cy;
        os << "}";
    }
    os << ",\"predictions\":[";
    for (size_t i=0;i<result.roi.detector_predictions.size();++i) {
        if (i) os << ",";
        const auto& d = result.roi.detector_predictions[i];
        float cx = (d.x1+d.x2)*0.5f, cy=(d.y1+d.y2)*0.5f;
        os << "{";
        os << "\"class_id\":" << d.class_id << ",\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << cx << "," << cy << "],\"center_x\":" << cx << ",\"center_y\":" << cy;
        os << "}";
    }
    os << "]},";

    os << "\"classifier\":{";
    os << "\"prediction\":";
    if (result.roi.classifier_topk.empty()) {
        os << "null";
    } else {
        const auto& p = result.roi.classifier_topk.front();
        os << "{\"class_id\":" << p.class_id << ",\"class_name\":\"" << json_escape(p.class_name) << "\",\"confidence\":" << std::setprecision(6) << p.confidence << std::setprecision(3) << ",\"logit\":" << p.logit << "}";
    }
    os << ",\"topk\":[";
    for (size_t i=0;i<result.roi.classifier_topk.size();++i) {
        if (i) os << ",";
        const auto& p = result.roi.classifier_topk[i];
        os << "{\"class_id\":" << p.class_id << ",\"class_name\":\"" << json_escape(p.class_name) << "\",\"confidence\":" << std::setprecision(6) << p.confidence << std::setprecision(3) << ",\"logit\":" << p.logit << "}";
    }
    os << "]},";

    os << "\"roi\":{";
    os << "\"mode\":\"" << json_escape(result.roi.roi_mode) << "\",";
    os << "\"pipeline_mode\":\"" << json_escape(result.roi.pipeline_roi_mode) << "\",";
    os << "\"padding_ratio\":" << result.roi.padding_ratio << ",";
    os << "\"bbox\":";
    if (result.roi.roi_bbox.size() >= 4) os << "[" << result.roi.roi_bbox[0] << "," << result.roi.roi_bbox[1] << "," << result.roi.roi_bbox[2] << "," << result.roi.roi_bbox[3] << "]"; else os << "null";
    os << ",\"base_bbox\":";
    if (result.roi.base_bbox.size() >= 4) os << "[" << result.roi.base_bbox[0] << "," << result.roi.base_bbox[1] << "," << result.roi.base_bbox[2] << "," << result.roi.base_bbox[3] << "]"; else os << "null";
    os << ",\"relative_box\":{";
    os << "\"x1\":" << result.roi.relative_box.at("x1") << ",\"y1\":" << result.roi.relative_box.at("y1") << ",\"x2\":" << result.roi.relative_box.at("x2") << ",\"y2\":" << result.roi.relative_box.at("y2");
    os << "},";
    os << "\"class_key\":\"" << json_escape(result.roi.class_key) << "\",";
    os << "\"matched_class_key\":\"" << json_escape(result.roi.matched_class_key) << "\",";
    os << "\"source\":\"" << json_escape(result.roi.source) << "\"";
    os << "},";

    os << "\"predictions\":[";
    for (size_t i=0;i<result.detections.size();++i) {
        if (i) os << ",";
        const auto& d = result.detections[i];
        float cx=(d.x1+d.x2)*0.5f, cy=(d.y1+d.y2)*0.5f;
        os << "{";
        os << "\"class_id\":" << d.class_id << ",\"class_name\":\"" << json_escape(d.class_name) << "\",";
        os << "\"confidence\":" << std::setprecision(6) << d.score << std::setprecision(3) << ",";
        os << "\"bbox\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "],";
        os << "\"center\":[" << cx << "," << cy << "],\"center_x\":" << cx << ",\"center_y\":" << cy << ",";
        os << "\"detector\":";
        if (result.roi.has_selected_detection) {
            const auto& sd = result.roi.selected_detection;
            os << "{\"class_id\":" << sd.class_id << ",\"class_name\":\"" << json_escape(sd.class_name) << "\",\"confidence\":" << std::setprecision(6) << sd.score << std::setprecision(3) << ",\"bbox\":[" << sd.x1 << "," << sd.y1 << "," << sd.x2 << "," << sd.y2 << "]}";
        } else os << "null";
        os << ",\"classifier\":";
        if (!result.roi.classifier_topk.empty()) {
            const auto& cp = result.roi.classifier_topk.front();
            os << "{\"class_id\":" << cp.class_id << ",\"class_name\":\"" << json_escape(cp.class_name) << "\",\"confidence\":" << std::setprecision(6) << cp.confidence << std::setprecision(3) << "}";
        } else os << "null";
        os << ",\"roi\":";
        if (result.roi.roi_bbox.size() >= 4) {
            os << "{\"mode\":\"" << json_escape(result.roi.roi_mode) << "\",\"pipeline_mode\":\"" << json_escape(result.roi.pipeline_roi_mode) << "\",\"bbox\":[" << result.roi.roi_bbox[0] << "," << result.roi.roi_bbox[1] << "," << result.roi.roi_bbox[2] << "," << result.roi.roi_bbox[3] << "],\"base_bbox\":[";
            if (result.roi.base_bbox.size()>=4) os << result.roi.base_bbox[0] << "," << result.roi.base_bbox[1] << "," << result.roi.base_bbox[2] << "," << result.roi.base_bbox[3];
            os << "],\"padding_ratio\":" << result.roi.padding_ratio << ",\"relative_box\":{\"x1\":" << result.roi.relative_box.at("x1") << ",\"y1\":" << result.roi.relative_box.at("y1") << ",\"x2\":" << result.roi.relative_box.at("x2") << ",\"y2\":" << result.roi.relative_box.at("y2") << "}}";
        } else os << "null";
        os << "}";
    }
    os << "]";
    os << "}";
    return os.str();
}


static std::string unsupported_task_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs
) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"status\":\"unsupported_task\",";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"task\":\"" << json_escape(result.task) << "\",";
    os << "\"model\":\"" << json_escape(args.model) << "\",";
    os << "\"latency_ms\":" << latency_ms << ",";
    os << "\"message\":\"" << json_escape(result.message) << "\",";
    os << "\"output_shapes\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < outputs[i].dims.size(); ++j) {
            if (j) os << ",";
            os << outputs[i].dims[j];
        }
        os << "]";
    }
    os << "],\"predictions\":[]}";
    return os.str();
}

static std::string inference_result_to_json(
    const Args& args,
    const InferenceResult& result,
    double latency_ms,
    const std::vector<Tensor>& outputs,
    const InferTiming* timing = nullptr,
    const PreprocessMeta* meta = nullptr
) {
    if (result.task == "classification") {
        return classification_to_json(args, result, latency_ms, outputs, timing, meta);
    }
    if (result.task == "detection") {
        return detections_to_json(args, result.detections, latency_ms, outputs, timing, meta);
    }
    if (result.task == "obb_detection") {
        return obb_to_json(args, result, latency_ms, outputs, timing, meta);
    }
    if (result.task == "segmentation") {
        return segmentation_to_json(args, result, latency_ms, outputs, timing, meta);
    }
    if (result.task == "roi_classification") {
        return roi_classification_to_json(args, result, latency_ms, outputs, timing, meta);
    }
    return unsupported_task_to_json(args, result, latency_ms, outputs);
}

static cv::Scalar color_for_class(int class_id) {
    // Deterministic but readable BGR color. Kept local to the visual debug overlay.
    static const cv::Scalar colors[] = {
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 128, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(128, 255, 0),
        cv::Scalar(255, 128, 0),
    };
    int idx = std::abs(class_id) % (int)(sizeof(colors) / sizeof(colors[0]));
    return colors[idx];
}

static cv::Mat draw_detections_on_frame(const cv::Mat& bgr, const std::vector<Detection>& dets) {
    cv::Mat annotated;
    if (bgr.empty()) return annotated;
    annotated = bgr.clone();

    const int w = annotated.cols;
    const int h = annotated.rows;
    const int thickness = std::max(2, (int)std::round(std::min(w, h) / 700.0));
    const double font_scale = std::max(0.55, std::min(w, h) / 1200.0);
    const int font_thickness = std::max(1, thickness - 1);
    const int baseline_pad = std::max(4, thickness * 2);

    for (const auto& d : dets) {
        int x1 = std::max(0, std::min(w - 1, (int)std::round(d.x1)));
        int y1 = std::max(0, std::min(h - 1, (int)std::round(d.y1)));
        int x2 = std::max(0, std::min(w - 1, (int)std::round(d.x2)));
        int y2 = std::max(0, std::min(h - 1, (int)std::round(d.y2)));
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Scalar color = color_for_class(d.class_id);
        cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness, cv::LINE_AA);

        float cx = (d.x1 + d.x2) * 0.5f;
        float cy = (d.y1 + d.y2) * 0.5f;
        int icx = std::max(0, std::min(w - 1, (int)std::round(cx)));
        int icy = std::max(0, std::min(h - 1, (int)std::round(cy)));
        cv::circle(annotated, cv::Point(icx, icy), std::max(4, thickness * 2), color, -1, cv::LINE_AA);
        cv::circle(annotated, cv::Point(icx, icy), std::max(7, thickness * 3), cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);

        std::ostringstream label_ss;
        label_ss << d.class_name << " " << std::fixed << std::setprecision(2) << d.score;
        std::string label = label_ss.str();

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        int text_x = x1;
        int text_y = y1 - baseline_pad;
        if (text_y - text_size.height - baseline_pad < 0) {
            text_y = std::min(h - 1, y1 + text_size.height + baseline_pad * 2);
        }
        int bg_x2 = std::min(w - 1, text_x + text_size.width + baseline_pad * 2);
        int bg_y1 = std::max(0, text_y - text_size.height - baseline_pad);
        int bg_y2 = std::min(h - 1, text_y + baseline_pad);
        cv::rectangle(annotated, cv::Point(text_x, bg_y1), cv::Point(bg_x2, bg_y2), color, -1, cv::LINE_AA);
        cv::putText(annotated, label, cv::Point(text_x + baseline_pad, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
    return annotated;
}

static cv::Mat draw_classification_on_frame(const cv::Mat& bgr, const std::vector<ClassificationItem>& topk) {
    cv::Mat annotated;
    if (bgr.empty()) return annotated;
    annotated = bgr.clone();

    const int w = annotated.cols;
    const int h = annotated.rows;
    const int pad = std::max(10, (int)std::round(std::min(w, h) / 70.0));
    const double font_scale = std::max(0.7, std::min(w, h) / 900.0);
    const int font_thickness = std::max(2, (int)std::round(font_scale * 2.0));

    std::string label = "classification: no result";
    if (!topk.empty()) {
        std::ostringstream ss;
        ss << topk.front().class_name << " " << std::fixed << std::setprecision(3) << topk.front().confidence;
        label = ss.str();
    }

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
    int x1 = pad;
    int y1 = pad;
    int x2 = std::min(w - 1, x1 + text_size.width + pad * 2);
    int y2 = std::min(h - 1, y1 + text_size.height + baseline + pad * 2);
    cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 0, 0), -1, cv::LINE_AA);
    cv::putText(annotated, label, cv::Point(x1 + pad, y2 - pad - baseline),
                cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    return annotated;
}

static cv::Mat draw_obb_on_frame(const cv::Mat& bgr, const std::vector<ObbDetection>& obbs) {
    cv::Mat annotated;
    if (bgr.empty()) return annotated;
    annotated = bgr.clone();
    const int w = annotated.cols;
    const int h = annotated.rows;
    const int thickness = std::max(2, (int)std::round(std::min(w, h) / 700.0));
    const double font_scale = std::max(0.55, std::min(w, h) / 1200.0);
    const int font_thickness = std::max(1, thickness - 1);
    const int baseline_pad = std::max(4, thickness * 2);
    for (const auto& d : obbs) {
        if (d.points.size() < 4) continue;
        cv::Scalar color = color_for_class(d.class_id);
        std::vector<cv::Point> pts;
        pts.reserve(d.points.size());
        for (const auto& p : d.points) {
            int px = std::max(0, std::min(w - 1, (int)std::round(p.x)));
            int py = std::max(0, std::min(h - 1, (int)std::round(p.y)));
            pts.emplace_back(px, py);
        }
        const cv::Point* poly_pts[1] = { pts.data() };
        int npts[] = { (int)pts.size() };
        cv::polylines(annotated, poly_pts, npts, 1, true, color, thickness, cv::LINE_AA);
        int icx = std::max(0, std::min(w - 1, (int)std::round(d.cx)));
        int icy = std::max(0, std::min(h - 1, (int)std::round(d.cy)));
        cv::circle(annotated, cv::Point(icx, icy), std::max(4, thickness * 2), color, -1, cv::LINE_AA);
        cv::circle(annotated, cv::Point(icx, icy), std::max(7, thickness * 3), cv::Scalar(255, 255, 255), thickness, cv::LINE_AA);
        std::ostringstream label_ss;
        label_ss << d.class_name << " " << std::fixed << std::setprecision(2) << d.score;
        std::string label = label_ss.str();
        int x1 = std::max(0, std::min(w - 1, (int)std::round(d.x1)));
        int y1 = std::max(0, std::min(h - 1, (int)std::round(d.y1)));
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        int text_x = x1;
        int text_y = y1 - baseline_pad;
        if (text_y - text_size.height - baseline_pad < 0) {
            text_y = std::min(h - 1, y1 + text_size.height + baseline_pad * 2);
        }
        int bg_x2 = std::min(w - 1, text_x + text_size.width + baseline_pad * 2);
        int bg_y1 = std::max(0, text_y - text_size.height - baseline_pad);
        int bg_y2 = std::min(h - 1, text_y + baseline_pad);
        cv::rectangle(annotated, cv::Point(text_x, bg_y1), cv::Point(bg_x2, bg_y2), color, -1, cv::LINE_AA);
        cv::putText(annotated, label, cv::Point(text_x + baseline_pad, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
    return annotated;
}


static cv::Mat draw_segmentation_on_frame(const cv::Mat& bgr, const std::vector<SegmentationDetection>& segs) {
    cv::Mat annotated;
    if (bgr.empty()) return annotated;
    annotated = bgr.clone();
    const int w = annotated.cols;
    const int h = annotated.rows;
    const int thickness = std::max(2, (int)std::round(std::min(w, h) / 700.0));
    const double font_scale = std::max(0.55, std::min(w, h) / 1200.0);
    const int font_thickness = std::max(1, thickness - 1);
    cv::Mat overlay = annotated.clone();

    for (const auto& s : segs) {
        const auto& d = s.det;
        cv::Scalar color = color_for_class(d.class_id);
        for (const auto& seg : s.segments) {
            if (seg.size() < 3) continue;
            std::vector<cv::Point> pts;
            pts.reserve(seg.size());
            for (const auto& p : seg) {
                int px = std::max(0, std::min(w - 1, (int)std::round(p.x)));
                int py = std::max(0, std::min(h - 1, (int)std::round(p.y)));
                pts.emplace_back(px, py);
            }
            const cv::Point* poly_pts[1] = { pts.data() };
            int npts[] = { (int)pts.size() };
            cv::fillPoly(overlay, poly_pts, npts, 1, color, cv::LINE_AA);
            cv::polylines(annotated, poly_pts, npts, 1, true, color, thickness, cv::LINE_AA);
        }
    }
    cv::addWeighted(overlay, 0.25, annotated, 0.75, 0.0, annotated);

    for (const auto& s : segs) {
        const auto& d = s.det;
        cv::Scalar color = color_for_class(d.class_id);
        int x1 = std::max(0, std::min(w - 1, (int)std::round(d.x1)));
        int y1 = std::max(0, std::min(h - 1, (int)std::round(d.y1)));
        int x2 = std::max(0, std::min(w - 1, (int)std::round(d.x2)));
        int y2 = std::max(0, std::min(h - 1, (int)std::round(d.y2)));
        if (x2 > x1 && y2 > y1) cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), color, thickness, cv::LINE_AA);
        std::ostringstream label_ss;
        label_ss << d.class_name << " " << std::fixed << std::setprecision(2) << d.score
                 << " area=" << std::setprecision(0) << s.mask_area;
        std::string label = label_ss.str();
        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
        int text_x = x1;
        int text_y = y1 - 4;
        if (text_y - text_size.height - 4 < 0) text_y = std::min(h - 1, y1 + text_size.height + 8);
        int bg_x2 = std::min(w - 1, text_x + text_size.width + 8);
        int bg_y1 = std::max(0, text_y - text_size.height - 4);
        int bg_y2 = std::min(h - 1, text_y + 4);
        cv::rectangle(annotated, cv::Point(text_x, bg_y1), cv::Point(bg_x2, bg_y2), color, -1, cv::LINE_AA);
        cv::putText(annotated, label, cv::Point(text_x + 4, text_y),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255, 255, 255), font_thickness, cv::LINE_AA);
    }
    return annotated;
}

static cv::Mat draw_roi_classification_on_frame(const cv::Mat& bgr, const InferenceResult& result) {
    cv::Mat annotated;
    if (bgr.empty()) return annotated;
    annotated = bgr.clone();
    const int w = annotated.cols;
    const int h = annotated.rows;
    const int thickness = std::max(2, (int)std::round(std::min(w, h) / 700.0));
    const double font_scale = std::max(0.55, std::min(w, h) / 1200.0);
    const int font_thickness = std::max(1, thickness - 1);
    cv::Scalar det_color(0, 255, 0);
    cv::Scalar roi_color(0, 128, 255);
    cv::Scalar final_color(255, 0, 255);
    if (result.roi.has_selected_detection) {
        const auto& d = result.roi.selected_detection;
        int x1 = std::max(0, std::min(w - 1, (int)std::round(d.x1)));
        int y1 = std::max(0, std::min(h - 1, (int)std::round(d.y1)));
        int x2 = std::max(0, std::min(w - 1, (int)std::round(d.x2)));
        int y2 = std::max(0, std::min(h - 1, (int)std::round(d.y2)));
        cv::rectangle(annotated, cv::Point(x1,y1), cv::Point(x2,y2), det_color, thickness, cv::LINE_AA);
    }
    if (result.roi.roi_bbox.size() >= 4) {
        int x1 = std::max(0, std::min(w - 1, (int)std::round(result.roi.roi_bbox[0])));
        int y1 = std::max(0, std::min(h - 1, (int)std::round(result.roi.roi_bbox[1])));
        int x2 = std::max(0, std::min(w - 1, (int)std::round(result.roi.roi_bbox[2])));
        int y2 = std::max(0, std::min(h - 1, (int)std::round(result.roi.roi_bbox[3])));
        cv::rectangle(annotated, cv::Point(x1,y1), cv::Point(x2,y2), roi_color, thickness + 1, cv::LINE_AA);
    }
    std::ostringstream label_ss;
    label_ss << "ROI " << (result.roi.final_decision.empty() ? result.roi.status : result.roi.final_decision)
             << " " << std::fixed << std::setprecision(2) << result.roi.final_confidence;
    std::string label = label_ss.str();
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, font_thickness, &baseline);
    int tx = 8, ty = std::max(text_size.height + 10, 24);
    cv::rectangle(annotated, cv::Point(tx-4, ty-text_size.height-6), cv::Point(std::min(w-1, tx+text_size.width+8), ty+6), final_color, -1, cv::LINE_AA);
    cv::putText(annotated, label, cv::Point(tx, ty), cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(255,255,255), font_thickness, cv::LINE_AA);
    return annotated;
}

static cv::Mat draw_inference_result_on_frame(const cv::Mat& bgr, const InferenceResult& result) {
    if (result.task == "classification") return draw_classification_on_frame(bgr, result.topk);
    if (result.task == "obb_detection") return draw_obb_on_frame(bgr, result.obbs);
    if (result.task == "segmentation") return draw_segmentation_on_frame(bgr, result.segmentations);
    if (result.task == "roi_classification") return draw_roi_classification_on_frame(bgr, result);
    return draw_detections_on_frame(bgr, result.detections);
}


// -----------------------------------------------------------------------------
// v0.8.4 ROI classification pipeline helpers
// -----------------------------------------------------------------------------
static std::string roi_trim(std::string s) {
    s.erase(0, s.find_first_not_of(" \t\r\n\"'"));
    size_t end = s.find_last_not_of(" \t\r\n\"'");
    if (end == std::string::npos) return "";
    s.erase(end + 1);
    return s;
}

static std::string roi_dirname(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return ".";
    if (pos == 0) return "/";
    return path.substr(0, pos);
}

static bool roi_is_abs_path(const std::string& p) {
    return !p.empty() && p[0] == '/';
}

static std::string roi_resolve_path(const std::string& base_dir, const std::string& value) {
    std::string v = roi_trim(value);
    if (v.empty()) return "";
    if (roi_is_abs_path(v)) return v;
    if (base_dir.empty() || base_dir == ".") return v;
    return base_dir + "/" + v;
}

static std::string json_string_array(const std::vector<std::string>& values) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << "\"" << json_escape(values[i]) << "\"";
    }
    os << "]";
    return os.str();
}

static std::string json_int_shape_array(const std::vector<std::vector<int>>& shapes) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < shapes.size(); ++i) {
        if (i) os << ",";
        os << "[";
        for (size_t j = 0; j < shapes[i].size(); ++j) {
            if (j) os << ",";
            os << shapes[i][j];
        }
        os << "]";
    }
    os << "]";
    return os.str();
}

static std::vector<std::vector<int>> tensor_shapes(const std::vector<Tensor>& outputs) {
    std::vector<std::vector<int>> shapes;
    shapes.reserve(outputs.size());
    for (const auto& t : outputs) shapes.push_back(t.dims);
    return shapes;
}

static bool roi_extract_scalar(const std::string& block, const std::string& key, std::string* out) {
    std::istringstream iss(block);
    std::string line;
    const std::string prefix = key + ":";
    while (std::getline(iss, line)) {
        std::string t = roi_trim(line);
        if (t.rfind(prefix, 0) == 0) {
            if (out) *out = roi_trim(t.substr(prefix.size()));
            return true;
        }
    }
    return false;
}

static std::string roi_section_block(const std::string& text, const std::string& section) {
    std::istringstream iss(text);
    std::string line;
    bool in = false;
    std::ostringstream out;
    while (std::getline(iss, line)) {
        std::string t = roi_trim(line);
        if (!in) {
            if (t == section + ":") in = true;
            continue;
        }
        bool top_level = !line.empty() && line[0] != ' ' && line[0] != '\t';
        if (top_level && !t.empty() && t.back() == ':') break;
        out << line << "\n";
    }
    return out.str();
}

static std::vector<int> roi_parse_size(const std::string& v, int dh, int dw) {
    std::string s = v;
    for (char& c : s) {
        if (c == '[' || c == ']' || c == ',' || c == 'x' || c == 'X') c = ' ';
    }
    std::istringstream iss(s);
    int h = 0, w = 0;
    iss >> h >> w;
    if (h <= 0 || w <= 0) return {dh, dw};
    return {h, w};
}

static std::vector<std::string> roi_parse_inline_names(std::string v) {
    std::vector<std::string> out;
    v = roi_trim(v);
    if (v.empty()) return out;
    size_t lb = v.find('['), rb = v.find(']');
    if (lb != std::string::npos && rb != std::string::npos && rb > lb) v = v.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(v);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = roi_trim(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

static std::map<std::string, float> roi_parse_relative_box(const std::string& block) {
    std::map<std::string, float> rel{{"x1",0.0f},{"y1",0.0f},{"x2",1.0f},{"y2",1.0f}};

    auto clamp01 = [](float v) -> float { return std::max(0.0f, std::min(1.0f, v)); };
    auto parse_float = [&](const std::string& raw, float defv) -> float {
        try { return clamp01(std::stof(roi_trim(raw))); } catch (...) { return defv; }
    };
    auto normalize = [&](std::map<std::string, float>& r) {
        if (r["x2"] < r["x1"]) std::swap(r["x1"], r["x2"]);
        if (r["y2"] < r["y1"]) std::swap(r["y1"], r["y2"]);
    };
    auto parse_inline = [&](const std::string& raw, std::map<std::string, float>& r) {
        auto getv = [&](const std::string& key, float defv) -> float {
            size_t p = raw.find(key + ":");
            if (p == std::string::npos) return defv;
            p += key.size() + 1;
            size_t e = raw.find_first_of(",}", p);
            std::string num = raw.substr(p, e == std::string::npos ? std::string::npos : e - p);
            return parse_float(num, defv);
        };
        r["x1"] = getv("x1", r["x1"]);
        r["y1"] = getv("y1", r["y1"]);
        r["x2"] = getv("x2", r["x2"]);
        r["y2"] = getv("y2", r["y2"]);
        normalize(r);
    };

    // Prefer the last relative_box in roi block so class_relative_box bundles
    // generated by VisionOps use the class-specific entry rather than the
    // disabled default full-box fallback. Supports both inline style:
    //   relative_box: {x1: 0.0, y1: 0.5, x2: 1.0, y2: 0.8}
    // and block style:
    //   relative_box:
    //     x1: 0.0
    //     y1: 0.5
    bool in_rel = false;
    std::map<std::string, float> cur = rel;
    std::istringstream iss(block);
    std::string line;
    while (std::getline(iss, line)) {
        std::string t = roi_trim(line);
        if (t.empty() || t[0] == '#') continue;

        if (t.rfind("relative_box:", 0) == 0) {
            cur = {{"x1",0.0f},{"y1",0.0f},{"x2",1.0f},{"y2",1.0f}};
            std::string raw = roi_trim(t.substr(std::string("relative_box:").size()));
            if (!raw.empty()) {
                parse_inline(raw, cur);
                rel = cur;
                in_rel = false;
            } else {
                in_rel = true;
                rel = cur;
            }
            continue;
        }

        if (in_rel) {
            auto colon = t.find(':');
            if (colon == std::string::npos) continue;
            std::string key = roi_trim(t.substr(0, colon));
            std::string val = roi_trim(t.substr(colon + 1));
            if (key == "x1" || key == "y1" || key == "x2" || key == "y2") {
                cur[key] = parse_float(val, cur[key]);
                normalize(cur);
                rel = cur;
                continue;
            }
            // Leaving the relative_box block.
            in_rel = false;
        }
    }

    normalize(rel);
    return rel;
}

struct RoiPipelineConfigCpp {
    std::string pipeline_config;
    std::string pipeline_name = "roi_classification";
    Args detector_args;
    Args classifier_args;
    std::vector<std::string> detector_names;
    std::vector<std::string> classifier_names;
    std::string roi_mode = "full_box";
    std::string select_policy = "conf_area";
    int target_class_id = -1;
    std::string target_class_name;
    float padding_ratio = 0.0f;
    int min_roi_width = 4;
    int min_roi_height = 4;
    std::map<std::string, float> relative_box{{"x1",0.0f},{"y1",0.0f},{"x2",1.0f},{"y2",1.0f}};
};

static RoiPipelineConfigCpp load_roi_pipeline_config_cpp(const std::string& path, const Args& base_args) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("cannot open ROI pipeline config: " + path);
    std::string text((std::istreambuf_iterator<char>(ifs)), {});
    std::string base_dir = roi_dirname(path);
    RoiPipelineConfigCpp cfg;
    cfg.pipeline_config = path;
    cfg.detector_args = base_args;
    cfg.classifier_args = base_args;
    cfg.detector_args.task = "detection";
    cfg.classifier_args.task = "classification";
    cfg.detector_args.input_h = 640; cfg.detector_args.input_w = 640;
    cfg.classifier_args.input_h = 224; cfg.classifier_args.input_w = 224;

    std::string name;
    if (roi_extract_scalar(text, "pipeline_name", &name) && !name.empty()) cfg.pipeline_name = name;
    const std::string s1 = roi_section_block(text, "stage1");
    const std::string s2 = roi_section_block(text, "stage2");
    const std::string roi = roi_section_block(text, "roi");

    auto apply_stage = [&](const std::string& block, Args& a, const std::string& default_task) {
        std::string v;
        if (roi_extract_scalar(block, "model_path", &v)) a.model = roi_resolve_path(base_dir, v);
        if (roi_extract_scalar(block, "meta_path", &v)) a.class_names_file = roi_resolve_path(base_dir, v);
        if (roi_extract_scalar(block, "task", &v)) a.task = normalize_task_name(v); else a.task = default_task;
        if (roi_extract_scalar(block, "input_size", &v)) { auto sz = roi_parse_size(v, a.input_h, a.input_w); a.input_h = sz[0]; a.input_w = sz[1]; }
        if (roi_extract_scalar(block, "num_classes", &v)) { try { a.num_classes = std::stoi(v); } catch (...) {} }
        if (roi_extract_scalar(block, "conf_threshold", &v)) { try { a.conf_threshold = std::stof(v); } catch (...) {} }
        if (roi_extract_scalar(block, "nms_threshold", &v)) { try { a.nms_threshold = std::stof(v); } catch (...) {} }
        if (roi_extract_scalar(block, "topk", &v)) { try { a.topk = std::stoi(v); } catch (...) {} }
    };
    apply_stage(s1, cfg.detector_args, "detection");
    apply_stage(s2, cfg.classifier_args, "classification");

    std::string v;
    if (roi_extract_scalar(s1, "select_policy", &v)) cfg.select_policy = roi_trim(v);
    if (roi_extract_scalar(s1, "target_class_id", &v)) { try { cfg.target_class_id = std::stoi(v); } catch (...) { cfg.target_class_id = -1; } }
    if (roi_extract_scalar(s1, "target_class_name", &v)) cfg.target_class_name = roi_trim(v);
    if (roi_extract_scalar(roi, "mode", &v)) cfg.roi_mode = roi_trim(v);
    if (roi_extract_scalar(roi, "padding_ratio", &v)) { try { cfg.padding_ratio = std::stof(v); } catch (...) {} }
    if (roi_extract_scalar(roi, "min_width", &v)) { try { cfg.min_roi_width = std::max(1, std::stoi(v)); } catch (...) {} }
    if (roi_extract_scalar(roi, "min_height", &v)) { try { cfg.min_roi_height = std::max(1, std::stoi(v)); } catch (...) {} }
    // First version: use the first relative_box found in roi block. This supports full_box, relative_box,
    // and common class_relative_box bundles generated by VisionOps.
    cfg.relative_box = roi_parse_relative_box(roi);

    if (cfg.detector_args.model.empty()) throw std::runtime_error("roi pipeline stage1.model_path is empty");
    if (cfg.classifier_args.model.empty()) throw std::runtime_error("roi pipeline stage2.model_path is empty");
    cfg.detector_names = load_class_names_simple_yaml(cfg.detector_args.class_names_file, cfg.detector_args.num_classes);
    cfg.classifier_names = load_class_names_simple_yaml(cfg.classifier_args.class_names_file, cfg.classifier_args.num_classes);
    return cfg;
}

static bool clip_bbox_xyxy_vec(const cv::Mat& image, const std::vector<float>& in, std::vector<float>& out) {
    if (image.empty() || in.size() < 4) return false;
    float w = (float)image.cols, h = (float)image.rows;
    float x1 = std::max(0.0f, std::min(w - 1.0f, in[0]));
    float y1 = std::max(0.0f, std::min(h - 1.0f, in[1]));
    float x2 = std::max(0.0f, std::min(w, in[2]));
    float y2 = std::max(0.0f, std::min(h, in[3]));
    if (x2 <= x1 || y2 <= y1) return false;
    out = {x1, y1, x2, y2};
    return true;
}

static bool crop_roi_cpp(const cv::Mat& image, const Detection& det, const RoiPipelineConfigCpp& cfg,
                         cv::Mat& roi, std::vector<float>& roi_box, std::vector<float>& base_box,
                         std::map<std::string, float>& rel_box) {
    std::vector<float> det_box{det.x1, det.y1, det.x2, det.y2};
    std::vector<float> clipped;
    if (!clip_bbox_xyxy_vec(image, det_box, clipped)) return false;
    float bw = std::max(1.0f, clipped[2] - clipped[0]);
    float bh = std::max(1.0f, clipped[3] - clipped[1]);
    float px = bw * cfg.padding_ratio;
    float py = bh * cfg.padding_ratio;
    if (!clip_bbox_xyxy_vec(image, {clipped[0]-px, clipped[1]-py, clipped[2]+px, clipped[3]+py}, base_box)) return false;
    rel_box = cfg.relative_box;
    std::string mode = cfg.roi_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    std::vector<float> final_box = base_box;
    if (mode == "relative_box" || mode == "class_relative_box") {
        float x1 = base_box[0], y1 = base_box[1], x2 = base_box[2], y2 = base_box[3];
        float rbw = std::max(1.0f, x2 - x1), rbh = std::max(1.0f, y2 - y1);
        final_box = {x1 + rbw * rel_box["x1"], y1 + rbh * rel_box["y1"], x1 + rbw * rel_box["x2"], y1 + rbh * rel_box["y2"]};
    }
    if (!clip_bbox_xyxy_vec(image, final_box, roi_box)) return false;
    int ix1 = (int)std::round(roi_box[0]);
    int iy1 = (int)std::round(roi_box[1]);
    int ix2 = (int)std::round(roi_box[2]);
    int iy2 = (int)std::round(roi_box[3]);
    ix1 = std::max(0, std::min(image.cols - 1, ix1));
    iy1 = std::max(0, std::min(image.rows - 1, iy1));
    ix2 = std::max(0, std::min(image.cols, ix2));
    iy2 = std::max(0, std::min(image.rows, iy2));
    if (ix2 - ix1 < cfg.min_roi_width || iy2 - iy1 < cfg.min_roi_height) return false;
    roi = image(cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1)).clone();
    return !roi.empty();
}

static const Detection* select_roi_detection_cpp(const std::vector<Detection>& dets, const RoiPipelineConfigCpp& cfg, int image_w, int image_h) {
    const Detection* best = nullptr;
    float best_score = -1.0f;
    float frame_area = std::max(1.0f, (float)image_w * (float)image_h);
    for (const auto& d : dets) {
        if (!cfg.target_class_name.empty() && d.class_name != cfg.target_class_name) continue;
        if (cfg.target_class_name.empty() && cfg.target_class_id >= 0 && d.class_id != cfg.target_class_id) continue;
        float area = std::max(0.0f, d.x2 - d.x1) * std::max(0.0f, d.y2 - d.y1);
        float s = d.score;
        std::string pol = cfg.select_policy;
        std::transform(pol.begin(), pol.end(), pol.begin(), ::tolower);
        if (pol == "largest_area" || pol == "area") s = area;
        else if (pol == "conf_area" || pol.empty()) s = d.score * 0.7f + std::min(area / frame_area, 1.0f) * 0.3f;
        if (s > best_score) { best_score = s; best = &d; }
    }
    return best;
}

class RoiPipelineRunnerCpp {
public:
    explicit RoiPipelineRunnerCpp(const RoiPipelineConfigCpp& cfg)
        : cfg_(cfg), detector_(cfg.detector_args), classifier_(cfg.classifier_args) {
        std::cout << "[INFO] ROI pipeline loaded: " << cfg_.pipeline_name << " config=" << cfg_.pipeline_config << "\n";
        std::cout << "[INFO] ROI stage1 detector=" << cfg_.detector_args.model << " input=" << cfg_.detector_args.input_h << "," << cfg_.detector_args.input_w << "\n";
        std::cout << "[INFO] ROI stage2 classifier=" << cfg_.classifier_args.model << " input=" << cfg_.classifier_args.input_h << "," << cfg_.classifier_args.input_w << "\n";
    }

    const RoiPipelineConfigCpp& cfg() const { return cfg_; }

    InferenceResult infer_bgr(const cv::Mat& frame, InferTiming* timing, PreprocessMeta* primary_meta) {
        if (frame.empty()) throw std::runtime_error("empty frame for ROI pipeline");
        InferenceResult out;
        out.task = "roi_classification";
        out.roi.valid = true;
        out.roi.relative_box = cfg_.relative_box;
        out.roi.roi_mode = cfg_.roi_mode;
        out.roi.pipeline_roi_mode = cfg_.roi_mode;
        out.roi.padding_ratio = cfg_.padding_ratio;
        out.roi.detector_model = cfg_.detector_args.model;
        out.roi.detector_meta_path = cfg_.detector_args.class_names_file;
        out.roi.detector_num_classes = cfg_.detector_args.num_classes;
        out.roi.detector_conf_threshold = cfg_.detector_args.conf_threshold;
        out.roi.detector_nms_threshold = cfg_.detector_args.nms_threshold;
        out.roi.detector_class_names = cfg_.detector_names;
        out.roi.detector_select_policy = cfg_.select_policy;
        out.roi.detector_target_class_id = cfg_.target_class_id;
        out.roi.detector_target_class_name = cfg_.target_class_name;
        double total0 = now_ms();

        InferTiming det_timing;
        PreprocessMeta det_meta;
        std::string det_backend, det_fallback;
        double s0 = now_ms();
        cv::Mat det_rgb = preprocess_rgb_uint8(frame, cfg_.detector_args, det_meta, det_backend, &det_fallback, &det_timing.preprocess_detail);
        det_timing.preprocess_ms = now_ms() - s0;
        det_timing.preprocess_backend = det_backend;
        det_timing.preprocess_detail.total_ms = det_timing.preprocess_ms;
        auto det_outputs = detector_.infer(det_rgb, &det_timing.rknn);
        out.roi.detector_output_shapes = tensor_shapes(det_outputs);
        s0 = now_ms();
        InferenceResult det_result = postprocess_by_task(det_outputs, cfg_.detector_args, cfg_.detector_names, det_meta);
        det_timing.postprocess_ms = now_ms() - s0;
        det_timing.total_ms = det_timing.preprocess_ms + det_timing.rknn.total_ms + det_timing.postprocess_ms;
        out.roi.detector_ms = det_timing.total_ms;
        out.roi.detector_predictions = det_result.detections;

        const Detection* selected = select_roi_detection_cpp(det_result.detections, cfg_, frame.cols, frame.rows);
        if (!selected) {
            out.roi.status = "no_target";
            out.roi.final_decision = "NO_TARGET";
            out.message = "no detector target selected";
            if (primary_meta) *primary_meta = det_meta;
            if (timing) { timing->total_ms = now_ms() - total0; timing->preprocess_ms = det_timing.preprocess_ms; timing->preprocess_backend = det_backend; timing->rknn = det_timing.rknn; }
            return out;
        }
        out.roi.selected_detection = *selected;
        out.roi.has_selected_detection = true;

        cv::Mat roi_img;
        std::vector<float> roi_box, base_box;
        std::map<std::string, float> rel_box;
        s0 = now_ms();
        bool crop_ok = crop_roi_cpp(frame, *selected, cfg_, roi_img, roi_box, base_box, rel_box);
        out.roi.crop_ms = now_ms() - s0;
        out.roi.roi_bbox = roi_box;
        out.roi.base_bbox = base_box;
        out.roi.relative_box = rel_box;
        out.roi.roi_mode = cfg_.roi_mode;
        out.roi.pipeline_roi_mode = cfg_.roi_mode;
        out.roi.padding_ratio = cfg_.padding_ratio;
        out.roi.class_key = std::to_string(selected->class_id) + ":" + selected->class_name;
        out.roi.source = cfg_.roi_mode;
        if (!crop_ok) {
            out.roi.status = "bad_roi";
            out.roi.final_decision = "REVIEW";
            out.message = "bad ROI crop";
            if (primary_meta) *primary_meta = det_meta;
            if (timing) { timing->total_ms = now_ms() - total0; timing->preprocess_ms = det_timing.preprocess_ms; timing->preprocess_backend = det_backend; timing->rknn = det_timing.rknn; }
            return out;
        }

        InferTiming cls_timing;
        PreprocessMeta cls_meta;
        s0 = now_ms();
        cv::Mat cls_rgb = classification_rgb_uint8_cpu_timed(roi_img, cfg_.classifier_args.input_w, cfg_.classifier_args.input_h, cls_meta, &cls_timing.preprocess_detail);
        cls_timing.preprocess_ms = now_ms() - s0;
        cls_timing.preprocess_backend = "cpu_classification_resize";
        auto cls_outputs = classifier_.infer(cls_rgb, &cls_timing.rknn);
        s0 = now_ms();
        InferenceResult cls_result = postprocess_by_task(cls_outputs, cfg_.classifier_args, cfg_.classifier_names, cls_meta);
        cls_timing.postprocess_ms = now_ms() - s0;
        cls_timing.total_ms = cls_timing.preprocess_ms + cls_timing.rknn.total_ms + cls_timing.postprocess_ms;
        out.roi.classifier_ms = cls_timing.total_ms;
        out.roi.classifier_topk = cls_result.topk;
        out.topk = cls_result.topk;

        ClassificationItem top;
        if (!cls_result.topk.empty()) top = cls_result.topk.front();
        out.roi.status = cls_result.topk.empty() ? "no_classification" : "ok";
        out.roi.final_decision = cls_result.topk.empty() ? "REVIEW" : top.class_name;
        out.roi.final_confidence = cls_result.topk.empty() ? 0.0f : top.confidence;

        Detection final_det = *selected;
        final_det.class_id = top.class_id;
        final_det.class_name = out.roi.final_decision;
        final_det.score = out.roi.final_confidence;
        out.detections.push_back(final_det);
        if (primary_meta) *primary_meta = det_meta;
        if (timing) {
            timing->preprocess_ms = det_timing.preprocess_ms;
            timing->preprocess_backend = det_backend;
            timing->rknn = det_timing.rknn;
            timing->postprocess_ms = det_timing.postprocess_ms + out.roi.crop_ms + cls_timing.postprocess_ms;
            timing->total_ms = now_ms() - total0;
        }
        return out;
    }

private:
    RoiPipelineConfigCpp cfg_;
    RknnRunner detector_;
    RknnRunner classifier_;
};

static std::string stats_json() {
    uint64_t total = g_total.load();
    uint64_t errors = g_errors.load();
    std::vector<double> lats;
    {
        std::lock_guard<std::mutex> lock(g_latency_mutex);
        lats = g_latencies;
    }
    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"backend\":\"cpp-rknn\",";
    os << "\"version\":\"v0.8.3.4\",";
    os << "\"total_inferences\":" << total << ",";
    os << "\"errors\":" << errors << ",";
    os << "\"successes\":" << (total >= errors ? total - errors : 0) << ",";
    os << "\"error_rate\":" << (total ? (double)errors / (double)total : 0.0);
    if (!lats.empty()) {
        std::sort(lats.begin(), lats.end());
        auto pct = [&](double p) {
            size_t idx = std::min(lats.size() - 1, (size_t)std::round((p / 100.0) * (lats.size() - 1)));
            return lats[idx];
        };
        double sum = std::accumulate(lats.begin(), lats.end(), 0.0);
        double mean = sum / lats.size();
        os << ",\"latency_ms\":{";
        os << "\"mean\":" << mean << ",";
        os << "\"p50\":" << pct(50) << ",";
        os << "\"p95\":" << pct(95) << ",";
        os << "\"p99\":" << pct(99) << ",";
        os << "\"min\":" << lats.front() << ",";
        os << "\"max\":" << lats.back();
        os << "},";
        os << "\"throughput_fps\":" << (mean > 0 ? 1000.0 / mean : 0.0);
    }
    os << "}";
    return os.str();
}

static std::string make_http_response(const std::string& body, const std::string& status = "200 OK", const std::string& content_type = "application/json") {
    std::ostringstream os;
    os << "HTTP/1.1 " << status << "\r\n";
    os << "Content-Type: " << content_type << "\r\n";
    os << "Content-Length: " << body.size() << "\r\n";
    os << "Connection: close\r\n";
    os << "\r\n";
    os << body;
    return os.str();
}

static bool contains(const std::string& s, const std::string& p) {
    return s.find(p) != std::string::npos;
}

static bool parse_bool_value(const std::string& value) {
    std::string v = value;
    std::transform(v.begin(), v.end(), v.begin(), ::tolower);
    return (v == "1" || v == "true" || v == "yes" || v == "on");
}

static std::string header_value(const std::string& headers, const std::string& key) {
    std::istringstream iss(headers);
    std::string line;
    std::string key_lower = key;
    std::transform(key_lower.begin(), key_lower.end(), key_lower.begin(), ::tolower);
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string k = line.substr(0, pos);
        std::transform(k.begin(), k.end(), k.begin(), ::tolower);
        if (k == key_lower) {
            std::string v = line.substr(pos + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            v.erase(v.find_last_not_of(" \t\r\n") + 1);
            return v;
        }
    }
    return "";
}

static std::string query_param_value(const std::string& query, const std::string& key, const std::string& default_value = "") {
    size_t pos = 0;
    while (pos <= query.size()) {
        size_t amp = query.find('&', pos);
        std::string part = query.substr(pos, amp == std::string::npos ? std::string::npos : amp - pos);
        size_t eq = part.find('=');
        std::string k = eq == std::string::npos ? part : part.substr(0, eq);
        std::string v = eq == std::string::npos ? "" : part.substr(eq + 1);
        if (k == key) return v;
        if (amp == std::string::npos) break;
        pos = amp + 1;
    }
    return default_value;
}

static std::vector<unsigned char> extract_image_body(const std::string& content_type, const std::string& body) {
    if (contains(content_type, "multipart/form-data")) {
        auto bpos = content_type.find("boundary=");
        if (bpos == std::string::npos) throw std::runtime_error("multipart boundary not found");
        std::string boundary = content_type.substr(bpos + 9);
        if (!boundary.empty() && boundary.front() == '"') {
            boundary.erase(0, 1);
            if (!boundary.empty() && boundary.back() == '"') boundary.pop_back();
        }

        std::string start_marker = "\r\n\r\n";
        auto h_end = body.find(start_marker);
        if (h_end == std::string::npos) throw std::runtime_error("multipart part header not found");
        size_t data_start = h_end + start_marker.size();

        std::string end_marker = "\r\n--" + boundary;
        auto data_end = body.find(end_marker, data_start);
        if (data_end == std::string::npos) data_end = body.size();

        return std::vector<unsigned char>(body.begin() + data_start, body.begin() + data_end);
    }

    return std::vector<unsigned char>(body.begin(), body.end());
}

static void send_all(int fd, const std::string& s) {
    const char* p = s.data();
    size_t n = s.size();
    while (n > 0) {
        ssize_t sent = ::send(fd, p, n, 0);
        if (sent <= 0) return;
        p += sent;
        n -= sent;
    }
}


class StreamWorker {
public:
    StreamWorker(RknnRunner& runner, const Args& args, const std::vector<std::string>& class_names)
        : runner_(&runner), roi_runner_(nullptr), args_(args), class_names_(class_names) {}

    StreamWorker(RoiPipelineRunnerCpp& roi_runner, const Args& args)
        : runner_(nullptr), roi_runner_(&roi_runner), args_(args), class_names_() {}

    ~StreamWorker() {
        stop();
    }

    bool start(std::string* message = nullptr, bool enable_inference = true) {
        if (args_.camera_source.empty()) {
            if (message) *message = "camera_source is empty; start with --camera-source <rtsp_url|/dev/videoX>";
            return false;
        }

        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) {
            // v0.8.3.4: stream may already be running in preview mode. In that case
            // /stream/start?mode=detect should simply enable inference without
            // reopening RTSP.
            set_inference_enabled(enable_inference);
            if (message) {
                *message = enable_inference
                    ? "stream already running; inference enabled"
                    : "stream already running; inference disabled, preview mode active";
            }
            return true;
        }

        reset_state_for_start(enable_inference);

        capture_thread_ = std::thread(&StreamWorker::capture_loop, this);
        infer_thread_ = std::thread(&StreamWorker::infer_loop, this);

        if (message) {
            *message = enable_inference
                ? "stream started in detect mode"
                : "stream started in preview mode";
        }
        return true;
    }

    bool start_preview(std::string* message = nullptr) {
        return start(message, false);
    }

    bool start_detect(std::string* message = nullptr) {
        return start(message, true);
    }

    bool enable_inference(std::string* message = nullptr) {
        if (!running_.load()) {
            if (message) *message = "stream is not running; call /stream/start?mode=detect or /stream/start first";
            return false;
        }
        set_inference_enabled(true);
        if (message) *message = "inference enabled";
        return true;
    }

    bool disable_inference(std::string* message = nullptr) {
        if (!running_.load()) {
            if (message) *message = "stream is not running; inference already disabled";
            return true;
        }
        set_inference_enabled(false);
        if (message) *message = "inference disabled; stream stays running in preview mode";
        return true;
    }

    void stop() {
        running_.store(false);
        if (capture_thread_.joinable()) capture_thread_.join();
        if (infer_thread_.joinable()) infer_thread_.join();
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            capture_thread_alive_ = false;
            infer_thread_alive_ = false;
        }
    }

    std::string status_json() {
        std::lock_guard<std::mutex> lock(state_mu_);
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << "{";
        os << "\"status\":\"ok\",";
        os << "\"backend\":\"cpp-rknn\",";
        os << "\"version\":\"v0.8.3.4\",";
        os << "\"running\":" << (running_.load() ? "true" : "false") << ",";
        os << "\"inference_enabled\":" << (inference_enabled_.load() ? "true" : "false") << ",";
        os << "\"stream_mode\":\"" << (running_.load() ? (inference_enabled_.load() ? "detect" : "preview") : "stopped") << "\",";
        os << "\"low_latency_mode\":true,";
        os << "\"capture_thread_alive\":" << (capture_thread_alive_ ? "true" : "false") << ",";
        os << "\"infer_thread_alive\":" << (infer_thread_alive_ ? "true" : "false") << ",";
        os << "\"camera_source_set\":" << (!args_.camera_source.empty() ? "true" : "false") << ",";
        os << "\"camera_source\":\"" << json_escape(args_.camera_source) << "\",";
        os << "\"camera_type\":\"" << json_escape(args_.camera_type) << "\",";
        os << "\"camera_width\":" << args_.camera_width << ",";
        os << "\"camera_height\":" << args_.camera_height << ",";
        os << "\"camera_fps\":" << args_.camera_fps << ",";
        os << "\"camera_buffer_size\":" << args_.camera_buffer_size << ",";
        os << "\"camera_fourcc\":\"" << json_escape(args_.camera_fourcc) << "\",";
        os << "\"stream_backend\":\"" << json_escape(args_.stream_backend) << "\",";
        os << "\"stream_codec\":\"" << json_escape(args_.stream_codec) << "\",";
        os << "\"gst_latency_ms\":" << args_.gst_latency_ms << ",";
        os << "\"rtsp_transport\":\"" << json_escape(args_.rtsp_transport) << "\",";
        os << "\"rtsp_timeout_ms\":" << args_.rtsp_timeout_ms << ",";
        os << "\"quiet_ffmpeg_log\":" << (args_.quiet_ffmpeg_log ? "true" : "false") << ",";
        os << "\"enable_snapshot\":" << (args_.enable_snapshot ? "true" : "false") << ",";
        os << "\"enable_annotated\":" << (args_.enable_annotated ? "true" : "false") << ",";
        os << "\"preprocess_backend_requested\":\"" << json_escape(args_.preprocess_status.requested_backend) << "\",";
        os << "\"preprocess_backend_active\":\"" << json_escape(args_.preprocess_status.active_backend) << "\",";
        os << "\"rga_mode_requested\":\"" << json_escape(args_.preprocess_status.requested_rga_mode) << "\",";
        os << "\"rga_mode_active\":\"" << json_escape(args_.preprocess_status.active_rga_mode) << "\",";
        os << "\"rga_compiled\":" << (args_.preprocess_status.rga_compiled ? "true" : "false") << ",";
        os << "\"rga_runtime_available\":" << (args_.preprocess_status.rga_runtime_available ? "true" : "false") << ",";
        os << "\"rga_available\":" << (args_.preprocess_status.rga_available ? "true" : "false") << ",";
        os << "\"rga_enabled_for_preprocess\":" << (args_.preprocess_status.rga_enabled_for_preprocess ? "true" : "false") << ",";
        os << "\"preprocess_backend_reason\":\"" << json_escape(args_.preprocess_status.reason) << "\",";
        os << "\"camera_read_fps_target\":" << args_.camera_read_fps << ",";
        os << "\"detect_fps_target\":" << args_.detect_fps << ",";
        os << "\"snapshot_fps_target\":" << args_.snapshot_fps << ",";
        os << "\"stream_fps_target\":" << args_.detect_fps << ",";
        os << "\"camera_fps\":" << camera_fps_measured_ << ",";
        os << "\"stream_fps\":" << camera_fps_measured_ << ",";
        os << "\"detect_fps\":" << detect_fps_measured_ << ",";
        os << "\"camera_frames\":" << camera_frames_ << ",";
        os << "\"stream_frames\":" << camera_frames_ << ",";
        os << "\"detect_frames\":" << detect_frames_ << ",";
        os << "\"latest_latency_ms\":" << latest_latency_ms_ << ",";
        os << "\"latest_frame_age_ms\":" << latest_frame_age_ms_locked() << ",";
        os << "\"latest_result_age_ms\":" << latest_result_age_ms_locked() << ",";
        os << "\"latest_snapshot_age_ms\":" << latest_snapshot_age_ms_locked() << ",";
        os << "\"latest_annotated_age_ms\":" << latest_annotated_age_ms_locked() << ",";
        os << "\"snapshot_available\":" << (!latest_capture_frame_.empty() ? "true" : "false") << ",";
        os << "\"annotated_available\":" << (!latest_annotated_frame_.empty() ? "true" : "false") << ",";
        os << "\"frame_seq\":" << latest_capture_seq_ << ",";
        os << "\"last_inferred_frame_seq\":" << last_inferred_frame_seq_ << ",";
        os << "\"dropped_overwrite_frames\":" << dropped_overwrite_frames_ << ",";
        os << "\"skipped_duplicate_frames\":" << skipped_duplicate_frames_ << ",";
        os << "\"read_failures\":" << read_failures_ << ",";
        os << "\"consecutive_read_failures\":" << consecutive_read_failures_ << ",";
        os << "\"reconnect_count\":" << reconnect_count_ << ",";
        os << "\"diagnostics\":{";
        os << "\"last_capture_read_ms\":" << last_capture_read_ms_ << ",";
        os << "\"last_snapshot_clone_ms\":" << last_snapshot_clone_ms_ << ",";
        os << "\"last_annotated_draw_ms\":" << last_annotated_draw_ms_ << ",";
        os << "\"last_stream_loop_ms\":" << last_stream_loop_ms_ << ",";
        os << "\"last_infer_loop_ms\":" << last_infer_loop_ms_ << ",";
        os << "\"latest_frame_age_ms\":" << latest_frame_age_ms_locked() << ",";
        os << "\"last_snapshot_request_clone_ms\":" << last_snapshot_request_clone_ms_ << ",";
        os << "\"last_annotated_request_clone_ms\":" << last_annotated_request_clone_ms_ << ",";
        os << "\"last_snapshot_encode_ms\":" << last_snapshot_encode_ms_ << ",";
        os << "\"last_annotated_encode_ms\":" << last_annotated_encode_ms_ << ",";
        os << "\"snapshot_encode_requests\":" << snapshot_encode_requests_ << ",";
        os << "\"annotated_encode_requests\":" << annotated_encode_requests_;
        os << "},";
        os << "\"last_capture_error\":\"" << json_escape(last_capture_error_) << "\",";
        os << "\"last_error\":\"" << json_escape(last_error_) << "\"";
        os << "}";
        return os.str();
    }

    std::string latest_result_json() {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (running_.load() && !inference_enabled_.load()) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(3);
            os << "{\"status\":\"inference_disabled\",\"message\":\"stream is running in preview mode; inference is disabled\",";
            os << "\"latest_frame_available\":" << (!latest_capture_frame_.empty() ? "true" : "false") << ",";
            os << "\"latest_frame_age_ms\":" << latest_frame_age_ms_locked() << ",";
            os << "\"capture_thread_alive\":" << (capture_thread_alive_ ? "true" : "false") << ",";
            os << "\"infer_thread_alive\":" << (infer_thread_alive_ ? "true" : "false") << "}";
            return os.str();
        }
        if (latest_result_json_.empty()) {
            std::ostringstream os;
            os << std::fixed << std::setprecision(3);
            os << "{\"status\":\"no_result\",\"message\":\"stream has not produced result yet\","
               << "\"latest_frame_available\":" << (!latest_capture_frame_.empty() ? "true" : "false") << ","
               << "\"latest_frame_age_ms\":" << latest_frame_age_ms_locked() << ","
               << "\"capture_thread_alive\":" << (capture_thread_alive_ ? "true" : "false") << ","
               << "\"infer_thread_alive\":" << (infer_thread_alive_ ? "true" : "false") << "}";
            return os.str();
        }
        return latest_result_json_;
    }

    std::vector<unsigned char> snapshot_jpeg() {
        std::vector<unsigned char> jpg;
        if (!args_.enable_snapshot) return jpg;

        cv::Mat frame;
        double clone0 = now_ms();
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            if (!latest_capture_frame_.empty()) frame = latest_capture_frame_.clone();
        }
        double clone_ms = now_ms() - clone0;
        if (frame.empty()) return jpg;

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, args_.snapshot_jpeg_quality};
        double enc0 = now_ms();
        cv::imencode(".jpg", frame, jpg, params);
        double enc_ms = now_ms() - enc0;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_snapshot_request_clone_ms_ = clone_ms;
            last_snapshot_encode_ms_ = enc_ms;
            snapshot_encode_requests_++;
        }
        return jpg;
    }

    std::vector<unsigned char> annotated_jpeg() {
        std::vector<unsigned char> jpg;
        if (!args_.enable_annotated) return jpg;

        cv::Mat frame;
        double clone0 = now_ms();
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            if (!latest_annotated_frame_.empty()) frame = latest_annotated_frame_.clone();
        }
        double clone_ms = now_ms() - clone0;
        if (frame.empty()) return jpg;

        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, args_.snapshot_jpeg_quality};
        double enc0 = now_ms();
        cv::imencode(".jpg", frame, jpg, params);
        double enc_ms = now_ms() - enc0;
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_annotated_request_clone_ms_ = clone_ms;
            last_annotated_encode_ms_ = enc_ms;
            annotated_encode_requests_++;
        }
        return jpg;
    }

private:
    void reset_state_for_start(bool enable_inference) {
        std::lock_guard<std::mutex> lock(state_mu_);
        inference_enabled_.store(enable_inference);
        capture_thread_alive_ = false;
        infer_thread_alive_ = false;
        last_error_.clear();
        last_capture_error_.clear();
        latest_result_json_ = enable_inference
            ? "{\"status\":\"starting\",\"message\":\"low latency stream worker is starting in detect mode\"}"
            : "{\"status\":\"inference_disabled\",\"message\":\"stream is starting in preview mode; inference is disabled\"}";
        latest_capture_frame_.release();
        latest_snapshot_frame_.release();
        latest_annotated_frame_.release();
        latest_capture_time_ms_ = 0.0;
        latest_result_time_ms_ = 0.0;
        latest_snapshot_time_ms_ = 0.0;
        latest_annotated_time_ms_ = 0.0;
        latest_capture_seq_ = 0;
        last_inferred_frame_seq_ = 0;
        camera_frames_ = 0;
        detect_frames_ = 0;
        dropped_overwrite_frames_ = 0;
        skipped_duplicate_frames_ = 0;
        read_failures_ = 0;
        consecutive_read_failures_ = 0;
        reconnect_count_ = 0;
        camera_fps_measured_ = 0.0;
        detect_fps_measured_ = 0.0;
        latest_latency_ms_ = 0.0;
        last_capture_read_ms_ = 0.0;
        last_snapshot_clone_ms_ = 0.0;
        last_annotated_draw_ms_ = 0.0;
        last_stream_loop_ms_ = 0.0;
        last_infer_loop_ms_ = 0.0;
        last_snapshot_encode_ms_ = 0.0;
        last_annotated_encode_ms_ = 0.0;
        last_snapshot_request_clone_ms_ = 0.0;
        last_annotated_request_clone_ms_ = 0.0;
        snapshot_encode_requests_ = 0;
        annotated_encode_requests_ = 0;
    }

    void set_inference_enabled(bool enabled) {
        inference_enabled_.store(enabled);
        std::lock_guard<std::mutex> lock(state_mu_);
        if (enabled) {
            latest_result_json_ = "{\"status\":\"starting\",\"message\":\"inference enabled; waiting for next frame\"}";
            latest_result_time_ms_ = 0.0;
            // Force infer_loop to consume the next available frame after enabling.
            last_inferred_frame_seq_ = 0;
        } else {
            latest_result_json_ = "{\"status\":\"inference_disabled\",\"message\":\"stream is running in preview mode; inference is disabled\"}";
            latest_result_time_ms_ = 0.0;
            latest_latency_ms_ = 0.0;
            detect_fps_measured_ = 0.0;
            latest_annotated_frame_.release();
            latest_annotated_time_ms_ = 0.0;
        }
    }

    double latest_frame_age_ms_locked() const {
        if (latest_capture_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_capture_time_ms_;
    }

    double latest_result_age_ms_locked() const {
        if (latest_result_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_result_time_ms_;
    }

    double latest_snapshot_age_ms_locked() const {
        // v0.8.3.4: snapshot.jpg is served from the latest captured frame, so this
        // age reflects the freshest capture frame rather than the inference loop.
        return latest_frame_age_ms_locked();
    }

    double latest_annotated_age_ms_locked() const {
        if (latest_annotated_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_annotated_time_ms_;
    }

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(state_mu_);
        last_error_ = err;
    }

    void clear_error_if_matches_capture_success() {
        std::lock_guard<std::mutex> lock(state_mu_);
        last_capture_error_.clear();
        if (last_error_ == "OpenCV failed to read frame" ||
            last_error_.find("failed to open") != std::string::npos ||
            last_error_.find("reconnecting") != std::string::npos) {
            last_error_.clear();
        }
    }

    std::unique_ptr<visionops::IStreamBackend> open_stream_backend(std::string* error_message) const {
        visionops::StreamOpenOptions stream_options;
        stream_options.backend = args_.stream_backend;
        stream_options.camera_source = args_.camera_source;
        stream_options.stream_codec = args_.stream_codec;
        stream_options.rtsp_transport = args_.rtsp_transport;
        stream_options.rtsp_timeout_ms = args_.rtsp_timeout_ms;
        stream_options.gst_latency_ms = args_.gst_latency_ms;
        stream_options.quiet_ffmpeg_log = args_.quiet_ffmpeg_log;
        stream_options.camera_type = args_.camera_type;
        stream_options.camera_width = args_.camera_width;
        stream_options.camera_height = args_.camera_height;
        stream_options.camera_fps = args_.camera_fps;
        stream_options.camera_buffer_size = args_.camera_buffer_size;
        stream_options.camera_fourcc = args_.camera_fourcc;

        auto backend = visionops::create_stream_backend(stream_options);
        std::string open_error;
        if (!backend->open(&open_error)) {
            if (error_message) *error_message = open_error.empty() ? "failed to open stream backend" : open_error;
            return nullptr;
        }
        return backend;
    }

    void capture_loop() {
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            capture_thread_alive_ = true;
        }

        std::unique_ptr<visionops::IStreamBackend> stream_backend;
        double fps_window_t0 = now_ms();
        uint64_t camera_frames_window = 0;

        auto reconnect = [&]() -> bool {
            if (stream_backend) stream_backend->close();
            stream_backend.reset();
            {
                std::lock_guard<std::mutex> lock(state_mu_);
                reconnect_count_++;
                last_capture_error_ = "reconnecting camera";
                last_error_ = "reconnecting camera";
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(800));

            std::string open_error;
            stream_backend = open_stream_backend(&open_error);
            if (!stream_backend) {
                std::lock_guard<std::mutex> lock(state_mu_);
                last_capture_error_ = open_error.empty() ? "failed to open stream backend" : open_error;
                last_error_ = last_capture_error_;
                return false;
            }

            {
                std::lock_guard<std::mutex> lock(state_mu_);
                consecutive_read_failures_ = 0;
                last_capture_error_.clear();
                if (last_error_ == "reconnecting camera" || last_error_.find("failed to open") != std::string::npos) {
                    last_error_.clear();
                }
            }
            return true;
        };

        std::cout << "[STREAM] low-latency capture thread opening camera: " << args_.camera_source << "\n";
        std::cout << "[STREAM] stream_backend=" << args_.stream_backend
                  << " stream_codec=" << args_.stream_codec
                  << " rtsp_transport=" << args_.rtsp_transport << "\n";

        reconnect();

        while (running_.load()) {
            if (!stream_backend) {
                reconnect();
                continue;
            }

            cv::Mat frame;
            std::string read_error;
            double read_t0 = now_ms();
            bool read_ok = stream_backend->read(frame, &read_error);
            double read_t1 = now_ms();
            double capture_read_ms = read_t1 - read_t0;

            {
                std::lock_guard<std::mutex> lock(state_mu_);
                last_capture_read_ms_ = capture_read_ms;
            }

            if (!read_ok || frame.empty()) {
                bool should_reconnect = false;
                {
                    std::lock_guard<std::mutex> lock(state_mu_);
                    read_failures_++;
                    consecutive_read_failures_++;
                    last_capture_error_ = read_error.empty() ? "OpenCV failed to read frame" : read_error;
                    last_error_ = last_capture_error_;
                    should_reconnect = consecutive_read_failures_ >= 3;
                }
                if (should_reconnect) {
                    reconnect();
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
                continue;
            }

            const double frame_time_ms = now_ms();
            {
                std::lock_guard<std::mutex> lock(state_mu_);
                if (!latest_capture_frame_.empty() && latest_capture_seq_ > last_inferred_frame_seq_) {
                    dropped_overwrite_frames_++;
                }
                latest_capture_frame_ = frame;
                latest_capture_time_ms_ = frame_time_ms;
                latest_capture_seq_++;
                camera_frames_++;
                camera_frames_window++;
                consecutive_read_failures_ = 0;
                last_capture_error_.clear();
                if (last_error_ == "OpenCV failed to read frame" || last_error_ == "reconnecting camera") {
                    last_error_.clear();
                }
            }

            double fps_now = now_ms();
            if (fps_now - fps_window_t0 >= 1000.0) {
                double sec = (fps_now - fps_window_t0) / 1000.0;
                std::lock_guard<std::mutex> lock(state_mu_);
                camera_fps_measured_ = camera_frames_window / sec;
                camera_frames_window = 0;
                fps_window_t0 = fps_now;
            }

            // Keep draining the decoder buffer. This is intentionally not throttled
            // by detect_fps; inference has its own loop and old frames are dropped.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (stream_backend) stream_backend->close();
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            capture_thread_alive_ = false;
        }
        std::cout << "[STREAM] capture thread stopped\n";
    }

    void infer_loop() {
        {
            std::lock_guard<std::mutex> lock(state_mu_);
            infer_thread_alive_ = true;
        }

        const double detect_interval_ms = args_.detect_fps > 0 ? 1000.0 / (double)args_.detect_fps : 100.0;
        const double snapshot_interval_ms = args_.snapshot_fps > 0 ? 1000.0 / (double)args_.snapshot_fps : 1000.0;
        double last_detect_ms = 0.0;
        double last_annotated_ms = 0.0;
        double fps_window_t0 = now_ms();
        uint64_t detect_frames_window = 0;

        while (running_.load()) {
            if (!inference_enabled_.load()) {
                {
                    std::lock_guard<std::mutex> lock(state_mu_);
                    detect_fps_measured_ = 0.0;
                    last_infer_loop_ms_ = 0.0;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }

            double now = now_ms();
            if (last_detect_ms > 0.0 && now - last_detect_ms < detect_interval_ms) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            double loop_t0 = now_ms();
            cv::Mat frame;
            uint64_t seq = 0;
            double frame_time_ms = 0.0;
            double capture_read_ms = 0.0;

            {
                std::lock_guard<std::mutex> lock(state_mu_);
                if (latest_capture_frame_.empty()) {
                    // Capture thread has not produced a frame yet.
                } else if (latest_capture_seq_ == last_inferred_frame_seq_) {
                    skipped_duplicate_frames_++;
                } else {
                    frame = latest_capture_frame_.clone();
                    seq = latest_capture_seq_;
                    frame_time_ms = latest_capture_time_ms_;
                    capture_read_ms = last_capture_read_ms_;
                    last_inferred_frame_seq_ = seq;
                }
            }

            if (frame.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            last_detect_ms = now_ms();

            InferTiming timing;
            timing.stream_detail.capture_read_ms = capture_read_ms;
            double total0 = now_ms();

            PreprocessMeta meta;
            std::vector<Tensor> outputs;
            InferenceResult infer_result;
            if (roi_runner_) {
                infer_result = roi_runner_->infer_bgr(frame, &timing, &meta);
            } else {
                double s0 = now_ms();
                std::string actual_preprocess_backend;
                std::string preprocess_fallback_reason;
                cv::Mat rgb = preprocess_rgb_uint8(frame, args_, meta, actual_preprocess_backend, &preprocess_fallback_reason, &timing.preprocess_detail);
                double s1 = now_ms();
                timing.preprocess_ms = s1 - s0;
                timing.preprocess_backend = actual_preprocess_backend;
                timing.preprocess_detail.total_ms = timing.preprocess_ms;
                if (!preprocess_fallback_reason.empty()) {
                    set_error("preprocess fallback: " + preprocess_fallback_reason);
                }

                outputs = runner_->infer(rgb, &timing.rknn);

                s0 = now_ms();
                infer_result = postprocess_by_task(outputs, args_, class_names_, meta);
                s1 = now_ms();
                timing.postprocess_ms = s1 - s0;
                timing.total_ms = now_ms() - total0;
            }

            cv::Mat annotated;
            double annotated_draw_ms = 0.0;
            bool should_update_annotated = args_.enable_annotated && (last_detect_ms - last_annotated_ms >= snapshot_interval_ms);
            if (should_update_annotated) {
                double draw0 = now_ms();
                annotated = draw_inference_result_on_frame(frame, infer_result);
                annotated_draw_ms = now_ms() - draw0;
                last_annotated_ms = last_detect_ms;
            }

            timing.stream_detail.snapshot_clone_ms = 0.0;
            timing.stream_detail.annotated_draw_ms = annotated_draw_ms;
            timing.stream_detail.loop_total_ms = now_ms() - loop_t0;
            timing.stream_detail.state_update_ms = 0.0;
            std::string result = inference_result_to_json(args_, infer_result, timing.total_ms, outputs, &timing, &meta);

            double state0 = now_ms();
            {
                std::lock_guard<std::mutex> lock(state_mu_);
                latest_result_json_ = result;
                if (should_update_annotated && !annotated.empty()) {
                    latest_annotated_frame_ = std::move(annotated);
                    latest_annotated_time_ms_ = now_ms();
                }
                latest_result_time_ms_ = now_ms();
                latest_latency_ms_ = timing.total_ms;
                detect_frames_++;
                detect_frames_window++;
                last_annotated_draw_ms_ = annotated_draw_ms;
                last_stream_loop_ms_ = timing.stream_detail.loop_total_ms;
                last_infer_loop_ms_ = timing.stream_detail.loop_total_ms;
            }
            timing.stream_detail.state_update_ms = now_ms() - state0;

            double fps_now = now_ms();
            if (fps_now - fps_window_t0 >= 1000.0) {
                double sec = (fps_now - fps_window_t0) / 1000.0;
                std::lock_guard<std::mutex> lock(state_mu_);
                detect_fps_measured_ = detect_frames_window / sec;
                detect_frames_window = 0;
                fps_window_t0 = fps_now;
            }
        }

        {
            std::lock_guard<std::mutex> lock(state_mu_);
            infer_thread_alive_ = false;
        }
        std::cout << "[STREAM] infer thread stopped\n";
    }

    RknnRunner* runner_ = nullptr;
    RoiPipelineRunnerCpp* roi_runner_ = nullptr;
    Args args_;
    std::vector<std::string> class_names_;

    std::atomic<bool> running_{false};
    std::atomic<bool> inference_enabled_{true};
    std::thread capture_thread_;
    std::thread infer_thread_;
    std::mutex state_mu_;

    bool capture_thread_alive_ = false;
    bool infer_thread_alive_ = false;

    cv::Mat latest_capture_frame_;
    cv::Mat latest_snapshot_frame_;   // kept for compatibility; snapshot.jpg now serves latest_capture_frame_.
    cv::Mat latest_annotated_frame_;
    std::string latest_result_json_;
    std::string last_error_;
    std::string last_capture_error_;

    double latest_capture_time_ms_ = 0.0;
    double latest_result_time_ms_ = 0.0;
    double latest_snapshot_time_ms_ = 0.0;
    double latest_annotated_time_ms_ = 0.0;
    double latest_latency_ms_ = 0.0;
    double camera_fps_measured_ = 0.0;
    double detect_fps_measured_ = 0.0;
    double last_capture_read_ms_ = 0.0;
    double last_snapshot_clone_ms_ = 0.0;
    double last_annotated_draw_ms_ = 0.0;
    double last_stream_loop_ms_ = 0.0;
    double last_infer_loop_ms_ = 0.0;
    double last_snapshot_request_clone_ms_ = 0.0;
    double last_annotated_request_clone_ms_ = 0.0;
    double last_snapshot_encode_ms_ = 0.0;
    double last_annotated_encode_ms_ = 0.0;

    uint64_t snapshot_encode_requests_ = 0;
    uint64_t annotated_encode_requests_ = 0;
    uint64_t camera_frames_ = 0;
    uint64_t detect_frames_ = 0;
    uint64_t latest_capture_seq_ = 0;
    uint64_t last_inferred_frame_seq_ = 0;
    uint64_t dropped_overwrite_frames_ = 0;
    uint64_t skipped_duplicate_frames_ = 0;
    uint64_t read_failures_ = 0;
    uint64_t consecutive_read_failures_ = 0;
    uint64_t reconnect_count_ = 0;
};


static void handle_client(int client_fd, RknnRunner* runner, RoiPipelineRunnerCpp* roi_runner, StreamWorker& stream_worker, const Args& args, const std::vector<std::string>& class_names) {
    try {
        double request_start = now_ms();
        std::string req;
        char buf[8192];
        size_t header_end = std::string::npos;

        while ((header_end = req.find("\r\n\r\n")) == std::string::npos) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) return;
            req.append(buf, buf + n);
            if (req.size() > 1024 * 1024 * 32) throw std::runtime_error("request too large");
        }

        std::string headers = req.substr(0, header_end);
        std::istringstream first_line(headers);
        std::string method, path, version;
        first_line >> method >> path >> version;
        std::string raw_path = path;
        std::string query_string;
        size_t query_pos = path.find('?');
        if (query_pos != std::string::npos) {
            query_string = path.substr(query_pos + 1);
            path = path.substr(0, query_pos);
        }

        // v0.1.4: support curl/libcurl HTTP Expect: 100-continue.
        // Without this, large multipart uploads may wait about 1 second before sending body.
        std::string expect_header = header_value(headers, "Expect");
        std::string expect_lower = expect_header;
        std::transform(expect_lower.begin(), expect_lower.end(), expect_lower.begin(), ::tolower);
        if (expect_lower.find("100-continue") != std::string::npos) {
            send_all(client_fd, "HTTP/1.1 100 Continue\r\n\r\n");
        }

        std::string body = req.substr(header_end + 4);
        std::string cl = header_value(headers, "Content-Length");
        size_t content_length = cl.empty() ? 0 : std::stoul(cl);
        while (body.size() < content_length) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) break;
            body.append(buf, buf + n);
        }
        double request_done = now_ms();

        if (method == "GET" && path == "/health") {
            const bool health_is_roi = (normalize_task_name(args.task) == "roi_classification" && roi_runner != nullptr);
            const RoiPipelineConfigCpp* health_roi_cfg = health_is_roi ? &roi_runner->cfg() : nullptr;
            const Args* health_primary_args = health_is_roi ? &health_roi_cfg->detector_args : &args;
            std::ostringstream os;
            os << "{"
               << "\"status\":\"ok\","
               << "\"backend\":\"cpp-rknn\","
               << "\"version\":\"v0.8.3.4\","
               << "\"output_mode\":\"" << json_escape(args.output_mode) << "\","
               << "\"preprocess_backend_requested\":\"" << json_escape(args.preprocess_status.requested_backend) << "\","
               << "\"preprocess_backend_active\":\"" << json_escape(args.preprocess_status.active_backend) << "\","
               << "\"rga_mode_requested\":\"" << json_escape(args.preprocess_status.requested_rga_mode) << "\","
               << "\"rga_mode_active\":\"" << json_escape(args.preprocess_status.active_rga_mode) << "\","
               << "\"preprocess_backend_reason\":\"" << json_escape(args.preprocess_status.reason) << "\","
               << "\"rga_compiled\":" << (args.preprocess_status.rga_compiled ? "true" : "false") << ","
               << "\"rga_runtime_available\":" << (args.preprocess_status.rga_runtime_available ? "true" : "false") << ","
               << "\"rga_available\":" << (args.preprocess_status.rga_available ? "true" : "false") << ","
               << "\"rga_enabled_for_preprocess\":" << (args.preprocess_status.rga_enabled_for_preprocess ? "true" : "false") << ","
               << "\"task\":\"" << json_escape(args.task) << "\","
               << "\"model\":\"" << json_escape(health_primary_args->model) << "\","
               << "\"class_names_file\":\"" << json_escape(health_primary_args->class_names_file) << "\","
               << "\"input_size\":[" << health_primary_args->input_h << "," << health_primary_args->input_w << "],"
               << "\"num_classes\":" << health_primary_args->num_classes << ","
               << "\"topk\":" << (normalize_task_name(args.task) == "classification" ? args.topk : (health_is_roi ? health_roi_cfg->classifier_args.topk : 0)) << ","
               << "\"mask_threshold\":" << (normalize_task_name(args.task) == "segmentation" ? args.mask_threshold : 0.0f) << ","
               << "\"active_model_kind\":\"" << (health_is_roi ? "roi_pipeline" : "single_model") << "\","
               << "\"classification_ready\":" << (normalize_task_name(args.task) == "classification" ? "true" : "false") << ","
               << "\"obb_ready\":" << (normalize_task_name(args.task) == "obb_detection" ? "true" : "false") << ","
               << "\"segmentation_ready\":" << (normalize_task_name(args.task) == "segmentation" ? "true" : "false") << ","
               << "\"camera_source_set\":" << (!args.camera_source.empty() ? "true" : "false") << ","
               << "\"camera_source\":\"" << json_escape(args.camera_source) << "\","
               << "\"camera_type\":\"" << json_escape(args.camera_type) << "\","
               << "\"camera_width\":" << args.camera_width << ","
               << "\"camera_height\":" << args.camera_height << ","
               << "\"camera_fps\":" << args.camera_fps << ","
               << "\"camera_buffer_size\":" << args.camera_buffer_size << ","
               << "\"camera_fourcc\":\"" << json_escape(args.camera_fourcc) << "\","
               << "\"stream_backend\":\"" << json_escape(args.stream_backend) << "\","
               << "\"stream_codec\":\"" << json_escape(args.stream_codec) << "\","
               << "\"gst_latency_ms\":" << args.gst_latency_ms << ","
               << "\"rtsp_transport\":\"" << json_escape(args.rtsp_transport) << "\","
               << "\"rtsp_timeout_ms\":" << args.rtsp_timeout_ms << ","
               << "\"quiet_ffmpeg_log\":" << (args.quiet_ffmpeg_log ? "true" : "false") << ","
               << "\"enable_snapshot\":" << (args.enable_snapshot ? "true" : "false") << ","
               << "\"enable_annotated\":" << (args.enable_annotated ? "true" : "false") << ","
               << "\"camera_read_fps\":" << args.camera_read_fps << ","
               << "\"detect_fps\":" << args.detect_fps << ","
               << "\"stream_fps\":" << args.detect_fps << ","
               << "\"snapshot_fps\":" << args.snapshot_fps << ","
               << "\"stream_modes\":[\"preview\",\"detect\"],"
               << "\"inference_control_endpoints\":[\"/inference/start\",\"/inference/stop\"],";
            if (normalize_task_name(args.task) == "roi_classification" && roi_runner) {
                const auto& rcfg = roi_runner->cfg();
                os << "\"pipeline_config\":\"" << json_escape(rcfg.pipeline_config) << "\","
                   << "\"pipeline_name\":\"" << json_escape(rcfg.pipeline_name) << "\","
                   << "\"roi_classification_ready\":true,"
                   << "\"stage1\":{"
                   << "\"task\":\"detection\","
                   << "\"model_path\":\"" << json_escape(rcfg.detector_args.model) << "\","
                   << "\"meta_path\":\"" << json_escape(rcfg.detector_args.class_names_file) << "\","
                   << "\"input_size\":[" << rcfg.detector_args.input_h << "," << rcfg.detector_args.input_w << "],"
                   << "\"num_classes\":" << rcfg.detector_args.num_classes << ","
                   << "\"class_names\":" << json_string_array(rcfg.detector_names) << ","
                   << "\"conf_threshold\":" << rcfg.detector_args.conf_threshold << ","
                   << "\"nms_threshold\":" << rcfg.detector_args.nms_threshold << ","
                   << "\"select_policy\":\"" << json_escape(rcfg.select_policy) << "\","
                   << "\"target_class_id\":" << rcfg.target_class_id << ","
                   << "\"target_class_name\":\"" << json_escape(rcfg.target_class_name) << "\""
                   << "},"
                   << "\"roi\":{"
                   << "\"mode\":\"" << json_escape(rcfg.roi_mode) << "\","
                   << "\"padding_ratio\":" << rcfg.padding_ratio << ","
                   << "\"min_width\":" << rcfg.min_roi_width << ","
                   << "\"min_height\":" << rcfg.min_roi_height << ","
                   << "\"relative_box\":{"
                   << "\"x1\":" << rcfg.relative_box.at("x1") << ","
                   << "\"y1\":" << rcfg.relative_box.at("y1") << ","
                   << "\"x2\":" << rcfg.relative_box.at("x2") << ","
                   << "\"y2\":" << rcfg.relative_box.at("y2") << "}"
                   << "},"
                   << "\"stage2\":{"
                   << "\"task\":\"classification\","
                   << "\"model_path\":\"" << json_escape(rcfg.classifier_args.model) << "\","
                   << "\"meta_path\":\"" << json_escape(rcfg.classifier_args.class_names_file) << "\","
                   << "\"input_size\":[" << rcfg.classifier_args.input_h << "," << rcfg.classifier_args.input_w << "],"
                   << "\"num_classes\":" << rcfg.classifier_args.num_classes << ","
                   << "\"class_names\":" << json_string_array(rcfg.classifier_names) << ","
                   << "\"topk\":" << rcfg.classifier_args.topk
                   << "},";
            } else {
                os << "\"roi_classification_ready\":false,";
            }
            os << "\"visual_debug_endpoint\":\"/stream/annotated.jpg\""
               << "}";
            send_all(client_fd, make_http_response(os.str()));
            return;
        }

        if (method == "GET" && path == "/stats") {
            send_all(client_fd, make_http_response(stats_json()));
            return;
        }

        if (method == "POST" && path == "/stream/start") {
            std::string mode = query_param_value(query_string, "mode", "detect");
            std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
            bool enable_inference = !(mode == "preview" || mode == "capture" || mode == "camera");
            std::string msg;
            bool ok = stream_worker.start(&msg, enable_inference);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"mode\":\"" + (enable_inference ? "detect" : "preview") +
                               "\",\"message\":\"" + json_escape(msg) + "\"}";
            send_all(client_fd, make_http_response(body, ok ? "200 OK" : "400 Bad Request"));
            return;
        }

        if (method == "POST" && path == "/stream/preview/start") {
            std::string msg;
            bool ok = stream_worker.start_preview(&msg);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"mode\":\"preview\",\"message\":\"" + json_escape(msg) + "\"}";
            send_all(client_fd, make_http_response(body, ok ? "200 OK" : "400 Bad Request"));
            return;
        }

        if (method == "POST" && path == "/stream/detect/start") {
            std::string msg;
            bool ok = stream_worker.start_detect(&msg);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"mode\":\"detect\",\"message\":\"" + json_escape(msg) + "\"}";
            send_all(client_fd, make_http_response(body, ok ? "200 OK" : "400 Bad Request"));
            return;
        }

        if (method == "POST" && path == "/inference/start") {
            std::string msg;
            bool ok = stream_worker.enable_inference(&msg);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"inference_enabled\":true,\"message\":\"" + json_escape(msg) + "\"}";
            send_all(client_fd, make_http_response(body, ok ? "200 OK" : "400 Bad Request"));
            return;
        }

        if (method == "POST" && path == "/inference/stop") {
            std::string msg;
            bool ok = stream_worker.disable_inference(&msg);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"inference_enabled\":false,\"message\":\"" + json_escape(msg) + "\"}";
            send_all(client_fd, make_http_response(body, ok ? "200 OK" : "400 Bad Request"));
            return;
        }

        if (method == "POST" && path == "/stream/stop") {
            stream_worker.stop();
            send_all(client_fd, make_http_response("{\"status\":\"ok\",\"message\":\"stream stopped\"}"));
            return;
        }

        if (method == "GET" && path == "/stream/status") {
            send_all(client_fd, make_http_response(stream_worker.status_json()));
            return;
        }

        if (method == "GET" && path == "/stream/latest_result") {
            send_all(client_fd, make_http_response(stream_worker.latest_result_json()));
            return;
        }

        if (method == "GET" && path == "/stream/snapshot.jpg") {
            if (!args.enable_snapshot) {
                send_all(client_fd, make_http_response("{\"status\":\"disabled\",\"message\":\"snapshot cache is disabled by --enable-snapshot false\"}", "404 Not Found"));
                return;
            }
            auto jpg = stream_worker.snapshot_jpeg();
            if (jpg.empty()) {
                send_all(client_fd, make_http_response("{\"status\":\"no_snapshot\",\"message\":\"snapshot is not ready; start stream and wait until snapshot_available=true in /stream/status\"}", "404 Not Found"));
            } else {
                std::string body(reinterpret_cast<const char*>(jpg.data()), jpg.size());
                send_all(client_fd, make_http_response(body, "200 OK", "image/jpeg"));
            }
            return;
        }

        if (method == "GET" && path == "/stream/annotated.jpg") {
            if (!args.enable_annotated) {
                send_all(client_fd, make_http_response("{\"status\":\"disabled\",\"message\":\"annotated cache is disabled by --enable-annotated false\"}", "404 Not Found"));
                return;
            }
            auto jpg = stream_worker.annotated_jpeg();
            if (jpg.empty()) {
                send_all(client_fd, make_http_response("{\"status\":\"no_annotated_frame\",\"message\":\"annotated frame is not ready; start stream and wait until annotated_available=true in /stream/status\"}", "404 Not Found"));
            } else {
                std::string body(reinterpret_cast<const char*>(jpg.data()), jpg.size());
                send_all(client_fd, make_http_response(body, "200 OK", "image/jpeg"));
            }
            return;
        }

        if (method == "POST" && path == "/infer") {
            g_total++;
            InferTiming timing;
            timing.request_read_ms = request_done - request_start;
            double t0 = now_ms();
            double stage0 = t0;
            double stage1 = t0;

            stage0 = now_ms();
            std::string content_type = header_value(headers, "Content-Type");
            auto image_bytes = extract_image_body(content_type, body);
            stage1 = now_ms();
            timing.body_extract_ms = stage1 - stage0;
            if (image_bytes.empty()) throw std::runtime_error("empty image body");

            stage0 = now_ms();
            cv::Mat raw(1, (int)image_bytes.size(), CV_8UC1, image_bytes.data());
            cv::Mat bgr = cv::imdecode(raw, cv::IMREAD_COLOR);
            stage1 = now_ms();
            timing.image_decode_ms = stage1 - stage0;
            if (bgr.empty()) throw std::runtime_error("cv::imdecode failed");

            PreprocessMeta meta;
            std::vector<Tensor> outputs;
            InferenceResult result;
            if (roi_runner) {
                result = roi_runner->infer_bgr(bgr, &timing, &meta);
            } else {
                if (!runner) throw std::runtime_error("RknnRunner is null for single-model inference");
                stage0 = now_ms();
                std::string actual_preprocess_backend;
                std::string preprocess_fallback_reason;
                cv::Mat rgb = preprocess_rgb_uint8(bgr, args, meta, actual_preprocess_backend, &preprocess_fallback_reason, &timing.preprocess_detail);
                stage1 = now_ms();
                timing.preprocess_ms = stage1 - stage0;
                timing.preprocess_backend = actual_preprocess_backend;
                timing.preprocess_detail.total_ms = timing.preprocess_ms;

                outputs = runner->infer(rgb, &timing.rknn);

                stage0 = now_ms();
                result = postprocess_by_task(outputs, args, class_names, meta);
                stage1 = now_ms();
                timing.postprocess_ms = stage1 - stage0;
            }

            double latency = now_ms() - t0;
            timing.total_ms = latency;
            {
                std::lock_guard<std::mutex> lock(g_latency_mutex);
                g_latencies.push_back(latency);
                if (g_latencies.size() > 200) g_latencies.erase(g_latencies.begin());
            }

            send_all(client_fd, make_http_response(inference_result_to_json(args, result, latency, outputs, &timing, &meta)));
            return;
        }

        send_all(client_fd, make_http_response("{\"status\":\"error\",\"message\":\"not found\"}", "404 Not Found"));
    } catch (const std::exception& e) {
        g_errors++;
        std::string body = std::string("{\"status\":\"error\",\"message\":\"") + json_escape(e.what()) + "\"}";
        send_all(client_fd, make_http_response(body, "500 Internal Server Error"));
    }
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto need = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + name);
            return argv[++i];
        };
        if (k == "--model") a.model = need(k);
        else if (k == "--pipeline-config") a.pipeline_config = need(k);
        else if (k == "--class-names-file") a.class_names_file = need(k);
        else if (k == "--task") a.task = need(k);
        else if (k == "--host") a.host = need(k);
        else if (k == "--port") a.port = std::stoi(need(k));
        else if (k == "--npu-core") a.npu_core = need(k);
        else if (k == "--num-classes") a.num_classes = std::stoi(need(k));
        else if (k == "--conf-threshold") a.conf_threshold = std::stof(need(k));
        else if (k == "--nms-threshold") a.nms_threshold = std::stof(need(k));
        else if (k == "--mask-threshold") {
            a.mask_threshold = std::stof(need(k));
            if (a.mask_threshold < 0.0f || a.mask_threshold > 1.0f) {
                throw std::runtime_error("invalid --mask-threshold, expected 0..1");
            }
        }
        else if (k == "--topk") a.topk = std::stoi(need(k));
        else if (k == "--max-det") a.max_det = std::stoi(need(k));
        else if (k == "--output-mode") {
            a.output_mode = need(k);
            std::transform(a.output_mode.begin(), a.output_mode.end(), a.output_mode.begin(), ::tolower);
            if (a.output_mode != "float" && a.output_mode != "quant") {
                throw std::runtime_error("invalid --output-mode, expected float or quant");
            }
        }
        else if (k == "--camera-source") {
            a.camera_source = need(k);
        }
        else if (k == "--camera-type") {
            a.camera_type = need(k);
            std::transform(a.camera_type.begin(), a.camera_type.end(), a.camera_type.begin(), ::tolower);
            if (a.camera_type != "auto" && a.camera_type != "rtsp" && a.camera_type != "usb") {
                throw std::runtime_error("invalid --camera-type, expected auto, rtsp, or usb");
            }
        }
        else if (k == "--camera-width") {
            a.camera_width = std::stoi(need(k));
            if (a.camera_width < 0) throw std::runtime_error("invalid --camera-width");
        }
        else if (k == "--camera-height") {
            a.camera_height = std::stoi(need(k));
            if (a.camera_height < 0) throw std::runtime_error("invalid --camera-height");
        }
        else if (k == "--camera-fps") {
            a.camera_fps = std::stoi(need(k));
            if (a.camera_fps < 0) throw std::runtime_error("invalid --camera-fps");
        }
        else if (k == "--camera-buffer-size") {
            a.camera_buffer_size = std::stoi(need(k));
            if (a.camera_buffer_size <= 0) throw std::runtime_error("invalid --camera-buffer-size");
        }
        else if (k == "--camera-fourcc") {
            a.camera_fourcc = need(k);
            std::transform(a.camera_fourcc.begin(), a.camera_fourcc.end(), a.camera_fourcc.begin(), ::toupper);
            if (!a.camera_fourcc.empty() && a.camera_fourcc.size() != 4) {
                throw std::runtime_error("invalid --camera-fourcc, expected 4 chars, e.g. YUYV or MJPG");
            }
        }
        else if (k == "--stream-fps") {
            // Legacy v0.2 argument: treat it as both camera read FPS and detect FPS.
            a.stream_fps = std::stoi(need(k));
            if (a.stream_fps <= 0) throw std::runtime_error("invalid --stream-fps");
            a.camera_read_fps = a.stream_fps;
            a.detect_fps = a.stream_fps;
        }
        else if (k == "--camera-read-fps") {
            a.camera_read_fps = std::stoi(need(k));
            if (a.camera_read_fps <= 0) throw std::runtime_error("invalid --camera-read-fps");
        }
        else if (k == "--detect-fps") {
            a.detect_fps = std::stoi(need(k));
            if (a.detect_fps <= 0) throw std::runtime_error("invalid --detect-fps");
            a.stream_fps = a.detect_fps;
        }
        else if (k == "--snapshot-fps") {
            a.snapshot_fps = std::stoi(need(k));
            if (a.snapshot_fps <= 0) throw std::runtime_error("invalid --snapshot-fps");
        }
        else if (k == "--snapshot-jpeg-quality") {
            a.snapshot_jpeg_quality = std::stoi(need(k));
            if (a.snapshot_jpeg_quality < 1 || a.snapshot_jpeg_quality > 100) {
                throw std::runtime_error("invalid --snapshot-jpeg-quality, expected 1..100");
            }
        }
        else if (k == "--enable-snapshot") {
            a.enable_snapshot = parse_bool_value(need(k));
        }
        else if (k == "--enable-annotated") {
            a.enable_annotated = parse_bool_value(need(k));
        }
        else if (k == "--preprocess-backend") {
            a.preprocess_backend = visionops::normalize_preprocess_backend(need(k));
            if (a.preprocess_backend == "invalid") {
                throw std::runtime_error("invalid --preprocess-backend, expected cpu, rga, or auto");
            }
        }
        else if (k == "--rga-mode") {
            a.rga_mode = visionops::normalize_rga_mode(need(k));
            if (a.rga_mode == "invalid") {
                throw std::runtime_error("invalid --rga-mode, expected off, resize_color, or resize_only");
            }
        }
        else if (k == "--stream-auto-start") {
            std::string v = need(k);
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            a.stream_auto_start = (v == "1" || v == "true" || v == "yes" || v == "on");
        }
        else if (k == "--stream-backend") {
            a.stream_backend = need(k);
            std::transform(a.stream_backend.begin(), a.stream_backend.end(), a.stream_backend.begin(), ::tolower);
            if (a.stream_backend != "opencv" && a.stream_backend != "gst-mpp") {
                throw std::runtime_error("invalid --stream-backend, expected opencv or gst-mpp");
            }
        }
        else if (k == "--stream-codec") {
            a.stream_codec = need(k);
            std::transform(a.stream_codec.begin(), a.stream_codec.end(), a.stream_codec.begin(), ::tolower);
            if (a.stream_codec != "h264" && a.stream_codec != "h265") {
                throw std::runtime_error("invalid --stream-codec, expected h264 or h265");
            }
        }
        else if (k == "--gst-latency-ms") {
            a.gst_latency_ms = std::stoi(need(k));
            if (a.gst_latency_ms < 0 || a.gst_latency_ms > 5000) {
                throw std::runtime_error("invalid --gst-latency-ms, expected 0..5000");
            }
        }
        else if (k == "--rtsp-transport") {
            a.rtsp_transport = need(k);
            std::transform(a.rtsp_transport.begin(), a.rtsp_transport.end(), a.rtsp_transport.begin(), ::tolower);
            if (a.rtsp_transport != "tcp" && a.rtsp_transport != "udp") {
                throw std::runtime_error("invalid --rtsp-transport, expected tcp or udp");
            }
        }
        else if (k == "--rtsp-timeout-ms") {
            a.rtsp_timeout_ms = std::stoi(need(k));
            if (a.rtsp_timeout_ms < 1000) throw std::runtime_error("invalid --rtsp-timeout-ms, expected >= 1000");
        }
        else if (k == "--quiet-ffmpeg-log") {
            std::string v = need(k);
            std::transform(v.begin(), v.end(), v.begin(), ::tolower);
            a.quiet_ffmpeg_log = (v == "1" || v == "true" || v == "yes" || v == "on");
        }
        else if (k == "--input-size") {
            std::string v = need(k);
            std::replace(v.begin(), v.end(), ',', ' ');
            std::istringstream iss(v);
            iss >> a.input_h >> a.input_w;
            if (a.input_h <= 0 || a.input_w <= 0) throw std::runtime_error("invalid --input-size");
        } else if (k == "-h" || k == "--help") {
            std::cout << "Usage: visionops_inference_cpp "
                      << "--model xxx.rknn --task detection --input-size 640,640 "
                      << "--class-names-file xxx.yaml --port 18080 [--camera-source rtsp://...|/dev/video7] "
                      << "[--camera-type rtsp|usb|auto] [--camera-width 1280 --camera-height 800 --camera-fps 10 --camera-fourcc YUYV] "
                      << "[--stream-backend opencv|gst-mpp] [--preprocess-backend cpu|rga|auto] [--rga-mode off|resize_color|resize_only] "
                      << "[--mask-threshold 0.5] [--enable-snapshot true|false] [--enable-annotated true|false] [--stream-codec h264|h265] "
                      << "[--camera-read-fps 10] [--detect-fps 10] [--rtsp-transport tcp] [--pipeline-config /path/pipeline.yaml]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown arg: " + k);
        }
    }
    return a;
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        args.preprocess_backend = visionops::normalize_preprocess_backend(args.preprocess_backend);
        if (args.preprocess_backend == "invalid") {
            throw std::runtime_error("invalid --preprocess-backend, expected cpu, rga, or auto");
        }
        args.rga_mode = visionops::normalize_rga_mode(args.rga_mode);
        if (args.rga_mode == "invalid") {
            throw std::runtime_error("invalid --rga-mode, expected off, resize_color, or resize_only");
        }
        args.preprocess_status = visionops::init_preprocess_backend(args.preprocess_backend, args.rga_mode);
        configure_opencv_ffmpeg_quiet_logging(args.quiet_ffmpeg_log);
        auto class_names = load_class_names_simple_yaml(args.class_names_file, args.num_classes);
        if ((int)class_names.size() != args.num_classes) {
            std::cout << "[WARN] class_names size=" << class_names.size()
                      << " num_classes=" << args.num_classes << "\n";
        }

        args.task = normalize_task_name(args.task);
        if (args.task != "detection" && args.task != "classification" &&
            args.task != "obb_detection" && args.task != "segmentation" &&
            args.task != "roi_classification") {
            throw std::runtime_error("unsupported --task: " + args.task);
        }
        if (args.task == "roi_classification" && args.pipeline_config.empty()) {
            throw std::runtime_error("--pipeline-config is required when --task roi_classification");
        }

        std::cout << "[INFO] task=" << args.task
                  << " model=" << args.model
                  << " input_size=" << args.input_h << "," << args.input_w
                  << " num_classes=" << args.num_classes
                  << " topk=" << args.topk
                  << " mask_threshold=" << args.mask_threshold << "\n";
        if (args.task == "segmentation") {
            std::cout << "[INFO] segmentation mask_threshold=" << args.mask_threshold << "\n";
        }

        std::cout << "[INFO] output_mode=" << args.output_mode << "\n";
        std::cout << "[INFO] preprocess_backend requested=" << args.preprocess_status.requested_backend
                  << " active=" << args.preprocess_status.active_backend
                  << " rga_mode_requested=" << args.preprocess_status.requested_rga_mode
                  << " rga_mode_active=" << args.preprocess_status.active_rga_mode
                  << " rga_compiled=" << (args.preprocess_status.rga_compiled ? "true" : "false")
                  << " rga_runtime_available=" << (args.preprocess_status.rga_runtime_available ? "true" : "false")
                  << " rga_available=" << (args.preprocess_status.rga_available ? "true" : "false")
                  << " reason=" << args.preprocess_status.reason << "\n";
        std::cout << "[INFO] stream_backend=" << args.stream_backend
                  << " stream_codec=" << args.stream_codec
                  << " gst_latency_ms=" << args.gst_latency_ms << "\n";
        std::cout << "[INFO] camera_type=" << args.camera_type
                  << " camera_source=" << args.camera_source
                  << " camera_width=" << args.camera_width
                  << " camera_height=" << args.camera_height
                  << " camera_fps=" << args.camera_fps
                  << " camera_fourcc=" << args.camera_fourcc
                  << " camera_buffer_size=" << args.camera_buffer_size << "\n";
        std::cout << "[INFO] rtsp_transport=" << args.rtsp_transport
                  << " rtsp_timeout_ms=" << args.rtsp_timeout_ms
                  << " quiet_ffmpeg_log=" << (args.quiet_ffmpeg_log ? "true" : "false") << "\n";
        std::cout << "[INFO] visual cache enable_snapshot=" << (args.enable_snapshot ? "true" : "false")
                  << " enable_annotated=" << (args.enable_annotated ? "true" : "false") << "\n";
        std::unique_ptr<RknnRunner> runner;
        std::unique_ptr<RoiPipelineRunnerCpp> roi_runner;
        std::unique_ptr<StreamWorker> stream_worker;
        if (args.task == "roi_classification") {
            RoiPipelineConfigCpp roi_cfg = load_roi_pipeline_config_cpp(args.pipeline_config, args);
            roi_runner.reset(new RoiPipelineRunnerCpp(roi_cfg));
            stream_worker.reset(new StreamWorker(*roi_runner, args));
        } else {
            runner.reset(new RknnRunner(args));
            stream_worker.reset(new StreamWorker(*runner, args, class_names));
        }
        if (args.stream_auto_start) {
            std::string msg;
            if (!stream_worker->start(&msg)) {
                std::cout << "[WARN] stream auto start failed: " << msg << "\n";
            }
        }

        int server_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) throw std::runtime_error("socket failed");

        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(args.port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            throw std::runtime_error("bind failed on port " + std::to_string(args.port) + ": " + std::strerror(errno));
        }

        if (listen(server_fd, 32) < 0) throw std::runtime_error("listen failed");

        std::cout << "[OK] visionops_inference_cpp v0.8.4.1 started at 0.0.0.0:" << args.port << "\n";
        std::cout << "[OK] endpoints: GET /health, POST /infer, GET /stats, POST /stream/start, POST /stream/stop, GET /stream/status, GET /stream/latest_result, GET /stream/snapshot.jpg, GET /stream/annotated.jpg, POST /inference/start, POST /inference/stop\n";

        while (true) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) continue;
            std::thread([client_fd, &runner, &roi_runner, &stream_worker, args, class_names]() {
                handle_client(client_fd, runner.get(), roi_runner.get(), *stream_worker, args, class_names);
                close(client_fd);
            }).detach();
        }
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
