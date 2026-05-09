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
#include <fstream>
#include <iomanip>
#include <limits>
#include <iostream>
#include <mutex>
#include <memory>
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

    // v0.4.3: force quiet OpenCV/FFmpeg logging.
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
    std::string host = "0.0.0.0";
    std::string npu_core = "auto";
    int port = 18080;
    int input_h = 640;
    int input_w = 640;
    int num_classes = 80;
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int topk = 5;
    int max_det = 100;
    std::string output_mode = "float";  // v0.1.6: float | quant

    // v0.2/v0.3: RTSP stream worker sidecar mode.
    std::string camera_source = "";
    int stream_fps = 10;          // legacy alias, kept for compatibility; equals detect_fps by default.
    int camera_read_fps = 10;     // v0.3: throttle RTSP read/decode rate on main stream.
    int detect_fps = 10;          // v0.3: target inference FPS.
    int snapshot_fps = 1;         // v0.3: how often to refresh cached snapshot/annotated caches.
    int snapshot_jpeg_quality = 80;
    // v0.4.3: decouple visual cache generation from realtime inference.
    // Disable these to measure RTSP + inference CPU without frame clone / overlay overhead.
    bool enable_snapshot = true;
    bool enable_annotated = true;
    bool stream_auto_start = false;

    // v0.4.3: RTSP robustness / log control. Keep TCP by default for main stream.
    std::string rtsp_transport = "tcp";   // tcp | udp
    int rtsp_timeout_ms = 5000;            // maps to FFmpeg stimeout, microseconds internally
    bool quiet_ffmpeg_log = true;          // reduce FFmpeg/OpenCV decode warning noise

    // v0.4.3: selectable stream backend.
    // opencv: low-risk OpenCV/FFmpeg path, kept as the default fallback.
    // gst-mpp: OpenCV + GStreamer pipeline using Rockchip mppvideodec.
    std::string stream_backend = "opencv"; // opencv | gst-mpp
    std::string stream_codec = "h264";     // h264 | h265
    int gst_latency_ms = 100;              // rtspsrc latency for gst-mpp backend

    // v0.4.3: preprocessing backend switch with RGA experiment modes.
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
            if (t[0] != '-') {
                if (t.find(':') != std::string::npos) in_class_names = false;
                continue;
            }
            std::string name = t.substr(1);
            name.erase(0, name.find_first_not_of(" \t\"'"));
            name.erase(name.find_last_not_of(" \t\"'\r\n") + 1);
            if (!name.empty()) names.push_back(name);
        }
    }

    if (names.empty()) {
        for (int i = 0; i < num_classes; ++i) names.push_back(std::to_string(i));
    }
    return names;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

    for (const auto& t : outputs) {
        if (t.dims.size() != 4 || t.dims[0] != 1 || t.dims[2] != h || t.dims[3] != w) {
            continue;
        }
        if (is_split_box_head(t)) box = &t;
        else if (is_split_cls_head(t, nc)) cls = &t;
        else if (is_split_obj_head(t)) obj = &t;
    }

    return box != nullptr && cls != nullptr && obj != nullptr;
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
    os << "\"version\":\"v0.4.3\",";
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
    os << "\"version\":\"v0.4.3\",";
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
        : runner_(runner), args_(args), class_names_(class_names) {}

    ~StreamWorker() {
        stop();
    }

    bool start(std::string* message = nullptr) {
        if (args_.camera_source.empty()) {
            if (message) *message = "camera_source is empty; start with --camera-source <rtsp_url>";
            return false;
        }

        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) {
            if (message) *message = "stream already running";
            return true;
        }

        {
            std::lock_guard<std::mutex> lock(state_mu_);
            last_error_.clear();
            latest_result_json_ = "{\"status\":\"starting\",\"message\":\"stream worker is starting\"}";
            latest_snapshot_frame_.release();
            latest_annotated_frame_.release();
            latest_annotated_time_ms_ = 0.0;
            camera_frames_ = 0;
            detect_frames_ = 0;
            camera_fps_measured_ = 0.0;
            detect_fps_measured_ = 0.0;
            latest_latency_ms_ = 0.0;
            latest_result_time_ms_ = 0.0;
            latest_snapshot_time_ms_ = 0.0;
            last_capture_read_ms_ = 0.0;
            last_snapshot_clone_ms_ = 0.0;
            last_annotated_draw_ms_ = 0.0;
            last_stream_loop_ms_ = 0.0;
            last_snapshot_encode_ms_ = 0.0;
            last_annotated_encode_ms_ = 0.0;
            last_snapshot_request_clone_ms_ = 0.0;
            last_annotated_request_clone_ms_ = 0.0;
            snapshot_encode_requests_ = 0;
            annotated_encode_requests_ = 0;
        }

        worker_ = std::thread(&StreamWorker::loop, this);
        if (message) *message = "stream started";
        return true;
    }

    void stop() {
        if (running_.exchange(false)) {
            if (worker_.joinable()) worker_.join();
        } else {
            if (worker_.joinable()) worker_.join();
        }
    }

    std::string status_json() {
        std::lock_guard<std::mutex> lock(state_mu_);
        std::ostringstream os;
        os << std::fixed << std::setprecision(3);
        os << "{";
        os << "\"status\":\"ok\",";
        os << "\"backend\":\"cpp-rknn\",";
        os << "\"version\":\"v0.4.3\",";
        os << "\"running\":" << (running_.load() ? "true" : "false") << ",";
        os << "\"camera_source_set\":" << (!args_.camera_source.empty() ? "true" : "false") << ",";
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
        // legacy fields kept for compatibility with v0.2 scripts
        os << "\"stream_fps_target\":" << args_.detect_fps << ",";
        os << "\"camera_fps\":" << camera_fps_measured_ << ",";
        os << "\"stream_fps\":" << camera_fps_measured_ << ",";
        os << "\"detect_fps\":" << detect_fps_measured_ << ",";
        os << "\"camera_frames\":" << camera_frames_ << ",";
        os << "\"stream_frames\":" << camera_frames_ << ",";
        os << "\"detect_frames\":" << detect_frames_ << ",";
        os << "\"latest_latency_ms\":" << latest_latency_ms_ << ",";
        os << "\"latest_result_age_ms\":" << latest_result_age_ms_locked() << ",";
        os << "\"latest_snapshot_age_ms\":" << latest_snapshot_age_ms_locked() << ",";
        os << "\"latest_annotated_age_ms\":" << latest_annotated_age_ms_locked() << ",";
        os << "\"snapshot_available\":" << (!latest_snapshot_frame_.empty() ? "true" : "false") << ",";
        os << "\"annotated_available\":" << (!latest_annotated_frame_.empty() ? "true" : "false") << ",";
        os << "\"diagnostics\":{";
        os << "\"last_capture_read_ms\":" << last_capture_read_ms_ << ",";
        os << "\"last_snapshot_clone_ms\":" << last_snapshot_clone_ms_ << ",";
        os << "\"last_annotated_draw_ms\":" << last_annotated_draw_ms_ << ",";
        os << "\"last_stream_loop_ms\":" << last_stream_loop_ms_ << ",";
        os << "\"last_snapshot_request_clone_ms\":" << last_snapshot_request_clone_ms_ << ",";
        os << "\"last_annotated_request_clone_ms\":" << last_annotated_request_clone_ms_ << ",";
        os << "\"last_snapshot_encode_ms\":" << last_snapshot_encode_ms_ << ",";
        os << "\"last_annotated_encode_ms\":" << last_annotated_encode_ms_ << ",";
        os << "\"snapshot_encode_requests\":" << snapshot_encode_requests_ << ",";
        os << "\"annotated_encode_requests\":" << annotated_encode_requests_;
        os << "},";
        os << "\"last_error\":\"" << json_escape(last_error_) << "\"";
        os << "}";
        return os.str();
    }

    std::string latest_result_json() {
        std::lock_guard<std::mutex> lock(state_mu_);
        if (latest_result_json_.empty()) {
            return "{\"status\":\"no_result\",\"message\":\"stream has not produced result yet\"}";
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
            if (!latest_snapshot_frame_.empty()) frame = latest_snapshot_frame_.clone();
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
    double latest_result_age_ms_locked() const {
        if (latest_result_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_result_time_ms_;
    }

    double latest_snapshot_age_ms_locked() const {
        if (latest_snapshot_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_snapshot_time_ms_;
    }

    double latest_annotated_age_ms_locked() const {
        if (latest_annotated_time_ms_ <= 0.0) return -1.0;
        return now_ms() - latest_annotated_time_ms_;
    }

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(state_mu_);
        last_error_ = err;
    }

    static void sleep_remaining_ms(double remaining_ms) {
        if (remaining_ms <= 0.0) return;
        int ms = (int)std::min(remaining_ms, 50.0);
        if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    void loop() {
        std::unique_ptr<visionops::IStreamBackend> stream_backend;
        try {
            std::cout << "[STREAM] opening camera: " << args_.camera_source << "\n";
            std::cout << "[STREAM] stream_backend=" << args_.stream_backend
                      << " stream_codec=" << args_.stream_codec
                      << " rtsp_transport=" << args_.rtsp_transport << "\n";

            visionops::StreamOpenOptions stream_options;
            stream_options.backend = args_.stream_backend;
            stream_options.camera_source = args_.camera_source;
            stream_options.stream_codec = args_.stream_codec;
            stream_options.rtsp_transport = args_.rtsp_transport;
            stream_options.rtsp_timeout_ms = args_.rtsp_timeout_ms;
            stream_options.gst_latency_ms = args_.gst_latency_ms;
            stream_options.quiet_ffmpeg_log = args_.quiet_ffmpeg_log;

            stream_backend = visionops::create_stream_backend(stream_options);
            std::string open_error;
            if (!stream_backend->open(&open_error)) {
                set_error(open_error.empty() ? "failed to open stream backend" : open_error);
                running_.store(false);
                return;
            }

            const double read_interval_ms = args_.camera_read_fps > 0 ? 1000.0 / (double)args_.camera_read_fps : 0.0;
            const double detect_interval_ms = args_.detect_fps > 0 ? 1000.0 / (double)args_.detect_fps : 100.0;
            const double snapshot_interval_ms = args_.snapshot_fps > 0 ? 1000.0 / (double)args_.snapshot_fps : 1000.0;

            double last_read_ms = 0.0;
            double last_detect_ms = 0.0;
            double last_snapshot_ms = 0.0;
            double last_annotated_ms = 0.0;
            double fps_window_t0 = now_ms();
            uint64_t camera_frames_window = 0;
            uint64_t detect_frames_window = 0;

            while (running_.load()) {
                double loop_t0 = now_ms();
                double before_read = now_ms();
                if (read_interval_ms > 0.0 && last_read_ms > 0.0) {
                    double since_last_read = before_read - last_read_ms;
                    if (since_last_read < read_interval_ms) {
                        sleep_remaining_ms(read_interval_ms - since_last_read);
                        continue;
                    }
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
                    set_error(read_error.empty() ? "failed to read frame from camera" : read_error);
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                double frame_time_ms = now_ms();
                last_read_ms = frame_time_ms;

                {
                    std::lock_guard<std::mutex> lock(state_mu_);
                    camera_frames_++;
                    camera_frames_window++;
                }

                // v0.4.3: snapshot cache can be disabled to isolate RTSP/decode/inference CPU.
                double snapshot_clone_ms = 0.0;
                if (args_.enable_snapshot && frame_time_ms - last_snapshot_ms >= snapshot_interval_ms) {
                    double snap0 = now_ms();
                    cv::Mat snapshot = frame.clone();
                    snapshot_clone_ms = now_ms() - snap0;
                    {
                        std::lock_guard<std::mutex> lock(state_mu_);
                        latest_snapshot_frame_ = std::move(snapshot);
                        latest_snapshot_time_ms_ = frame_time_ms;
                        last_snapshot_clone_ms_ = snapshot_clone_ms;
                    }
                    last_snapshot_ms = frame_time_ms;
                }

                if (frame_time_ms - last_detect_ms < detect_interval_ms) {
                    double loop_ms = now_ms() - loop_t0;
                    {
                        std::lock_guard<std::mutex> lock(state_mu_);
                        last_stream_loop_ms_ = loop_ms;
                    }
                    continue;
                }
                last_detect_ms = frame_time_ms;

                InferTiming timing;
                timing.stream_detail.capture_read_ms = capture_read_ms;
                timing.stream_detail.snapshot_clone_ms = snapshot_clone_ms;
                double total0 = now_ms();

                double s0 = now_ms();
                PreprocessMeta meta;
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

                auto outputs = runner_.infer(rgb, &timing.rknn);

                s0 = now_ms();
                auto dets = postprocess_detection(outputs, args_, class_names_, meta);
                s1 = now_ms();
                timing.postprocess_ms = s1 - s0;
                timing.total_ms = now_ms() - total0;

                // v0.4.3: visual debug image is refreshed at snapshot_fps, not every
                // inference, so the validation endpoint does not distort the FPS baseline.
                cv::Mat annotated;
                double annotated_draw_ms = 0.0;
                bool should_update_annotated = args_.enable_annotated && (frame_time_ms - last_annotated_ms >= snapshot_interval_ms);
                if (should_update_annotated) {
                    double draw0 = now_ms();
                    annotated = draw_detections_on_frame(frame, dets);
                    annotated_draw_ms = now_ms() - draw0;
                    last_annotated_ms = frame_time_ms;
                }

                timing.stream_detail.annotated_draw_ms = annotated_draw_ms;
                timing.stream_detail.loop_total_ms = now_ms() - loop_t0;
                timing.stream_detail.state_update_ms = 0.0;
                std::string result = detections_to_json(args_, dets, timing.total_ms, outputs, &timing, &meta);
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
                    last_error_.clear();
                }
                timing.stream_detail.state_update_ms = now_ms() - state0;

                double fps_now = now_ms();
                if (fps_now - fps_window_t0 >= 1000.0) {
                    double sec = (fps_now - fps_window_t0) / 1000.0;
                    std::lock_guard<std::mutex> lock(state_mu_);
                    camera_fps_measured_ = camera_frames_window / sec;
                    detect_fps_measured_ = detect_frames_window / sec;
                    camera_frames_window = 0;
                    detect_frames_window = 0;
                    fps_window_t0 = fps_now;
                }
            }

            if (stream_backend) stream_backend->close();
            std::cout << "[STREAM] stopped\n";
        } catch (const std::exception& e) {
            if (stream_backend) stream_backend->close();
            set_error(e.what());
            running_.store(false);
        }
    }

    RknnRunner& runner_;
    Args args_;
    std::vector<std::string> class_names_;

    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex state_mu_;

    cv::Mat latest_snapshot_frame_;
    cv::Mat latest_annotated_frame_;
    std::string latest_result_json_;
    std::string last_error_;
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
    double last_snapshot_request_clone_ms_ = 0.0;
    double last_annotated_request_clone_ms_ = 0.0;
    double last_snapshot_encode_ms_ = 0.0;
    double last_annotated_encode_ms_ = 0.0;
    uint64_t snapshot_encode_requests_ = 0;
    uint64_t annotated_encode_requests_ = 0;
    uint64_t camera_frames_ = 0;
    uint64_t detect_frames_ = 0;
};

static void handle_client(int client_fd, RknnRunner& runner, StreamWorker& stream_worker, const Args& args, const std::vector<std::string>& class_names) {
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
            std::ostringstream os;
            os << "{"
               << "\"status\":\"ok\","
               << "\"backend\":\"cpp-rknn\","
               << "\"version\":\"v0.4.3\","
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
               << "\"model\":\"" << json_escape(args.model) << "\","
               << "\"class_names_file\":\"" << json_escape(args.class_names_file) << "\","
               << "\"input_size\":[" << args.input_h << "," << args.input_w << "],"
               << "\"num_classes\":" << args.num_classes << ","
               << "\"camera_source_set\":" << (!args.camera_source.empty() ? "true" : "false") << ","
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
               << "\"visual_debug_endpoint\":\"/stream/annotated.jpg\""
               << "}";
            send_all(client_fd, make_http_response(os.str()));
            return;
        }

        if (method == "GET" && path == "/stats") {
            send_all(client_fd, make_http_response(stats_json()));
            return;
        }

        if (method == "POST" && path == "/stream/start") {
            std::string msg;
            bool ok = stream_worker.start(&msg);
            std::string body = std::string("{\"status\":\"") + (ok ? "ok" : "error") +
                               "\",\"message\":\"" + json_escape(msg) + "\"}";
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

            stage0 = now_ms();
            PreprocessMeta meta;
            std::string actual_preprocess_backend;
            std::string preprocess_fallback_reason;
            cv::Mat rgb = preprocess_rgb_uint8(bgr, args, meta, actual_preprocess_backend, &preprocess_fallback_reason, &timing.preprocess_detail);
            stage1 = now_ms();
            timing.preprocess_ms = stage1 - stage0;
            timing.preprocess_backend = actual_preprocess_backend;
            timing.preprocess_detail.total_ms = timing.preprocess_ms;

            auto outputs = runner.infer(rgb, &timing.rknn);

            stage0 = now_ms();
            auto dets = postprocess_detection(outputs, args, class_names, meta);
            stage1 = now_ms();
            timing.postprocess_ms = stage1 - stage0;

            double latency = now_ms() - t0;
            timing.total_ms = latency;
            {
                std::lock_guard<std::mutex> lock(g_latency_mutex);
                g_latencies.push_back(latency);
                if (g_latencies.size() > 200) g_latencies.erase(g_latencies.begin());
            }

            send_all(client_fd, make_http_response(detections_to_json(args, dets, latency, outputs, &timing, &meta)));
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
        else if (k == "--class-names-file") a.class_names_file = need(k);
        else if (k == "--task") a.task = need(k);
        else if (k == "--host") a.host = need(k);
        else if (k == "--port") a.port = std::stoi(need(k));
        else if (k == "--npu-core") a.npu_core = need(k);
        else if (k == "--num-classes") a.num_classes = std::stoi(need(k));
        else if (k == "--conf-threshold") a.conf_threshold = std::stof(need(k));
        else if (k == "--nms-threshold") a.nms_threshold = std::stof(need(k));
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
                      << "--class-names-file xxx.yaml --port 18080 [--camera-source rtsp://...] [--stream-backend opencv|gst-mpp] [--preprocess-backend cpu|rga|auto] [--rga-mode off|resize_color|resize_only] [--enable-snapshot true|false] [--enable-annotated true|false] [--stream-codec h264|h265] [--camera-read-fps 10] [--detect-fps 10] [--rtsp-transport tcp]\n";
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
        std::cout << "[INFO] rtsp_transport=" << args.rtsp_transport
                  << " rtsp_timeout_ms=" << args.rtsp_timeout_ms
                  << " quiet_ffmpeg_log=" << (args.quiet_ffmpeg_log ? "true" : "false") << "\n";
        std::cout << "[INFO] visual cache enable_snapshot=" << (args.enable_snapshot ? "true" : "false")
                  << " enable_annotated=" << (args.enable_annotated ? "true" : "false") << "\n";
        RknnRunner runner(args);
        StreamWorker stream_worker(runner, args, class_names);
        if (args.stream_auto_start) {
            std::string msg;
            if (!stream_worker.start(&msg)) {
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

        std::cout << "[OK] visionops_inference_cpp v0.4.3 started at 0.0.0.0:" << args.port << "\n";
        std::cout << "[OK] endpoints: GET /health, POST /infer, GET /stats, POST /stream/start, POST /stream/stop, GET /stream/status, GET /stream/latest_result, GET /stream/snapshot.jpg, GET /stream/annotated.jpg\n";

        while (true) {
            int client_fd = accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) continue;
            std::thread([client_fd, &runner, &stream_worker, args, class_names]() {
                handle_client(client_fd, runner, stream_worker, args, class_names);
                close(client_fd);
            }).detach();
        }
    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
