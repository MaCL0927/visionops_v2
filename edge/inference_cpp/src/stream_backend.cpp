#include "visionops/stream_backend.hpp"

#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <iostream>

#include <opencv2/core/utils/logger.hpp>

namespace visionops {
namespace {

static bool looks_like_integer_camera_index(const std::string& source) {
    if (source.empty()) return false;
    for (char c : source) {
        if (c < '0' || c > '9') return false;
    }
    return true;
}

static bool starts_with(const std::string& s, const std::string& prefix) {
    return s.rfind(prefix, 0) == 0;
}

static bool looks_like_rtsp_source(const std::string& source) {
    return starts_with(source, "rtsp://") || starts_with(source, "rtsps://");
}

static bool looks_like_usb_v4l2_source(const std::string& source) {
    return starts_with(source, "/dev/video") || starts_with(source, "/dev/v4l/");
}

static int make_fourcc(const std::string& fourcc) {
    if (fourcc.size() != 4) return 0;
    return cv::VideoWriter::fourcc(fourcc[0], fourcc[1], fourcc[2], fourcc[3]);
}

static std::string fourcc_to_string(double fourcc_value) {
    int v = static_cast<int>(fourcc_value);
    std::string s;
    s.push_back(static_cast<char>(v & 0xFF));
    s.push_back(static_cast<char>((v >> 8) & 0xFF));
    s.push_back(static_cast<char>((v >> 16) & 0xFF));
    s.push_back(static_cast<char>((v >> 24) & 0xFF));
    return s;
}

static void configure_quiet_logs(bool quiet) {
    if (!quiet) return;
    setenv("OPENCV_LOG_LEVEL", "ERROR", 1);
    setenv("OPENCV_VIDEOIO_DEBUG", "0", 1);
    setenv("OPENCV_FFMPEG_LOGLEVEL", "16", 1);  // AV_LOG_ERROR
    unsetenv("OPENCV_FFMPEG_DEBUG");
    try {
        cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);
    } catch (...) {
    }
}

static void configure_ffmpeg_capture_options(const StreamOpenOptions& options) {
    // OpenCV FFmpeg expects stimeout in microseconds.
    long long stimeout_us = static_cast<long long>(options.rtsp_timeout_ms) * 1000LL;
    std::ostringstream os;
    os << "rtsp_transport;" << options.rtsp_transport
       << "|stimeout;" << stimeout_us
       // Low-latency RTSP hints. Some OpenCV/FFmpeg builds may ignore part of
       // these options, so v0.5.4 still relies primarily on the split
       // capture/infer workers to keep draining the decoder buffer.
       << "|fflags;nobuffer"
       << "|flags;low_delay"
       << "|max_delay;0"
       << "|reorder_queue_size;0";
    setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", os.str().c_str(), 1);
}

class OpenCvStreamBackend final : public IStreamBackend {
public:
    explicit OpenCvStreamBackend(StreamOpenOptions options) : options_(std::move(options)) {}

    bool open(std::string* error_message = nullptr) override {
        configure_quiet_logs(options_.quiet_ffmpeg_log);
        configure_ffmpeg_capture_options(options_);

        if (options_.camera_source.empty()) {
            if (error_message) *error_message = "camera_source is empty";
            return false;
        }

        bool ok = false;
        if (looks_like_integer_camera_index(options_.camera_source)) {
            int idx = std::stoi(options_.camera_source);
            ok = cap_.open(idx, cv::CAP_V4L2);
            if (!ok) ok = cap_.open(idx);
        } else if (looks_like_usb_v4l2_source(options_.camera_source)) {
            // v0.7.1: USB/UVC cameras such as Orbbec RGB should use V4L2, not FFmpeg.
            ok = cap_.open(options_.camera_source, cv::CAP_V4L2);
            if (!ok) ok = cap_.open(options_.camera_source);
        } else if (looks_like_rtsp_source(options_.camera_source)) {
            ok = cap_.open(options_.camera_source, cv::CAP_FFMPEG);
            if (!ok) ok = cap_.open(options_.camera_source);
        } else {
            // Keep the old fallback behavior for local video files or other OpenCV sources.
            ok = cap_.open(options_.camera_source);
        }

        if (!ok || !cap_.isOpened()) {
            if (error_message) *error_message = "OpenCV failed to open camera source: " + options_.camera_source;
            return false;
        }

        // Keep latency low where backend supports these properties.
        int buffer_size = options_.camera_buffer_size > 0 ? options_.camera_buffer_size : 1;
        cap_.set(cv::CAP_PROP_BUFFERSIZE, buffer_size);

        if (!looks_like_rtsp_source(options_.camera_source)) {
            int fourcc = make_fourcc(options_.camera_fourcc);
            if (fourcc != 0) cap_.set(cv::CAP_PROP_FOURCC, fourcc);
            if (options_.camera_width > 0) cap_.set(cv::CAP_PROP_FRAME_WIDTH, options_.camera_width);
            if (options_.camera_height > 0) cap_.set(cv::CAP_PROP_FRAME_HEIGHT, options_.camera_height);
            if (options_.camera_fps > 0) cap_.set(cv::CAP_PROP_FPS, options_.camera_fps);
        }

        std::ostringstream info;
        info << "[STREAM] OpenCV source=" << options_.camera_source
             << " camera_type=" << options_.camera_type
             << " width=" << cap_.get(cv::CAP_PROP_FRAME_WIDTH)
             << " height=" << cap_.get(cv::CAP_PROP_FRAME_HEIGHT)
             << " fps=" << cap_.get(cv::CAP_PROP_FPS)
             << " fourcc=" << fourcc_to_string(cap_.get(cv::CAP_PROP_FOURCC))
             << " buffer_size=" << buffer_size;
        std::cout << info.str() << std::endl;

        return true;
    }

    bool read(cv::Mat& frame_bgr, std::string* error_message = nullptr) override {
        if (!cap_.isOpened()) {
            if (error_message) *error_message = "OpenCV stream is not opened";
            return false;
        }
        if (!cap_.read(frame_bgr) || frame_bgr.empty()) {
            if (error_message) *error_message = "OpenCV failed to read frame";
            return false;
        }
        return true;
    }

    void close() override {
        if (cap_.isOpened()) cap_.release();
    }

    bool is_opened() const override {
        return cap_.isOpened();
    }

    std::string name() const override {
        return "opencv";
    }

private:
    StreamOpenOptions options_;
    cv::VideoCapture cap_;
};

static std::string make_gst_mpp_pipeline(const StreamOpenOptions& options) {
    // This is intentionally kept as an optional RTSP path. v0.3.1 does not rely on MPP.
    // It is useful only when the board's OpenCV is built with GStreamer and Rockchip plugins exist.
    std::string depay = options.stream_codec == "h265" ? "rtph265depay ! h265parse" : "rtph264depay ! h264parse";
    std::string protocols = options.rtsp_transport == "udp" ? "udp" : "tcp";

    std::ostringstream ss;
    ss << "rtspsrc location=\"" << options.camera_source << "\" "
       << "latency=" << options.gst_latency_ms << " protocols=" << protocols << " ! "
       << depay << " ! "
       << "mppvideodec ! "
       << "videoconvert ! video/x-raw,format=BGR ! "
       << "appsink sync=false drop=true max-buffers=1";
    return ss.str();
}

class GstMppStreamBackend final : public IStreamBackend {
public:
    explicit GstMppStreamBackend(StreamOpenOptions options) : options_(std::move(options)) {}

    bool open(std::string* error_message = nullptr) override {
        configure_quiet_logs(options_.quiet_ffmpeg_log);
        if (options_.camera_source.empty()) {
            if (error_message) *error_message = "camera_source is empty";
            return false;
        }
        if (!looks_like_rtsp_source(options_.camera_source)) {
            if (error_message) *error_message = "gst-mpp backend only supports RTSP sources; use --stream-backend opencv for USB";
            return false;
        }
        pipeline_ = make_gst_mpp_pipeline(options_);
        bool ok = cap_.open(pipeline_, cv::CAP_GSTREAMER);
        if (!ok || !cap_.isOpened()) {
            if (error_message) *error_message = "GStreamer/MPP failed to open pipeline: " + pipeline_;
            return false;
        }
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        return true;
    }

    bool read(cv::Mat& frame_bgr, std::string* error_message = nullptr) override {
        if (!cap_.isOpened()) {
            if (error_message) *error_message = "GStreamer/MPP stream is not opened";
            return false;
        }
        if (!cap_.read(frame_bgr) || frame_bgr.empty()) {
            if (error_message) *error_message = "GStreamer/MPP failed to read frame";
            return false;
        }
        return true;
    }

    void close() override {
        if (cap_.isOpened()) cap_.release();
    }

    bool is_opened() const override {
        return cap_.isOpened();
    }

    std::string name() const override {
        return "gst-mpp";
    }

private:
    StreamOpenOptions options_;
    std::string pipeline_;
    cv::VideoCapture cap_;
};

}  // namespace

std::unique_ptr<IStreamBackend> create_stream_backend(const StreamOpenOptions& options) {
    if (options.backend == "opencv" || options.backend.empty()) {
        return std::make_unique<OpenCvStreamBackend>(options);
    }
    if (options.backend == "gst-mpp") {
        return std::make_unique<GstMppStreamBackend>(options);
    }
    throw std::runtime_error("unknown stream backend: " + options.backend);
}

}  // namespace visionops
