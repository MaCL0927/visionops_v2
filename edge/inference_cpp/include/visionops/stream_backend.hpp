#pragma once

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>

namespace visionops {

struct StreamOpenOptions {
    // opencv: low-risk default path, works for RTSP and USB camera indexes/paths supported by OpenCV.
    // gst-mpp: optional GStreamer pipeline using Rockchip mppvideodec for RTSP only.
    std::string backend = "opencv";
    std::string camera_source;
    std::string stream_codec = "h264";       // h264 | h265, only used by gst-mpp
    std::string rtsp_transport = "tcp";      // tcp | udp
    int rtsp_timeout_ms = 5000;               // OpenCV FFmpeg stimeout in ms
    int gst_latency_ms = 100;                 // rtspsrc latency, only used by gst-mpp
    bool quiet_ffmpeg_log = true;
};

class IStreamBackend {
public:
    virtual ~IStreamBackend() = default;
    virtual bool open(std::string* error_message = nullptr) = 0;
    virtual bool read(cv::Mat& frame_bgr, std::string* error_message = nullptr) = 0;
    virtual void close() = 0;
    virtual bool is_opened() const = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<IStreamBackend> create_stream_backend(const StreamOpenOptions& options);

}  // namespace visionops
