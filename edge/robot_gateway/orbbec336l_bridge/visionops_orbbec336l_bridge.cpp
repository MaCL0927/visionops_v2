// VisionOps Orbbec Gemini 336L SDK bridge
// HTTP endpoints:
//   GET  /health
//   GET  /stream/status
//   GET  /stream/snapshot.jpg
//   GET  /stream/depth.png       16-bit PNG depth, millimeters when scale is available
//   GET  /stream/depth_vis.jpg   visualized depth JPEG
//   GET  /stream/depth_meta
//   GET  /stream.mjpeg, /stream/mjpeg, /stream.mjpg
//   POST /stream/start, /stream/stop  compatibility no-op endpoints

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "libobsensor/ObSensor.hpp"

namespace {

std::atomic<bool> g_running{true};

static std::string getenv_str(const char *name, const std::string &fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    return std::string(v);
}

static int getenv_int(const char *name, int fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    try { return std::stoi(v); } catch (...) { return fallback; }
}

static bool getenv_bool(const char *name, bool fallback) {
    const char *v = std::getenv(name);
    if (!v || !*v) return fallback;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return (s == "1" || s == "true" || s == "yes" || s == "on");
}

static std::string now_string() {
    char buf[64] = {0};
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
    localtime_r(&t, &tmv);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tmv);
    return std::string(buf);
}

static std::string json_escape(const std::string &s) {
    std::ostringstream os;
    for (char c : s) {
        switch (c) {
        case '\\': os << "\\\\"; break;
        case '"': os << "\\\""; break;
        case '\n': os << "\\n"; break;
        case '\r': os << "\\r"; break;
        case '\t': os << "\\t"; break;
        default: os << c; break;
        }
    }
    return os.str();
}

static bool encode_jpeg(const cv::Mat &bgr, int quality, std::vector<uchar> &out) {
    if (bgr.empty()) return false;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, quality))};
    return cv::imencode(".jpg", bgr, out, params);
}

static bool encode_png16(const cv::Mat &u16, std::vector<uchar> &out) {
    if (u16.empty() || u16.type() != CV_16UC1) return false;
    std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 3};
    return cv::imencode(".png", u16, out, params);
}

static cv::Mat depth_to_vis(const cv::Mat &depth_mm) {
    if (depth_mm.empty()) return cv::Mat();
    double minv = 0.0, maxv = 0.0;
    cv::Mat mask = depth_mm > 0;
    if (cv::countNonZero(mask) <= 0) {
        return cv::Mat(depth_mm.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    }
    cv::minMaxLoc(depth_mm, &minv, &maxv, nullptr, nullptr, mask);
    if (maxv <= minv) maxv = minv + 1.0;
    cv::Mat u8;
    depth_mm.convertTo(u8, CV_8U, 255.0 / (maxv - minv), -minv * 255.0 / (maxv - minv));
    u8.setTo(0, ~mask);
    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_JET);
    color.setTo(cv::Scalar(0, 0, 0), ~mask);
    return color;
}

static std::string frame_format_to_string(OBFormat f) {
    switch (f) {
    case OB_FORMAT_RGB: return "RGB";
    case OB_FORMAT_BGR: return "BGR";
    case OB_FORMAT_MJPG: return "MJPG";
    case OB_FORMAT_YUYV: return "YUYV";
    case OB_FORMAT_NV12: return "NV12";
    case OB_FORMAT_NV21: return "NV21";
    case OB_FORMAT_Y16: return "Y16";
    case OB_FORMAT_Y8: return "Y8";
    default: return "UNKNOWN";
    }
}

class OrbbecBridge {
public:
    OrbbecBridge()
        : http_host_(getenv_str("VISIONOPS_ORBBEC336L_HTTP_HOST", "127.0.0.1")),
          http_port_(getenv_int("VISIONOPS_ORBBEC336L_HTTP_PORT", 18182)),
          color_width_(getenv_int("VISIONOPS_ORBBEC336L_COLOR_WIDTH", 640)),
          color_height_(getenv_int("VISIONOPS_ORBBEC336L_COLOR_HEIGHT", 480)),
          depth_width_(getenv_int("VISIONOPS_ORBBEC336L_DEPTH_WIDTH", 640)),
          depth_height_(getenv_int("VISIONOPS_ORBBEC336L_DEPTH_HEIGHT", 480)),
          fps_(getenv_int("VISIONOPS_ORBBEC336L_FPS", 30)),
          jpeg_quality_(getenv_int("VISIONOPS_ORBBEC336L_JPEG_QUALITY", 85)),
          mjpeg_fps_(getenv_int("VISIONOPS_ORBBEC336L_MJPEG_FPS", 10)),
          flip_vertical_(getenv_bool("VISIONOPS_ORBBEC336L_FLIP_VERTICAL", false)),
          flip_horizontal_(getenv_bool("VISIONOPS_ORBBEC336L_FLIP_HORIZONTAL", false)),
          serial_(getenv_str("VISIONOPS_ORBBEC336L_SERIAL", "")) {}

    bool start_camera() {
        try {
            ctx_ = std::make_shared<ob::Context>();
            if (!serial_.empty()) {
                auto dev_list = ctx_->queryDeviceList();
                bool found = false;
                for (uint32_t i = 0; i < dev_list->deviceCount(); ++i) {
                    auto dev = dev_list->getDevice(i);
                    auto info = dev->getDeviceInfo();
                    std::string sn = info ? info->serialNumber() : "";
                    if (sn == serial_) {
                        pipeline_ = std::make_shared<ob::Pipeline>(dev);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    std::cerr << "[ERROR] Orbbec serial not found: " << serial_ << std::endl;
                    return false;
                }
            } else {
                pipeline_ = std::make_shared<ob::Pipeline>();
            }

            auto cfg = std::make_shared<ob::Config>();
            enable_color_stream(cfg);
            enable_depth_stream(cfg);
            cfg->setAlignMode(ALIGN_D2C_SW_MODE);
            pipeline_->start(cfg);
            camera_started_ = true;
            start_ms_ = now_ms();
            std::cerr << "[INFO] Orbbec Gemini 336L bridge camera started" << std::endl;
            camera_thread_ = std::thread([this]() { this->camera_loop(); });
            return true;
        } catch (const ob::Error &e) {
            std::cerr << "[ERROR] Orbbec SDK error while starting camera: " << e.getMessage() << std::endl;
            last_error_ = e.getMessage();
            return false;
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] start camera failed: " << e.what() << std::endl;
            last_error_ = e.what();
            return false;
        }
    }

    void stop_camera() {
        g_running = false;
        if (camera_thread_.joinable()) camera_thread_.join();
        try {
            if (pipeline_) pipeline_->stop();
        } catch (...) {}
        camera_started_ = false;
    }

    bool start_http() {
        server_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd_ < 0) {
            std::cerr << "[FATAL] socket failed" << std::endl;
            return false;
        }
        int yes = 1;
        setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(http_port_);
        if (http_host_ == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            if (inet_pton(AF_INET, http_host_.c_str(), &addr.sin_addr) != 1) {
                std::cerr << "[FATAL] invalid host: " << http_host_ << std::endl;
                return false;
            }
        }
        if (::bind(server_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[FATAL] bind failed on port " << http_port_ << ": " << std::strerror(errno) << std::endl;
            return false;
        }
        if (::listen(server_fd_, 16) < 0) {
            std::cerr << "[FATAL] listen failed" << std::endl;
            return false;
        }
        std::cerr << "[INFO] HTTP listening on " << http_host_ << ":" << http_port_ << std::endl;
        while (g_running) {
            sockaddr_in cli{};
            socklen_t len = sizeof(cli);
            int fd = ::accept(server_fd_, reinterpret_cast<sockaddr *>(&cli), &len);
            if (fd < 0) {
                if (errno == EINTR) continue;
                if (!g_running) break;
                continue;
            }
            std::thread(&OrbbecBridge::handle_client, this, fd).detach();
        }
        return true;
    }

private:
    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    void enable_color_stream(const std::shared_ptr<ob::Config> &cfg) {
        try {
            auto profiles = pipeline_->getStreamProfileList(OB_SENSOR_COLOR);
            std::shared_ptr<ob::VideoStreamProfile> profile;
            try {
                if (color_width_ > 0 && color_height_ > 0) {
                    profile = profiles->getVideoStreamProfile(color_width_, color_height_, OB_FORMAT_RGB, fps_);
                }
            } catch (...) {}
            if (!profile) {
                try {
                    profile = profiles->getVideoStreamProfile(color_width_, color_height_, OB_FORMAT_MJPG, fps_);
                } catch (...) {}
            }
            if (!profile) profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
            cfg->enableStream(profile);
            std::cerr << "[INFO] color profile " << profile->width() << "x" << profile->height()
                      << " fps=" << profile->fps() << " fmt=" << frame_format_to_string(profile->format()) << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[WARN] enable color failed: " << e.what() << std::endl;
        }
    }

    void enable_depth_stream(const std::shared_ptr<ob::Config> &cfg) {
        try {
            auto profiles = pipeline_->getStreamProfileList(OB_SENSOR_DEPTH);
            std::shared_ptr<ob::VideoStreamProfile> profile;
            try {
                if (depth_width_ > 0 && depth_height_ > 0) {
                    profile = profiles->getVideoStreamProfile(depth_width_, depth_height_, OB_FORMAT_Y16, fps_);
                }
            } catch (...) {}
            if (!profile) profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
            cfg->enableStream(profile);
            std::cerr << "[INFO] depth profile " << profile->width() << "x" << profile->height()
                      << " fps=" << profile->fps() << " fmt=" << frame_format_to_string(profile->format()) << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "[WARN] enable depth failed: " << e.what() << std::endl;
        }
    }

    cv::Mat color_frame_to_bgr(const std::shared_ptr<ob::ColorFrame> &frame) {
        if (!frame) return cv::Mat();
        int w = static_cast<int>(frame->width());
        int h = static_cast<int>(frame->height());
        OBFormat fmt = frame->format();
        const uint8_t *data = reinterpret_cast<const uint8_t *>(frame->data());
        if (!data || w <= 0 || h <= 0) return cv::Mat();
        cv::Mat bgr;
        try {
            if (fmt == OB_FORMAT_RGB) {
                cv::Mat rgb(h, w, CV_8UC3, const_cast<uint8_t *>(data));
                cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
            } else if (fmt == OB_FORMAT_BGR) {
                bgr = cv::Mat(h, w, CV_8UC3, const_cast<uint8_t *>(data)).clone();
            } else if (fmt == OB_FORMAT_MJPG) {
                std::vector<uint8_t> buf(data, data + frame->dataSize());
                bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
            } else if (fmt == OB_FORMAT_YUYV) {
                cv::Mat yuyv(h, w, CV_8UC2, const_cast<uint8_t *>(data));
                cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUY2);
            } else if (fmt == OB_FORMAT_NV12) {
                cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV12);
            } else if (fmt == OB_FORMAT_NV21) {
                cv::Mat yuv(h + h / 2, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR_NV21);
            } else if (fmt == OB_FORMAT_Y8) {
                cv::Mat gray(h, w, CV_8UC1, const_cast<uint8_t *>(data));
                cv::cvtColor(gray, bgr, cv::COLOR_GRAY2BGR);
            }
        } catch (const std::exception &e) {
            std::cerr << "[WARN] color conversion failed: " << e.what() << std::endl;
        }
        if (!bgr.empty()) apply_flips(bgr);
        return bgr;
    }

    cv::Mat depth_frame_to_mm(const std::shared_ptr<ob::DepthFrame> &frame) {
        if (!frame) return cv::Mat();
        int w = static_cast<int>(frame->width());
        int h = static_cast<int>(frame->height());
        const uint16_t *data = reinterpret_cast<const uint16_t *>(frame->data());
        if (!data || w <= 0 || h <= 0) return cv::Mat();
        cv::Mat raw(h, w, CV_16UC1, const_cast<uint16_t *>(data));
        cv::Mat mm;
        float scale = 1.0f;
        try { scale = frame->getValueScale(); } catch (...) { scale = 1.0f; }
        if (scale > 0.0f && scale != 1.0f) {
            cv::Mat f32;
            raw.convertTo(f32, CV_32F, scale);
            f32.convertTo(mm, CV_16U);
        } else {
            mm = raw.clone();
        }
        if (!mm.empty()) apply_flips(mm);
        depth_scale_ = scale;
        return mm;
    }

    void apply_flips(cv::Mat &m) {
        if (flip_vertical_ && flip_horizontal_) cv::flip(m, m, -1);
        else if (flip_vertical_) cv::flip(m, m, 0);
        else if (flip_horizontal_) cv::flip(m, m, 1);
    }

    void camera_loop() {
        while (g_running) {
            try {
                auto frames = pipeline_->waitForFrames(1000);
                if (!frames) continue;
                cv::Mat bgr;
                cv::Mat depth_mm;
                auto c = frames->colorFrame();
                auto d = frames->depthFrame();
                if (c) bgr = color_frame_to_bgr(c);
                if (d) depth_mm = depth_frame_to_mm(d);
                auto now = now_ms();
                {
                    std::lock_guard<std::mutex> lk(mtx_);
                    if (!bgr.empty()) {
                        latest_bgr_ = bgr;
                        last_color_ms_ = now;
                        color_format_ = frame_format_to_string(c->format());
                        color_w_ = bgr.cols;
                        color_h_ = bgr.rows;
                    }
                    if (!depth_mm.empty()) {
                        latest_depth_mm_ = depth_mm;
                        last_depth_ms_ = now;
                        depth_w_ = depth_mm.cols;
                        depth_h_ = depth_mm.rows;
                    }
                    frame_count_++;
                }
                cv_.notify_all();
            } catch (const ob::Error &e) {
                last_error_ = e.getMessage();
                std::cerr << "[WARN] Orbbec wait frame error: " << e.getMessage() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            } catch (const std::exception &e) {
                last_error_ = e.what();
                std::cerr << "[WARN] camera loop error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        }
    }

    void send_all(int fd, const void *data, size_t len) {
        const char *p = reinterpret_cast<const char *>(data);
        while (len > 0) {
            ssize_t n = ::send(fd, p, len, MSG_NOSIGNAL);
            if (n <= 0) return;
            p += n;
            len -= static_cast<size_t>(n);
        }
    }

    void send_text(int fd, int code, const std::string &ctype, const std::string &body) {
        std::ostringstream h;
        h << "HTTP/1.1 " << code << " " << (code == 200 ? "OK" : code == 404 ? "Not Found" : "Error") << "\r\n"
          << "Content-Type: " << ctype << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());
        send_all(fd, body.data(), body.size());
    }

    void send_binary(int fd, int code, const std::string &ctype, const std::vector<uchar> &body) {
        std::ostringstream h;
        h << "HTTP/1.1 " << code << " OK\r\n"
          << "Content-Type: " << ctype << "\r\n"
          << "Content-Length: " << body.size() << "\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());
        if (!body.empty()) send_all(fd, body.data(), body.size());
    }

    std::string status_json() {
        std::lock_guard<std::mutex> lk(mtx_);
        int64_t now = now_ms();
        std::ostringstream os;
        os << "{"
           << "\"ok\":true,"
           << "\"component\":\"orbbec336l_bridge\","
           << "\"camera_started\":" << (camera_started_ ? "true" : "false") << ","
           << "\"frame_count\":" << frame_count_ << ","
           << "\"color_width\":" << color_w_ << ","
           << "\"color_height\":" << color_h_ << ","
           << "\"depth_width\":" << depth_w_ << ","
           << "\"depth_height\":" << depth_h_ << ","
           << "\"color_format\":\"" << json_escape(color_format_) << "\","
           << "\"depth_scale\":" << std::fixed << std::setprecision(6) << depth_scale_ << ","
           << "\"last_color_age_ms\":" << (last_color_ms_ > 0 ? now - last_color_ms_ : -1) << ","
           << "\"last_depth_age_ms\":" << (last_depth_ms_ > 0 ? now - last_depth_ms_ : -1) << ","
           << "\"uptime_ms\":" << (start_ms_ > 0 ? now - start_ms_ : 0) << ","
           << "\"time\":\"" << json_escape(now_string()) << "\","
           << "\"last_error\":\"" << json_escape(last_error_) << "\""
           << "}";
        return os.str();
    }

    std::string depth_meta_json() {
        std::lock_guard<std::mutex> lk(mtx_);
        std::ostringstream os;
        os << "{"
           << "\"ok\":" << (!latest_depth_mm_.empty() ? "true" : "false") << ","
           << "\"width\":" << depth_w_ << ","
           << "\"height\":" << depth_h_ << ","
           << "\"encoding\":\"16UC1\","
           << "\"unit\":\"mm\","
           << "\"scale\":" << std::fixed << std::setprecision(6) << depth_scale_ << ","
           << "\"last_depth_ms\":" << last_depth_ms_
           << "}";
        return os.str();
    }

    void send_snapshot(int fd) {
        cv::Mat img;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (latest_bgr_.empty()) cv_.wait_for(lk, std::chrono::milliseconds(1500));
            if (!latest_bgr_.empty()) img = latest_bgr_.clone();
        }
        std::vector<uchar> jpg;
        if (!encode_jpeg(img, jpeg_quality_, jpg)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no color frame\"}");
            return;
        }
        send_binary(fd, 200, "image/jpeg", jpg);
    }

    void send_depth_png(int fd) {
        cv::Mat depth;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (latest_depth_mm_.empty()) cv_.wait_for(lk, std::chrono::milliseconds(1500));
            if (!latest_depth_mm_.empty()) depth = latest_depth_mm_.clone();
        }
        std::vector<uchar> png;
        if (!encode_png16(depth, png)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no depth frame\"}");
            return;
        }
        send_binary(fd, 200, "image/png", png);
    }

    void send_depth_vis(int fd) {
        cv::Mat depth;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            if (latest_depth_mm_.empty()) cv_.wait_for(lk, std::chrono::milliseconds(1500));
            if (!latest_depth_mm_.empty()) depth = latest_depth_mm_.clone();
        }
        std::vector<uchar> jpg;
        if (!encode_jpeg(depth_to_vis(depth), jpeg_quality_, jpg)) {
            send_text(fd, 503, "application/json", "{\"ok\":false,\"error\":\"no depth frame\"}");
            return;
        }
        send_binary(fd, 200, "image/jpeg", jpg);
    }

    void send_mjpeg(int fd) {
        std::string boundary = "visionops-orbbec336l";
        std::ostringstream h;
        h << "HTTP/1.1 200 OK\r\n"
          << "Content-Type: multipart/x-mixed-replace; boundary=" << boundary << "\r\n"
          << "Cache-Control: no-cache\r\n"
          << "Connection: close\r\n\r\n";
        auto hs = h.str();
        send_all(fd, hs.data(), hs.size());

        int delay_ms = std::max(10, 1000 / std::max(1, mjpeg_fps_));
        while (g_running) {
            cv::Mat img;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                if (latest_bgr_.empty()) cv_.wait_for(lk, std::chrono::milliseconds(1000));
                if (!latest_bgr_.empty()) img = latest_bgr_.clone();
            }
            std::vector<uchar> jpg;
            if (encode_jpeg(img, jpeg_quality_, jpg)) {
                std::ostringstream ph;
                ph << "--" << boundary << "\r\n"
                   << "Content-Type: image/jpeg\r\n"
                   << "Content-Length: " << jpg.size() << "\r\n\r\n";
                auto ps = ph.str();
                if (::send(fd, ps.data(), ps.size(), MSG_NOSIGNAL) <= 0) break;
                if (::send(fd, jpg.data(), jpg.size(), MSG_NOSIGNAL) <= 0) break;
                std::string tail = "\r\n";
                if (::send(fd, tail.data(), tail.size(), MSG_NOSIGNAL) <= 0) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    void handle_client(int fd) {
        char buf[4096] = {0};
        ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) { ::close(fd); return; }
        std::string req(buf, static_cast<size_t>(n));
        std::istringstream is(req);
        std::string method, path, proto;
        is >> method >> path >> proto;

        if (path == "/health" || path == "/stream/status") {
            send_text(fd, 200, "application/json", status_json());
        } else if (path == "/stream/snapshot.jpg" || path == "/snapshot.jpg") {
            send_snapshot(fd);
        } else if (path == "/stream/depth.png") {
            send_depth_png(fd);
        } else if (path == "/stream/depth_vis.jpg" || path == "/stream/depth.jpg") {
            send_depth_vis(fd);
        } else if (path == "/stream/depth_meta") {
            send_text(fd, 200, "application/json", depth_meta_json());
        } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
            send_mjpeg(fd);
        } else if ((method == "POST" || method == "GET") && (path == "/stream/start" || path == "/stream/stop")) {
            send_text(fd, 200, "application/json", "{\"ok\":true,\"note\":\"camera bridge keeps streaming\"}");
        } else {
            send_text(fd, 404, "application/json", "{\"ok\":false,\"error\":\"not found\"}");
        }
        ::close(fd);
    }

private:
    std::string http_host_;
    int http_port_;
    int color_width_;
    int color_height_;
    int depth_width_;
    int depth_height_;
    int fps_;
    int jpeg_quality_;
    int mjpeg_fps_;
    bool flip_vertical_;
    bool flip_horizontal_;
    std::string serial_;

    std::shared_ptr<ob::Context> ctx_;
    std::shared_ptr<ob::Pipeline> pipeline_;
    std::thread camera_thread_;
    std::atomic<bool> camera_started_{false};

    std::mutex mtx_;
    std::condition_variable cv_;
    cv::Mat latest_bgr_;
    cv::Mat latest_depth_mm_;
    int64_t last_color_ms_ = 0;
    int64_t last_depth_ms_ = 0;
    int64_t start_ms_ = 0;
    uint64_t frame_count_ = 0;
    int color_w_ = 0;
    int color_h_ = 0;
    int depth_w_ = 0;
    int depth_h_ = 0;
    float depth_scale_ = 1.0f;
    std::string color_format_ = "";
    std::string last_error_ = "";

    int server_fd_ = -1;
};

void signal_handler(int) {
    g_running = false;
}

} // namespace

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    std::cerr << "[INFO] VisionOps Orbbec Gemini 336L SDK bridge starting" << std::endl;
    OrbbecBridge bridge;
    if (!bridge.start_camera()) {
        std::cerr << "[FATAL] camera start failed" << std::endl;
        return 2;
    }
    bool ok = bridge.start_http();
    bridge.stop_camera();
    return ok ? 0 : 1;
}
