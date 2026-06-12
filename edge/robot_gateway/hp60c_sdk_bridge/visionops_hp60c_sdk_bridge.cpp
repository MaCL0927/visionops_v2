// VisionOps HP60C Angstrong SDK bridge
// Provides the same HTTP surface as the previous ROS1 bridge:
//   GET /health
//   GET /stream/snapshot.jpg
//   GET /stream/depth.png       (16-bit PNG depth, millimeters if SDK depthImg is in mm)
//   GET /stream.mjpeg, /stream/mjpeg, /stream.mjpg
//   GET /stream/status
//   POST /stream/start, POST /stream/stop  (no-op; camera stays running)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <ctime>
#include <iostream>
#include <list>
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

#include "as_camera_sdk_api.h"
#include "as_camera_sdk_def.h"

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

static ssize_t send_all(int fd, const void *buf, size_t len) {
    const char *p = static_cast<const char *>(buf);
    size_t left = len;
    while (left > 0) {
        ssize_t n = ::send(fd, p, left, MSG_NOSIGNAL);
        if (n <= 0) return n;
        p += n;
        left -= static_cast<size_t>(n);
    }
    return static_cast<ssize_t>(len);
}

static bool send_string(int fd, const std::string &s) {
    return send_all(fd, s.data(), s.size()) == static_cast<ssize_t>(s.size());
}

struct FrameSnapshot {
    std::vector<uchar> jpeg;
    int width = 0;
    int height = 0;
    uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point ts;
};

struct DepthSnapshot {
    std::vector<uchar> png;
    int width = 0;
    int height = 0;
    uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point ts;
};

class Hp60cSdkBridge {
public:
    Hp60cSdkBridge()
        : config_path_(getenv_str("VISIONOPS_HP60C_CONFIG", "")),
          bind_host_(getenv_str("VISIONOPS_HP60C_HTTP_HOST", "127.0.0.1")),
          http_port_(getenv_int("VISIONOPS_HP60C_HTTP_PORT", 18181)),
          jpeg_quality_(getenv_int("VISIONOPS_HP60C_JPEG_QUALITY", 85)),
          mjpeg_fps_(getenv_int("VISIONOPS_HP60C_MJPEG_FPS", 10)),
          flip_vertical_(getenv_bool("VISIONOPS_HP60C_FLIP_VERTICAL", true)),
          prefer_mjpeg_(getenv_bool("VISIONOPS_HP60C_PREFER_MJPEG", true)),
          rgb_order_(getenv_str("VISIONOPS_HP60C_RGB_ORDER", "bgr")) {}

    ~Hp60cSdkBridge() { stop(); }

    bool start() {
        int ret = AS_SDK_Init();
        if (ret != 0) {
            last_error_ = "AS_SDK_Init failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
            return false;
        }
        sdk_inited_ = true;

        char sdk_ver[128] = {0};
        if (AS_SDK_GetSwVersion(sdk_ver, sizeof(sdk_ver)) == 0) {
            sdk_version_ = sdk_ver;
        }
        std::cerr << "[INFO] Angstrong SDK version: " << sdk_version_ << std::endl;
        std::cerr << "[INFO] config: " << config_path_ << std::endl;

        AS_LISTENER_CALLBACK_S cb{};
        cb.onAttached = &Hp60cSdkBridge::on_attached;
        cb.onDetached = &Hp60cSdkBridge::on_detached;
        cb.privateData = this;

        ret = AS_SDK_StartListener(cb, AS_LISTENNER_TYPE_USB, true);
        if (ret != 0) {
            last_error_ = "AS_SDK_StartListener USB failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
            return false;
        }

        http_thread_ = std::thread(&Hp60cSdkBridge::http_loop, this);
        return true;
    }

    void stop() {
        if (stopped_.exchange(true)) return;
        g_running = false;
        if (listen_fd_ >= 0) {
            ::shutdown(listen_fd_, SHUT_RDWR);
            ::close(listen_fd_);
            listen_fd_ = -1;
        }
        if (http_thread_.joinable()) http_thread_.join();

        std::lock_guard<std::mutex> lk(cam_mutex_);
        for (AS_CAM_PTR cam : cameras_) {
            AS_SDK_StopStream(cam);
            AS_SDK_CloseCamera(cam);
            AS_SDK_DestoryCamHandle(cam);
        }
        cameras_.clear();
        if (sdk_inited_) {
            AS_SDK_StopListener();
            AS_SDK_Deinit();
            sdk_inited_ = false;
        }
    }

private:
    static void on_attached(AS_CAM_ATTR_S *attr, void *privateData) {
        auto *self = static_cast<Hp60cSdkBridge *>(privateData);
        if (self) self->handle_attached(attr);
    }

    static void on_detached(AS_CAM_ATTR_S *attr, void *privateData) {
        auto *self = static_cast<Hp60cSdkBridge *>(privateData);
        if (self) self->handle_detached(attr);
    }

    static void on_frame(AS_CAM_PTR pCamera, const AS_SDK_Data_s *data, void *privateData) {
        auto *self = static_cast<Hp60cSdkBridge *>(privateData);
        if (self && data) self->handle_frame(pCamera, data);
    }

    void handle_attached(AS_CAM_ATTR_S *attr) {
        if (!attr) return;
        std::lock_guard<std::mutex> lk(cam_mutex_);
        std::cerr << "[INFO] camera attached" << std::endl;

        AS_CAM_PTR cam = nullptr;
        int ret = AS_SDK_CreateCamHandle(cam, attr);
        if (ret != 0 || !cam) {
            last_error_ = "AS_SDK_CreateCamHandle failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
            return;
        }

        AS_SDK_CAM_MODEL_E model = AS_SDK_CAM_MODEL_UNKNOWN;
        AS_SDK_GetCameraModel(cam, model);
        cam_model_ = static_cast<int>(model);
        std::cerr << "[INFO] camera model: " << cam_model_ << std::endl;

        ret = AS_SDK_OpenCamera(cam, config_path_.c_str());
        if (ret != 0) {
            last_error_ = "AS_SDK_OpenCamera failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
            AS_SDK_DestoryCamHandle(cam);
            return;
        }

        char sn[128] = {0};
        if (AS_SDK_GetSerialNumber(cam, sn, sizeof(sn)) == 0) {
            serial_ = sn;
        }

        AS_CAM_Stream_Cb_s stream_cb{};
        stream_cb.callback = &Hp60cSdkBridge::on_frame;
        stream_cb.privateData = this;
        ret = AS_SDK_RegisterStreamCallback(cam, &stream_cb);
        if (ret != 0) {
            last_error_ = "AS_SDK_RegisterStreamCallback failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
        }

        ret = AS_SDK_StartStream(cam, 0);
        if (ret != 0) {
            last_error_ = "AS_SDK_StartStream failed: " + std::to_string(ret);
            std::cerr << "[ERROR] " << last_error_ << std::endl;
            AS_SDK_CloseCamera(cam);
            AS_SDK_DestoryCamHandle(cam);
            return;
        }

        cameras_.push_back(cam);
        camera_opened_ = true;
        last_error_.clear();
        std::cerr << "[OK] HP60C camera opened, serial=" << serial_ << std::endl;
    }

    void handle_detached(AS_CAM_ATTR_S * /*attr*/) {
        std::lock_guard<std::mutex> lk(cam_mutex_);
        std::cerr << "[WARN] camera detached" << std::endl;
        for (AS_CAM_PTR cam : cameras_) {
            AS_SDK_StopStream(cam);
            AS_SDK_CloseCamera(cam);
            AS_SDK_DestoryCamHandle(cam);
        }
        cameras_.clear();
        camera_opened_ = false;
        last_error_ = "camera detached";
    }

    void handle_frame(AS_CAM_PTR /*pCamera*/, const AS_SDK_Data_s *data) {
        cv::Mat bgr;
        int src_w = 0, src_h = 0;

        // Keep the latest depth frame as a 16-bit PNG.
        // For HP60C, SDK demo saves depth as raw .yuv, but the callback depthImg
        // buffer is typically a single-channel depth plane. If it is 16-bit, keep
        // values unchanged so downstream Python can read it with cv2.IMREAD_UNCHANGED.
        if (data->depthImg.size > 0 && data->depthImg.data && data->depthImg.width > 0 && data->depthImg.height > 0) {
            const int dw = static_cast<int>(data->depthImg.width);
            const int dh = static_cast<int>(data->depthImg.height);
            const size_t expected_u16 = static_cast<size_t>(dw) * static_cast<size_t>(dh) * 2u;
            const size_t expected_u8 = static_cast<size_t>(dw) * static_cast<size_t>(dh);
            cv::Mat depth_for_png;

            if (data->depthImg.size >= expected_u16) {
                cv::Mat raw16(dh, dw, CV_16UC1, data->depthImg.data);
                depth_for_png = raw16.clone();
            } else if (data->depthImg.size >= expected_u8) {
                cv::Mat raw8(dh, dw, CV_8UC1, data->depthImg.data);
                // Preserve shape and expose as 16-bit PNG for a stable HTTP API.
                raw8.convertTo(depth_for_png, CV_16UC1);
            }

            if (!depth_for_png.empty()) {
                std::vector<uchar> depth_png;
                std::vector<int> png_params = {cv::IMWRITE_PNG_COMPRESSION, 1};
                if (cv::imencode(".png", depth_for_png, depth_png, png_params)) {
                    std::lock_guard<std::mutex> lk(depth_mutex_);
                    latest_depth_.png = std::move(depth_png);
                    latest_depth_.width = depth_for_png.cols;
                    latest_depth_.height = depth_for_png.rows;
                    latest_depth_.frame_id++;
                    latest_depth_.ts = std::chrono::steady_clock::now();
                    depth_frame_count_++;
                }
            }
        }

        if (prefer_mjpeg_ && data->mjpegImg.size > 0 && data->mjpegImg.data) {
            std::vector<uchar> bytes(
                static_cast<uchar *>(data->mjpegImg.data),
                static_cast<uchar *>(data->mjpegImg.data) + data->mjpegImg.size);
            bgr = cv::imdecode(bytes, cv::IMREAD_COLOR);
            src_w = static_cast<int>(data->mjpegImg.width);
            src_h = static_cast<int>(data->mjpegImg.height);
            if (!bgr.empty() && flip_vertical_) cv::flip(bgr, bgr, 0);
        }

        if (bgr.empty() && data->rgbImg.size > 0 && data->rgbImg.data) {
            src_w = static_cast<int>(data->rgbImg.width);
            src_h = static_cast<int>(data->rgbImg.height);
            if (data->rgbImg.size >= data->rgbImg.width * data->rgbImg.height * 3) {
                cv::Mat raw(src_h, src_w, CV_8UC3, data->rgbImg.data);
                if (rgb_order_ == "rgb") {
                    cv::cvtColor(raw, bgr, cv::COLOR_RGB2BGR);
                } else {
                    bgr = raw.clone();
                }
            }
        }

        if (bgr.empty() && data->yuyvImg.size > 0 && data->yuyvImg.data) {
            src_w = static_cast<int>(data->yuyvImg.width);
            src_h = static_cast<int>(data->yuyvImg.height);
            cv::Mat yuyv(src_h, src_w, CV_8UC2, data->yuyvImg.data);
            cv::cvtColor(yuyv, bgr, cv::COLOR_YUV2BGR_YUYV);
        }

        if (bgr.empty()) {
            return;
        }

        std::vector<uchar> jpg;
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, std::max(10, std::min(100, jpeg_quality_))};
        if (!cv::imencode(".jpg", bgr, jpg, params)) {
            return;
        }

        {
            std::lock_guard<std::mutex> lk(frame_mutex_);
            latest_.jpeg = std::move(jpg);
            latest_.width = bgr.cols > 0 ? bgr.cols : src_w;
            latest_.height = bgr.rows > 0 ? bgr.rows : src_h;
            latest_.frame_id++;
            latest_.ts = std::chrono::steady_clock::now();
            frame_count_++;
        }
        frame_cv_.notify_all();
    }

    bool wait_frame(FrameSnapshot &out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(frame_mutex_);
        if (latest_.jpeg.empty()) {
            frame_cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&] { return !latest_.jpeg.empty() || !g_running; });
        }
        if (latest_.jpeg.empty()) return false;
        out = latest_;
        return true;
    }

    bool wait_depth(DepthSnapshot &out, int timeout_ms) {
        std::unique_lock<std::mutex> lk(depth_mutex_);
        if (latest_depth_.png.empty()) {
            // Depth and RGB arrive in the same SDK callback. A short wait is enough.
            lk.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(std::max(1, timeout_ms)));
            lk.lock();
        }
        if (latest_depth_.png.empty()) return false;
        out = latest_depth_;
        return true;
    }

    double latest_depth_age_ms() {
        std::lock_guard<std::mutex> lk(depth_mutex_);
        if (latest_depth_.png.empty()) return -1.0;
        auto d = std::chrono::steady_clock::now() - latest_depth_.ts;
        return std::chrono::duration<double, std::milli>(d).count();
    }

    double latest_age_ms() {
        std::lock_guard<std::mutex> lk(frame_mutex_);
        if (latest_.jpeg.empty()) return -1.0;
        auto d = std::chrono::steady_clock::now() - latest_.ts;
        return std::chrono::duration<double, std::milli>(d).count();
    }

    std::string health_json() {
        FrameSnapshot snap;
        bool has = wait_frame(snap, 1);
        std::ostringstream os;
        os << "{\n";
        os << "  \"ok\": true,\n";
        os << "  \"backend\": \"hp60c-sdk-cpp-bridge\",\n";
        os << "  \"running\": true,\n";
        os << "  \"camera_opened\": " << (camera_opened_ ? "true" : "false") << ",\n";
        os << "  \"snapshot_available\": " << (has ? "true" : "false") << ",\n";
        os << "  \"has_frame\": " << (has ? "true" : "false") << ",\n";
        os << "  \"latest_snapshot_age_ms\": " << latest_age_ms() << ",\n";
        os << "  \"width\": " << (has ? snap.width : 0) << ",\n";
        os << "  \"height\": " << (has ? snap.height : 0) << ",\n";
        os << "  \"frame_count\": " << frame_count_.load() << ",\n";
        os << "  \"depth_available\": " << (depth_frame_count_.load() > 0 ? "true" : "false") << ",\n";
        os << "  \"depth_frame_count\": " << depth_frame_count_.load() << ",\n";
        os << "  \"latest_depth_age_ms\": " << latest_depth_age_ms() << ",\n";
        os << "  \"camera_model\": " << cam_model_ << ",\n";
        os << "  \"serial\": \"" << json_escape(serial_) << "\",\n";
        os << "  \"sdk_version\": \"" << json_escape(sdk_version_) << "\",\n";
        os << "  \"config\": \"" << json_escape(config_path_) << "\",\n";
        os << "  \"error\": \"" << json_escape(last_error_) << "\",\n";
        os << "  \"time\": \"" << now_string() << "\"\n";
        os << "}\n";
        return os.str();
    }

    void http_loop() {
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) {
            std::cerr << "[ERROR] socket failed" << std::endl;
            return;
        }
        int yes = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<uint16_t>(http_port_));
        if (bind_host_ == "0.0.0.0") {
            addr.sin_addr.s_addr = INADDR_ANY;
        } else {
            inet_pton(AF_INET, bind_host_.c_str(), &addr.sin_addr);
        }

        if (::bind(listen_fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            std::cerr << "[ERROR] bind failed on " << bind_host_ << ":" << http_port_ << " errno=" << errno << std::endl;
            return;
        }
        if (::listen(listen_fd_, 16) < 0) {
            std::cerr << "[ERROR] listen failed" << std::endl;
            return;
        }
        std::cerr << "[OK] HTTP bridge listening on http://" << bind_host_ << ":" << http_port_ << std::endl;

        while (g_running) {
            sockaddr_in cli{};
            socklen_t clen = sizeof(cli);
            int fd = ::accept(listen_fd_, reinterpret_cast<sockaddr *>(&cli), &clen);
            if (fd < 0) {
                if (!g_running) break;
                continue;
            }
            std::thread(&Hp60cSdkBridge::handle_http_client, this, fd).detach();
        }
    }

    void handle_http_client(int fd) {
        char buf[4096] = {0};
        ssize_t n = ::recv(fd, buf, sizeof(buf) - 1, 0);
        if (n <= 0) { ::close(fd); return; }
        std::string req(buf, static_cast<size_t>(n));
        std::istringstream is(req);
        std::string method, path, version;
        is >> method >> path >> version;

        if (path == "/health" || path == "/stream/status") {
            std::string body = health_json();
            std::ostringstream h;
            h << "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " << body.size()
              << "\r\nConnection: close\r\n\r\n";
            send_string(fd, h.str());
            send_string(fd, body);
        } else if (path == "/stream/snapshot.jpg" || path == "/snapshot.jpg") {
            FrameSnapshot snap;
            if (!wait_frame(snap, 2000)) {
                std::string body = "no frame\n";
                std::ostringstream h;
                h << "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: " << body.size()
                  << "\r\nConnection: close\r\n\r\n";
                send_string(fd, h.str());
                send_string(fd, body);
            } else {
                std::ostringstream h;
                h << "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: " << snap.jpeg.size()
                  << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
                send_string(fd, h.str());
                send_all(fd, snap.jpeg.data(), snap.jpeg.size());
            }
        } else if (path == "/stream/depth.png" || path == "/depth.png") {
            DepthSnapshot depth;
            if (!wait_depth(depth, 500)) {
                std::string body = "no depth\n";
                std::ostringstream h;
                h << "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: " << body.size()
                  << "\r\nConnection: close\r\n\r\n";
                send_string(fd, h.str());
                send_string(fd, body);
            } else {
                std::ostringstream h;
                h << "HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: " << depth.png.size()
                  << "\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
                send_string(fd, h.str());
                send_all(fd, depth.png.data(), depth.png.size());
            }
        } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
            handle_mjpeg(fd);
            return;
        } else if (path.find("/stream/start") == 0 || path.find("/stream/stop") == 0) {
            std::string body = "{\"ok\":true,\"message\":\"sdk bridge keeps camera streaming\"}\n";
            std::ostringstream h;
            h << "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " << body.size()
              << "\r\nConnection: close\r\n\r\n";
            send_string(fd, h.str());
            send_string(fd, body);
        } else {
            std::string body = "not found\n";
            std::ostringstream h;
            h << "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: " << body.size()
              << "\r\nConnection: close\r\n\r\n";
            send_string(fd, h.str());
            send_string(fd, body);
        }
        ::close(fd);
    }

    void handle_mjpeg(int fd) {
        std::string header =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache, no-store, must-revalidate\r\n"
            "Pragma: no-cache\r\n"
            "Connection: close\r\n\r\n";
        if (!send_string(fd, header)) { ::close(fd); return; }
        int delay_ms = std::max(1, 1000 / std::max(1, mjpeg_fps_));
        uint64_t last_id = 0;
        while (g_running) {
            FrameSnapshot snap;
            if (!wait_frame(snap, 2000)) break;
            if (snap.frame_id == last_id) {
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                continue;
            }
            last_id = snap.frame_id;
            std::ostringstream part;
            part << "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " << snap.jpeg.size() << "\r\n\r\n";
            if (!send_string(fd, part.str())) break;
            if (send_all(fd, snap.jpeg.data(), snap.jpeg.size()) <= 0) break;
            if (!send_string(fd, "\r\n")) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
        ::close(fd);
    }

private:
    std::string config_path_;
    std::string bind_host_;
    int http_port_ = 18181;
    int jpeg_quality_ = 85;
    int mjpeg_fps_ = 10;
    bool flip_vertical_ = true;
    bool prefer_mjpeg_ = true;
    std::string rgb_order_ = "bgr";

    std::atomic<bool> stopped_{false};
    std::atomic<bool> camera_opened_{false};
    std::atomic<uint64_t> frame_count_{0};
    bool sdk_inited_ = false;
    int listen_fd_ = -1;
    std::thread http_thread_;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    FrameSnapshot latest_;

    std::mutex depth_mutex_;
    DepthSnapshot latest_depth_;
    std::atomic<uint64_t> depth_frame_count_{0};

    std::mutex cam_mutex_;
    std::list<AS_CAM_PTR> cameras_;
    int cam_model_ = 0;
    std::string serial_;
    std::string sdk_version_;
    std::string last_error_;
};

void signal_handler(int) {
    g_running = false;
}

} // namespace

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    Hp60cSdkBridge bridge;
    if (!bridge.start()) {
        return 1;
    }
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    bridge.stop();
    return 0;
}
