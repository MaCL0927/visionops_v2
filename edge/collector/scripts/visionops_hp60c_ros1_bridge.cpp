#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::mutex g_frame_mutex;
cv::Mat g_latest_bgr;
ros::Time g_latest_stamp;

std::mutex g_depth_mutex;
cv::Mat g_latest_depth16;              // 16UC1, unit: mm
ros::Time g_latest_depth_stamp;
std::string g_latest_depth_encoding;

std::atomic<bool> g_running{true};
std::atomic<bool> g_exit{false};
std::string g_rgb_topic = "/ascamera_hp60c/rgb0/image";
std::string g_depth_topic = "/ascamera_hp60c/depth0/image_raw";
int g_jpeg_quality = 80;
int g_depth_png_compression = 1;
int g_http_port = 18181;

std::string json_escape(const std::string& s) {
  std::ostringstream oss;
  for (char c : s) {
    switch (c) {
      case '"': oss << "\\\""; break;
      case '\\': oss << "\\\\"; break;
      case '\b': oss << "\\b"; break;
      case '\f': oss << "\\f"; break;
      case '\n': oss << "\\n"; break;
      case '\r': oss << "\\r"; break;
      case '\t': oss << "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          oss << "\\u" << std::hex << std::uppercase << static_cast<int>(c);
        } else {
          oss << c;
        }
    }
  }
  return oss.str();
}

std::string now_iso() {
  auto now = std::chrono::system_clock::now();
  std::time_t t = std::chrono::system_clock::to_time_t(now);
  char buf[64];
  std::strftime(buf, sizeof(buf), "%F %T", std::localtime(&t));
  return std::string(buf);
}

void send_all(int fd, const std::string& s) {
  const char* p = s.data();
  size_t n = s.size();
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w <= 0) return;
    p += w;
    n -= static_cast<size_t>(w);
  }
}

void send_all_bytes(int fd, const std::vector<uchar>& data) {
  const char* p = reinterpret_cast<const char*>(data.data());
  size_t n = data.size();
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w <= 0) return;
    p += w;
    n -= static_cast<size_t>(w);
  }
}

bool get_latest_rgb_meta(double* age_ms = nullptr, int* width = nullptr, int* height = nullptr) {
  ros::Time stamp;
  int w = 0, h = 0;
  {
    std::lock_guard<std::mutex> lk(g_frame_mutex);
    if (g_latest_bgr.empty()) return false;
    w = g_latest_bgr.cols;
    h = g_latest_bgr.rows;
    stamp = g_latest_stamp;
  }
  if (width) *width = w;
  if (height) *height = h;
  if (age_ms) {
    ros::Duration age = ros::Time::now() - stamp;
    *age_ms = age.toSec() * 1000.0;
  }
  return true;
}

bool encode_latest(std::vector<uchar>& out, double* age_ms = nullptr, int* width = nullptr, int* height = nullptr) {
  cv::Mat frame;
  ros::Time stamp;
  {
    std::lock_guard<std::mutex> lk(g_frame_mutex);
    if (g_latest_bgr.empty()) return false;
    frame = g_latest_bgr.clone();
    stamp = g_latest_stamp;
  }
  if (width) *width = frame.cols;
  if (height) *height = frame.rows;
  if (age_ms) {
    ros::Duration age = ros::Time::now() - stamp;
    *age_ms = age.toSec() * 1000.0;
  }
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, g_jpeg_quality};
  return cv::imencode(".jpg", frame, out, params);
}

void image_cb(const sensor_msgs::ImageConstPtr& msg) {
  try {
    cv_bridge::CvImageConstPtr cv_ptr;
    if (msg->encoding == "bgr8") {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } else if (msg->encoding == "rgb8") {
      cv_ptr = cv_bridge::toCvShare(msg, "rgb8");
    } else {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    }
    cv::Mat bgr;
    if (msg->encoding == "rgb8") cv::cvtColor(cv_ptr->image, bgr, cv::COLOR_RGB2BGR);
    else bgr = cv_ptr->image.clone();
    {
      std::lock_guard<std::mutex> lk(g_frame_mutex);
      g_latest_bgr = bgr;
      g_latest_stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
    }
  } catch (const std::exception& e) {
    ROS_WARN_THROTTLE(2.0, "HP60C bridge RGB conversion failed: %s", e.what());
  }
}

void depth_cb(const sensor_msgs::ImageConstPtr& msg) {
  try {
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg);
    const cv::Mat& src = cv_ptr->image;
    cv::Mat depth16;

    if (src.type() == CV_16UC1) {
      depth16 = src.clone();
    } else if (src.type() == CV_32FC1) {
      // Convert meter float depth to millimeter uint16 depth.
      depth16.create(src.rows, src.cols, CV_16UC1);
      for (int y = 0; y < src.rows; ++y) {
        const float* sp = src.ptr<float>(y);
        unsigned short* dp = depth16.ptr<unsigned short>(y);
        for (int x = 0; x < src.cols; ++x) {
          float v = sp[x];
          if (!std::isfinite(v) || v <= 0.0f) {
            dp[x] = 0;
          } else {
            float mm = v * 1000.0f;
            if (mm < 0.0f) mm = 0.0f;
            if (mm > 65535.0f) mm = 65535.0f;
            dp[x] = static_cast<unsigned short>(std::lround(mm));
          }
        }
      }
    } else {
      ROS_WARN_THROTTLE(2.0, "HP60C bridge unsupported depth encoding=%s type=%d", msg->encoding.c_str(), src.type());
      return;
    }

    {
      std::lock_guard<std::mutex> lk(g_depth_mutex);
      g_latest_depth16 = depth16;
      g_latest_depth_stamp = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
      g_latest_depth_encoding = msg->encoding;
    }
  } catch (const std::exception& e) {
    ROS_WARN_THROTTLE(2.0, "HP60C bridge depth conversion failed: %s", e.what());
  }
}

bool clone_latest_depth(cv::Mat& depth16, ros::Time* stamp = nullptr, std::string* encoding = nullptr) {
  std::lock_guard<std::mutex> lk(g_depth_mutex);
  if (g_latest_depth16.empty()) return false;
  depth16 = g_latest_depth16.clone();
  if (stamp) *stamp = g_latest_depth_stamp;
  if (encoding) *encoding = g_latest_depth_encoding;
  return true;
}

bool depth_stats(const cv::Mat& depth16, int* valid_count, unsigned short* min_v, unsigned short* max_v) {
  if (depth16.empty() || depth16.type() != CV_16UC1) return false;
  int count = 0;
  unsigned short mn = std::numeric_limits<unsigned short>::max();
  unsigned short mx = 0;
  for (int y = 0; y < depth16.rows; ++y) {
    const unsigned short* p = depth16.ptr<unsigned short>(y);
    for (int x = 0; x < depth16.cols; ++x) {
      unsigned short v = p[x];
      if (v == 0) continue;
      ++count;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
  }
  if (valid_count) *valid_count = count;
  if (min_v) *min_v = (count > 0 ? mn : 0);
  if (max_v) *max_v = (count > 0 ? mx : 0);
  return true;
}

bool encode_latest_depth_png(std::vector<uchar>& out, double* age_ms = nullptr, int* width = nullptr, int* height = nullptr,
                             std::string* encoding = nullptr, int* valid_count = nullptr,
                             unsigned short* min_v = nullptr, unsigned short* max_v = nullptr) {
  cv::Mat depth16;
  ros::Time stamp;
  std::string enc;
  if (!clone_latest_depth(depth16, &stamp, &enc)) return false;
  if (width) *width = depth16.cols;
  if (height) *height = depth16.rows;
  if (encoding) *encoding = enc;
  if (age_ms) {
    ros::Duration age = ros::Time::now() - stamp;
    *age_ms = age.toSec() * 1000.0;
  }
  depth_stats(depth16, valid_count, min_v, max_v);
  std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, g_depth_png_compression};
  return cv::imencode(".png", depth16, out, params);
}

bool encode_latest_depth_vis(std::vector<uchar>& out, double* age_ms = nullptr, int* width = nullptr, int* height = nullptr) {
  cv::Mat depth16;
  ros::Time stamp;
  std::string enc;
  if (!clone_latest_depth(depth16, &stamp, &enc)) return false;
  int valid = 0;
  unsigned short mn = 0, mx = 0;
  depth_stats(depth16, &valid, &mn, &mx);
  if (width) *width = depth16.cols;
  if (height) *height = depth16.rows;
  if (age_ms) {
    ros::Duration age = ros::Time::now() - stamp;
    *age_ms = age.toSec() * 1000.0;
  }
  cv::Mat gray(depth16.rows, depth16.cols, CV_8UC1, cv::Scalar(0));
  if (valid > 0 && mx > mn) {
    const double scale = 255.0 / static_cast<double>(mx - mn);
    for (int y = 0; y < depth16.rows; ++y) {
      const unsigned short* sp = depth16.ptr<unsigned short>(y);
      unsigned char* gp = gray.ptr<unsigned char>(y);
      for (int x = 0; x < depth16.cols; ++x) {
        unsigned short v = sp[x];
        if (v == 0) {
          gp[x] = 0;
        } else {
          int iv = static_cast<int>((static_cast<double>(v - mn) * scale) + 0.5);
          gp[x] = static_cast<unsigned char>(std::max(0, std::min(255, iv)));
        }
      }
    }
  }
  cv::Mat color;
  cv::applyColorMap(gray, color, cv::COLORMAP_JET);
  std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, g_jpeg_quality};
  return cv::imencode(".jpg", color, out, params);
}

void http_json(int fd, const std::string& body, int code=200, const std::string& status="OK") {
  std::ostringstream oss;
  oss << "HTTP/1.1 " << code << " " << status << "\r\n"
      << "Content-Type: application/json; charset=utf-8\r\n"
      << "Cache-Control: no-store\r\n"
      << "Content-Length: " << body.size() << "\r\n\r\n" << body;
  send_all(fd, oss.str());
}

std::string status_json() {
  double rgb_age = -1.0;
  int rgb_w = 0, rgb_h = 0;
  bool rgb_ok = get_latest_rgb_meta(&rgb_age, &rgb_w, &rgb_h);

  cv::Mat depth16;
  ros::Time depth_stamp;
  std::string depth_enc;
  bool depth_ok = clone_latest_depth(depth16, &depth_stamp, &depth_enc);
  double depth_age = -1.0;
  int depth_w = 0, depth_h = 0, depth_valid = 0;
  unsigned short depth_min = 0, depth_max = 0;
  if (depth_ok) {
    depth_w = depth16.cols;
    depth_h = depth16.rows;
    ros::Duration age = ros::Time::now() - depth_stamp;
    depth_age = age.toSec() * 1000.0;
    depth_stats(depth16, &depth_valid, &depth_min, &depth_max);
  }

  std::ostringstream oss;
  oss << "{"
      << "\"ok\":true,"
      << "\"backend\":\"hp60c-ros1-cpp-bridge\","
      << "\"running\":" << (g_running.load() ? "true" : "false") << ","
      << "\"topic\":\"" << json_escape(g_rgb_topic) << "\","
      << "\"rgb_topic\":\"" << json_escape(g_rgb_topic) << "\","
      << "\"depth_topic\":\"" << json_escape(g_depth_topic) << "\","
      << "\"snapshot_available\":" << ((rgb_ok && g_running.load()) ? "true" : "false") << ","
      << "\"has_frame\":" << (rgb_ok ? "true" : "false") << ","
      << "\"latest_snapshot_age_ms\":" << (rgb_ok ? std::to_string(rgb_age) : "null") << ","
      << "\"width\":" << rgb_w << ","
      << "\"height\":" << rgb_h << ","
      << "\"depth_available\":" << ((depth_ok && g_running.load()) ? "true" : "false") << ","
      << "\"has_depth\":" << (depth_ok ? "true" : "false") << ","
      << "\"latest_depth_age_ms\":" << (depth_ok ? std::to_string(depth_age) : "null") << ","
      << "\"depth_width\":" << depth_w << ","
      << "\"depth_height\":" << depth_h << ","
      << "\"depth_encoding\":\"" << json_escape(depth_enc) << "\","
      << "\"depth_output_encoding\":\"16UC1_mm_png\","
      << "\"depth_valid_count\":" << depth_valid << ","
      << "\"depth_min_mm\":" << depth_min << ","
      << "\"depth_max_mm\":" << depth_max << ","
      << "\"time\":\"" << now_iso() << "\""
      << "}";
  return oss.str();
}

void http_mjpeg_stream(int fd) {
  const std::string boundary = "visionops_hp60c_frame";
  std::ostringstream hdr;
  hdr << "HTTP/1.1 200 OK\r\n"
      << "Content-Type: multipart/x-mixed-replace; boundary=" << boundary << "\r\n"
      << "Cache-Control: no-store\r\n"
      << "Connection: close\r\n\r\n";
  send_all(fd, hdr.str());

  while (!g_exit.load() && ros::ok()) {
    if (!g_running.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }
    std::vector<uchar> jpg;
    double age = 0;
    int w = 0, h = 0;
    if (!encode_latest(jpg, &age, &w, &h)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }
    std::ostringstream part;
    part << "--" << boundary << "\r\n"
         << "Content-Type: image/jpeg\r\n"
         << "Content-Length: " << jpg.size() << "\r\n\r\n";
    if (::send(fd, part.str().data(), part.str().size(), MSG_NOSIGNAL) <= 0) break;
    if (!jpg.empty() && ::send(fd, reinterpret_cast<const char*>(jpg.data()), jpg.size(), MSG_NOSIGNAL) <= 0) break;
    if (::send(fd, "\r\n", 2, MSG_NOSIGNAL) <= 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
  }
}

void send_image_response(int fd, const std::string& content_type, const std::vector<uchar>& data) {
  std::ostringstream hdr;
  hdr << "HTTP/1.1 200 OK\r\n"
      << "Content-Type: " << content_type << "\r\n"
      << "Cache-Control: no-store\r\n"
      << "Content-Length: " << data.size() << "\r\n\r\n";
  send_all(fd, hdr.str());
  send_all_bytes(fd, data);
}

void handle_client(int fd) {
  char buf[2048];
  ssize_t n = ::recv(fd, buf, sizeof(buf)-1, 0);
  if (n <= 0) { ::close(fd); return; }
  buf[n] = 0;
  std::string req(buf);
  std::istringstream iss(req);
  std::string method, path, version;
  iss >> method >> path >> version;
  auto qpos = path.find('?');
  if (qpos != std::string::npos) path = path.substr(0, qpos);

  if (path == "/" || path == "/health" || path == "/stream/status" || path == "/stream/depth_meta" || path == "/stream/depth/status") {
    http_json(fd, status_json());
  } else if (path == "/stream/start") {
    g_running = true;
    http_json(fd, std::string("{\"ok\":true,\"message\":\"HP60C ROS1 bridge started\",") + "\"status\":" + status_json() + "}");
  } else if (path == "/stream/stop") {
    // Preview page may call stop when leaving. For ROS1 cameras the bridge is
    // the shared frame provider for snapshot/preview/depth, so stopping it here
    // would make the next page entry fail. Keep the HTTP bridge alive and only
    // report success. Use systemctl stop if the service really needs to stop.
    g_running = true;
    http_json(fd, std::string("{\"ok\":true,\"message\":\"HP60C ROS1 bridge keep-alive; stop is a no-op for shared preview/depth\",") + "\"status\":" + status_json() + "}");
  } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
    http_mjpeg_stream(fd);
  } else if (path == "/stream/snapshot.jpg" || path == "/snapshot.jpg") {
    if (!g_running.load()) {
      http_json(fd, "{\"ok\":false,\"error\":\"bridge stopped\"}", 503, "Service Unavailable");
    } else {
      std::vector<uchar> jpg; double age=0; int w=0,h=0;
      if (!encode_latest(jpg, &age, &w, &h)) {
        http_json(fd, "{\"ok\":false,\"error\":\"no RGB frame yet\"}", 503, "Service Unavailable");
      } else {
        send_image_response(fd, "image/jpeg", jpg);
      }
    }
  } else if (path == "/stream/depth.png" || path == "/depth.png") {
    if (!g_running.load()) {
      http_json(fd, "{\"ok\":false,\"error\":\"bridge stopped\"}", 503, "Service Unavailable");
    } else {
      std::vector<uchar> png; double age=0; int w=0,h=0;
      if (!encode_latest_depth_png(png, &age, &w, &h)) {
        http_json(fd, "{\"ok\":false,\"error\":\"no depth frame yet\"}", 503, "Service Unavailable");
      } else {
        send_image_response(fd, "image/png", png);
      }
    }
  } else if (path == "/stream/depth_vis.jpg" || path == "/depth_vis.jpg") {
    if (!g_running.load()) {
      http_json(fd, "{\"ok\":false,\"error\":\"bridge stopped\"}", 503, "Service Unavailable");
    } else {
      std::vector<uchar> jpg; double age=0; int w=0,h=0;
      if (!encode_latest_depth_vis(jpg, &age, &w, &h)) {
        http_json(fd, "{\"ok\":false,\"error\":\"no depth frame yet\"}", 503, "Service Unavailable");
      } else {
        send_image_response(fd, "image/jpeg", jpg);
      }
    }
  } else {
    http_json(fd, "{\"ok\":false,\"error\":\"not found\"}", 404, "Not Found");
  }
  ::close(fd);
}

void http_server() {
  int srv = ::socket(AF_INET, SOCK_STREAM, 0);
  if (srv < 0) { perror("socket"); return; }
  int yes = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(g_http_port));
  if (::bind(srv, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) { perror("bind"); ::close(srv); return; }
  if (::listen(srv, 16) < 0) { perror("listen"); ::close(srv); return; }
  ROS_INFO("HP60C ROS1 bridge HTTP listening on 127.0.0.1:%d", g_http_port);
  while (!g_exit.load() && ros::ok()) {
    sockaddr_in caddr{}; socklen_t clen = sizeof(caddr);
    int c = ::accept(srv, reinterpret_cast<sockaddr*>(&caddr), &clen);
    if (c < 0) continue;
    std::thread(handle_client, c).detach();
  }
  ::close(srv);
}

void on_signal(int) { g_exit = true; ros::shutdown(); }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "visionops_hp60c_ros1_bridge");
  ros::NodeHandle nh("~");
  const char* env_legacy_topic = std::getenv("VISIONOPS_HP60C_ROS1_TOPIC");
  const char* env_rgb_topic = std::getenv("VISIONOPS_HP60C_RGB_TOPIC");
  const char* env_depth_topic = std::getenv("VISIONOPS_HP60C_DEPTH_TOPIC");
  const char* env_quality = std::getenv("VISIONOPS_HP60C_JPEG_QUALITY");
  const char* env_depth_png = std::getenv("VISIONOPS_HP60C_DEPTH_PNG_COMPRESSION");
  const char* env_port = std::getenv("VISIONOPS_HP60C_ROS1_BRIDGE_PORT");
  if (env_legacy_topic && std::strlen(env_legacy_topic) > 0) g_rgb_topic = env_legacy_topic;
  if (env_rgb_topic && std::strlen(env_rgb_topic) > 0) g_rgb_topic = env_rgb_topic;
  if (env_depth_topic && std::strlen(env_depth_topic) > 0) g_depth_topic = env_depth_topic;
  if (env_quality) g_jpeg_quality = std::max(10, std::min(100, std::atoi(env_quality)));
  if (env_depth_png) g_depth_png_compression = std::max(0, std::min(9, std::atoi(env_depth_png)));
  if (env_port) g_http_port = std::max(1024, std::min(65535, std::atoi(env_port)));
  nh.param<std::string>("topic", g_rgb_topic, g_rgb_topic);      // backward compatible
  nh.param<std::string>("rgb_topic", g_rgb_topic, g_rgb_topic);
  nh.param<std::string>("depth_topic", g_depth_topic, g_depth_topic);
  nh.param<int>("jpeg_quality", g_jpeg_quality, g_jpeg_quality);
  nh.param<int>("depth_png_compression", g_depth_png_compression, g_depth_png_compression);
  nh.param<int>("http_port", g_http_port, g_http_port);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  ros::Subscriber rgb_sub = nh.subscribe(g_rgb_topic, 1, image_cb);
  ros::Subscriber depth_sub = nh.subscribe(g_depth_topic, 1, depth_cb);
  std::thread server_thread(http_server);
  ROS_INFO("VisionOps HP60C ROS1 C++ bridge subscribing RGB: %s", g_rgb_topic.c_str());
  ROS_INFO("VisionOps HP60C ROS1 C++ bridge subscribing Depth: %s", g_depth_topic.c_str());
  ros::spin();
  g_exit = true;
  if (server_thread.joinable()) server_thread.join();
  return 0;
}
