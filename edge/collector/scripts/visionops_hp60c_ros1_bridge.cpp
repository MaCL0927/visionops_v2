#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {
std::mutex g_frame_mutex;
cv::Mat g_latest_bgr;
ros::Time g_latest_stamp;
std::atomic<bool> g_running{true};
std::atomic<bool> g_exit{false};
std::string g_topic = "/ascamera_hp60c/rgb0/image";
int g_jpeg_quality = 80;
int g_http_port = 18181;

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
    ROS_WARN_THROTTLE(2.0, "HP60C bridge image conversion failed: %s", e.what());
  }
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
  double age = -1.0; int w = 0, h = 0;
  std::vector<uchar> tmp;
  bool ok = encode_latest(tmp, &age, &w, &h);
  std::ostringstream oss;
  oss << "{"
      << "\"ok\":true,"
      << "\"backend\":\"hp60c-ros1-cpp-bridge\","
      << "\"running\":" << (g_running.load() ? "true" : "false") << ","
      << "\"topic\":\"" << g_topic << "\","
      << "\"snapshot_available\":" << ((ok && g_running.load()) ? "true" : "false") << ","
      << "\"has_frame\":" << (ok ? "true" : "false") << ","
      << "\"latest_snapshot_age_ms\":" << (ok ? std::to_string(age) : "null") << ","
      << "\"width\":" << w << ","
      << "\"height\":" << h << ","
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

  if (path == "/" || path == "/health" || path == "/stream/status") {
    http_json(fd, status_json());
  } else if (path == "/stream/start") {
    g_running = true;
    http_json(fd, std::string("{\"ok\":true,\"message\":\"HP60C ROS1 bridge started\",") + "\"status\":" + status_json() + "}");
  } else if (path == "/stream/stop") {
    // Preview page may call stop when leaving. For ROS1 cameras the bridge is
    // the shared frame provider for snapshot/preview, so stopping it here would
    // make the next page entry fail. Keep the HTTP bridge alive and only report
    // success. Use systemctl stop if the service really needs to be stopped.
    g_running = true;
    http_json(fd, std::string("{\"ok\":true,\"message\":\"HP60C ROS1 bridge keep-alive; stop is a no-op for shared preview\",") + "\"status\":" + status_json() + "}");
  } else if (path == "/stream.mjpeg" || path == "/stream/mjpeg" || path == "/stream.mjpg") {
    http_mjpeg_stream(fd);
  } else if (path == "/stream/snapshot.jpg" || path == "/snapshot.jpg") {
    if (!g_running.load()) {
      http_json(fd, "{\"ok\":false,\"error\":\"bridge stopped\"}", 503, "Service Unavailable");
    } else {
      std::vector<uchar> jpg; double age=0; int w=0,h=0;
      if (!encode_latest(jpg, &age, &w, &h)) {
        http_json(fd, "{\"ok\":false,\"error\":\"no frame yet\"}", 503, "Service Unavailable");
      } else {
        std::ostringstream hdr;
        hdr << "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nCache-Control: no-store\r\nContent-Length: " << jpg.size() << "\r\n\r\n";
        send_all(fd, hdr.str());
        send_all_bytes(fd, jpg);
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
  const char* env_topic = std::getenv("VISIONOPS_HP60C_ROS1_TOPIC");
  const char* env_quality = std::getenv("VISIONOPS_HP60C_JPEG_QUALITY");
  const char* env_port = std::getenv("VISIONOPS_HP60C_ROS1_BRIDGE_PORT");
  if (env_topic && std::strlen(env_topic) > 0) g_topic = env_topic;
  if (env_quality) g_jpeg_quality = std::max(10, std::min(100, std::atoi(env_quality)));
  if (env_port) g_http_port = std::max(1024, std::min(65535, std::atoi(env_port)));
  nh.param<std::string>("topic", g_topic, g_topic);
  nh.param<int>("jpeg_quality", g_jpeg_quality, g_jpeg_quality);
  nh.param<int>("http_port", g_http_port, g_http_port);

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  ros::Subscriber sub = nh.subscribe(g_topic, 1, image_cb);
  std::thread server_thread(http_server);
  ROS_INFO("VisionOps HP60C ROS1 C++ bridge subscribing: %s", g_topic.c_str());
  ros::spin();
  g_exit = true;
  if (server_thread.joinable()) server_thread.join();
  return 0;
}
