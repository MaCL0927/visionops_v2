cmake_minimum_required(VERSION 3.10)
project(visionops_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

find_path(RKNN_INCLUDE_DIR
  NAMES rknn_api.h
  PATHS
    /usr/include
    /usr/local/include
    /usr/include/rknn
    /usr/local/include/rknn
    /opt/rknn/include
    /opt/visionops/include
)

find_library(RKNN_LIBRARY
  NAMES rknnrt librknnrt.so
  PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/lib/aarch64-linux-gnu
    /opt/rknn/lib
    /opt/visionops/lib
)

if(NOT RKNN_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot find rknn_api.h. Install RKNN runtime headers or set CMAKE_INCLUDE_PATH.")
endif()

if(NOT RKNN_LIBRARY)
  message(FATAL_ERROR "Cannot find librknnrt.so. Put it in /usr/lib or set CMAKE_LIBRARY_PATH.")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "RKNN include: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN library: ${RKNN_LIBRARY}")

# v0.4.2: optional RGA path. When headers/libs are present, the binary enables
# RGA resize + BGR->RGB color conversion for preprocessing; CPU padding and CPU
# fallback are kept for robustness.
find_path(RGA_INCLUDE_DIR
  NAMES im2d.hpp rga/im2d.hpp RgaUtils.h rga/RgaUtils.h rga.h rga/rga.h
  PATHS
    /usr/include
    /usr/include/rga
    /usr/local/include
    /usr/local/include/rga
    /opt/rga/include
    /opt/visionops/include
)

find_library(RGA_LIBRARY
  NAMES rga librga.so
  PATHS
    /usr/lib
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /usr/local/lib/aarch64-linux-gnu
    /opt/rga/lib
    /opt/visionops/lib
)

if(RGA_INCLUDE_DIR AND RGA_LIBRARY)
  set(VISIONOPS_WITH_RGA 1)
  message(STATUS "RGA include: ${RGA_INCLUDE_DIR}")
  message(STATUS "RGA library: ${RGA_LIBRARY}")
else()
  set(VISIONOPS_WITH_RGA 0)
  message(WARNING "RGA headers/library not found. Building with CPU preprocessing fallback only.")
endif()

add_executable(visionops_inference_cpp
  src/main.cpp
  src/stream_backend.cpp
  src/rga_preprocess.cpp
)

target_include_directories(visionops_inference_cpp PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${RKNN_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_compile_options(visionops_inference_cpp PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_compile_definitions(visionops_inference_cpp PRIVATE VISIONOPS_WITH_RGA=${VISIONOPS_WITH_RGA})

if(VISIONOPS_WITH_RGA)
  target_include_directories(visionops_inference_cpp PRIVATE ${RGA_INCLUDE_DIR})
endif()

target_link_libraries(visionops_inference_cpp PRIVATE
  ${OpenCV_LIBS}
  ${RKNN_LIBRARY}
  pthread
  dl
)

if(VISIONOPS_WITH_RGA)
  target_link_libraries(visionops_inference_cpp PRIVATE ${RGA_LIBRARY})
endif()

install(TARGETS visionops_inference_cpp RUNTIME DESTINATION bin)
