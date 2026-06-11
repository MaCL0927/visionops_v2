cmake_minimum_required(VERSION 3.10)
project(visionops_hp60c_sdk_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ANGSTRONG_SDK_ROOT "$ENV{VISIONOPS_HP60C_SDK_ROOT}" CACHE PATH "Path to Angstrong demo/linux_ros root")
if(NOT ANGSTRONG_SDK_ROOT)
  set(ANGSTRONG_SDK_ROOT "/home/neardi/AngstrongCameraSdk_v1.2.61.20250910/demo/linux_ros")
endif()

set(ANGSTRONG_INCLUDE_DIR "${ANGSTRONG_SDK_ROOT}/libs/include")
set(ANGSTRONG_LIB_DIR "$ENV{VISIONOPS_HP60C_SDK_LIB_DIR}" CACHE PATH "Path to Angstrong SDK lib dir")
if(NOT ANGSTRONG_LIB_DIR)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ANGSTRONG_LIB_DIR "${ANGSTRONG_SDK_ROOT}/libs/lib/aarch64-linux-gnu-gcc-5")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    set(ANGSTRONG_LIB_DIR "${ANGSTRONG_SDK_ROOT}/libs/lib/arm-linux-gnueabihf-gcc-4.8")
  else()
    set(ANGSTRONG_LIB_DIR "${ANGSTRONG_SDK_ROOT}/libs/lib/x86_64-linux-gnu-gcc-5")
  endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(visionops_hp60c_sdk_bridge visionops_hp60c_sdk_bridge.cpp)
target_include_directories(visionops_hp60c_sdk_bridge PRIVATE ${ANGSTRONG_INCLUDE_DIR})
target_link_directories(visionops_hp60c_sdk_bridge PRIVATE ${ANGSTRONG_LIB_DIR})
target_link_libraries(visionops_hp60c_sdk_bridge PRIVATE AngstrongCameraSdk ${OpenCV_LIBS} pthread dl)
set_target_properties(visionops_hp60c_sdk_bridge PROPERTIES
  BUILD_RPATH "${ANGSTRONG_LIB_DIR}"
  INSTALL_RPATH "${ANGSTRONG_LIB_DIR}"
)
