cmake_minimum_required(VERSION 3.10)
project(visionops_orbbec336l_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORBBEC_SDK_ROOT "$ENV{VISIONOPS_ORBBEC336L_SDK_ROOT}" CACHE PATH "Orbbec SDK root")
if(NOT ORBBEC_SDK_ROOT)
  set(ORBBEC_SDK_ROOT "/opt/OrbbecSDK")
endif()

set(ORBBEC_INCLUDE_DIR "$ENV{VISIONOPS_ORBBEC336L_SDK_INCLUDE_DIR}" CACHE PATH "Orbbec SDK include dir")
if(NOT ORBBEC_INCLUDE_DIR)
  if(EXISTS "${ORBBEC_SDK_ROOT}/include")
    set(ORBBEC_INCLUDE_DIR "${ORBBEC_SDK_ROOT}/include")
  elseif(EXISTS "${ORBBEC_SDK_ROOT}/include/libobsensor")
    set(ORBBEC_INCLUDE_DIR "${ORBBEC_SDK_ROOT}/include")
  endif()
endif()

set(ORBBEC_LIB_DIR "$ENV{VISIONOPS_ORBBEC336L_SDK_LIB_DIR}" CACHE PATH "Orbbec SDK lib dir")
if(NOT ORBBEC_LIB_DIR)
  if(EXISTS "${ORBBEC_SDK_ROOT}/lib")
    set(ORBBEC_LIB_DIR "${ORBBEC_SDK_ROOT}/lib")
  elseif(EXISTS "${ORBBEC_SDK_ROOT}/lib/aarch64")
    set(ORBBEC_LIB_DIR "${ORBBEC_SDK_ROOT}/lib/aarch64")
  endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(visionops_orbbec336l_bridge visionops_orbbec336l_bridge.cpp)
target_include_directories(visionops_orbbec336l_bridge PRIVATE ${ORBBEC_INCLUDE_DIR})
target_link_directories(visionops_orbbec336l_bridge PRIVATE ${ORBBEC_LIB_DIR})
target_link_libraries(visionops_orbbec336l_bridge PRIVATE OrbbecSDK ${OpenCV_LIBS} pthread dl)

set_target_properties(visionops_orbbec336l_bridge PROPERTIES
  BUILD_RPATH "${ORBBEC_LIB_DIR}"
  INSTALL_RPATH "${ORBBEC_LIB_DIR}"
)
