cmake_minimum_required(VERSION 3.11.0)

project(oneflow-lite-runtime C CXX)

option(BUILD_X86 "" OFF)
option(BUILD_CUDA "" OFF)
option(BUILD_ASCEND_NPU "" OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty thirdparty)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../schemas oneflow-lite/schemas)

add_subdirectory(oneflow-lite)
