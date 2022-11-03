cmake_minimum_required(VERSION 3.11.0)

project(oneflow-lite-compiler C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty thirdparty)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../schemas oneflow-lite/schemas)
add_subdirectory(oneflow-lite)
