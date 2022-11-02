cmake_minimum_required(VERSION 3.11.0)

project(oneflow-lite C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(thirdparty)
add_subdirectory(schemas)
add_subdirectory(core)
add_subdirectory(delegates)

add_library(oneflow-lite-runtime dummy.cc)
target_link_libraries(oneflow-lite-runtime schemas flatcc-runtime)
