cmake_minimum_required(VERSION 3.18.0)

project(oneflow-lite-compiler)

option(LITE_USE_CUDA "" OFF)
option(LITE_USE_ASCEND_NPU "" OFF)

set(THIRD_PARTY_MIRROR "" CACHE STRING "")

include(ExternalProject)

set(ONEFLOW_PREFIX oneflow-prefix)
set(ONEFLOW_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/${ONEFLOW_PREFIX}/src/oneflow-build)
set(ONEFLOW_LITE_COMPILE_BIN ${ONEFLOW_BUILD_DIR}/oneflow/ir/bin/oneflow-lite-compile)

ExternalProject_Add(oneflow
  PREFIX ${ONEFLOW_PREFIX}
  URL ${PROJECT_SOURCE_DIR}/oneflow
  UPDATE_COMMAND ""
  BUILD_BYPRODUCTS ${ONEFLOW_LITE_COMPILE_BIN}
  CMAKE_CACHE_ARGS
    -DWITH_MLIR:BOOL=ON
    -DBUILD_CUDA:BOOL=OFF
    -DBUILD_CPP_API:BOOL=OFF
    -DBUILD_PYTHON:BOOL=ON
    -DWITH_ONEDNN:BOOL=OFF
    -DWITH_COCOAPI:BOOL=OFF
    -DMAYBE_NEED_ERROR_MSG_CHECK:BOOL=OFF
    -DTHIRD_PARTY_MIRROR:STRING=${THIRD_PARTY_MIRROR}
    -DLITE_USE_CUDA:BOOL=${LITE_USE_CUDA}
    -DLITE_USE_ASCEND_NPU:BOOL=${LITE_USE_ASCEND_NPU}
)
add_executable(oneflow-lite-compile IMPORTED)
set_property(TARGET oneflow-lite-compile PROPERTY IMPORTED_LOCATION ${ONEFLOW_LITE_COMPILE_BIN})
add_dependencies(oneflow-lite-compile oneflow)

install(IMPORTED_RUNTIME_ARTIFACTS oneflow-lite-compile
  COMPONENT install_component
  RUNTIME DESTINATION bin
)
add_custom_target(install_target
  COMMAND "${CMAKE_COMMAND}" -DCMAKE_INSTALL_COMPONENT=install_component
  -DCMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_install.cmake"
  DEPENDS oneflow
)
add_library(dummy dummy.cc)
add_dependencies(dummy install_target)
