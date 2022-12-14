include(cmake/FindAscendSdk.cmake)

include_directories(${ASCEND_INCLUDE_DIR})

file(GLOB_RECURSE ASCEND_OP_SRCS ops/*.cc ops/*.cpp)

set(OFLITE_RUNTIME_DELEGATE_X86_SRCS
  ascend_alloca.cc
  ascend_create_op.cc
  ascend_device.cc
  ascend_driver.cc
  ascend_utils.cc
  ${ASCEND_OP_SRCS}
)
add_library(oneflow-lite-runtime-delegate-ascend OBJECT ${OFLITE_RUNTIME_DELEGATE_X86_SRCS})
add_dependencies(oneflow-lite-runtime-delegate-ascend schemas ${ASCEND_LIBRARIES})
target_link_libraries(oneflow-lite-runtime-delegate-ascend flatcc-runtime)
  
set_property(GLOBAL APPEND PROPERTY
  OFLITE_THIRDPARTY_LIBS
  ${ASCEND_LIBRARIES}
)
set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-delegate-ascend>
)
