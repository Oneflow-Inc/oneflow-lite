include(cmake/FindAscendSdk.cmake)

include_directories(${ASCEND_INCLUDE_DIR})

set(OFLITE_RUNTIME_DELEGATE_X86_SRCS
  ascend_alloca.cc
  ascend_device.cc
  ascend_driver.cc
)
add_library(oneflow-lite-runtime-delegate-ascend OBJECT ${OFLITE_RUNTIME_DELEGATE_X86_SRCS})
add_dependencies(oneflow-lite-runtime-delegate-ascend schemas ${ASCEND_LIBRARIES})
  
set_property(GLOBAL APPEND PROPERTY
  OFLITE_THIRDPARTY_LIBS
  ${ASCEND_LIBRARIES}
)
set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-delegate-ascend>
)
