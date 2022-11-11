set(OFLITE_RUNTIME_BASE_SRCS
  dims.cc
)
add_library(oneflow-lite-runtime-base OBJECT ${OFLITE_RUNTIME_BASE_SRCS})
add_dependencies(oneflow-lite-runtime-base schemas)

set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-base>
)
