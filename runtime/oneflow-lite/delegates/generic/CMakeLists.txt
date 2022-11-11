set(OFLITE_RUNTIME_DELEGATE_GENERIC_SRCS
  generic_allocator.cc
)
add_library(oneflow-lite-runtime-delegate-generic OBJECT
  ${OFLITE_RUNTIME_DELEGATE_GENERIC_SRCS})
add_dependencies(oneflow-lite-runtime-delegate-generic schemas)

set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-delegate-generic>
)
