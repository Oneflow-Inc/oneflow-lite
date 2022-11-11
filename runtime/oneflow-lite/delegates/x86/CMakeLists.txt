set(OFLITE_RUNTIME_DELEGATE_X86_SRCS
  x86_allocator.cc
  x86_device.cc
)
add_library(oneflow-lite-runtime-delegate-x86 OBJECT ${OFLITE_RUNTIME_DELEGATE_X86_SRCS})
add_dependencies(oneflow-lite-runtime-delegate-x86 schemas)

set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-delegate-x86>
)
