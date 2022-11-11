set(OFLITE_RUNTIME_DELEGATE_CUDA_SRCS
  cuda_allocator.cc
  cuda_device.cc
)
add_library(oneflow-lite-runtime-delegate-cuda OBJECT ${OFLITE_RUNTIME_DELEGATE_CUDA_SRCS})
add_dependencies(oneflow-lite-runtime-delegate-cuda schemas)

set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-delegate-cuda>
)
