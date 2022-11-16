file(GLOB_RECURSE OFLITE_RUNTIME_CORE_SRCS
    allocator.cc
    buffer.cc
    device.cc
    event.cc
    executable.cc
    execution_context.cc
    operator.cc
    stream.cc
    tensor.cc
)
add_library(oneflow-lite-runtime-core OBJECT ${OFLITE_RUNTIME_CORE_SRCS})
add_dependencies(oneflow-lite-runtime-core schemas)
target_link_libraries(oneflow-lite-runtime-core flatcc-runtime)

set_property(GLOBAL APPEND PROPERTY
  OFLITE_RUNTIME_OBJECT_LIBS
  $<TARGET_OBJECTS:oneflow-lite-runtime-core>
)
