file(GLOB_RECURSE OFLITE_RUNTIME_CORE_SRCS
  executable.cc
)
add_library(oneflow-lite-runtime-core ${OFLITE_RUNTIME_CORE_SRCS})
add_dependencies(oneflow-lite-runtime-core schemas)
target_link_libraries(oneflow-lite-runtime-core flatcc-runtime)

