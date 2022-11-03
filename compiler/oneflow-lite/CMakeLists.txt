add_library(oneflow-lite-compile dummy.cc)
target_link_libraries(oneflow-lite-compile schemas flatcc-runtime)
