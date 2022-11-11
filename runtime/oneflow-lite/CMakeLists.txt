add_subdirectory(base)
add_subdirectory(core)
add_subdirectory(delegates)

get_property(
  OFLITE_RUNTIME_OBJECT_LIBS GLOBAL PROPERTY OFLITE_RUNTIME_OBJECT_LIBS)

add_library(oneflow-lite-runtime dummy.cc ${OFLITE_RUNTIME_OBJECT_LIBS})
target_link_libraries(oneflow-lite-runtime flatcc-runtime)

add_subdirectory(tests)
