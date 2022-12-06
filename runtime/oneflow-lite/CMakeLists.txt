add_subdirectory(base)
add_subdirectory(core)
add_subdirectory(delegates)

get_property(
  OFLITE_RUNTIME_OBJECT_LIBS GLOBAL PROPERTY OFLITE_RUNTIME_OBJECT_LIBS)
get_property(
  OFLITE_THIRDPARTY_LIBS GLOBAL PROPERTY OFLITE_THIRDPARTY_LIBS)

add_library(oneflow-lite-runtime SHARED dummy.cc ${OFLITE_RUNTIME_OBJECT_LIBS})
target_link_libraries(oneflow-lite-runtime flatcc-runtime ${OFLITE_THIRDPARTY_LIBS})

add_subdirectory(tests)
