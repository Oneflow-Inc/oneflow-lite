add_subdirectory(attributes)

file(GLOB SCHEMA_FILES *.fbs)
flatcc_generate(SCHEMA_SRCS ${SCHEMA_FILES})

add_custom_target(schema_gen DEPENDS ${SCHEMA_SRCS} flatcc-runtime)
add_library(schemas INTERFACE)
add_dependencies(schemas schema_gen attribute_schema_gen)
