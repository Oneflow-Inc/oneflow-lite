file(GLOB ATTRIBUTE_SCHEMA_FILES *.fbs)
flatcc_generate(ATTRIBUTE_SCHEMA_SRCS ${ATTRIBUTE_SCHEMA_FILES})

add_custom_target(attribute_schema_gen DEPENDS ${ATTRIBUTE_SCHEMA_SRCS} flatcc-runtime)
