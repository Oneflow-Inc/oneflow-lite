find_path(ASCEND_INCLUDE_DIR acl/acl_rt.h
          PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/include $ENV{ASCEND_HOME_PATH}
                $ENV{ASCEND_HOME_PATH}/include)

find_library(
  ASCEND_ASCENDCL_LIB
  NAMES ascendcl
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

find_library(
  ASCEND_GE_COMPILER_LIB
  NAMES ge_compiler
  PATHS ${ASCEND_HOME_PATH} ${ASCEND_HOME_PATH}/lib64 $ENV{ASCEND_HOME_PATH}
        $ENV{ASCEND_HOME_PATH}/lib64
)

if(NOT ASCEND_INCLUDE_DIR OR NOT ASCEND_ASCENDCL_LIB OR NOT ASCEND_GE_COMPILER_LIB)
  message(
    FATAL_ERROR
      "Ascend Sdk was not found. You can set ASCEND_HOME_PATH to specify the search path."
  )
endif()

add_library(ascend_ascendcl SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_ascendcl PROPERTY IMPORTED_LOCATION ${ASCEND_ASCENDCL_LIB})

add_library(ascend_ge_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET ascend_ge_compiler PROPERTY IMPORTED_LOCATION ${ASCEND_GE_COMPILER_LIB})

set(ASCEND_LIBRARIES ascend_ascendcl ascend_ge_compiler)
