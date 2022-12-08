add_subdirectory(generic)

if(BUILD_X86)
  add_subdirectory(x86)
endif()

if(BUILD_ASCEND_NPU)
  add_subdirectory(ascend)
endif()
