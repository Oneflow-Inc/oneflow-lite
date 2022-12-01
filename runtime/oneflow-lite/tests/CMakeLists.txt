add_executable(test-executable test_executable.cc)
target_link_libraries(test-executable oneflow-lite-runtime)

add_executable(test-resnet50 test_resnet50.cc)
target_link_libraries(test-resnet50 oneflow-lite-runtime)
