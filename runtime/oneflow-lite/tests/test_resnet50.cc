#include <iostream>

#include "oneflow-lite/core/executable.h"
#include "oneflow-lite/core/execution_context.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "./test_resnet50 [model.bin]" << std::endl;
    return 1;
  }
  std::cout << "start to load model " << argv[1] << std::endl;
  OfLiteExecutable* executable = NULL;
  OfLiteExecutableCreate(&executable, OfLiteStringRefCreate(argv[1]));
  if (!executable) {
    std::cout << "failed to load model" << std::endl;
  }

  OfLiteExecutionContext* context = NULL;
  OfLiteExecutionOption option;
  OfLiteExecutionContextCreate(executable, option, &context);
  if (!context) {
    std::cout << "failed to create execution context" << std::endl;
  }

  OfLiteExecutionContextInvoke(context);
  return 0;
}
