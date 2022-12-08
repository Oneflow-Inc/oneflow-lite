#include <iostream>

#include "oneflow-lite/core/executable.h"
#include "oneflow-lite/core/execution_context.h"

int main(int argc, char* argv[]) {
  if (argc != 2) {
    OFLITE_FAIL("./test_resnet50 [model.bin]\n");
  }
  std::cout << "start to load model " << argv[1] << std::endl;
  OfLiteExecutable* executable = NULL;
  OfLiteExecutableCreate(&executable, OfLiteStringRefCreate(argv[1]));
  if (!executable) {
    OFLITE_FAIL("failed to load model\n");
  }

  OfLiteExecutionContext* context = NULL;
  OfLiteExecutionOption option;
  OfLiteExecutionContextCreate(executable, option, &context);
  if (!context) {
    OFLITE_FAIL("failed to create execution context\n");
  }

  OfLiteTensor* input = NULL;
  OfLiteExecutionContextInput(context, 0, &input);
  if (!OfLiteTensorIsHost(input)) {
    OFLITE_FAIL("input tensor should be allocated by host\n");
  }
  const OfLiteTensorDesc* input_desc = NULL;
  OfLiteTensorTensorDesc(input, &input_desc);
  if (input_desc->dtype != OfLiteDataType_F32) {
    OFLITE_FAIL("input tensor data type should be float\n");
  }
  float* input_data = reinterpret_cast<float*>(OfLiteTensorData(input));
  for (size_t i = 0; i < OfLiteDimsCount(input_desc->dims); ++i) {
    input_data[i] = 1.0f;
  }

  OfLiteExecutionContextInvoke(context);

  OfLiteTensor* output = NULL;
  OfLiteExecutionContextOutput(context, 0, &output);
  if (!OfLiteTensorIsHost(output)) {
    OFLITE_FAIL("output tensor should be allocated by host\n");
  }
  const OfLiteTensorDesc* output_desc = NULL;
  OfLiteTensorTensorDesc(output, &output_desc);
  if (output_desc->dtype != OfLiteDataType_F32) {
    OFLITE_FAIL("output tensor data type should be float\n");
  }
  const float* output_data = reinterpret_cast<float*>(OfLiteTensorData(output));
  size_t size = OfLiteDimsCount(output_desc->dims);
  std::cout << "out = [";
  for (size_t i = 0; i < (size > 10 ? 10 : size); ++i) {
    std::cout << output_data[i] << ", ";
  }
  std::cout << "...]" << std::endl;

  OfLiteExecutionContextDestory(context);
  OfLiteExecutableDestory(executable);
  return 0;
}
