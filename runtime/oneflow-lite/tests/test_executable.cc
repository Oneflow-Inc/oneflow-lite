#include "oneflow-lite/core/executable.h"

int main() {
  OfLiteExecutable* executable = NULL;
  OfLiteExecutableCreate(&executable, OfLiteStringRefCreate("xxx"));

  OfLiteStringRef name;
  OfLiteExecutableName(executable, &name);

  size_t input_size = 0;
  OfLiteExecutableInputSize(executable, &input_size);

  size_t output_size = 0;
  OfLiteExecutableOutputSize(executable, &output_size);

  size_t input = -1;
  OfLiteExecutableInput(executable, 0, &input);

  size_t output = -1;
  OfLiteExecutableOutput(executable, 0, &output);

  OfLiteStringRef input_name;
  OfLiteExecutableInputName(executable, 0, &input_name);

  OfLiteStringRef output_name;
  OfLiteExecutableOutputName(executable, 0, &output_name);

  size_t operand_size = 0;
  OfLiteExecutableOperandSize(executable, &operand_size);

  const OfLiteTensorDef* operand = NULL;
  OfLiteExecutableOperand(executable, 0, &operand);

  size_t device_size = 0;
  OfLiteExecutableDeviceSize(executable, &device_size);

  OfLiteStringRef device_name;
  OfLiteExecutableDevice(executable, 0, &device_name);

  size_t buffer_segment_size = 0;
  OfLiteExecutableBufferSegmentSize(executable, &buffer_segment_size);

  const OfLiteBufferSegmentDef* buffer_segment = NULL;
  OfLiteExecutableBufferSegment(executable, 0, &buffer_segment);

  size_t op_size = 0;
  OfLiteExecutableOpSize(executable, &op_size);

  const OfLiteOpDef* op = NULL;
  OfLiteExecutableOp(executable, 0, &op);

  size_t function_size = 0;
  OfLiteExecutableFunctionSize(executable, &function_size);

  const OfLiteOpFunctionDef* function = NULL;
  OfLiteExecutableFunction(executable, 0, &function);
}
