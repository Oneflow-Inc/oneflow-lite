/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow-lite/core/executable.h"

#include <stdio.h>

#include "oneflow-lite/schemas/executable_generated.h"

typedef struct OfLiteNativeExecutable {
  uint8_t* buf;
  size_t buf_length;
  oneflow_lite_ExecutableDef_table_t def;
} OfLiteNativeExecutable;

OFLITE_API void OfLiteExecutableCreate(OfLiteExecutable** executable,
                                       OfLiteStringRef filepath) {
  FILE* file = fopen(filepath.data, "rb");
  if (file == NULL) {
    // TODO()
  }
  if (fseek(file, 0, SEEK_END) != -1) {
    // TODO()
  }
  size_t buf_length = ftell(file);
  if (fseek(file, 0, SEEK_SET) != -1) {
    // TODO()
  }
  uint8_t* buf = reinterpret_cast<uint8_t*>(
      malloc(sizeof(OfLiteNativeExecutable) + buf_length + 1));
  if (fread(buf, buf_length, 1, file) != 1) {
    // TODO()
  }
  buf[buf_length] = 0;

  fclose(file);
  if (flatcc_verify_ok !=
      oneflow_lite_ExecutableDef_verify_as_root(buf, buf_length)) {
    // TODO()
  }
  OfLiteNativeExecutable* native_executable =
      reinterpret_cast<OfLiteNativeExecutable*>(buf + buf_length + 1);
  native_executable->buf = buf;
  native_executable->buf_length = buf_length;
  native_executable->def = oneflow_lite_ExecutableDef_as_root(buf);
  *executable = reinterpret_cast<OfLiteExecutable*>(native_executable);
}

OFLITE_API void OfLiteExecutableDestory(OfLiteExecutable* executable) {
  OfLiteNativeExecutable* native_executable =
      reinterpret_cast<OfLiteNativeExecutable*>(executable);
  free(native_executable->buf);
}

OFLITE_API void OfLiteExecutableName(const OfLiteExecutable* executable,
                                     OfLiteStringRef* name) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  flatbuffers_string_t flatc_name =
      oneflow_lite_ExecutableDef_name(native_executable->def);
  *name = OfLiteStringRef{flatc_name, flatbuffers_string_len(flatc_name)};
}

OFLITE_API void OfLiteExecutableInputSize(const OfLiteExecutable* executable,
                                          size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t inputs =
      oneflow_lite_ExecutableDef_inputs(native_executable->def);
  *size = oneflow_lite_TensorDef_vec_len(inputs);
}

OFLITE_API void OfLiteExecutableOutputSize(const OfLiteExecutable* executable,
                                           size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t outputs =
      oneflow_lite_ExecutableDef_outputs(native_executable->def);
  *size = oneflow_lite_TensorDef_vec_len(outputs);
}

OFLITE_API void OfLiteExecutableInput(const OfLiteExecutable* executable,
                                      size_t index,
                                      const OfLiteTensorDef** input) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t inputs =
      oneflow_lite_ExecutableDef_inputs(native_executable->def);
  if (index >= oneflow_lite_TensorDef_vec_len(inputs)) {
    // TODO()
  }
  oneflow_lite_TensorDef_table_t flatc_input =
      oneflow_lite_TensorDef_vec_at(inputs, index);
  *input = reinterpret_cast<const OfLiteTensorDef*>(flatc_input);
}

OFLITE_API void OfLiteExecutableInputName(const OfLiteExecutable* executable,
                                          size_t index,
                                          OfLiteStringRef* input_name) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  flatbuffers_string_vec_t input_names =
      oneflow_lite_ExecutableDef_input_names(native_executable->def);
  if (index >= flatbuffers_string_vec_len(input_names)) {
    // TODO()
  }
  flatbuffers_string_t flatc_name =
      flatbuffers_string_vec_at(input_names, index);
  *input_name = OfLiteStringRef{flatc_name, flatbuffers_string_len(flatc_name)};
}

OFLITE_API void OfLiteExecutableOutput(const OfLiteExecutable* executable,
                                       size_t index,
                                       const OfLiteTensorDef** output) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t outputs =
      oneflow_lite_ExecutableDef_outputs(native_executable->def);
  oneflow_lite_TensorDef_table_t flatc_output =
      oneflow_lite_TensorDef_vec_at(outputs, index);
  *output = reinterpret_cast<const OfLiteTensorDef*>(flatc_output);
}

OFLITE_API void OfLiteExecutableOutputName(const OfLiteExecutable* executable,
                                           size_t index,
                                           OfLiteStringRef* output_name) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  flatbuffers_string_vec_t output_names =
      oneflow_lite_ExecutableDef_output_names(native_executable->def);
  if (index >= flatbuffers_string_vec_len(output_names)) {
    // TODO()
  }
  flatbuffers_string_t flatc_name =
      flatbuffers_string_vec_at(output_names, index);
  *output_name =
      OfLiteStringRef{flatc_name, flatbuffers_string_len(flatc_name)};
}

OFLITE_API void OfLiteExecutableOperandSize(const OfLiteExecutable* executable,
                                            size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t operands =
      oneflow_lite_ExecutableDef_operands(native_executable->def);
  *size = oneflow_lite_TensorDef_vec_len(operands);
}

OFLITE_API void OfLiteExecutableOperand(const OfLiteExecutable* executable,
                                        size_t index,
                                        const OfLiteTensorDef** operand) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_TensorDef_vec_t operands =
      oneflow_lite_ExecutableDef_operands(native_executable->def);
  if (index >= oneflow_lite_TensorDef_vec_len(operands)) {
    // TODO()
  }
  oneflow_lite_TensorDef_table_t flatc_operand =
      oneflow_lite_TensorDef_vec_at(operands, index);
  *operand = reinterpret_cast<const OfLiteTensorDef*>(flatc_operand);
}

OFLITE_API void OfLiteExecutableDeviceSize(const OfLiteExecutable* executable,
                                           size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  flatbuffers_string_vec_t devices =
      oneflow_lite_ExecutableDef_devices(native_executable->def);
  *size = flatbuffers_string_vec_len(devices);
}

OFLITE_API void OfLiteExecutableDevice(const OfLiteExecutable* executable,
                                       size_t index, OfLiteStringRef* device) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  flatbuffers_string_vec_t devices =
      oneflow_lite_ExecutableDef_devices(native_executable->def);
  if (index >= flatbuffers_string_vec_len(devices)) {
    // TODO()
  }
  flatbuffers_string_t flatc_device = flatbuffers_string_vec_at(devices, index);
  *device = OfLiteStringRef{flatc_device, flatbuffers_string_len(flatc_device)};
}

OFLITE_API void OfLiteExecutableBufferSegmentSize(
    const OfLiteExecutable* executable, size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_BufferSegmentDef_vec_t buffer_sgements =
      oneflow_lite_ExecutableDef_segments(native_executable->def);
  *size = oneflow_lite_BufferSegmentDef_vec_len(buffer_sgements);
}

OFLITE_API void OfLiteExecutableBufferSegment(
    const OfLiteExecutable* executable, size_t index,
    const OfLiteBufferSegmentDef** buffer_sgement) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_BufferSegmentDef_vec_t buffer_sgements =
      oneflow_lite_ExecutableDef_segments(native_executable->def);
  if (index >= oneflow_lite_BufferSegmentDef_vec_len(buffer_sgements)) {
    // TODO()
  }
  oneflow_lite_BufferSegmentDef_table_t flatc_buffer_sgement =
      oneflow_lite_BufferSegmentDef_vec_at(buffer_sgements, index);
  *buffer_sgement =
      reinterpret_cast<const OfLiteBufferSegmentDef*>(flatc_buffer_sgement);
}

OFLITE_API void OfLiteExecutableOpSize(const OfLiteExecutable* executable,
                                       size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_OpDef_vec_t ops =
      oneflow_lite_ExecutableDef_ops(native_executable->def);
  *size = oneflow_lite_OpDef_vec_len(ops);
}

OFLITE_API void OfLiteExecutableOp(const OfLiteExecutable* executable,
                                   size_t index, const OfLiteOpDef** op) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_OpDef_vec_t ops =
      oneflow_lite_ExecutableDef_ops(native_executable->def);
  if (index >= oneflow_lite_OpDef_vec_len(ops)) {
    // TODO()
  }
  oneflow_lite_OpDef_table_t flatc_op = oneflow_lite_OpDef_vec_at(ops, index);
  *op = reinterpret_cast<const OfLiteOpDef*>(flatc_op);
}

OFLITE_API void OfLiteExecutableFunctionSize(const OfLiteExecutable* executable,
                                             size_t* size) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_OpFunctionDef_vec_t functions =
      oneflow_lite_ExecutableDef_functions(native_executable->def);
  *size = oneflow_lite_OpFunctionDef_vec_len(functions);
}

OFLITE_API void OfLiteExecutableFunction(const OfLiteExecutable* executable,
                                         size_t index,
                                         const OfLiteOpFunctionDef** function) {
  const OfLiteNativeExecutable* native_executable =
      reinterpret_cast<const OfLiteNativeExecutable*>(executable);
  oneflow_lite_OpFunctionDef_vec_t functions =
      oneflow_lite_ExecutableDef_functions(native_executable->def);
  if (index >= oneflow_lite_OpFunctionDef_vec_len(functions)) {
    // TODO()
  }
  oneflow_lite_OpFunctionDef_table_t flatc_function =
      oneflow_lite_OpFunctionDef_vec_at(functions, index);
  *function = reinterpret_cast<const OfLiteOpFunctionDef*>(flatc_function);
}
