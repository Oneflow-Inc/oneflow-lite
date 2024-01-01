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
#include "oneflow-lite/core/execution_context.h"

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/device_context.h"
#include "oneflow-lite/core/device_util.h"
#include "oneflow-lite/core/execution_unit.h"
#include "oneflow-lite/core/operator.h"
#include "oneflow-lite/schemas/executable_generated.h"

static const int OFLITE_DRIVER_NUM_LIMIT = 8;

typedef struct OfLiteExecutionContext {
  OfLiteAlloca* host_alloca;

  size_t driver_size;
  OfLiteDriver* drivers[OFLITE_DRIVER_NUM_LIMIT];

  OfLiteDeviceContext** device_contexts;
  size_t device_context_size;

  OfLiteBuffer** buffer_segments;
  size_t buffer_segment_size;

  OfLiteTensor** operands;
  size_t operand_size;
  OfLiteTensorSpan* inputs;
  OfLiteTensorSpan* outputs;

  OfLiteExecutionUnit** execution_units;
  size_t execution_unit_size;
} OfLiteExecutionContext;

static void OfLiteExecutionContextGetOrCreateDriver(
    OfLiteExecutionContext* context, OfLiteStringRef backend,
    OfLiteDriver** driver) {
  for (size_t i = 0; i < context->driver_size; ++i) {
    if (context->drivers[i]) {
      OfLiteStringRef identifier;
      OfLiteDriverQueryIdentifier(context->drivers[i], &identifier);
      if (OfLiteStringRefEqual(backend, identifier)) {
        *driver = context->drivers[i];
        return;
      }
    }
  }
  if (context->driver_size >= OFLITE_DRIVER_NUM_LIMIT) {
    OFLITE_FAIL(
        "The number of drivers alive at the same time has reached the upper "
        "limit 8\n");
  }
  // create a new driver and reallocate space for drivers
  OfLiteDriverCreate(backend, driver);
  context->drivers[context->driver_size] = *driver;
  ++context->driver_size;
}

static void OfLiteExecutionContextCreateImpl(
    const OfLiteExecutable* executable, const OfLiteExecutionOption& option,
    OfLiteExecutionContext* context) {
  context->driver_size = 0;
  OfLiteHostAllocaCreate(&context->host_alloca);

  size_t device_size = 0;
  OfLiteExecutableDeviceSize(executable, &device_size);
  context->device_contexts = reinterpret_cast<OfLiteDeviceContext**>(
      OfLiteMalloc(device_size * sizeof(OfLiteDeviceContext*)));
  for (size_t i = 0; i < device_size; ++i) {
    OfLiteStringRef device;
    OfLiteStringRef backend;
    size_t ordinal = 0;
    OfLiteExecutableDevice(executable, i, &device);
    OfLiteParseBackendAndOrdinal(device, &backend, &ordinal);
    OfLiteDriver* driver = nullptr;
    OfLiteExecutionContextGetOrCreateDriver(context, backend, &driver);
    OfLiteDeviceContextCreate(driver, ordinal, context->device_contexts + i);
  }
  context->device_context_size = device_size;

  size_t buffer_segment_size = 0;
  OfLiteExecutableBufferSegmentSize(executable, &buffer_segment_size);
  context->buffer_segments = reinterpret_cast<OfLiteBuffer**>(
      OfLiteMalloc(buffer_segment_size * sizeof(OfLiteBuffer*)));
  for (size_t i = 0; i < buffer_segment_size; ++i) {
    const OfLiteBufferSegmentDef* segment = nullptr;
    OfLiteExecutableBufferSegment(executable, i, &segment);
    oneflow_lite_BufferSegmentDef_table_t flatcc_segment =
        reinterpret_cast<oneflow_lite_BufferSegmentDef_table_t>(segment);
    size_t size = oneflow_lite_BufferSegmentDef_size(flatcc_segment);
    int32_t device = oneflow_lite_BufferSegmentDef_device(flatcc_segment);
    // int32_t alignment =
    // oneflow_lite_BufferSegmentDef_alignment(flatcc_segment);
    if (device >= context->device_context_size) {
      OFLITE_FAIL("Buffer segment device id should less than device size\n");
    }
    OfLiteBufferCreate(context->device_contexts[device]->device_alloca, size,
                       context->buffer_segments + i);
  }
  context->buffer_segment_size = buffer_segment_size;

  OfLiteExecutableOperandSize(executable, &context->operand_size);
  context->operands = reinterpret_cast<OfLiteTensor**>(
      OfLiteMalloc(context->operand_size * sizeof(OfLiteTensor*)));
  for (size_t i = 0; i < context->operand_size; ++i) {
    const OfLiteTensorDef* tensor = nullptr;
    OfLiteExecutableOperand(executable, i, &tensor);
    OfLiteTensorDesc desc;
    OfLiteTensorDescCreateFromTensorDef(tensor, &desc);
    oneflow_lite_TensorDef_table_t flatcc_tensor =
        reinterpret_cast<oneflow_lite_TensorDef_table_t>(tensor);
    int32_t segment_id = oneflow_lite_TensorDef_segment_id(flatcc_tensor);
    int64_t segment_offset =
        oneflow_lite_TensorDef_segment_offset(flatcc_tensor);
    if (segment_id >= context->buffer_segment_size) {
      OFLITE_FAIL("Tensor segment id should less than buffer segment size\n");
    }
    OfLiteTensorCreateFromBuffer(desc, context->buffer_segments[segment_id],
                                 segment_offset, &(context->operands[i]));
  }

  size_t input_size = 0;
  OfLiteExecutableInputSize(executable, &input_size);
  OfLiteTensorSpanCreate(input_size, &context->inputs);
  for (size_t i = 0; i < input_size; ++i) {
    size_t input_index = -1;
    OfLiteExecutableInput(executable, i, &input_index);
    context->inputs->items[i] = context->operands[input_index];
  }
  size_t output_size = 0;
  OfLiteExecutableOutputSize(executable, &output_size);
  OfLiteTensorSpanCreate(output_size, &context->outputs);
  for (size_t i = 0; i < output_size; ++i) {
    size_t output_index = -1;
    OfLiteExecutableOutput(executable, i, &output_index);
    context->outputs->items[i] = context->operands[output_index];
  }

  size_t execution_unit_size = 0;
  OfLiteExecutableOpSize(executable, &execution_unit_size);
  context->execution_units = reinterpret_cast<OfLiteExecutionUnit**>(
      OfLiteMalloc(execution_unit_size * sizeof(OfLiteExecutionUnit*)));
  for (size_t i = 0; i < execution_unit_size; ++i) {
    const OfLiteOpDef* op = nullptr;
    OfLiteExecutableOp(executable, i, &op);
    OfLiteExecutionUnitCreate(
        op, context->operands, context->operand_size, context->device_contexts,
        context->device_context_size, context->execution_units + i);
  }
  context->execution_unit_size = execution_unit_size;
}

OFLITE_API void OfLiteExecutionContextCreate(
    const OfLiteExecutable* executable, const OfLiteExecutionOption& option,
    OfLiteExecutionContext** context) {
  *context = reinterpret_cast<OfLiteExecutionContext*>(
      OfLiteMalloc(sizeof(OfLiteExecutionContext)));
  OfLiteExecutionContextCreateImpl(executable, option, *context);
}

OFLITE_API void OfLiteExecutionContextDestory(OfLiteExecutionContext* context) {
  for (size_t i = 0; i < context->execution_unit_size; ++i) {
    OfLiteExecutionUnitDestory(context->execution_units[i]);
  }
  OfLiteFree(context->execution_units);

  OfLiteTensorSpanDestory(context->inputs);
  OfLiteTensorSpanDestory(context->outputs);
  for (size_t i = 0; i < context->operand_size; ++i) {
    OfLiteTensorDestory(context->operands[i]);
  }
  OfLiteFree(context->operands);
  for (size_t i = 0; i < context->buffer_segment_size; ++i) {
    OfLiteBufferDestory(context->buffer_segments[i]);
  }
  OfLiteFree(context->buffer_segments);
  for (size_t i = 0; i < context->device_context_size; ++i) {
    OfLiteDeviceContextDestory(context->device_contexts[i]);
  }
  OfLiteFree(context->device_contexts);
  for (size_t i = 0; i < context->driver_size; ++i) {
    OfLiteDriverDestory(context->drivers[i]);
  }
  OfLiteAllocaDestory(context->host_alloca);
}

OFLITE_API void OfLiteExecutionContextInputSize(
    const OfLiteExecutionContext* context, size_t* input_size) {
  *input_size = context->inputs->size;
}

OFLITE_API void OfLiteExecutionContextOutputSize(
    const OfLiteExecutionContext* context, size_t* output_size) {
  *output_size = context->outputs->size;
}

OFLITE_API void OfLiteExecutionContextInput(
    const OfLiteExecutionContext* context, size_t index, OfLiteTensor** input) {
  if (index >= context->inputs->size) {
    OFLITE_FAIL("input index is out of boundary\n");
  }
  *input = context->inputs->items[index];
}

OFLITE_API void OfLiteExecutionContextOutput(
    const OfLiteExecutionContext* context, size_t index,
    OfLiteTensor** output) {
  if (index >= context->outputs->size) {
    OFLITE_FAIL("output index is out of boundary\n");
  }
  *output = context->outputs->items[index];
}

OFLITE_API void OfLiteExecutionContextInvoke(OfLiteExecutionContext* context) {
  for (size_t i = 0; i < context->execution_unit_size; ++i) {
    OfLiteExecutionUnitInvoke(context->execution_units[i]);
  }
}
