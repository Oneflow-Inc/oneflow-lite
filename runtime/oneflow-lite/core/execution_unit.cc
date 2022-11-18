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
#include "oneflow-lite/core/execution_unit.h"

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/schemas/executable_generated.h"

void OfLiteExecutionUnitCreate(const OfLiteOpDef* op, OfLiteTensor** operands,
                               size_t operand_size,
                               OfLiteDeviceContext** device_contexts,
                               size_t device_context_size,
                               OfLiteExecutionUnit** execution_unit) {
  *execution_unit = reinterpret_cast<OfLiteExecutionUnit*>(
      OfLiteMalloc(sizeof(OfLiteExecutionUnit)));
  oneflow_lite_OpDef_table_t flatcc_op =
      reinterpret_cast<oneflow_lite_OpDef_table_t>(op);

  size_t device = oneflow_lite_OpDef_device(flatcc_op);
  if (device >= device_context_size) {
    // TODO(): Op device should less than device context size
  }
  (*execution_unit)->device_context = device_contexts[device];

  OfLitePopulateDeviceContext((*execution_unit)->device_context);
  OfLiteOperatorCreate(op, &((*execution_unit)->op));

  flatbuffers_int32_vec_t inputs = oneflow_lite_OpDef_inputs(flatcc_op);
  OfLiteTensorSpanCreate(flatbuffers_int32_vec_len(inputs),
                         &((*execution_unit)->inputs));
  for (size_t i = 0; i < flatbuffers_int32_vec_len(inputs); ++i) {
    size_t input = flatbuffers_int32_vec_at(inputs, i);
    if (input >= operand_size) {
      // TODO(): Op input index should less than operand size
    }
    (*execution_unit)->inputs->items[i] = operands[input];
  }
  flatbuffers_int32_vec_t outputs = oneflow_lite_OpDef_outputs(flatcc_op);
  OfLiteTensorSpanCreate(flatbuffers_int32_vec_len(outputs),
                         &((*execution_unit)->outputs));
  for (size_t i = 0; i < flatbuffers_int32_vec_len(outputs); ++i) {
    size_t output = flatbuffers_int32_vec_at(outputs, i);
    if (output >= operand_size) {
      // TODO(): Op output index should less than operand size
    }
    (*execution_unit)->outputs->items[i] = operands[output];
  }
}

void OfLiteExecutionUnitDestory(OfLiteExecutionUnit* execution_unit) {
  OfLiteOperatorDestory(execution_unit->op);
  OfLiteTensorSpanDestory(execution_unit->inputs);
  OfLiteTensorSpanDestory(execution_unit->outputs);
  OfLiteFree(execution_unit);
}

void OfLiteExecutionUnitInvoke(OfLiteExecutionUnit* execution_unit) {
  OfLitePopulateDeviceContext(execution_unit->device_context);
  OfLiteOperatorCompute(execution_unit->op, *execution_unit->inputs,
                        *execution_unit->outputs);
}
