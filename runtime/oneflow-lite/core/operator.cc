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
#include "oneflow-lite/core/operator.h"
#include "oneflow-lite/core/device.h"

#include "oneflow-lite/core/vtable_handle.h"

OFLITE_API void OfLiteOperatorCreate(OfLiteDevice* device, const OfLiteOpDef* def,
                                     OfLiteOperator** op) {
  OfLiteDeviceCreateOp(device, def, op);
}

#define OP_VTABLE_CAST(op)                 \
  reinterpret_cast<OfLiteOperatorVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(op)->vtable)

OFLITE_API void OfLiteOperatorDestory(OfLiteOperator* op) {
  OP_VTABLE_CAST(op)->destory(op);
}

OFLITE_API void OfLiteOperatorCompute(OfLiteOperator* op,
                                      const OfLiteTensorSpan& inputs,
                                      const OfLiteTensorSpan& outputs) {
  OP_VTABLE_CAST(op)->compute(op, inputs, outputs);
}

#undef OP_VTABLE_CAST
