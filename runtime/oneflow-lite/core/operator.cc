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

OFLITE_API void OfLiteOperatorCreate(const OfLiteOpDef* def,
                                     OfLiteOperator** op) {}

OFLITE_API void OfLiteOperatorDestory(OfLiteOperator* op) {
  reinterpret_cast<OfLiteOperatorVTable*>(op)->destory(op);
}

OFLITE_API void OfLiteOperatorCompute(OfLiteOperator* op,
                                      const OfLiteTensorSpan& inputs,
                                      const OfLiteTensorSpan& outputs) {
  reinterpret_cast<OfLiteOperatorVTable*>(op)->compute(op, inputs, outputs);
}
