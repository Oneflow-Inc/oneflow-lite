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
#ifndef ONEFLOW_LITE_CORE_OPERATOR_H_
#define ONEFLOW_LITE_CORE_OPERATOR_H_

#include "oneflow-lite/core/span.h"
#include "oneflow-lite/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteOperator OfLiteOperator;
typedef struct OfLiteOpDef OfLiteOpDef;
typedef struct OfLiteDevice OfLiteDevice;

OFLITE_API void OfLiteOperatorCreate(OfLiteDevice* device,
                                     const OfLiteOpDef* def,
                                     OfLiteOperator** op);
OFLITE_API void OfLiteOperatorDestory(OfLiteOperator* op);

OFLITE_API void OfLiteOperatorCompute(OfLiteOperator* op,
                                      const OfLiteTensorSpan& inputs,
                                      const OfLiteTensorSpan& outputs);

typedef struct OfLiteOperatorVTable {
  void (*destory)(OfLiteOperator* op);
  void (*compute)(OfLiteOperator* op, const OfLiteTensorSpan& inputs,
                  const OfLiteTensorSpan& outputs);
} OfLiteOperatorVTable;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_OPERATOR_H_
