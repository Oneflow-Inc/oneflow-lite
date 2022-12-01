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
#ifndef ONEFLOW_LITE_CORE_EXECUTION_CONTEXT_H_
#define ONEFLOW_LITE_CORE_EXECUTION_CONTEXT_H_

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/core/executable.h"
#include "oneflow-lite/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteExecutionContext OfLiteExecutionContext;
typedef struct OfLiteExecutionOption {
  // TODO
} OfLiteExecutionOption;

OFLITE_API void OfLiteExecutionContextCreate(
    const OfLiteExecutable* executable, const OfLiteExecutionOption& option,
    OfLiteExecutionContext** context);

OFLITE_API void OfLiteExecutionContextDestory(OfLiteExecutionContext* context);

OFLITE_API void OfLiteExecutionContextInputSize(
    const OfLiteExecutionContext* context, size_t* input_size);

OFLITE_API void OfLiteExecutionContextOutputSize(
    const OfLiteExecutionContext* context, size_t* output_size);

OFLITE_API void OfLiteExecutionContextInput(
    const OfLiteExecutionContext* context, size_t index, OfLiteTensor** input);

OFLITE_API void OfLiteExecutionContextOutput(
    const OfLiteExecutionContext* context, size_t index, OfLiteTensor** output);

OFLITE_API void OfLiteExecutionContextInvoke(OfLiteExecutionContext* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_EXECUTION_CONTEXT_H_
