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

typedef struct OfLiteExecutionContext {
  OfLiteTensorSpan operands;
  OfLiteTensorSpan inputs;
  OfLiteTensorSpan outputs;

} OfLiteExecutionContext;

OFLITE_API void OfLiteExecutionContextCreate(
    const OfLiteExecutable* executable, const OfLiteExecutionOption& option,
    OfLiteExecutionContext** context) {}

OFLITE_API void OfLiteExecutionContextDestory(OfLiteExecutionContext* context) {
}

OFLITE_API void OfLiteExecutionContextInputs(
    const OfLiteExecutionContext* context, OfLiteTensorSpan* inputs) {}

OFLITE_API void OfLiteExecutionContextOutputs(
    const OfLiteExecutionContext* context, OfLiteTensorSpan* outputs) {}

OFLITE_API void OfLiteExecutionContextInvoke(OfLiteExecutionContext* context) {}
