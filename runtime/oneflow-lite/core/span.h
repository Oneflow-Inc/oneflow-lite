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
#ifndef ONEFLOW_LITE_CORE_SPAN_H_
#define ONEFLOW_LITE_CORE_SPAN_H_

#include "oneflow-lite/base/datatype.h"
#include "oneflow-lite/base/dims.h"
#include "oneflow-lite/core/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define OFLITE_SPAN(className, T) \
  typedef struct className##Span { \
    T** items; \
    size_t size; \
  } className##Span; \
                      \
  OFLITE_API void className##SpanCreate(size_t size, className##Span** span); \
  OFLITE_API void className##SpanDestory(className##Span* span); \

OFLITE_SPAN(OfLiteI32, int32_t);
OFLITE_SPAN(OfLiteI64, int64_t);
OFLITE_SPAN(OfLiteF32, float);
OFLITE_SPAN(OfLiteDataType, OfLiteDataType);
OFLITE_SPAN(OfLiteDims, OfLiteDims);
OFLITE_SPAN(OfLiteString, OfLiteStringRef);
OFLITE_SPAN(OfLiteTensor, OfLiteTensor);

#undef OFLITE_SPAN

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_SPAN_H_
