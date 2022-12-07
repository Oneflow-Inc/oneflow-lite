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
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/span.h"

#define OFLITE_SPAN_IMPL(className, T) \
  OFLITE_API void className##SpanCreate(size_t size, className##Span** span) { \
    *span = reinterpret_cast<className##Span*>( \
      OfLiteMalloc(sizeof(className##Span))); \
    (*span)->items = reinterpret_cast<T**>( \
        OfLiteMalloc(size * sizeof(T*))); \
    (*span)->size = size; \
  } \
  OFLITE_API void className##SpanDestory(className##Span* span) { \
    OfLiteFree(span->items); \
    OfLiteFree(span); \
  }

OFLITE_SPAN_IMPL(OfLiteI32, int32_t);
OFLITE_SPAN_IMPL(OfLiteI64, int64_t);
OFLITE_SPAN_IMPL(OfLiteF32, float);
OFLITE_SPAN_IMPL(OfLiteDataType, OfLiteDataType);
OFLITE_SPAN_IMPL(OfLiteDims, OfLiteDims);
OFLITE_SPAN_IMPL(OfLiteString, OfLiteStringRef);
OFLITE_SPAN_IMPL(OfLiteTensor, OfLiteTensor);

#undef OFLITE_SPAN_IMPL
