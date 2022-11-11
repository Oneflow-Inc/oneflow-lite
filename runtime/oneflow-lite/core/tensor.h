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
#ifndef ONEFLOW_LITE_CORE_TENSOR_H_
#define ONEFLOW_LITE_CORE_TENSOR_H_

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/datatype.h"
#include "oneflow-lite/base/dims.h"
#include "oneflow-lite/base/layout.h"
#include "oneflow-lite/core/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteTensor OfLiteTensor;

typedef struct OfLiteTensorDesc {
  OfLiteDims dims;
  OfLiteDataType dtype;
  OfLiteLayout layout;
  size_t alignment;
} OfLiteTensorDesc;

typedef struct OfLiteTensorSpan {
  OfLiteTensor** vals;
  size_t size;
} OfLiteTensorSpan;

OFLITE_API void OfLiteTensorCreate(const OfLiteTensorDesc& desc,
                                   OfLiteAllocator* alloca,
                                   const OfLiteTensor** tensor);
OFLITE_API void OfLiteTensorDestory(OfLiteTensor* tensor);
OFLITE_API void OfLiteTensorDims(const OfLiteTensor* tensor, OfLiteDims* dims);
OFLITE_API void OfLiteTensorDataType(const OfLiteTensor* tensor,
                                     OfLiteDataType* dtype);
OFLITE_API void OfLiteTensorLayout(const OfLiteTensor* tensor,
                                   OfLiteLayout* layout);
OFLITE_API void OfLiteTensorAllocator(const OfLiteTensor* tensor,
                                      const OfLiteAllocator** alloca);

OFLITE_API void OfLiteTensorStorage(const OfLiteTensor* tensor,
                                    const void** storage);

OFLITE_API size_t OfLiteTensorSpanSize(const OfLiteTensorSpan& span);
OFLITE_API void OfLiteTensorSpanAt(const OfLiteTensorSpan& span, size_t index,
                                   const OfLiteTensor** tensor);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_TENSOR_H_