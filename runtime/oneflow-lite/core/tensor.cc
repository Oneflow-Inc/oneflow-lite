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
#include "oneflow-lite/core/tensor.h"

#include <cstring>

typedef struct OfLiteTensorImpl {
  OfLiteTensorDesc desc;
  OfLiteAllocator* alloca;
  void* storage;
} OfLiteTensorImpl;

OFLITE_API void OfLiteTensorCreate(const OfLiteTensorDesc& desc,
                                   OfLiteAllocator* alloca,
                                   const OfLiteTensor** tensor) {
  size_t size = OfLiteDimsCount(desc.dims) * OfLiteDataTypeByteSize(desc.dtype);
  OfLiteTensorImpl* tensor_impl = new OfLiteTensorImpl;
  memcpy(&tensor_impl->desc, &desc, sizeof(desc));
  tensor_impl->alloca = alloca;
  OfLiteAllocatorMalloc(alloca, size, desc.alignment, &tensor_impl->storage);
  *tensor = reinterpret_cast<OfLiteTensor*>(tensor_impl);
}

OFLITE_API void OfLiteTensorDestory(OfLiteTensor* tensor) {
  OfLiteTensorImpl* tensor_impl = reinterpret_cast<OfLiteTensorImpl*>(tensor);
  OfLiteAllocatorFree(tensor_impl->alloca, tensor_impl->storage);
  delete tensor_impl;
}

OFLITE_API void OfLiteTensorDims(const OfLiteTensor* tensor, OfLiteDims* dims) {
  memcpy(dims, &reinterpret_cast<const OfLiteTensorImpl*>(tensor)->desc.dims,
         sizeof(OfLiteDims));
}

OFLITE_API void OfLiteTensorDataType(const OfLiteTensor* tensor,
                                     OfLiteDataType* dtype) {
  *dtype = reinterpret_cast<const OfLiteTensorImpl*>(tensor)->desc.dtype;
}

OFLITE_API void OfLiteTensorLayout(const OfLiteTensor* tensor,
                                   OfLiteLayout* layout) {
  *layout = reinterpret_cast<const OfLiteTensorImpl*>(tensor)->desc.layout;
}

OFLITE_API void OfLiteTensorAllocator(const OfLiteTensor* tensor,
                                      const OfLiteAllocator** alloca) {
  *alloca = reinterpret_cast<const OfLiteTensorImpl*>(tensor)->alloca;
}

OFLITE_API void OfLiteTensorStorage(const OfLiteTensor* tensor,
                                    const void** storage) {
  *storage = reinterpret_cast<const OfLiteTensorImpl*>(tensor)->storage;
}

OFLITE_API size_t OfLiteTensorSpanSize(const OfLiteTensorSpan& span) {
  return span.size;
}
OFLITE_API void OfLiteTensorSpanAt(const OfLiteTensorSpan& span, size_t index,
                                   const OfLiteTensor** tensor) {
  if (index >= span.size) {
    // TODO
  }
  *tensor = span.vals[index];
}
