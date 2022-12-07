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

#include <string.h>

#include "oneflow-lite/base/memory.h"

typedef struct OfLiteTensor {
  OfLiteTensorDesc desc;
  OfLiteBuffer* buffer;
  size_t buffer_offset;
  size_t buffer_length;
} OfLiteTensor;

OFLITE_API void OfLiteTensorCreate(const OfLiteTensorDesc& desc,
                                   OfLiteAlloca* alloca,
                                   OfLiteTensor** tensor) {
  size_t size = OfLiteDimsCount(desc.dims) * OfLiteDataTypeByteSize(desc.dtype);
  OfLiteTensor* tensor_impl =
      reinterpret_cast<OfLiteTensor*>(OfLiteMalloc(sizeof(OfLiteTensor)));
  memcpy(&tensor_impl->desc, &desc, sizeof(desc));
  OfLiteBufferCreate(alloca, size, &tensor_impl->buffer);
  tensor_impl->buffer_offset = 0;
  tensor_impl->buffer_length = size;
  *tensor = reinterpret_cast<OfLiteTensor*>(tensor_impl);
}

OFLITE_API void OfLiteTensorCreateFromBuffer(const OfLiteTensorDesc& desc,
                                             OfLiteBuffer* buffer,
                                             size_t offset,
                                             OfLiteTensor** tensor) {
  size_t size = OfLiteDimsCount(desc.dims) * OfLiteDataTypeByteSize(desc.dtype);
  if (size + offset > OfLiteBufferByteSize(buffer)) {
    OFLITE_FAIL("the buffer space is not enough\n");
  }
  OfLiteTensor* tensor_impl =
      reinterpret_cast<OfLiteTensor*>(OfLiteMalloc(sizeof(OfLiteTensor)));
  memcpy(&tensor_impl->desc, &desc, sizeof(desc));
  OfLiteBufferRetain(buffer);
  tensor_impl->buffer = buffer;
  tensor_impl->buffer_offset = offset;
  tensor_impl->buffer_length = size;
  *tensor = reinterpret_cast<OfLiteTensor*>(tensor_impl);
}

OFLITE_API void OfLiteTensorDestory(OfLiteTensor* tensor) {
  OfLiteTensor* tensor_impl = reinterpret_cast<OfLiteTensor*>(tensor);
  OfLiteBufferDestory(tensor_impl->buffer);
  OfLiteFree(tensor_impl);
}

OFLITE_API void OfLiteTensorDims(const OfLiteTensor* tensor, OfLiteDims* dims) {
  memcpy(dims, &tensor->desc.dims, sizeof(OfLiteDims));
}

OFLITE_API void OfLiteTensorDataType(const OfLiteTensor* tensor,
                                     OfLiteDataType* dtype) {
  *dtype = tensor->desc.dtype;
}

OFLITE_API void OfLiteTensorLayout(const OfLiteTensor* tensor,
                                   OfLiteLayout* layout) {
  *layout = tensor->desc.layout;
}

OFLITE_API void OfLiteTensorAlloca(const OfLiteTensor* tensor,
                                   OfLiteAlloca** alloca) {
  OfLiteBufferAlloca(tensor->buffer, alloca);
}

OFLITE_API void* OfLiteTensorData(const OfLiteTensor* tensor) {
  return OfLiteBufferBytes(tensor->buffer) + tensor->buffer_offset;
}

OFLITE_API size_t OfLiteTensorDataSize(const OfLiteTensor* tensor) {
  return tensor->buffer_length;
}

OFLITE_API bool OfLiteTensorIsHost(const OfLiteTensor* tensor) {
  OfLiteAlloca* alloca = nullptr;
  OfLiteBufferAlloca(tensor->buffer, &alloca);
  OfLiteMemType mem_type;
  OfLiteAllocaQueryMemType(alloca, &mem_type);
  return mem_type != OfLiteMemType_Device;
}

OFLITE_API OfLiteMemType OfLiteTensorMemType(const OfLiteTensor* tensor) {
  OfLiteAlloca* alloca = nullptr;
  OfLiteBufferAlloca(tensor->buffer, &alloca);
  OfLiteMemType mem_type;
  OfLiteAllocaQueryMemType(alloca, &mem_type);
  return mem_type;
}

OFLITE_API void OfLiteTensorSpanCreate(size_t size, OfLiteTensorSpan** span) {
  *span = reinterpret_cast<OfLiteTensorSpan*>(
      OfLiteMalloc(sizeof(OfLiteTensorSpan)));
  (*span)->items = reinterpret_cast<OfLiteTensor**>(
      OfLiteMalloc(size * sizeof(OfLiteTensor*)));
  (*span)->size = size;
}

OFLITE_API void OfLiteTensorSpanDestory(OfLiteTensorSpan* span) {
  OfLiteFree(span->items);
  OfLiteFree(span);
}
