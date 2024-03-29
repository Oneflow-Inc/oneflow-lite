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
#include "oneflow-lite/schemas/executable_generated.h"

OFLITE_API void OfLiteTensorDescCreateFromTensorDef(
    const OfLiteTensorDef* tensor, OfLiteTensorDesc* desc) {
  oneflow_lite_TensorDef_table_t flatcc_tensor =
      reinterpret_cast<oneflow_lite_TensorDef_table_t>(tensor);
  desc->dtype = OfLiteDataTypeConvertFromString(
      oneflow_lite_TensorDef_type(flatcc_tensor));
  desc->layout = OfLiteLayoutConvertFromString(
      oneflow_lite_TensorDef_layout(flatcc_tensor));
  flatbuffers_int64_vec_t sizes = oneflow_lite_TensorDef_sizes(flatcc_tensor);
  desc->dims.ndim = flatbuffers_int64_vec_len(sizes);
  if (!OfLiteDimsCheck(desc->dims)) {
    OFLITE_FAIL("Tensor sizes is too large, only supports up to 10D array\n");
  }
  for (size_t i = 0; i < desc->dims.ndim; ++i) {
    desc->dims.sizes[i] = flatbuffers_int64_vec_at(sizes, i);
  }
}

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

OFLITE_API void OfLiteTensorTensorDesc(const OfLiteTensor* tensor,
                                       const OfLiteTensorDesc** desc) {
  *desc = &tensor->desc;
}

OFLITE_API void OfLiteTensorDims(const OfLiteTensor* tensor,
                                 const OfLiteDims** dims) {
  *dims = &tensor->desc.dims;
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
