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
#ifndef ONEFLOW_LITE_CORE_ALLOCA_H_
#define ONEFLOW_LITE_CORE_ALLOCA_H_

#include <stdint.h>
#include <stdlib.h>

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/stringref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteAlloca OfLiteAlloca;
typedef struct OfLiteDevice OfLiteDevice;

OFLITE_API void OfLiteHostAllocaCreate(OfLiteAlloca** alloca);

typedef enum OfLiteMemType {
  OfLiteMemType_Host,
  OfLiteMemType_Device,
  // Page-locked host memory and accessible to the device
  OfLiteMemType_Device_Host,
} OfLiteMemType;

OFLITE_API void OfLiteAllocaCreate(OfLiteDevice* device, OfLiteMemType type,
                                   OfLiteAlloca** alloca);

OFLITE_API void OfLiteAllocaDestory(OfLiteAlloca* alloca);

OFLITE_API void OfLiteAllocaMalloc(OfLiteAlloca* alloca, size_t size,
                                   void** ptr);
OFLITE_API void OfLiteAllocaFree(OfLiteAlloca* alloca, void* ptr);

OFLITE_API void OfLiteAllocaAlignedAlloc(OfLiteAlloca* alloca, size_t alignment,
                                         size_t size, void** ptr);

OFLITE_API void OfLiteAllocaQueryMemType(OfLiteAlloca* alloca, OfLiteMemType* type);

typedef struct OfLiteAllocaVTable {
  void (*destory)(OfLiteAlloca* alloca);
  void (*malloc)(OfLiteAlloca* alloca, size_t size, void** ptr);
  void (*aligned_alloc)(OfLiteAlloca* alloca, size_t alignment, size_t size,
                        void** ptr);
  void (*free)(OfLiteAlloca* alloca, void* ptr);
  void (*query_mem_type)(OfLiteAlloca* alloca, OfLiteMemType* type);
} OfLiteAllocaVTable;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_ALLOCA_H_
