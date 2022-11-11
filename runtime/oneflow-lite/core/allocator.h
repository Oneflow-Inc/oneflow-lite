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
#ifndef ONEFLOW_LITE_CORE_ALLOCATOR_H_
#define ONEFLOW_LITE_CORE_ALLOCATOR_H_

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/core/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteAllocator OfLiteAllocator;

enum OfLiteAllocatorType {
  OfLiteAllocatorType_Device,
  OfLiteAllocatorType_Host,
};

OFLITE_API void OfLiteAllocatorCreate(OfLiteDevice* device,
                                      OfLiteAllocatorType type,
                                      OfLiteAllocator** alloca);
OFLITE_API void OfLiteAllocatorDestory(OfLiteAllocator* alloca);

OFLITE_API void OfLiteAllocatorMalloc(OfLiteAllocator* alloca, size_t size,
                                      size_t alignment, void** ptr);
OFLITE_API void OfLiteAllocatorFree(OfLiteAllocator* alloca, void* ptr);

typedef struct OfLiteAllocatorVTable {
  void (*destory)(OfLiteAllocator* alloca);
  void (*malloc)(OfLiteAllocator* alloca, size_t size, size_t alignment,
                 void** ptr);
  void (*free)(OfLiteAllocator* alloca, void* ptr);
} OfLiteAllocatorVTable;

typedef OfLiteAllocator* (*OfLiteAllocatorFactory)(OfLiteDevice*);

OFLITE_API void OfLiteAllocatorRegisterFactory(OfLiteDeviceId device,
                                               OfLiteAllocatorType type,
                                               OfLiteAllocatorFactory factory);

#define OFLITE_REGISTER_ALLOCATOR(device, type, factory)          \
  static int OFLITE_CAT(_oflite_allocator_rrgistry_, __COUNTER__) \
      OFLITE_UNUSED = {                                           \
          ((void)OfLiteAllocatorRegisterFactory(device, type, factory), 0)};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_ALLOCATOR_H_
