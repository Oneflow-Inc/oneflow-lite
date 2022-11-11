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
#include "oneflow-lite/core/allocator.h"
#include "oneflow-lite/core/device.h"

typedef struct OfLiteGenericAllocator {
  OfLiteAllocatorVTable* vtable;
  OfLiteDevice* device;
} OfLiteGenericAllocator;

static void OfLiteGenericAllocatorDestory(OfLiteAllocator* alloca) {
  delete reinterpret_cast<OfLiteGenericAllocator*>(alloca);
}

static void OfLiteGenericAllocatorMalloc(OfLiteAllocator* alloca, size_t size,
                                         size_t alignment, void** ptr) {
  OfLiteDevice* device =
      reinterpret_cast<OfLiteGenericAllocator*>(alloca)->device;
  OfLiteDeviceMalloc(device, size, ptr);
}

static void OfLiteGenericAllocatorFree(OfLiteAllocator* alloca, void* ptr) {
  OfLiteDevice* device =
      reinterpret_cast<OfLiteGenericAllocator*>(alloca)->device;
  OfLiteDeviceFree(device, ptr);
}

static OfLiteAllocatorVTable vtable = {
    .destory = OfLiteGenericAllocatorDestory,
    .malloc = OfLiteGenericAllocatorMalloc,
    .free = OfLiteGenericAllocatorFree,
};

OFLITE_API OfLiteAllocator* OfLiteGenericAllocatorCreate(OfLiteDevice* device) {
  OfLiteGenericAllocator* alloca = new OfLiteGenericAllocator;
  alloca->vtable = &vtable;
  alloca->device = device;
  return reinterpret_cast<OfLiteAllocator*>(alloca);
}
