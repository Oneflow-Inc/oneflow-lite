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
#include "oneflow-lite/core/alloca.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"

typedef struct OfLiteGenericAlloca {
  OfLiteVTableHandle handle;
  OfLiteDevice* device;
} OfLiteGenericAlloca;

static void OfLiteGenericAllocaDestory(OfLiteAlloca* alloca) {
  delete reinterpret_cast<OfLiteGenericAlloca*>(alloca);
}

static void OfLiteGenericAllocaMalloc(OfLiteAlloca* alloca, size_t size,
                                         void** ptr) {
  OfLiteDevice* device =
      reinterpret_cast<OfLiteGenericAlloca*>(alloca)->device;
  OfLiteDeviceMalloc(device, size, ptr);
}

static void OfLiteGenericAllocaFree(OfLiteAlloca* alloca, void* ptr) {
  OfLiteDevice* device =
      reinterpret_cast<OfLiteGenericAlloca*>(alloca)->device;
  OfLiteDeviceFree(device, ptr);
}

static OfLiteAllocaVTable vtable = {
    .destory = OfLiteGenericAllocaDestory,
    .malloc = OfLiteGenericAllocaMalloc,
    .aligned_alloc = 0,
    .free = OfLiteGenericAllocaFree,
};

OFLITE_API OfLiteAlloca* OfLiteGenericAllocaCreate(OfLiteDevice* device) {
  OfLiteGenericAlloca* alloca = new OfLiteGenericAlloca;
  alloca->handle.vtable = &vtable;
  alloca->device = device;
  return reinterpret_cast<OfLiteAlloca*>(alloca);
}
