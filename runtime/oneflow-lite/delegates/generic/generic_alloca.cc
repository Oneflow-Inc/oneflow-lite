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
#include "oneflow-lite/delegates/generic/generic_alloca.h"

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/alloca.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"

typedef struct OfLiteGenericAlloca {
  OfLiteVTableHandle handle;
  OfLiteDevice* device;
  OfLiteMemType mem_type;
} OfLiteGenericAlloca;

static void OfLiteGenericAllocaDestory(OfLiteAlloca* alloca) {
  OfLiteFree(alloca);
}

static void OfLiteGenericAllocaMalloc(OfLiteAlloca* alloca, size_t size,
                                      void** ptr) {
  OfLiteGenericAlloca* impl = reinterpret_cast<OfLiteGenericAlloca*>(alloca);
  if (impl->mem_type == OfLiteMemType_Device) {
    OfLiteDeviceMalloc(impl->device, size, ptr);
  } else {
    OfLiteDeviceMallocHost(impl->device, size, ptr);
  }
}

static void OfLiteGenericAllocaFree(OfLiteAlloca* alloca, void* ptr) {
  OfLiteGenericAlloca* impl = reinterpret_cast<OfLiteGenericAlloca*>(alloca);
  if (impl->mem_type == OfLiteMemType_Device) {
    OfLiteDeviceFree(impl->device, ptr);
  } else {
    OfLiteDeviceFreeHost(impl->device, ptr);
  }
}

static void OfLiteGenericAllocaQueryMemType(OfLiteAlloca* alloca,
                                            OfLiteMemType* type) {
  *type = reinterpret_cast<OfLiteGenericAlloca*>(alloca)->mem_type;
}

static OfLiteAllocaVTable vtable = {
    .destory = OfLiteGenericAllocaDestory,
    .malloc = OfLiteGenericAllocaMalloc,
    .aligned_alloc = 0,
    .free = OfLiteGenericAllocaFree,
    .query_mem_type = OfLiteGenericAllocaQueryMemType,
};

OFLITE_API OfLiteAlloca* OfLiteGenericAllocaCreate(OfLiteDevice* device,
                                                   OfLiteMemType mem_type) {
  OfLiteGenericAlloca* alloca = reinterpret_cast<OfLiteGenericAlloca*>(
      OfLiteMalloc(sizeof(OfLiteGenericAlloca)));
  alloca->handle.vtable = &vtable;
  alloca->device = device;
  alloca->mem_type = mem_type;
  return reinterpret_cast<OfLiteAlloca*>(alloca);
}
