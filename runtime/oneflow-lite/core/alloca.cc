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

#include <assert.h>

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"

typedef struct OfLiteHostAlloca {
  OfLiteAllocaVTable* vtable;
} OfLiteHostAlloca;

static void OfLiteHostAllocaDestory(OfLiteAlloca* alloca) {
  OfLiteFree(alloca);
}

static void OfLiteHostAllocaMalloc(OfLiteAlloca* alloca, size_t size,
                                   void** ptr) {
  *ptr = OfLiteMalloc(size);
}

static void OfLiteHostAllocaAlignedAlloc(OfLiteAlloca* alloca, size_t alignment,
                                         size_t size, void** ptr) {
  *ptr = OfLiteAlignedAlloc(alignment, size);
}

static void OfLiteHostAllocaFree(OfLiteAlloca* alloca, void* ptr) {
  OfLiteFree(ptr);
}

OFLITE_API void OfLiteHostAllocaCreate(OfLiteAlloca** alloca) {
  static OfLiteAllocaVTable vtable = {
      .destory = OfLiteHostAllocaDestory,
      .malloc = OfLiteHostAllocaMalloc,
      .aligned_alloc = OfLiteHostAllocaAlignedAlloc,
      .free = OfLiteHostAllocaFree,
  };
  OfLiteHostAlloca* host_alloca = reinterpret_cast<OfLiteHostAlloca*>(
      OfLiteMalloc(sizeof(OfLiteHostAlloca)));
  host_alloca->vtable = &vtable;
  *alloca = reinterpret_cast<OfLiteAlloca*>(host_alloca);
}

OFLITE_API void OfLiteAllocaCreate(OfLiteDevice* device, OfLiteAllocaType type,
                                   OfLiteAlloca** alloca) {
  OfLiteDeviceCreateAlloca(device, type, alloca);
}

#define ALLOCA_VTABLE_CAST(alloca)       \
  reinterpret_cast<OfLiteAllocaVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(alloca)->vtable)

OFLITE_API void OfLiteAllocaDestory(OfLiteAlloca* alloca) {
  ALLOCA_VTABLE_CAST(alloca)->destory(alloca);
}

OFLITE_API void OfLiteAllocaMalloc(OfLiteAlloca* alloca, size_t size,
                                   void** ptr) {
  ALLOCA_VTABLE_CAST(alloca)->malloc(alloca, size, ptr);
}

OFLITE_API void OfLiteAllocaFree(OfLiteAlloca* alloca, void* ptr) {
  ALLOCA_VTABLE_CAST(alloca)->free(alloca, ptr);
}

OFLITE_API void OfLiteAllocaAlignedAlloc(OfLiteAlloca* alloca, size_t alignment,
                                         size_t size, void** ptr) {
  ALLOCA_VTABLE_CAST(alloca)->aligned_alloc(alloca, alignment, size, ptr);
}

#undef ALLOCA_VTABLE_CAST
