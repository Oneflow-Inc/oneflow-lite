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

#include <assert.h>

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/vtable_handle.h"

typedef struct OfLiteHostAllocator {
  OfLiteAllocatorVTable* vtable;
} OfLiteHostAllocator;

static void OfLiteHostAllocatorDestory(OfLiteAllocator* alloca) {
  OfLiteFree(alloca);
}

static void OfLiteHostAllocatorMalloc(OfLiteAllocator* alloca, size_t size,
                                      void** ptr) {
  *ptr = OfLiteMalloc(size);
}

static void OfLiteHostAllocatorAlignedAlloc(OfLiteAllocator* alloca,
                                            size_t alignment, size_t size,
                                            void** ptr) {
  *ptr = OfLiteAlignedAlloc(alignment, size);
}

static void OfLiteHostAllocatorFree(OfLiteAllocator* alloca, void* ptr) {
  OfLiteFree(ptr);
}

OFLITE_API void OfLiteHostAllocatorCreate(OfLiteAllocator** alloca) {
  static OfLiteAllocatorVTable vtable = {
      .destory = OfLiteHostAllocatorDestory,
      .malloc = OfLiteHostAllocatorMalloc,
      .aligned_alloc = OfLiteHostAllocatorAlignedAlloc,
      .free = OfLiteHostAllocatorFree,
  };
  OfLiteHostAllocator* host_alloca = reinterpret_cast<OfLiteHostAllocator*>(
      OfLiteMalloc(sizeof(OfLiteHostAllocator)));
  host_alloca->vtable = &vtable;
  *alloca = reinterpret_cast<OfLiteAllocator*>(host_alloca);
}

static const size_t OFLITE_ALLOCATOR_COUNT_LIMIT = 64;

typedef struct OfLiteAllocatorFactoryItem {
  OfLiteDeviceId device_id;
  OfLiteAllocatorType type;
  OfLiteAllocatorFactory factory;
} OfLiteAllocatorFactoryItem;

typedef struct OfLiteAllocatorRegistry {
  size_t size;
  OfLiteAllocatorFactoryItem items[OFLITE_ALLOCATOR_COUNT_LIMIT];
} OfLiteAllocatorRegistry;

OfLiteAllocatorRegistry* GetOfLiteAllocatorRegistry() {
  static OfLiteAllocatorRegistry oflite_allocator_registry{.size = 0};
  return &oflite_allocator_registry;
}

OFLITE_API void OfLiteAllocatorCreate(OfLiteDevice* device,
                                      OfLiteAllocatorType type,
                                      OfLiteAllocator** alloca) {
  OfLiteDeviceId device_id;
  OfLiteDeviceQueryId(device, &device_id);
  OfLiteAllocatorRegistry* registry = GetOfLiteAllocatorRegistry();
  for (size_t i = 0; i < registry->size; ++i) {
    const OfLiteAllocatorFactoryItem& item = registry->items[i];
    if (item.device_id == device_id && item.type == type) {
      *alloca = item.factory(device);
      return;
    }
  }
  // fallback to host allocator for device host memory allocation
  if (type == OfLiteAllocatorType_Device_Host) {
    OfLiteHostAllocatorCreate(alloca);
  }
  // TODO(): create allocator error
}

#define ALLOCATOR_VTABLE_CAST(alloca)       \
  reinterpret_cast<OfLiteAllocatorVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(alloca)->vtable)

OFLITE_API void OfLiteAllocatorDestory(OfLiteAllocator* alloca) {
  ALLOCATOR_VTABLE_CAST(alloca)->destory(alloca);
}

OFLITE_API void OfLiteAllocatorMalloc(OfLiteAllocator* alloca, size_t size,
                                      void** ptr) {
  ALLOCATOR_VTABLE_CAST(alloca)->malloc(alloca, size, ptr);
}

OFLITE_API void OfLiteAllocatorFree(OfLiteAllocator* alloca, void* ptr) {
  ALLOCATOR_VTABLE_CAST(alloca)->free(alloca, ptr);
}

OFLITE_API void OfLiteAllocatorAlignedAlloc(OfLiteAllocator* alloca,
                                            size_t alignment, size_t size,
                                            void** ptr) {
  ALLOCATOR_VTABLE_CAST(alloca)->aligned_alloc(alloca, alignment, size, ptr);
}

#undef ALLOCATOR_VTABLE_CAST

OFLITE_API void OfLiteAllocatorRegisterFactory(OfLiteDeviceId device,
                                               OfLiteAllocatorType type,
                                               OfLiteAllocatorFactory factory) {
  OfLiteAllocatorRegistry* registry = GetOfLiteAllocatorRegistry();
  assert(registry->size < OFLITE_ALLOCATOR_COUNT_LIMIT &&
         "failed to register allocator");

  OfLiteAllocatorFactoryItem* item = &registry->items[registry->size];
  item->device_id = device;
  item->type = type;
  item->factory = factory;
  registry->size += 1;
}
