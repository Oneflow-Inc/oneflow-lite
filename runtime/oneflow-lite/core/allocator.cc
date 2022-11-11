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
  // TODO(): create allocator error
}

OFLITE_API void OfLiteAllocatorDestory(OfLiteAllocator* alloca) {
  reinterpret_cast<OfLiteAllocatorVTable*>(alloca)->destory(alloca);
}

OFLITE_API void OfLiteAllocatorMalloc(OfLiteAllocator* alloca, size_t size,
                                      size_t alignment, void** ptr) {
  reinterpret_cast<OfLiteAllocatorVTable*>(alloca)->malloc(alloca, size,
                                                           alignment, ptr);
}

OFLITE_API void OfLiteAllocatorFree(OfLiteAllocator* alloca, void* ptr) {
  reinterpret_cast<OfLiteAllocatorVTable*>(alloca)->free(alloca, ptr);
}

OFLITE_API void OfLiteAllocatorRegisterFactory(OfLiteDeviceId device,
                                               OfLiteAllocatorType type,
                                               OfLiteAllocatorFactory factory) {
  OfLiteAllocatorRegistry* registry = GetOfLiteAllocatorRegistry();
  if (registry->size == OFLITE_ALLOCATOR_COUNT_LIMIT) {
    // TODO(): failed to register allocator
    return;
  }
  OfLiteAllocatorFactoryItem* item = &registry->items[registry->size];
  item->device_id = device;
  item->type = type;
  item->factory = factory;
  registry->size += 1;
}
