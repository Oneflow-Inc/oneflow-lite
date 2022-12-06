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
#include "oneflow-lite/core/driver.h"
#include "oneflow-lite/core/device.h"

#include <assert.h>

#include "oneflow-lite/core/vtable_handle.h"

static const size_t OFLITE_DRIVER_COUNT_LIMIT = 64;
static const size_t OFLITE_DRIVER_TYPE_LENGTH_LIMIT = 128;

typedef struct OfLiteDriverFactoryItem {
  char type[OFLITE_DRIVER_TYPE_LENGTH_LIMIT];
  OfLiteDriverFactory factory;
} OfLiteDriverFactoryItem;

typedef struct OfLiteDriverRegistry {
  size_t size;
  OfLiteDriverFactoryItem items[OFLITE_DRIVER_COUNT_LIMIT];
} OfLiteDriverRegistry;

OfLiteDriverRegistry* GetOfLiteDriverRegistry() {
  static OfLiteDriverRegistry oflite_driver_registry{.size = 0};
  return &oflite_driver_registry;
}

OFLITE_API void OfLiteDriverCreate(OfLiteStringRef type,
                                   OfLiteDriver** driver) {
  OfLiteDriverRegistry* registry = GetOfLiteDriverRegistry();
  for (size_t i = 0; i < registry->size; ++i) {
    const OfLiteDriverFactoryItem& item = registry->items[i];
    if (OfLiteStringRefEqual(type, OfLiteStringRefCreate(item.type))) {
      *driver = item.factory();
      return;
    }
  }
  assert(false && "failed to create a driver");
}

#define DRIVER_VTABLE_CAST(driver)       \
  reinterpret_cast<OfLiteDriverVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(driver)->vtable)

OFLITE_API void OfLiteDriverDestory(OfLiteDriver* driver) {
  DRIVER_VTABLE_CAST(driver)->destory(driver);
}

void OfLiteDriverQueryIdentifier(OfLiteDriver* driver, OfLiteStringRef* identifier) {
  DRIVER_VTABLE_CAST(driver)->query_identifier(driver, identifier);
}

OFLITE_API void OfLiteDriverCreateDevice(OfLiteDriver* driver, size_t ordinal,
                                         OfLiteDevice** device) {
  DRIVER_VTABLE_CAST(driver)->create_device(driver, ordinal, device);
}

#undef DRIVER_VTABLE_CAST

OFLITE_API void OfLiteDriverRegisterFactory(OfLiteStringRef type,
                                            OfLiteDriverFactory factory) {
  assert(type.size < OFLITE_DRIVER_TYPE_LENGTH_LIMIT &&
         "the length of driver type name should less than 128");
  OfLiteDriverRegistry* registry = GetOfLiteDriverRegistry();
  assert(registry->size < OFLITE_DRIVER_COUNT_LIMIT);

  OfLiteDriverFactoryItem* item = &registry->items[registry->size];
  registry->size += 1;
  strncpy(item->type, type.data, type.size);
  item->type[type.size] = 0;
  item->factory = factory;
}
