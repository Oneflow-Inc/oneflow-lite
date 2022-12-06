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
#include "oneflow-lite/core/vtable_handle.h"

static const size_t OFLITE_DRIVER_COUNT_LIMIT = 64;
extern const size_t OFLITE_DRIVER_TYPE_LENGTH_LIMIT = 128;

typedef struct OfLiteDriverRegisterEntry {
  char type[OFLITE_DRIVER_TYPE_LENGTH_LIMIT];
  OfLiteDriverFactory factory;
} OfLiteDriverRegisterEntry;

typedef struct OfLiteDriverRegistry {
  size_t size;
  OfLiteDriverRegisterEntry entries[OFLITE_DRIVER_COUNT_LIMIT];
} OfLiteDriverRegistry;

OfLiteDriverRegistry* GetOfLiteDriverRegistry() {
  static OfLiteDriverRegistry oflite_driver_registry{.size = 0};
  return &oflite_driver_registry;
}

OFLITE_API void OfLiteDriverCreate(OfLiteStringRef type,
                                   OfLiteDriver** driver) {
  OfLiteDriverRegistry* registry = GetOfLiteDriverRegistry();
  for (size_t i = 0; i < registry->size; ++i) {
    const OfLiteDriverRegisterEntry& entry = registry->entries[i];
    if (OfLiteStringRefEqual(type, OfLiteStringRefCreate(entry.type))) {
      *driver = entry.factory();
      return;
    }
  }
  OFLITE_FAIL("failed to create a driver for %s\n", type.data);
}

#define DRIVER_VTABLE_CAST(driver)       \
  reinterpret_cast<OfLiteDriverVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(driver)->vtable)

OFLITE_API void OfLiteDriverDestory(OfLiteDriver* driver) {
  DRIVER_VTABLE_CAST(driver)->destory(driver);
}

void OfLiteDriverQueryIdentifier(OfLiteDriver* driver,
                                 OfLiteStringRef* identifier) {
  DRIVER_VTABLE_CAST(driver)->query_identifier(driver, identifier);
}

OFLITE_API void OfLiteDriverCreateDevice(OfLiteDriver* driver, size_t ordinal,
                                         OfLiteDevice** device) {
  DRIVER_VTABLE_CAST(driver)->create_device(driver, ordinal, device);
}

#undef DRIVER_VTABLE_CAST

OFLITE_API void OfLiteDriverRegisterFactory(OfLiteStringRef type,
                                            OfLiteDriverFactory factory) {
  if (type.size >= OFLITE_DRIVER_TYPE_LENGTH_LIMIT) {
    OFLITE_FAIL("the length of driver type name should less than 128\n");
  }
  OfLiteDriverRegistry* registry = GetOfLiteDriverRegistry();
  if (registry->size >= OFLITE_DRIVER_COUNT_LIMIT) {
    OFLITE_FAIL("The number of drivers has reached the upper limit\n");
  }
  OfLiteDriverRegisterEntry* entry = &registry->entries[registry->size];
  registry->size += 1;
  strncpy(entry->type, type.data, type.size);
  entry->type[type.size] = 0;
  entry->factory = factory;
}
