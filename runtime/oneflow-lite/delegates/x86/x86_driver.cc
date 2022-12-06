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
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/driver.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/x86/x86_device.h"

typedef struct OfLiteX86Driver {
  OfLiteVTableHandle handle;
} OfLiteX86Driver;

void OfLiteX86DriverDestory(OfLiteDriver* driver) { OfLiteFree(driver); }

void OfLiteX86DriverQueryIdentifier(OfLiteDriver* driver,
                                    OfLiteStringRef* identifier) {
  *identifier = OfLiteStringRefCreate(OfLiteX86Identifier);
}

void OfLiteX86DriverCreateDevice(OfLiteDriver* driver, size_t ordinal,
                                 OfLiteDevice** device) {
  *device = OfLiteX86DeviceCreate(ordinal);
}

static OfLiteDriverVTable vtable = {
    .destory = OfLiteX86DriverDestory,
    .query_identifier = OfLiteX86DriverQueryIdentifier,
    .create_device = OfLiteX86DriverCreateDevice,
};

static OfLiteDriver* OfLiteX86DriverCreate() {
  OfLiteX86Driver* driver =
      reinterpret_cast<OfLiteX86Driver*>(OfLiteMalloc(sizeof(OfLiteX86Driver)));
  driver->handle.vtable = &vtable;
  return reinterpret_cast<OfLiteDriver*>(driver);
}

OFLITE_REGISTER_HOST_DRIVER(OfLiteX86Identifier, OfLiteX86DriverCreate);
