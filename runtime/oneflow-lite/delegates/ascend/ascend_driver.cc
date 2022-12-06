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
#include "acl/acl.h"
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/driver.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/ascend/ascend_device.h"

typedef struct OfLiteAscendDriver {
  OfLiteVTableHandle handle;
} OfLiteAscendDriver;

void OfLiteAscendDriverDestory(OfLiteDriver* driver) {
  OfLiteFree(driver);
  // Finalize ascendcl
  aclFinalize();
}

void OfLiteAscendDriverQueryIdentifier(OfLiteDriver* driver,
                                       OfLiteStringRef* identifier) {
  *identifier = OfLiteStringRefCreate(OfLiteAscendIdentifier);
}

void OfLiteAscendDriverCreateDevice(OfLiteDriver* driver, size_t ordinal,
                                    OfLiteDevice** device) {
  *device = OfLiteAscendDeviceCreate(ordinal);
}

static OfLiteDriverVTable vtable = {
    .destory = OfLiteAscendDriverDestory,
    .query_identifier = OfLiteAscendDriverQueryIdentifier,
    .create_device = OfLiteAscendDriverCreateDevice,
};

static OfLiteDriver* OfLiteAscendDriverCreate() {
  // Initialize ascendcl
  aclInit(NULL);
  OfLiteAscendDriver* driver = reinterpret_cast<OfLiteAscendDriver*>(
      OfLiteMalloc(sizeof(OfLiteAscendDriver)));
  driver->handle.vtable = &vtable;
  return reinterpret_cast<OfLiteDriver*>(driver);
}

OFLITE_REGISTER_DRIVER(OfLiteAscendIdentifier, OfLiteAscendDriverCreate);
