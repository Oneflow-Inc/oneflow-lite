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
#ifndef ONEFLOW_LITE_CORE_DRIVER_H_
#define ONEFLOW_LITE_CORE_DRIVER_H_

#include <stdint.h>
#include <stdlib.h>

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/stringref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteDriver OfLiteDriver;
typedef struct OfLiteDevice OfLiteDevice;

OFLITE_API void OfLiteDriverCreate(OfLiteStringRef type, OfLiteDriver** driver);

OFLITE_API void OfLiteDriverDestory(OfLiteDriver* driver);

OFLITE_API void OfLiteDriverQueryIdentifier(OfLiteDriver* driver, OfLiteStringRef* identifier);

OFLITE_API void OfLiteDriverCreateDevice(OfLiteDriver* driver, size_t ordinal, OfLiteDevice** device);

typedef struct OfLiteDriverVTable {
  void (*destory)(OfLiteDriver* driver);
  void (*query_identifier)(OfLiteDriver* driver, OfLiteStringRef* identifier);
  void (*create_device)(OfLiteDriver* driver, size_t ordinal, OfLiteDevice** device);
} OfLiteDriverVTable;

typedef OfLiteDriver* (*OfLiteDriverFactory)();

OFLITE_API void OfLiteDriverRegisterFactory(OfLiteStringRef type,
                                            OfLiteDriverFactory factory);

#define OFLITE_REGISTER_DRIVER(type, factory)                      \
  static int OFLITE_CAT(_oflite_driver_registry_, __COUNTER__)     \
      OFLITE_UNUSED = {((void)OfLiteDriverRegisterFactory(         \
                            OfLiteStringRefCreate(type), factory), \
                        0)};

#define OFLITE_REGISTER_HOST_DRIVER(type, factory) \
  OFLITE_REGISTER_DRIVER("host", factory)          \
  OFLITE_REGISTER_DRIVER(type, factory)

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_DRIVER_H_
