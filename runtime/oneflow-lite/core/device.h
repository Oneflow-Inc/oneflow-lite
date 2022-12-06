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
#ifndef ONEFLOW_LITE_CORE_DEVICE_H_
#define ONEFLOW_LITE_CORE_DEVICE_H_

#include <stdint.h>
#include <stdlib.h>

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/alloca.h"
#include "oneflow-lite/core/driver.h"
#include "oneflow-lite/core/event.h"
#include "oneflow-lite/core/stream.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteDevice OfLiteDevice;

OFLITE_API void OfLiteDeviceCreate(OfLiteDriver* driver, size_t ordinal,
                                   OfLiteDevice** device);

OFLITE_API void OfLiteDeviceDestory(OfLiteDevice* device);

OFLITE_API void OfLiteDeviceQueryName(const OfLiteDevice* device,
                                      OfLiteStringRef* name);

OFLITE_API void OfLiteDeviceQueryOrdinal(const OfLiteDevice* device,
                                         size_t* ordinal);

OFLITE_API void OfLiteDeviceCreateEvent(OfLiteDevice* device,
                                        OfLiteEvent** event);

OFLITE_API void OfLiteDeviceCreateStream(OfLiteDevice* device,
                                         OfLiteStream** stream);

OFLITE_API void OfLiteDeviceCreateAlloca(OfLiteDevice* device, OfLiteAllocaType alloca_type, OfLiteAlloca** alloca);

OFLITE_API void OfLiteDeviceMalloc(OfLiteDevice* device, size_t size,
                                   void** ptr);

OFLITE_API void OfLiteDeviceFree(OfLiteDevice* device, void* ptr);

OFLITE_API void OfLiteDeviceMallocHost(OfLiteDevice* device, size_t size,
                                       void** ptr);

OFLITE_API void OfLiteDeviceFreeHost(OfLiteDevice* device, void* ptr);

typedef struct OfLiteDeviceVTable {
  void (*destory)(OfLiteDevice* device);
  void (*query_name)(const OfLiteDevice* device, OfLiteStringRef* name);
  void (*query_ordinal)(const OfLiteDevice* device, size_t* ordinal);
  void (*create_event)(OfLiteDevice* device, OfLiteEvent** event);
  void (*create_stream)(OfLiteDevice* device, OfLiteStream** stream);
  void (*create_alloca)(OfLiteDevice* device, OfLiteAllocaType alloca_type, OfLiteAlloca** alloca);
  void (*malloc)(OfLiteDevice* device, size_t size, void** ptr);
  void (*free)(OfLiteDevice* device, void* ptr);
  void (*malloc_host)(OfLiteDevice* device, size_t size, void** ptr);
  void (*free_host)(OfLiteDevice* device, void* ptr);
} OfLiteDeviceVTable;

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_DEVICE_H_
