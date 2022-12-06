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
#include "oneflow-lite/core/device.h"

#include "oneflow-lite/core/driver.h"
#include "oneflow-lite/core/vtable_handle.h"

OFLITE_API void OfLiteDeviceCreate(OfLiteDriver* driver, size_t ordinal,
                                   OfLiteDevice** device) {
  OfLiteDriverCreateDevice(driver, ordinal, device);
}

#define DEVICE_VTABLE_CAST(device)       \
  reinterpret_cast<OfLiteDeviceVTable*>( \
      reinterpret_cast<const OfLiteVTableHandle*>(device)->vtable)

OFLITE_API void OfLiteDeviceDestory(OfLiteDevice* device) {
  DEVICE_VTABLE_CAST(device)->destory(device);
}

OFLITE_API void OfLiteDeviceQueryName(const OfLiteDevice* device,
                                      OfLiteStringRef* name) {
  DEVICE_VTABLE_CAST(device)->query_name(device, name);
}

OFLITE_API void OfLiteDeviceQueryOrdinal(const OfLiteDevice* device,
                                         size_t* ordinal) {
  DEVICE_VTABLE_CAST(device)->query_ordinal(device, ordinal);
}

OFLITE_API void OfLiteDeviceCreateEvent(OfLiteDevice* device,
                                        OfLiteEvent** event) {
  DEVICE_VTABLE_CAST(device)->create_event(device, event);
}

OFLITE_API void OfLiteDeviceCreateStream(OfLiteDevice* device,
                                         OfLiteStream** stream) {
  DEVICE_VTABLE_CAST(device)->create_stream(device, stream);
}

OFLITE_API void OfLiteDeviceCreateAlloca(OfLiteDevice* device,
                                         OfLiteAllocaType alloca_type,
                                         OfLiteAlloca** alloca) {
  DEVICE_VTABLE_CAST(device)->create_alloca(device, alloca_type, alloca);
}

OFLITE_API void OfLiteDeviceCreateOp(OfLiteDevice* device,
                                     const OfLiteOpDef* def,
                                     OfLiteOperator** op) {
  DEVICE_VTABLE_CAST(device)->create_op(device, def, op);
}

OFLITE_API void OfLiteDeviceMalloc(OfLiteDevice* device, size_t size,
                                   void** ptr) {
  DEVICE_VTABLE_CAST(device)->malloc(device, size, ptr);
}

OFLITE_API void OfLiteDeviceFree(OfLiteDevice* device, void* ptr) {
  DEVICE_VTABLE_CAST(device)->free(device, ptr);
}

OFLITE_API void OfLiteDeviceMallocHost(OfLiteDevice* device, size_t size,
                                       void** ptr) {
  DEVICE_VTABLE_CAST(device)->malloc_host(device, size, ptr);
}

OFLITE_API void OfLiteDeviceFreeHost(OfLiteDevice* device, void* ptr) {
  DEVICE_VTABLE_CAST(device)->free_host(device, ptr);
}

#undef DEVICE_VTABLE_CAST
