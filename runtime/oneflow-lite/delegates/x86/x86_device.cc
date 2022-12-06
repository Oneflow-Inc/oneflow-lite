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
#include "oneflow-lite/delegates/x86/x86_device.h"
#include "oneflow-lite/delegates/x86/x86_alloca.h"

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"

typedef struct OfLiteX86Device {
  OfLiteVTableHandle handle;
  size_t ordinal;
} OfLiteX86Device;

void OfLiteX86DeviceDestory(OfLiteDevice* device) { OfLiteFree(device); }

void OfLiteX86DeviceQueryName(const OfLiteDevice* device,
                              OfLiteStringRef* name) {
  *name = OfLiteStringRefCreate(OfLiteX86Identifier);
}

void OfLiteX86DeviceQueryOrdinal(const OfLiteDevice* device, size_t* ordinal) {
  *ordinal = reinterpret_cast<const OfLiteX86Device*>(device)->ordinal;
}

void OfLiteX86DeviceCreateEvent(OfLiteDevice* device, OfLiteEvent** event) {}

void OfLiteX86DeviceCreateStream(OfLiteDevice* device, OfLiteStream** stream) {}

void OfLiteX86DeviceCreateAlloca(OfLiteDevice* device, OfLiteAllocaType alloca_type, OfLiteAlloca** alloca) {
  *alloca = OfLiteX86AllocaCreate(device, alloca_type);
}

void OfLiteX86DeviceMalloc(OfLiteDevice* device, size_t size, void** ptr) {
  *ptr = OfLiteMalloc(size);
}

void OfLiteX86DeviceFree(OfLiteDevice* device, void* ptr) { OfLiteFree(ptr); }

void OfLiteX86DeviceMallocHost(OfLiteDevice* device, size_t size, void** ptr) {
  OfLiteX86DeviceMalloc(device, size, ptr);
}

void OfLiteX86DeviceFreeHost(OfLiteDevice* device, void* ptr) {
  OfLiteX86DeviceFree(device, ptr);
}

static OfLiteDeviceVTable vtable = {
    .destory = OfLiteX86DeviceDestory,
    .query_name = OfLiteX86DeviceQueryName,
    .query_ordinal = OfLiteX86DeviceQueryOrdinal,
    .create_event = OfLiteX86DeviceCreateEvent,
    .create_stream = OfLiteX86DeviceCreateStream,
    .create_alloca = OfLiteX86DeviceCreateAlloca,
    .malloc = OfLiteX86DeviceMalloc,
    .free = OfLiteX86DeviceFree,
    .malloc_host = OfLiteX86DeviceMallocHost,
    .free_host = OfLiteX86DeviceFreeHost,
};

OfLiteDevice* OfLiteX86DeviceCreate(size_t ordinal) {
  OfLiteX86Device* device =
      reinterpret_cast<OfLiteX86Device*>(OfLiteMalloc(sizeof(OfLiteX86Device)));
  device->handle.vtable = &vtable;
  device->ordinal = ordinal;
  return reinterpret_cast<OfLiteDevice*>(device);
}
