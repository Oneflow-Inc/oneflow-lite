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
// #include "cuda.h"
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"

extern const OfLiteDeviceId OfLiteCUDADeviceId = 0;
static const char* OfLiteCUDADeviceName = "cuda";

typedef struct OfLiteCUDADevice {
  OfLiteVTableHandle handle;
  size_t ordinal;
} OfLiteCUDADevice;

void OfLiteCUDADeviceDestory(OfLiteDevice* device) { OfLiteFree(device); }

void OfLiteCUDADeviceQueryId(const OfLiteDevice* device, OfLiteDeviceId* id) {
  *id = OfLiteCUDADeviceId;
}

void OfLiteCUDADeviceQueryName(const OfLiteDevice* device,
                               OfLiteStringRef* name) {
  *name = OfLiteStringRefCreate(OfLiteCUDADeviceName);
}

void OfLiteCUDADeviceQueryOrdinal(const OfLiteDevice* device, size_t* ordinal) {
  *ordinal = reinterpret_cast<const OfLiteCUDADevice*>(device)->ordinal;
}

void OfLiteCUDADeviceCreateEvent(OfLiteDevice* device, OfLiteEvent** event) {}

void OfLiteCUDADeviceCreateStream(OfLiteDevice* device, OfLiteStream** stream) {
}

void OfLiteCUDADeviceMalloc(OfLiteDevice* device, size_t size, void** ptr) {
  OfLiteCUDADevice* cuda_device = reinterpret_cast<OfLiteCUDADevice*>(device);
  // int ordinal = -1;
  // cudaGetDevice(&ordinal);
  // cudaSetDevice(static_cast<int>(cuda_device->ordinal));
  // cudaMalloc(ptr, size);
  // cudaSetDevice(ordinal);
}

void OfLiteCUDADeviceFree(OfLiteDevice* device, void* ptr) {
  OfLiteCUDADevice* cuda_device = reinterpret_cast<OfLiteCUDADevice*>(device);
  // int ordinal = -1;
  // cudaGetDevice(&ordinal);
  // cudaSetDevice(static_cast<int>(cuda_device->ordinal));
  // cudaFree(ptr);
  // cudaSetDevice(ordinal);
}

void OfLiteCUDADeviceMallocHost(OfLiteDevice* device, size_t size, void** ptr) {
  OfLiteCUDADevice* cuda_device = reinterpret_cast<OfLiteCUDADevice*>(device);
  // int ordinal = -1;
  // cudaGetDevice(&ordinal);
  // cudaSetDevice(static_cast<int>(cuda_device->ordinal));
  // cudaMallocHost(ptr, size);
  // cudaSetDevice(ordinal);
}

void OfLiteCUDADeviceFreeHost(OfLiteDevice* device, void* ptr) {
  OfLiteCUDADevice* cuda_device = reinterpret_cast<OfLiteCUDADevice*>(device);
  // int ordinal = -1;
  // cudaGetDevice(&ordinal);
  // cudaSetDevice(static_cast<int>(cuda_device->ordinal));
  // cudaFreeHost(ptr);
  // cudaSetDevice(ordinal);
}

static OfLiteDeviceVTable vtable = {
    .destory = OfLiteCUDADeviceDestory,
    .query_id = OfLiteCUDADeviceQueryId,
    .query_name = OfLiteCUDADeviceQueryName,
    .query_ordinal = OfLiteCUDADeviceQueryOrdinal,
    .create_event = OfLiteCUDADeviceCreateEvent,
    .create_stream = OfLiteCUDADeviceCreateStream,
    .malloc = OfLiteCUDADeviceMalloc,
    .free = OfLiteCUDADeviceFree,
    .malloc_host = OfLiteCUDADeviceMallocHost,
    .free_host = OfLiteCUDADeviceFreeHost,
};

static OfLiteDevice* OfLiteCUDADeviceCreate(size_t ordinal) {
  OfLiteCUDADevice* device = reinterpret_cast<OfLiteCUDADevice*>(
      OfLiteMalloc(sizeof(OfLiteCUDADevice)));
  device->handle.vtable = &vtable;
  device->ordinal = ordinal;
  return reinterpret_cast<OfLiteDevice*>(device);
}

OFLITE_REGISTER_DEVICE(OfLiteCUDADeviceName, OfLiteCUDADeviceCreate);
