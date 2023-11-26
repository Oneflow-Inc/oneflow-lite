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
#include "oneflow-lite/core/device_context.h"

#include <thread>

#include "oneflow-lite/base/memory.h"

OFLITE_API void OfLiteDeviceContextCreate(OfLiteDriver* driver, size_t ordinal,
                                          OfLiteDeviceContext** context) {
  *context = reinterpret_cast<OfLiteDeviceContext*>(
      OfLiteMalloc(sizeof(OfLiteDeviceContext)));
  OfLiteDevice* device = nullptr;
  OfLiteAlloca* device_alloca = nullptr;
  OfLiteAlloca* device_host_alloca = nullptr;
  OfLiteDeviceCreate(driver, ordinal, &device);
  OfLiteAllocaCreate(device, OfLiteMemType_Device, &device_alloca);
  OfLiteAllocaCreate(device, OfLiteMemType_Device_Host, &device_host_alloca);
  (*context)->device = device;
  (*context)->device_alloca = device_alloca;
  (*context)->device_host_alloca = device_host_alloca;
}

OFLITE_API void OfLiteDeviceContextDestory(OfLiteDeviceContext* context) {
  OfLiteAllocaDestory(context->device_host_alloca);
  OfLiteAllocaDestory(context->device_alloca);
  OfLiteDeviceDestory(context->device);
  OfLiteFree(context);
}

static OfLiteDeviceContext** OfLiteObtainThreadLocalDeviceContext() {
  static thread_local OfLiteDeviceContext* device_context = nullptr;
  return &device_context;
}

OFLITE_API void OfLitePopulateDeviceContext(
    OfLiteDeviceContext* device_context) {
  *OfLiteObtainThreadLocalDeviceContext() = device_context;
}

OFLITE_API void OfLiteObtainDeviceContext(
    OfLiteDeviceContext** device_context) {
  *device_context = *OfLiteObtainThreadLocalDeviceContext();
}
