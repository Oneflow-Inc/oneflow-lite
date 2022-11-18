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
#ifndef ONEFLOW_LITE_CORE_DEVICE_CONTEXT_H_
#define ONEFLOW_LITE_CORE_DEVICE_CONTEXT_H_

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "oneflow-lite/core/allocator.h"
#include "oneflow-lite/core/device.h"

typedef struct OfLiteDeviceContext {
  OfLiteDevice* device;
  OfLiteAllocator* device_alloca;
  OfLiteAllocator* device_host_alloca;
} OfLiteDeviceContext;

OFLITE_API void OfLiteDeviceContextCreate(OfLiteStringRef device_type,
                                          size_t ordinal,
                                          OfLiteDeviceContext** context);

OFLITE_API void OfLiteDeviceContextDestory(OfLiteDeviceContext* context);

OFLITE_API void OfLitePopulateDeviceContext(
    OfLiteDeviceContext* device_context);

OFLITE_API void OfLiteObtainDeviceContext(OfLiteDeviceContext** device_context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_DEVICE_CONTEXT_H_
