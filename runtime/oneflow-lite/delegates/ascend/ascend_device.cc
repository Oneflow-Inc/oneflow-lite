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
#include "oneflow-lite/delegates/ascend/ascend_device.h"

#include "acl/acl_rt.h"
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/ascend/ascend_alloca.h"
#include "oneflow-lite/delegates/ascend/ascend_create_op.h"
#include "oneflow-lite/delegates/ascend/ascend_utils.h"

typedef struct OfLiteAscendDevice {
  OfLiteVTableHandle handle;
  size_t ordinal;
  aclrtContext context;
} OfLiteAscendDevice;

void OfLiteAscendDeviceDestory(OfLiteDevice* device) {
  ACL_CHECK(aclrtDestroyContext(
      reinterpret_cast<OfLiteAscendDevice*>(device)->context));
  OfLiteFree(device);
}

void OfLiteAscendDeviceQueryName(const OfLiteDevice* device,
                                 OfLiteStringRef* name) {
  *name = OfLiteStringRefCreate(OfLiteAscendIdentifier);
}

void OfLiteAscendDeviceQueryOrdinal(const OfLiteDevice* device,
                                    size_t* ordinal) {
  *ordinal = reinterpret_cast<const OfLiteAscendDevice*>(device)->ordinal;
}

void OfLiteAscendDeviceCreateEvent(OfLiteDevice* device, OfLiteEvent** event) {}

void OfLiteAscendDeviceCreateStream(OfLiteDevice* device,
                                    OfLiteStream** stream) {}

void OfLiteAscendDeviceCreateAlloca(OfLiteDevice* device,
                                    OfLiteMemType mem_type,
                                    OfLiteAlloca** alloca) {
  *alloca = OfLiteAscendAllocaCreate(device, mem_type);
}

void OfLiteAscendDeviceCreateOp(OfLiteDevice* device, const OfLiteOpDef* def,
                                OfLiteOperator** op) {
  *op = OfLiteAscendCreateOp(device, def);
}

void OfLiteAscendDeviceMalloc(OfLiteDevice* device, size_t size, void** ptr) {
  ACL_CHECK(aclrtMalloc(ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
}

void OfLiteAscendDeviceFree(OfLiteDevice* device, void* ptr) {
  ACL_CHECK(aclrtFree(ptr));
}

void OfLiteAscendDeviceMallocHost(OfLiteDevice* device, size_t size,
                                  void** ptr) {
  ACL_CHECK(aclrtMallocHost(ptr, size));
}

void OfLiteAscendDeviceFreeHost(OfLiteDevice* device, void* ptr) {
  ACL_CHECK(aclrtFreeHost(ptr));
}

static OfLiteDeviceVTable vtable = {
    .destory = OfLiteAscendDeviceDestory,
    .query_name = OfLiteAscendDeviceQueryName,
    .query_ordinal = OfLiteAscendDeviceQueryOrdinal,
    .create_event = OfLiteAscendDeviceCreateEvent,
    .create_stream = OfLiteAscendDeviceCreateStream,
    .create_alloca = OfLiteAscendDeviceCreateAlloca,
    .create_op = OfLiteAscendDeviceCreateOp,
    .malloc = OfLiteAscendDeviceMalloc,
    .free = OfLiteAscendDeviceFree,
    .malloc_host = OfLiteAscendDeviceMallocHost,
    .free_host = OfLiteAscendDeviceFreeHost,
};

OfLiteDevice* OfLiteAscendDeviceCreate(size_t ordinal) {
  OfLiteAscendDevice* device = reinterpret_cast<OfLiteAscendDevice*>(
      OfLiteMalloc(sizeof(OfLiteAscendDevice)));
  device->handle.vtable = &vtable;
  device->ordinal = ordinal;
  ACL_CHECK(aclrtSetDevice(ordinal));
  ACL_CHECK(aclrtCreateContext(&device->context, ordinal));
  return reinterpret_cast<OfLiteDevice*>(device);
}
