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
#include "oneflow-lite/delegates/ascend/ascend_utils.h"

#include "oneflow-lite/core/device_context.h"

OfLiteAscendDevice* OfLiteAscendObtainDevice() {
  OfLiteDeviceContext* context;
  OfLiteObtainDeviceContext(&context);
  return reinterpret_cast<OfLiteAscendDevice*>(context->device);
}

aclrtMemcpyKind OfLiteAscendComputeMemcpyKind(OfLiteMemType src_type,
                                              OfLiteMemType dst_type) {
  if (src_type == OfLiteMemType_Device) {
    if (dst_type == OfLiteMemType_Device) {
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else {
      return ACL_MEMCPY_DEVICE_TO_HOST;
    }
  } else {
    if (dst_type == OfLiteMemType_Device) {
      return ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
      return ACL_MEMCPY_HOST_TO_HOST;
    }
  }
}
