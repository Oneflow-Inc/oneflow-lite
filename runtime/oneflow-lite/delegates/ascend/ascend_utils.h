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
#ifndef ONEFLOW_LITE_DELAGATES_ASCEND_ASCEND_UTILS_H_
#define ONEFLOW_LITE_DELAGATES_ASCEND_ASCEND_UTILS_H_

#include "acl/acl_rt.h"
#include "oneflow-lite/base/common.h"

inline aclrtMemcpyKind OfLiteAscendComputeMemcpyKind(OfLiteMemType src_type,
                                                     OfLiteMemType dst_type) {
  if (src_type == OfLiteMemType_Device) {
    if (dst_type = OfLiteMemType_Device) {
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
    } else {
      return ACL_MEMCPY_DEVICE_TO_HOST;
    }
  } else {
    if (dst_type = OfLiteMemType_Device) {
      return ACL_MEMCPY_HOST_TO_DEVICE;
    } else {
      return ACL_MEMCPY_HOST_TO_HOST;
    }
  }
}

#define ACL_CHECK(status)                                        \
  if (status != ACL_SUCCESS) {                                   \
    OFLITE_FAIL("failed to call ACL runtime api: %d\n", status); \
  }

#define ATC_CHECK(status)                                     \
  if (status != ge::GRAPH_SUCCESS) {                          \
    OFLITE_FAIL("failed to call ATC graph api: %d\n", status) \
  }

#endif  // ONEFLOW_LITE_DELAGATES_ASCEND_ASCEND_UTILS_H_
