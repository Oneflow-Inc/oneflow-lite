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
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/ascend/ascend_create_op.h"

namespace {

typedef struct CopyOp {
  OfLiteVTableHandle handle;
} CopyOp;

void destory(OfLiteOperator* op) { OfLiteFree(op); }

aclrtMemcpyKind OfLiteAscendComputeMemcpyKind(OfLiteMemType src_type,
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

void compute(OfLiteOperator* op, const OfLiteTensorSpan& inputs,
             const OfLiteTensorSpan& outputs) {
  if (inputs.size != outputs.size || inputs.size != 1) {
    OFLITE_FAIL("the number of inputs and outputs should all be 1\n");
  }
  aclrtMemcpyKind kind =
      OfLiteAscendComputeMemcpyKind(OfLiteTensorMemType(inputs.items[0]),
                                    OfLiteTensorMemType(outputs.items[0]));
  size_t length = OfLiteTensorDataSize(outputs.items[0]);
  ACL_CHECK(aclrtMemcpy(OfLiteTensorData(outputs.items[0]), length,
                        OfLiteTensorData(inputs.items[0]), length, kind));
}

static OfLiteOperatorVTable vtable = {
    .destory = destory,
    .compute = compute,
};

}  // namespace

ASCEND_CREATE_OP(copy) {
  CopyOp* op = reinterpret_cast<CopyOp*>(OfLiteMalloc(sizeof(CopyOp)));
  op->handle.vtable = &vtable;
  return reinterpret_cast<OfLiteOperator*>(op);
}
