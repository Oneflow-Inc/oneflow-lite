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
#ifndef ONEFLOW_LITE_DELEGATES_ASCEND_ASCEND_CREATE_OP_H_
#define ONEFLOW_LITE_DELEGATES_ASCEND_ASCEND_CREATE_OP_H_

#include "acl/acl_rt.h"
#include "oneflow-lite/core/device.h"
#include "oneflow-lite/core/operator.h"
#include "oneflow-lite/delegates/ascend/ascend_utils.h"
#include "oneflow-lite/schemas/executable_generated.h"

OfLiteOperator* OfLiteAscendCreateOp(OfLiteDevice* device,
                                     const OfLiteOpDef* def);

#define ASCEND_CREATE_OP(type)                                       \
  OfLiteOperator* OfLiteAscendCreate##type##Op(OfLiteDevice* device, \
                                               const OfLiteOpDef* def)

ASCEND_CREATE_OP(copy);
ASCEND_CREATE_OP(mlir_jit);

#endif  // ONEFLOW_LITE_DELEGATES_ASCEND_ASCEND_CREATE_OP_H_
