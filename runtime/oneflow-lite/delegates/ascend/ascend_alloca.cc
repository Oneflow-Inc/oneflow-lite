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
#include "oneflow-lite/delegates/ascend/ascend_alloca.h"

#include "oneflow-lite/core/alloca.h"
#include "oneflow-lite/delegates/generic/generic_alloca.h"

OfLiteAlloca* OfLiteAscendAllocaCreate(OfLiteDevice* device, OfLiteAllocaType alloca_type) {
  if (alloca_type == OfLiteAllocaType_Device) {
    return OfLiteGenericAllocaCreate(device);
  }
  OfLiteAlloca* host_alloca = nullptr;
  OfLiteHostAllocaCreate(&host_alloca);
  return host_alloca;
}
