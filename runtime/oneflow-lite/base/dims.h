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
#ifndef ONEFLOW_LITE_BASE_DIMS_H_
#define ONEFLOW_LITE_BASE_DIMS_H_

#include <cstdlib>

#include "oneflow-lite/base/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

static const size_t OFLITE_MAX_DIMS_NUM = 10;

typedef struct OfLiteDims {
  int64_t sizes[OFLITE_MAX_DIMS_NUM];
  size_t ndim;
} OfLiteDims;

inline bool OfLiteDimsCheck(const OfLiteDims& dims) {
  return dims.ndim < OFLITE_MAX_DIMS_NUM;
}

OFLITE_API int64_t OfLiteDimsCount(const OfLiteDims& dims, size_t start = 0,
                                   size_t end = OFLITE_MAX_DIMS_NUM);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_BASE_DIMS_H_
