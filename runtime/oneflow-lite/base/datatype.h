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
#ifndef ONEFLOW_LITE_BASE_DATATYPE_H_
#define ONEFLOW_LITE_BASE_DATATYPE_H_

#include <cstdlib>

#include "oneflow-lite/base/common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum OfLiteDataType {
  OfLiteDataType_BEGIN = 0,
  OfLiteDataType_Bool,
  OfLiteDataType_I8,
  OfLiteDataType_I16,
  OfLiteDataType_I32,
  OfLiteDataType_I64,
  OfLiteDataType_U8,
  OfLiteDataType_U16,
  OfLiteDataType_U32,
  OfLiteDataType_U64,
  OfLiteDataType_F8,
  OfLiteDataType_F16,
  OfLiteDataType_F32,
  OfLiteDataType_F64,
  OfLiteDataType_BF16,

  OfLiteDataType_END,
};

inline bool OfLiteDataTypeCheck(OfLiteDataType dtype) {
  return dtype > OfLiteDataType_BEGIN && dtype < OfLiteDataType_END;
}
inline bool OfLiteDataTypeIsIntergral(OfLiteDataType dtype) {
  return dtype >= OfLiteDataType_Bool && dtype <= OfLiteDataType_U64;
}
inline bool OfLiteDataTypeIsFloating(OfLiteDataType dtype) {
  return dtype >= OfLiteDataType_F8 && dtype <= OfLiteDataType_BF16;
}

inline size_t OfLiteDataTypeByteSize(OfLiteDataType dtype) {
  switch (dtype) {
    case OfLiteDataType_Bool:
    case OfLiteDataType_I8:
    case OfLiteDataType_U8:
    case OfLiteDataType_F8:
      return 1;
    case OfLiteDataType_I16:
    case OfLiteDataType_U16:
    case OfLiteDataType_F16:
    case OfLiteDataType_BF16:
      return 2;
    case OfLiteDataType_I32:
    case OfLiteDataType_U32:
    case OfLiteDataType_F32:
      return 4;
    case OfLiteDataType_I64:
    case OfLiteDataType_U64:
    case OfLiteDataType_F64:
      return 8;
    default:
      return 0;
  }
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_BASE_DATATYPE_H_
