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
#include "oneflow-lite/base/datatype.h"

#include <string.h>

OFLITE_API OfLiteDataType OfLiteDataTypeConvertFromString(const char* dtype) {
  if (0 == strcmp(dtype, "bool")) {
    return OfLiteDataType_Bool;
  } else if (0 == strcmp(dtype, "i8")) {
    return OfLiteDataType_I8;
  } else if (0 == strcmp(dtype, "i16")) {
    return OfLiteDataType_I16;
  } else if (0 == strcmp(dtype, "i32")) {
    return OfLiteDataType_I32;
  } else if (0 == strcmp(dtype, "i64")) {
    return OfLiteDataType_I64;
  } else if (0 == strcmp(dtype, "u8")) {
    return OfLiteDataType_U8;
  } else if (0 == strcmp(dtype, "u16")) {
    return OfLiteDataType_U16;
  } else if (0 == strcmp(dtype, "u32")) {
    return OfLiteDataType_U32;
  } else if (0 == strcmp(dtype, "u64")) {
    return OfLiteDataType_U64;
  } else if (0 == strcmp(dtype, "f8")) {
    return OfLiteDataType_F8;
  } else if (0 == strcmp(dtype, "f16")) {
    return OfLiteDataType_F16;
  } else if (0 == strcmp(dtype, "f32")) {
    return OfLiteDataType_F32;
  } else if (0 == strcmp(dtype, "f64")) {
    return OfLiteDataType_F64;
  } else if (0 == strcmp(dtype, "bf16")) {
    return OfLiteDataType_BF16;
  } else {
    return OfLiteDataType_END;
  }
}
