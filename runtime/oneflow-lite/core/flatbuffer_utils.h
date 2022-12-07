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
#ifndef ONEFLOW_LITE_CORE_FLATBUFFER_UTILS_H_
#define ONEFLOW_LITE_CORE_FLATBUFFER_UTILS_H_

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/stringref.h"
#include "oneflow-lite/core/span.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteOpDef OfLiteOpDef;
typedef struct OfLiteAttrDefVec OfLiteAttrDefVec;
typedef struct OfLiteAttrDef OfLiteAttrDef;
typedef struct OfLiteAttrDefValue OfLiteAttrDefValue;

OFLITE_API const OfLiteAttrDefVec* OfLiteOpDefQueryAttrs(const OfLiteOpDef* def);

OFLITE_API const OfLiteAttrDef* OfLiteOpDefQueryAttr(const OfLiteOpDef* def, int index);
OFLITE_API const OfLiteAttrDef* OfLiteOpDefQueryAttrByName(const OfLiteOpDef* def, OfLiteStringRef name);

OFLITE_API OfLiteStringRef OfLiteAttrDefQueryName(const OfLiteAttrDef* def);

OFLITE_API OfLiteStringRef OfLiteAttrDefQueryType(const OfLiteAttrDef* def);

#define OFLITE_ATTRDEF_QUERY_VALUE(return_type, as_type) \
  OFLITE_API return_type OfLiteAttrDefQueryValue_As##as_type(const OfLiteAttrDef* def); \
  OFLITE_API return_type OfLiteOpDefQueryAttrValue_As##as_type(const OfLiteOpDef* def, int index); \
  OFLITE_API return_type OfLiteOpDefQueryAttrValueByName_As##as_type(const OfLiteOpDef* def, OfLiteStringRef name);

OFLITE_ATTRDEF_QUERY_VALUE(int32_t, I32);
OFLITE_ATTRDEF_QUERY_VALUE(int64_t, I64);
OFLITE_ATTRDEF_QUERY_VALUE(bool, Bool);
OFLITE_ATTRDEF_QUERY_VALUE(float, F32);
OFLITE_ATTRDEF_QUERY_VALUE(double, F64);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteStringRef, String);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDims, Shape);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDims, Stride);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDataType, DataType);

OFLITE_ATTRDEF_QUERY_VALUE(OfLiteI32Span, I32s);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteI64Span, I64s);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteF32Span, F32s);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteStringSpan, Strs);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDimsSpan, Shapes);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDimsSpan, Strides);
OFLITE_ATTRDEF_QUERY_VALUE(OfLiteDataTypeSpan, DataTypes);

#undef OFLITE_ATTRDEF_QUERY_VALUE

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_FLATBUFFER_UTILS_H_
