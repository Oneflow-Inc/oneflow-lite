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
#include "oneflow-lite/core/flatbuffer_utils.h"

#include "oneflow-lite/schemas/attributes/bool_generated.h"
#include "oneflow-lite/schemas/attributes/f32_generated.h"
#include "oneflow-lite/schemas/attributes/f32s_generated.h"
#include "oneflow-lite/schemas/attributes/f64_generated.h"
#include "oneflow-lite/schemas/attributes/i32_generated.h"
#include "oneflow-lite/schemas/attributes/i32s_generated.h"
#include "oneflow-lite/schemas/attributes/i64_generated.h"
#include "oneflow-lite/schemas/attributes/i64s_generated.h"
#include "oneflow-lite/schemas/attributes/shape_generated.h"
#include "oneflow-lite/schemas/attributes/shapes_generated.h"
#include "oneflow-lite/schemas/attributes/str_generated.h"
#include "oneflow-lite/schemas/attributes/strs_generated.h"
#include "oneflow-lite/schemas/executable_generated.h"

OFLITE_API void OfLiteOpDefName(const OfLiteOpDef* def, OfLiteStringRef* name) {
  flatbuffers_string_t flatc_name = oneflow_lite_OpDef_name(
      reinterpret_cast<const oneflow_lite_OpDef_table_t>(def));
  *name = OfLiteStringRef{flatc_name, flatbuffers_string_len(flatc_name)};
}

OFLITE_API const OfLiteAttrDefVec* OfLiteOpDefQueryAttrs(
    const OfLiteOpDef* def) {
  oneflow_lite_AttrDef_vec_t flatc_attrs = oneflow_lite_OpDef_attrs(
      reinterpret_cast<const oneflow_lite_OpDef_table_t>(def));
  return reinterpret_cast<const OfLiteAttrDefVec*>(flatc_attrs);
}

OFLITE_API const OfLiteAttrDef* OfLiteOpDefQueryAttr(const OfLiteOpDef* def,
                                                     int index) {
  oneflow_lite_AttrDef_vec_t flatc_attrs = oneflow_lite_OpDef_attrs(
      reinterpret_cast<const oneflow_lite_OpDef_table_t>(def));
  if (index >= oneflow_lite_AttrDef_vec_len(flatc_attrs)) {
    OFLITE_FAIL("index is out of boundary\n");
  }
  return reinterpret_cast<const OfLiteAttrDef*>(
      oneflow_lite_AttrDef_vec_at(flatc_attrs, index));
}

OFLITE_API const OfLiteAttrDef* OfLiteOpDefQueryAttrByName(
    const OfLiteOpDef* def, OfLiteStringRef name) {
  oneflow_lite_AttrDef_vec_t flatc_attrs = oneflow_lite_OpDef_attrs(
      reinterpret_cast<const oneflow_lite_OpDef_table_t>(def));
  for (int i = 0; i < oneflow_lite_AttrDef_vec_len(flatc_attrs); ++i) {
    oneflow_lite_AttrDef_table_t flatc_attr =
        oneflow_lite_AttrDef_vec_at(flatc_attrs, i);
    flatbuffers_string_t flatc_attr_key = oneflow_lite_AttrDef_key(flatc_attr);
    if (OfLiteStringRefEqual(
            name, OfLiteStringRef{flatc_attr_key,
                                  flatbuffers_string_len(flatc_attr_key)})) {
      return reinterpret_cast<const OfLiteAttrDef*>(flatc_attr);
    }
  }
  OFLITE_FAIL("failed to look up attribute %s\n", name.data);
  return nullptr;
}

OFLITE_API OfLiteStringRef OfLiteAttrDefQueryName(const OfLiteAttrDef* def) {
  flatbuffers_string_t name = oneflow_lite_AttrDef_key(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  return OfLiteStringRef{name, flatbuffers_string_len(name)};
}

OFLITE_API OfLiteStringRef OfLiteAttrDefQueryType(const OfLiteAttrDef* def) {
  flatbuffers_string_t type = oneflow_lite_AttrDef_type(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  return OfLiteStringRef{type, flatbuffers_string_len(type)};
}

OFLITE_API int32_t OfLiteAttrDefQueryValue_AsI32(const OfLiteAttrDef* def) {
  flatbuffers_int8_vec_t value = oneflow_lite_AttrDef_value(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  if (flatcc_verify_ok != oneflow_lite_I32Def_verify_as_root(
                              value, flatbuffers_int8_vec_len(value))) {
    OFLITE_FAIL("failed to verify buffer as I32Def root\n");
  }
  return oneflow_lite_I32Def_value(oneflow_lite_I32Def_as_root(value));
}

OFLITE_API int32_t OfLiteOpDefQueryAttrValue_AsI32(const OfLiteOpDef* def,
                                                   int index) {
  return OfLiteAttrDefQueryValue_AsI32(OfLiteOpDefQueryAttr(def, index));
}

OFLITE_API int32_t OfLiteOpDefQueryAttrValueByName_AsI32(const OfLiteOpDef* def,
                                                         OfLiteStringRef name) {
  return OfLiteAttrDefQueryValue_AsI32(OfLiteOpDefQueryAttrByName(def, name));
}

OFLITE_API int64_t OfLiteAttrDefQueryValue_AsI64(const OfLiteAttrDef* def) {
  flatbuffers_int8_vec_t value = oneflow_lite_AttrDef_value(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  if (flatcc_verify_ok != oneflow_lite_I64Def_verify_as_root(
                              value, flatbuffers_int8_vec_len(value))) {
    OFLITE_FAIL("failed to verify buffer as I64Def root\n");
  }
  return oneflow_lite_I64Def_value(oneflow_lite_I64Def_as_root(value));
}

OFLITE_API int64_t OfLiteOpDefQueryAttrValue_AsI64(const OfLiteOpDef* def,
                                                   int index) {
  return OfLiteAttrDefQueryValue_AsI64(OfLiteOpDefQueryAttr(def, index));
}

OFLITE_API int64_t OfLiteOpDefQueryAttrValueByName_AsI64(const OfLiteOpDef* def,
                                                         OfLiteStringRef name) {
  return OfLiteAttrDefQueryValue_AsI64(OfLiteOpDefQueryAttrByName(def, name));
}

OFLITE_API float OfLiteAttrDefQueryValue_AsF32(const OfLiteAttrDef* def) {
  flatbuffers_int8_vec_t value = oneflow_lite_AttrDef_value(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  if (flatcc_verify_ok != oneflow_lite_F32Def_verify_as_root(
                              value, flatbuffers_int8_vec_len(value))) {
    OFLITE_FAIL("failed to verify buffer as F32Def root\n");
  }
  return oneflow_lite_F32Def_value(oneflow_lite_F32Def_as_root(value));
}

OFLITE_API float OfLiteOpDefQueryAttrValue_AsF32(const OfLiteOpDef* def,
                                                 int index) {
  return OfLiteAttrDefQueryValue_AsF32(OfLiteOpDefQueryAttr(def, index));
}

OFLITE_API float OfLiteOpDefQueryAttrValueByName_AsF32(const OfLiteOpDef* def,
                                                       OfLiteStringRef name) {
  return OfLiteAttrDefQueryValue_AsF32(OfLiteOpDefQueryAttrByName(def, name));
}

OFLITE_API double OfLiteAttrDefQueryValue_AsF64(const OfLiteAttrDef* def) {
  flatbuffers_int8_vec_t value = oneflow_lite_AttrDef_value(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  if (flatcc_verify_ok != oneflow_lite_F64Def_verify_as_root(
                              value, flatbuffers_int8_vec_len(value))) {
    OFLITE_FAIL("failed to verify buffer as F64Def root\n");
  }
  return oneflow_lite_F64Def_value(oneflow_lite_F64Def_as_root(value));
}

OFLITE_API double OfLiteOpDefQueryAttrValue_AsF64(const OfLiteOpDef* def,
                                                  int index) {
  return OfLiteAttrDefQueryValue_AsF64(OfLiteOpDefQueryAttr(def, index));
}

OFLITE_API double OfLiteOpDefQueryAttrValueByName_AsF64(const OfLiteOpDef* def,
                                                        OfLiteStringRef name) {
  return OfLiteAttrDefQueryValue_AsF64(OfLiteOpDefQueryAttrByName(def, name));
}

OFLITE_API OfLiteStringRef
OfLiteAttrDefQueryValue_AsString(const OfLiteAttrDef* def) {
  flatbuffers_int8_vec_t value = oneflow_lite_AttrDef_value(
      reinterpret_cast<const oneflow_lite_AttrDef_table_t>(def));
  if (flatcc_verify_ok != oneflow_lite_StringDef_verify_as_root(
                              value, flatbuffers_int8_vec_len(value))) {
    OFLITE_FAIL("failed to verify buffer as StringDef root\n");
  }
  flatbuffers_string_t s =
      oneflow_lite_StringDef_value(oneflow_lite_StringDef_as_root(value));
  return OfLiteStringRef{s, flatbuffers_string_len(s)};
}

OFLITE_API OfLiteStringRef
OfLiteOpDefQueryAttrValue_AsString(const OfLiteOpDef* def, int index) {
  return OfLiteAttrDefQueryValue_AsString(OfLiteOpDefQueryAttr(def, index));
}

OFLITE_API OfLiteStringRef OfLiteOpDefQueryAttrValueByName_AsString(
    const OfLiteOpDef* def, OfLiteStringRef name) {
  return OfLiteAttrDefQueryValue_AsString(
      OfLiteOpDefQueryAttrByName(def, name));
}
