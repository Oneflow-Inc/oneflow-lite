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
#include "oneflow-lite/base/stringref.h"

#include <string.h>

OFLITE_API OfLiteStringRef OfLiteStringRefCreate(const char* str) {
  OfLiteStringRef strref = {str, strlen(str)};
  return strref;
}

OFLITE_API OfLiteStringRef OfLiteStringRefSubStr(OfLiteStringRef value,
                                                 size_t pos, size_t len) {
  pos = OFLITE_MIN(pos, value.size);
  len = OFLITE_MIN(len, value.size - pos);
  OfLiteStringRef strref = {value.data + pos, len};
  return strref;
}

OFLITE_API bool OfLiteStringRefEqual(OfLiteStringRef lhs, OfLiteStringRef rhs) {
  if (lhs.size != rhs.size) {
    return false;
  }
  return 0 == strncmp(lhs.data, rhs.data, lhs.size);
}
