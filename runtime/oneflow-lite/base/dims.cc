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
#include "oneflow-lite/base/dims.h"

OFLITE_API int64_t OfLiteDimsCount(const OfLiteDims& dims, size_t start,
                                   size_t end) {
  end = end >= dims.ndim ? dims.ndim : end;
  int64_t count = 1;
  for (size_t pos = start; pos < end; ++pos) {
    count *= dims.sizes[pos];
  }
  return count;
}
