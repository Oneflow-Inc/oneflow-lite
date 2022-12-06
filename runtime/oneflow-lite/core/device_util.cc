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
#include "oneflow-lite/core/device_util.h"

#include <stdlib.h>

void OfLiteParseBackendAndOrdinal(OfLiteStringRef device,
                                  OfLiteStringRef* backend, size_t* ordinal) {
  size_t pos = 0;
  for (; pos < device.size; ++pos) {
    if (device.data[pos] == ':') {
      break;
    }
  }
  *backend = OfLiteStringRefSubStr(device, 0, pos);
  if (pos == device.size - 1) {
    // TODO(): invalid device
  } else if (pos == device.size) {
    *ordinal = 0;
  } else {
    char temp[24] = {0};
    memcpy(temp, device.data + pos + 1, device.size - pos - 1);
    char* end = NULL;
    *ordinal = strtoull(temp, &end, 0);
  }
}
