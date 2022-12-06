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
#ifndef ONEFLOW_LITE_CORE_BUFFER_H_
#define ONEFLOW_LITE_CORE_BUFFER_H_

#include "oneflow-lite/core/alloca.h"

typedef struct OfLiteBuffer OfLiteBuffer;

void OfLiteBufferCreate(OfLiteAlloca* alloca, size_t bytesize,
                        OfLiteBuffer** buffer);

void OfLiteBufferDestory(OfLiteBuffer* buffer);

void OfLiteBufferRetain(OfLiteBuffer* buffer);

size_t OfLiteBufferByteSize(const OfLiteBuffer* buffer);

uint8_t* OfLiteBufferBytes(const OfLiteBuffer* buffer);

void OfLiteBufferAlloca(const OfLiteBuffer* buffer,
                           const OfLiteAlloca** alloca);

#endif  // ONEFLOW_LITE_CORE_BUFFER_H_
