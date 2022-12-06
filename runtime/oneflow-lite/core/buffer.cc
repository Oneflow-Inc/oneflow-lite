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
#include "oneflow-lite/core/buffer.h"

#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/base/refcount.h"

typedef struct OfLiteBuffer {
  OfLiteRefCount refcount;
  OfLiteAlloca* alloca;
  uint8_t* bytes;
  size_t bytesize;
} OfLiteBuffer;

void OfLiteBufferCreate(OfLiteAlloca* alloca, size_t bytesize,
                        OfLiteBuffer** buffer) {
  *buffer = reinterpret_cast<OfLiteBuffer*>(OfLiteMalloc(sizeof(OfLiteBuffer)));
  OfLiteRefCountInitialize(&(buffer[0]->refcount), 1);
  buffer[0]->alloca = alloca;
  OfLiteAllocaMalloc(alloca, bytesize,
                        reinterpret_cast<void**>(&buffer[0]->bytes));
  buffer[0]->bytesize = bytesize;
}

void OfLiteBufferDestory(OfLiteBuffer* buffer) {
  OfLiteRefCountDecrease(&buffer->refcount);
  if (OfLiteRefCountEqual(buffer->refcount, 0)) {
    OfLiteAllocaFree(buffer->alloca, buffer->bytes);
  }
}

void OfLiteBufferRetain(OfLiteBuffer* buffer) {
  OfLiteRefCountIncrease(&buffer->refcount);
}

size_t OfLiteBufferByteSize(const OfLiteBuffer* buffer) {
  return buffer->bytesize;
}

uint8_t* OfLiteBufferBytes(const OfLiteBuffer* buffer) { return buffer->bytes; }

void OfLiteBufferAlloca(const OfLiteBuffer* buffer,
                           const OfLiteAlloca** alloca) {
  *alloca = buffer->alloca;
}
