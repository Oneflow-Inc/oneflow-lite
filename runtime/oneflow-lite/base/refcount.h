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
#ifndef ONEFLOW_LITE_BASE_REF_COUNT_H_
#define ONEFLOW_LITE_BASE_REF_COUNT_H_

#include <stdint.h>
#include <stdlib.h>

#ifndef __cplusplus
#include <stdatomic.h>
#else
#include <atomic>
#define _Atomic(X) std::atomic<X>
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteRefCount {
  _Atomic(size_t) refcount;
} OfLiteRefCount;

void OfLiteRefCountInitialize(OfLiteRefCount* ref, size_t value);
void OfLiteRefCountIncrease(OfLiteRefCount* ref);
void OfLiteRefCountDecrease(OfLiteRefCount* ref);

bool OfLiteRefCountEqual(const OfLiteRefCount& ref, size_t value);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_BASE_REF_COUNT_H_
