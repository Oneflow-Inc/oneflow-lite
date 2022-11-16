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
#ifndef ONEFLOW_LITE_BASE_MEMORY_H_
#define ONEFLOW_LITE_BASE_MEMORY_H_

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

inline void* OfLiteMalloc(size_t size) { return malloc(size); }

inline void OfLiteFree(void* ptr) { free(ptr); }

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_BASE_MEMORY_H_
