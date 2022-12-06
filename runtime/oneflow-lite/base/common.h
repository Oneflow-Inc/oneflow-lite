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
#ifndef ONEFLOW_LITE_BASE_COMMON_H_
#define ONEFLOW_LITE_BASE_COMMON_H_

#include <stdio.h>

#if defined(_WIN32)
#define OFLITE_EXPORT __declspec(dllexport)
#else
#define OFLITE_EXPORT __attribute__((visibility("default")))
#endif  // _WIN32

#ifdef __cplusplus
#define OFLITE_API extern "C" OFLITE_EXPORT
#else
#define OFLITE_API OFLITE_EXPORT
#endif  // __cplusplus

#if defined(__clang__)
#define OFLITE_UNUSED __attribute__((maybe_unused))
#elif defined(__GNUC__) && !defined(__clang__)
#define OFLITE_UNUSED __attribute__((unused))
#else
#define OFLITE_UNUSED
#endif  // __clang__

#define OFLITE_CAT(x, y) OFLITE_CAT_IMPL(x, y)
#define OFLITE_CAT_IMPL(x, y) x##y

#define OFLITE_MAX(x, y) ((x) > (y) ? (x) : (y))
#define OFLITE_MIN(x, y) ((x) > (y) ? (y) : (x))

#define OFLITE_FAIL(...)          \
  {                               \
    fprintf(stderr, __VA_ARGS__); \
    exit(1);                      \
  }

#endif  // ONEFLOW_LITE_BASE_COMMON_H_
