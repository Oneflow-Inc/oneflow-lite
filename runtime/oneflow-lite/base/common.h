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

#endif  // ONEFLOW_LITE_BASE_COMMON_H_
