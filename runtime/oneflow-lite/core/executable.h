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
#ifndef ONEFLOW_LITE_CORE_EXECUTABLE_H_
#define ONEFLOW_LITE_CORE_EXECUTABLE_H_

#include "oneflow-lite/base/common.h"
#include "oneflow-lite/base/stringref.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct OfLiteExecutable OfLiteExecutable;
typedef struct OfLiteTensorDef OfLiteTensorDef;
typedef struct OfLiteBufferSegmentDef OfLiteBufferSegmentDef;
typedef struct OfLiteOpDef OfLiteOpDef;
typedef struct OfLiteOpFunctionDef OfLiteOpFunctionDef;

OFLITE_API void OfLiteExecutableCreateFromPath(OfLiteExecutable** executable,
                                               OfLiteStringRef filepath);

OFLITE_API void OfLiteExecutableName(const OfLiteExecutable* executable,
                                     OfLiteStringRef* name);

OFLITE_API void OfLiteExecutableInputSize(const OfLiteExecutable* executable,
                                          size_t* size);
OFLITE_API void OfLiteExecutableOutputSize(const OfLiteExecutable* executable,
                                           size_t* size);

OFLITE_API void OfLiteExecutableInput(const OfLiteExecutable* executable,
                                      size_t index,
                                      const OfLiteTensorDef** input);
OFLITE_API void OfLiteExecutableInputName(const OfLiteExecutable* executable,
                                          size_t index,
                                          OfLiteStringRef* input_name);

OFLITE_API void OfLiteExecutableOutput(const OfLiteExecutable* executable,
                                       size_t index,
                                       const OfLiteTensorDef** output);
OFLITE_API void OfLiteExecutableOutputName(const OfLiteExecutable* executable,
                                           size_t index,
                                           OfLiteStringRef* output_name);

OFLITE_API void OfLiteExecutableOperandSize(const OfLiteExecutable* executable,
                                            size_t* size);
OFLITE_API void OfLiteExecutableOperand(const OfLiteExecutable* executable,
                                        size_t index,
                                        const OfLiteTensorDef** operand);

OFLITE_API void OfLiteExecutableDeviceSize(const OfLiteExecutable* executable,
                                           size_t* size);
OFLITE_API void OfLiteExecutableDevice(const OfLiteExecutable* executable,
                                       size_t index, OfLiteStringRef* device);

OFLITE_API void OfLiteExecutableBufferSegmentSize(
    const OfLiteExecutable* executable, size_t* size);

OFLITE_API void OfLiteExecutableBufferSegment(
    const OfLiteExecutable* executable, size_t index,
    const OfLiteBufferSegmentDef** buffer_sgement);

OFLITE_API void OfLiteExecutableOpSize(const OfLiteExecutable* executable,
                                       size_t* size);
OFLITE_API void OfLiteExecutableOp(const OfLiteExecutable* executable,
                                   size_t index, const OfLiteOpDef** op);

OFLITE_API void OfLiteExecutableFunctionSize(const OfLiteExecutable* executable,
                                             size_t* size);
OFLITE_API void OfLiteExecutableFunction(const OfLiteExecutable* executable,
                                         size_t index,
                                         const OfLiteOpFunctionDef** function);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // ONEFLOW_LITE_CORE_EXECUTABLE_H_
