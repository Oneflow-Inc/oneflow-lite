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
#include "graph/graph.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/flatbuffer_utils.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/ascend/ascend_create_op.h"

namespace {

typedef struct MlitJitOp {
  OfLiteVTableHandle handle;
} MlitJitOp;

void destory(OfLiteOperator* op) { OfLiteFree(op); }

void compute(OfLiteOperator* op, const OfLiteTensorSpan& inputs,
             const OfLiteTensorSpan& outputs) {
}

static OfLiteOperatorVTable vtable = {
    .destory = destory,
    .compute = compute,
};

ge::Graph OfLiteAscendLoadGraph(const void* buffer, size_t size) {
  const char* temp_filename = ".__TMP__ascend_graph";
  FILE* fp = fopen(temp_filename, "wb");
  if (!fp) {
    OFLITE_FAIL("failed to open temp file %s\n", temp_filename);
  }
  if (fwrite(buffer, 1, size, fp) != size) {
    OFLITE_FAIL("failed to write ascend graph\n");
  }
  fclose(fp);

  ge::Graph graph("ascend-graph");
  ATC_CHECK(graph.LoadFromFile(temp_filename));

  // clean up temp file
  if (0 != remove(temp_filename)) {
    OFLITE_FAIL("failed to clean up temp file\n");
  }
  return graph;
}

void OfLiteAscendGraphBuilderInitialize() {
  std::map<ge::AscendString, ge::AscendString> options;
  const char* soc_version = aclrtGetSocName();
  if (soc_version) {
    options[ge::ir_option::SOC_VERSION] = soc_version;
  } else {
    options[ge::ir_option::SOC_VERSION] = "Ascend310";
  }
  options[ge::ir_option::OP_DEBUG_LEVEL] = "0";
  options[ge::ir_option::DEBUG_DIR] = "/tmp/";
  // options[ge::ir_option::AUTO_TUNE_MODE] = "GA|RL";
  ge::aclgrphBuildInitialize(options);
}

void OfLiteAscendGraphBuilderFinialize() {
  ge::aclgrphBuildFinalize();
}

}  // namespace

ASCEND_CREATE_OP(mlir_jit) {
  MlitJitOp* op = reinterpret_cast<MlitJitOp*>(OfLiteMalloc(sizeof(MlitJitOp)));
  op->handle.vtable = &vtable;

  OfLiteStringRef mlir_assembly =
    OfLiteOpDefQueryAttrValueByName_AsString(def, OfLiteStringRefCreate("mlir_assembly"));
  ge::Graph graph = OfLiteAscendLoadGraph(mlir_assembly.data, mlir_assembly.size);

  std::map<ge::AscendString, ge::AscendString> options;
  options.insert(std::make_pair(ge::ir_option::LOG_LEVEL, "error"));
  options.insert(std::make_pair(ge::ir_option::OP_DEBUG_LEVEL, "0"));
  options.insert(std::make_pair(ge::ir_option::INPUT_FORMAT, "NCHW"));

  OfLiteAscendGraphBuilderInitialize();
  ge::ModelBufferData buffer;
  ATC_CHECK(aclgrphBuildModel(graph, options, buffer));
  OfLiteAscendGraphBuilderFinialize();

  return reinterpret_cast<OfLiteOperator*>(op);
}
