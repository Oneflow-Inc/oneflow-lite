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
#include "acl/acl_mdl.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "graph/graph.h"
#include "oneflow-lite/base/memory.h"
#include "oneflow-lite/core/flatbuffer_utils.h"
#include "oneflow-lite/core/vtable_handle.h"
#include "oneflow-lite/delegates/ascend/ascend_create_op.h"

namespace {

typedef struct MlitJitOp {
  OfLiteVTableHandle handle;

  ge::ModelBufferData model;
  uint32_t model_id;
  aclmdlDesc* model_desc;

  int32_t input_count;
  int32_t output_count;
  aclmdlDataset* input_dataset;
  aclmdlDataset* output_dataset;
} MlitJitOp;

void OfLiteAscendDestoryDataset(aclmdlDataset* dataset) {
  for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); ++i) {
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(dataset, i);
    ACL_CHECK(aclDestroyDataBuffer(data_buffer));
  }
  ACL_CHECK(aclmdlDestroyDataset(dataset));
}

void OfLiteAscendMlitJitOpDestory(OfLiteOperator* op) {
  MlitJitOp* impl = reinterpret_cast<MlitJitOp*>(op);
  OfLiteAscendDestoryDataset(impl->input_dataset);
  OfLiteAscendDestoryDataset(impl->output_dataset);
  ACL_CHECK(aclmdlUnload(impl->model_id));
  ACL_CHECK(aclmdlDestroyDesc(impl->model_desc));
  impl->model.~ModelBufferData();
  OfLiteFree(op);
}

void OfLiteAscendMlitJitOpCompute(OfLiteOperator* op,
                                  const OfLiteTensorSpan& inputs,
                                  const OfLiteTensorSpan& outputs) {
  OfLiteAscendDevice* device = OfLiteAscendObtainDevice();
  ACL_CHECK(aclrtSetCurrentContext(device->context));

  MlitJitOp* impl = reinterpret_cast<MlitJitOp*>(op);
  if (inputs.size != impl->input_count) {
    OFLITE_FAIL("mlit_jit input count mismatch\n");
  }
  if (outputs.size != impl->output_count) {
    OFLITE_FAIL("mlit_jit output count mismatch\n");
  }
  for (size_t i = 0; i < impl->input_count; ++i) {
    OfLiteTensor* input = inputs.items[i];
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(impl->input_dataset, i);
    ACL_CHECK(aclUpdateDataBuffer(data_buffer, OfLiteTensorData(input), OfLiteTensorDataSize(input)));
  }
  for (size_t i = 0; i < impl->output_count; ++i) {
    OfLiteTensor* output = outputs.items[i];
    aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(impl->output_dataset, i);
    ACL_CHECK(aclUpdateDataBuffer(data_buffer, OfLiteTensorData(output), OfLiteTensorDataSize(output)));
  }
  ACL_CHECK(aclmdlExecute(impl->model_id, impl->input_dataset, impl->output_dataset));
}

static OfLiteOperatorVTable vtable = {
    .destory = OfLiteAscendMlitJitOpDestory,
    .compute = OfLiteAscendMlitJitOpCompute,
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

void OfLiteAscendGraphBuilderFinialize() { ge::aclgrphBuildFinalize(); }

aclmdlDataset* OfLiteAscendCreateDataset(size_t count) {
  aclmdlDataset* dataset = aclmdlCreateDataset();
  if (!dataset) {
    OFLITE_FAIL("failed to create ACL dataset\n");
  }
  for (size_t i = 0; i < count; ++i) {
    aclDataBuffer* data_buffer = aclCreateDataBuffer(nullptr, 0);
    if (!data_buffer) {
      OFLITE_FAIL("failed to create ACL data buffer\n");
    }
    ACL_CHECK(aclmdlAddDatasetBuffer(dataset, data_buffer));
  }
  return dataset;
}

}  // namespace

ASCEND_CREATE_OP(mlir_jit) {
  MlitJitOp* op = reinterpret_cast<MlitJitOp*>(OfLiteMalloc(sizeof(MlitJitOp)));
  op->handle.vtable = &vtable;

  OfLiteStringRef mlir_assembly = OfLiteOpDefQueryAttrValueByName_AsString(
      def, OfLiteStringRefCreate("mlir_assembly"));
  ge::Graph graph =
      OfLiteAscendLoadGraph(mlir_assembly.data, mlir_assembly.size);

  OfLiteAscendGraphBuilderInitialize();
  std::map<ge::AscendString, ge::AscendString> options = {
    {ge::ir_option::LOG_LEVEL, "error"},
    {ge::ir_option::OP_DEBUG_LEVEL, "0"},
    {ge::ir_option::INPUT_FORMAT, "NCHW"}
  };
  ATC_CHECK(aclgrphBuildModel(graph, options, op->model));
  OfLiteAscendGraphBuilderFinialize();

  ACL_CHECK(aclmdlLoadFromMem(reinterpret_cast<void*>(op->model.data.get()),
                              op->model.length, &op->model_id));
  op->model_desc = aclmdlCreateDesc();
  if (!op->model_desc) {
    OFLITE_FAIL("failed to create ACL model description\n");
  }
  ACL_CHECK(aclmdlGetDesc(op->model_desc, op->model_id));

  op->input_count = aclmdlGetNumInputs(op->model_desc);
  op->output_count = aclmdlGetNumOutputs(op->model_desc);
  op->input_dataset = OfLiteAscendCreateDataset(op->input_count);
  op->output_dataset = OfLiteAscendCreateDataset(op->output_count);
  return reinterpret_cast<OfLiteOperator*>(op);
}
