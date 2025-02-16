// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_allocator_with_mem_pattern.h"
#include "simple_tensor_allocator.h"

namespace onnxruntime {

AllocatorPtr ITensorAllocator::GetAllocator(const OrtMemoryInfo& memory_info) {
  return exec_providers_.GetAllocator(memory_info);
}

std::unique_ptr<ITensorAllocator> ITensorAllocator::Create(bool enable_mem_pattern,
                                                           const ExecutionPlanBase& execution_plan,
                                                           const ExecutionProviders& exec_providers,
                                                           std::vector<BufferUniquePtr>& weights_buffers) {
  if (enable_mem_pattern) {
    return std::make_unique<TensorAllocatorWithMemPattern>(execution_plan, exec_providers, weights_buffers);
  }
  return std::make_unique<SimpleTensorAllocator>(execution_plan, exec_providers, weights_buffers);
}

}  // namespace onnxruntime
