// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/unique.h"

#include "core/framework/utils.h"
#include "core/providers/common.h"

#include <map>

namespace onnxruntime {

/*
ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema()
        .SetDoc(Unique_ver11_doc)
        .Attr(
            "sorted",
            "(Optional) Whether to sort the unique elements in ascending order before returning as output. "
            "Must be one of 0, or 1 (default).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "axis",
            "(Optional) The dimension to apply unique. If not specified, the unique elements of the "
            "flattened input are returned. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL)
        .Input(0, "X", "A N-D input tensor that is to be processed.", "T")
        .Output(
            0,
            "Y",
            "A tensor of the same type as 'X' "
            "containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted "
            "or maintained in the same order they occur in input 'X'",
            "T")
        .Output(
            1,
            "indices",
            "A 1-D INT64 tensor "
            "containing indices of 'Y' elements' first occurance in 'X'. "
            "When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in the flattened input tensor. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            2,
            "inverse_indices",
            "A 1-D INT64 tensor "
            "containing, for elements of 'X', its corresponding indices in 'Y'. "
            "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(
            3,
            "counts",
            "A 1-D INT64 tensor containing "
            "the count of each element "
            "of 'Y' in input 'X'",
            "tensor(int64)",
            OpSchema::Optional)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types(),
            "Input can be of any tensor type.")    
*/
ONNX_CPU_OPERATOR_KERNEL(
    Unique,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Unique);

Status Unique::Compute(OpKernelContext* context) const {
  const Tensor& input = *context->Input<Tensor>(0);

  Status status;
  auto data_type = input.DataType();
  int64_t axis = HandleNegativeAxis(axis_, input.Shape().NumDimensions());

  DispatchOnTensorTypeWithReturn(data_type, status, ComputeImpl, *context, axis);

  return status;
}

//template <typename T>
//static bool LessThan(const std::vector<T>& lhs, const std::vector<T>& rhs) {
//  for (size_t i = 0, end = lhs.size(); i < end; ++i) {
//    if (lhs[i] < rhs[i]) {
//      return true;
//    } else if (lhs[i] > rhs[i]) {
//      return false;
//    }
//  }
//
//  return false;  // equal
//}
//
//template <>
//static bool LessThan(const std::vector<std::string>& lhs, const std::vector<std::string>& rhs) {
//  for (size_t i = 0, end = lhs.size(); i < end; ++i) {
//    if (lhs[i] < rhs[i]) {
//      return true;
//    } else if (lhs[i] > rhs[i]) {
//      return false;
//    }
//  }
//
//  return false;  // equal
//}

template <typename T>
class Entry {
 public:
  Entry(const T* data, const TensorShape& shape, int64_t axis, int64_t n) : n_{n} {
    // iterate the 'n' entry of axis.

    // int offset, int stride, int n, int blocksize
    // offset is n in axis dim
    // stride is Size()/blocksize
    // blocksize is SizeFromDim(axis)
    int64_t n_axis = shape[axis];
    int64_t columns = shape.SizeFromDimension(axis);  // columns for each entry of the axis
    int64_t total_rows = shape.Size() / columns;
    int64_t tr2 = shape.SizeToDimension(axis);
    int64_t rows = total_rows / n_axis;  // rows for this entry
    int64_t offset = n * columns;

    items_.reserve(rows * columns);
    const T* end = data + shape.Size();

    for (int r = 0; r < rows; ++r) {
      T* cur = data + offset + (r * columns * n_axis);
      for (int c = 0; c < columns; ++c) {
        items_.push_back(*cur);
        ++cur;
      }
      assert(cur < end);
    }
  }

  bool operator<(const Entry& rhs) const {
    // we only expect to be comparing entries with the same shape
    assert(items_.size() == rhs.items_.size());
    bool c = items_ < rhs.items_;
    bool less_than = false;
    for (size_t i = 0, end = items_.size(); i < end; ++i) {
      if (items_[i] < rhs.items_[i]) {
        less_than = true;
        break;
      } else if (items_[i] > rhs.items_[i]) {
        break;
      }
    }

    ORT_ENFORCE(less_than == c);

    return less_than;
  }

 private:
  int64_t n_;
  std::vector<std::reference_wrapper<const T>> items_;
};

template <typename T>
static void CreateFlattenedOutput(OpKernelContext& context,
                                  const std::map<const T, size_t>& offsets,          // sorted:unsorted idx
                                  const std::vector<std::vector<int64_t>>& indices,  // unsorted
                                  const std::vector<int64_t>& inverse_index,         // unsorted
                                  bool sorted) {
  int64_t num_unique = static_cast<int64_t>(indices.size());
  Tensor& Y = *context.Output(0, TensorShape({num_unique /*, <Entry shape> if not flattened */}));
  Tensor* indices_out = context.Output(1, TensorShape({num_unique}));
  Tensor* inverse_indices = context.Output(2, TensorShape({static_cast<int64_t>(inverse_index.size())}));
  Tensor* counts = context.Output(3, TensorShape({num_unique}));

  auto Y_data = Y.MutableDataAsSpan<T>();
  gsl::span<int64_t> indices_data = indices_out != nullptr ? indices_out->MutableDataAsSpan<int64_t>()
                                                           : gsl::span<int64_t>();
  gsl::span<int64_t> inverse_indices_data = inverse_indices != nullptr ? inverse_indices->MutableDataAsSpan<int64_t>()
                                                                       : gsl::span<int64_t>();
  gsl::span<int64_t> counts_data = counts != nullptr ? counts->MutableDataAsSpan<int64_t>()
                                                     : gsl::span<int64_t>();

  // iterate using 'offsets' which is sorted, but contains the offset of the unsorted entry
  auto offsets_iter = offsets.begin();
  for (int64_t i = 0, end = num_unique; i < end; ++i, ++offsets_iter) {
    auto unsorted_idx = offsets_iter->second;
    // write sequentially if we want sorted output
    auto output_idx = sorted ? i : unsorted_idx;

    Y_data[output_idx] = offsets_iter->first;

    if (indices_out) {
      indices_data[output_idx] = indices[unsorted_idx].front();
    }

    if (counts) {
      counts_data[output_idx] = indices[unsorted_idx].size();
    }
  }

  if (inverse_indices) {
    if (sorted) {
      std::vector<int64_t> unsorted_to_sorted;
      unsorted_to_sorted.reserve(num_unique);
      for (const auto& offset : offsets) {
        // entry 0 is the offset of the first sorted entry
        unsorted_to_sorted.push_back(offset.second);
      }

      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = unsorted_to_sorted[inverse_index[i]];
      }
    } else {
      // memcpy or gsl::copy
      for (size_t i = 0, end = inverse_index.size(); i < end; ++i) {
        inverse_indices_data[i] = inverse_index[i];
      }
    }
  }
}

template <typename T>
Status Unique::ComputeImpl(OpKernelContext& context, int64_t axis) const {
  const Tensor& input = *context.Input<Tensor>(0);

  if (flatten_) {
    auto data = input.DataAsSpan<T>();

    // offset of entry in indices
    std::map<const T, size_t> offsets;
    std::vector<std::vector<int64_t>> indices;
    std::vector<int64_t> inverse_index;

    for (int64_t i = 0, end = input.Shape().Size(); i < end; ++i) {
      auto entry = offsets.find(data[i]);

      if (entry == offsets.end()) {
        // new value
        inverse_index.push_back({static_cast<int64_t>(indices.size())});
        indices.push_back({i});
      } else {
        size_t indices_idx = entry->second;
        indices[indices_idx].push_back(i);
        inverse_index.push_back(indices_idx);
      }
    }

    CreateFlattenedOutput(context, offsets, indices, inverse_index, sort_);
  }

  return Status::OK();
}

}  // namespace onnxruntime
