// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// Disable TensorRT on the tests because axis=0 is not supported

template <typename T>
void RunUniqueTest(const std::vector<int64_t>& X_dims,
                   const std::vector<T>& X,
                   const int64_t* axis,
                   bool sorted,
                   const std::vector<int64_t>& Y_dims,
                   const std::vector<T>& Y,
                   const std::vector<int64_t>& indices_dims,
                   const std::vector<int64_t>& indices,
                   const std::vector<int64_t>& inverse_indices_dims,
                   const std::vector<int64_t>& inverse_indices,
                   const std::vector<int64_t>& counts_dims,
                   const std::vector<int64_t>& counts) {
  OpTester test("Unique", 11);

  if (axis) {
    test.AddAttribute("axis", *axis);
  }

  test.AddAttribute("sorted", static_cast<int64_t>(sorted));

  test.AddInput<T>("X", X_dims, X);
  test.AddOutput<T>("Y", Y_dims, Y);
  test.AddOutput<int64_t>("indices", indices_dims, indices);
  test.AddOutput<int64_t>("inverse_indices", inverse_indices_dims, inverse_indices);
  test.AddOutput<int64_t>("counts", counts_dims, counts);

  test.Run();
  // test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(UniqueTest, Unique_Flatten_Unsorted) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, 4.f, 1.f, 2.f, 2.f, 0.f};
  const int64_t* axis = nullptr;
  bool sorted = false;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{1.f, 4.f, 2.f, 0.f};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{0, 1, 3, 5};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{0, 1, 0, 2, 2, 3};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{2, 1, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

TEST(UniqueTest, Unique_Flatten_Sorted) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<float> X{1.f, 4.f, 1.f, 2.f, 2.f, 0.f};
  const int64_t* axis = nullptr;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<float> Y{0.f, 1.f, 2.f, 4.f};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{5, 0, 3, 1};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{1, 3, 1, 2, 2, 0};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{1, 2, 2, 1};

  RunUniqueTest<float>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                       inverse_indices_dims, inverse_indices, counts_dims, counts);
}

// string tests!
TEST(UniqueTest, Unique_Flatten_Sorted_String) {
  const std::vector<int64_t> X_dims{2, 3};
  const std::vector<std::string> X{"1.f", "4.f", "1.f", "2.f", "2.f", "0.f"};
  const int64_t* axis = nullptr;
  bool sorted = true;
  const std::vector<int64_t> Y_dims{4};
  const std::vector<std::string> Y{"0.f", "1.f", "2.f", "4.f"};

  const std::vector<int64_t> indices_dims{4};
  const std::vector<int64_t> indices{5, 0, 3, 1};
  const std::vector<int64_t> inverse_indices_dims{6};
  const std::vector<int64_t> inverse_indices{1, 3, 1, 2, 2, 0};
  const std::vector<int64_t> counts_dims{4};
  const std::vector<int64_t> counts{1, 2, 2, 1};

  RunUniqueTest<std::string>(X_dims, X, axis, sorted, Y_dims, Y, indices_dims, indices,
                             inverse_indices_dims, inverse_indices, counts_dims, counts);
}

}  // namespace test
}  // namespace onnxruntime
