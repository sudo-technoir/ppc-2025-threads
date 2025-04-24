#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/belov_a_radix_sort_with_batcher_mergesort/include/ops_omp.hpp"

using namespace belov_a_radix_batcher_mergesort_omp;

namespace {
vector<Bigint> GenerateMixedValuesArray(int n) {
  random_device rd;
  mt19937 gen(rd());

  uniform_int_distribution<Bigint> small_range(-999LL, 999LL);
  uniform_int_distribution<Bigint> medium_range(-10000LL, 10000LL);
  uniform_int_distribution<Bigint> large_range(-10000000000LL, 10000000000LL);
  uniform_int_distribution<int> choice(0, 2);

  vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (choice(gen) == 0) {
      arr.push_back(small_range(gen));
    } else if (choice(gen) == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(large_range(gen));
    }
  }
  return arr;
}

vector<Bigint> GenerateIntArray(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(numeric_limits<int>::min(), numeric_limits<int>::max());

  vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}

vector<Bigint> GenerateBigintArray(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<Bigint> dist(numeric_limits<Bigint>::min() / 2, numeric_limits<Bigint>::max() / 2);

  vector<Bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}
}  // namespace

TEST(belov_a_radix_batcher_mergesort_omp, test_random_small_BigintV_vector) {
  int n = 1024;
  vector<Bigint> arr = GenerateBigintArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_medium_BigintV_vector) {
  int n = 8192;
  vector<Bigint> arr = GenerateBigintArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_large_BigintV_vector) {
  int n = 65536;
  vector<Bigint> arr = GenerateBigintArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_small_intV_vector) {
  int n = 4096;
  vector<Bigint> arr = GenerateIntArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_medium_intV_vector) {
  int n = 16384;
  vector<Bigint> arr = GenerateIntArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_large_intV_vector) {
  int n = 65536;
  vector<Bigint> arr = GenerateIntArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_small_mixedV_vector) {
  int n = 2048;
  vector<Bigint> arr = GenerateMixedValuesArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_medium_mixedV_vector) {
  int n = 16384;
  vector<Bigint> arr = GenerateMixedValuesArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_random_large_mixedV_vector) {
  int n = 65536;
  vector<Bigint> arr = GenerateMixedValuesArray(n);

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_predefined_intV_vector) {
  int n = 16;
  vector<Bigint> arr = {74685421,  -53749, 2147483647, -1000, -2147483648, 1001, 0,       124,
                        315986930, -123,   42,         -43,   2,           -1,   -999999, 999998};

  vector<Bigint> expected_solution = arr;
  std::ranges::sort(expected_solution);

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_one_element_input_Bigint) {
  int n = 1;
  vector<Bigint> arr = {8888};

  vector<Bigint> expected_solution = arr;

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);
  ASSERT_EQ(tesk_task_omp.Validation(), true);
  tesk_task_omp.PreProcessing();
  tesk_task_omp.Run();
  tesk_task_omp.PostProcessing();

  EXPECT_EQ(arr, expected_solution);
}

TEST(belov_a_radix_batcher_mergesort_omp, test_array_size_missmatch) {
  int n = 3;
  vector<Bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);

  EXPECT_FALSE(tesk_task_omp.Validation());
}

TEST(belov_a_radix_batcher_mergesort_omp, test_invalid_inputs_count) {
  int n = 3;
  vector<Bigint> arr = {-53742329, -2147483648, 123265244, 0, 315986930, 42, 2147483647, -853960, 472691};

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);

  EXPECT_FALSE(tesk_task_omp.Validation());
}

TEST(belov_a_radix_batcher_mergesort_omp, test_empty_input_Validation) {
  int n = 0;
  vector<Bigint> arr = {};

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);

  EXPECT_FALSE(tesk_task_omp.Validation());
}

TEST(belov_a_radix_batcher_mergesort_omp, test_empty_output_Validation) {
  int n = 3;
  vector<Bigint> arr = {789, 654, 231, 0, 123456789, 792012345678, -22475942, -853960, 59227648};

  shared_ptr<ppc::core::TaskData> task_data_omp = make_shared<ppc::core::TaskData>();
  task_data_omp->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->inputs_count.emplace_back(arr.size());
  task_data_omp->inputs_count.emplace_back(n);
  // task_data_omp->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
  task_data_omp->outputs_count.emplace_back(arr.size());

  RadixBatcherMergesortParallel tesk_task_omp(task_data_omp);

  EXPECT_FALSE(tesk_task_omp.Validation());
}