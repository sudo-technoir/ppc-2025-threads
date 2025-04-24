#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "omp/kovalchuk_a_shell_sort_omp/include/ops_omp.hpp"

TEST(kovalchuk_a_shell_sort_omp, Test_EmptyArray) {
  std::vector<int> input = {};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  EXPECT_TRUE(output.empty());
}

TEST(kovalchuk_a_shell_sort_omp, Test_SingleElement) {
  std::vector<int> input = {42};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  EXPECT_EQ(input, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_ReverseSorted) {
  std::vector<int> input = {9, 7, 5, 3, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 3, 5, 7, 9};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_Duplicates) {
  std::vector<int> input = {5, 2, 5, 1, 2};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 2, 5, 5};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_NegativeNumbers) {
  std::vector<int> input = {-5, 0, -3, 10, -1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {-5, -3, -1, 0, 10};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_ExtremeValues) {
  std::vector<int> input = {INT32_MIN, 0, INT32_MAX};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {INT32_MIN, 0, INT32_MAX};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_OddSizeArray) {
  std::vector<int> input = {4, 1, 7, 2, 9};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 4, 7, 9};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_EvenSizeArray) {
  std::vector<int> input = {5, 2, 8, 1};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 5, 8};
  EXPECT_EQ(expected, output);
}

TEST(kovalchuk_a_shell_sort_omp, Test_DoubleReverseOrder) {
  std::vector<int> input = {9, 7, 5, 3, 1, 10, 8, 6, 4, 2};
  std::vector<int> output(input.size());

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto task = std::make_shared<kovalchuk_a_shell_sort_omp::ShellSortOMP>(task_data);

  ASSERT_TRUE(task->Validation());
  task->PreProcessing();
  task->Run();
  task->PostProcessing();

  std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  EXPECT_EQ(expected, output);
}