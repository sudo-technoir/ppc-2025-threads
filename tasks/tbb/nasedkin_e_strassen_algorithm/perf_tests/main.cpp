#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/nasedkin_e_strassen_algorithm/include/ops_tbb.hpp"

namespace {
std::vector<double> GenerateRandomMatrix(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> distrib(-100.0, 100.0);
  std::vector<double> matrix(size * size);
  for (size_t i = 0; i < size * size; ++i) {
    matrix[i] = distrib(gen);
  }
  return matrix;
}
}  // namespace

TEST(nasedkin_e_strassen_algorithm_tbb, test_pipeline_run) {
  constexpr size_t kMatrixSize = 512;
  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  std::vector<double> expected = nasedkin_e_strassen_algorithm_tbb::StandardMultiply(in_a, in_b, kMatrixSize);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<nasedkin_e_strassen_algorithm_tbb::StrassenTbb>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(expected[i], out[i], 1e-6);
  }
}

TEST(nasedkin_e_strassen_algorithm_tbb, test_task_run) {
  constexpr size_t kMatrixSize = 512;
  std::vector<double> in_a = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> in_b = GenerateRandomMatrix(kMatrixSize);
  std::vector<double> out(kMatrixSize * kMatrixSize, 0.0);

  std::vector<double> expected = nasedkin_e_strassen_algorithm_tbb::StandardMultiply(in_a, in_b, kMatrixSize);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_a.data()));
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_b.data()));
  task_data->inputs_count.emplace_back(in_a.size());
  task_data->inputs_count.emplace_back(in_b.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  task_data->outputs_count.emplace_back(out.size());

  auto test_task = std::make_shared<nasedkin_e_strassen_algorithm_tbb::StrassenTbb>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  for (size_t i = 0; i < out.size(); ++i) {
    EXPECT_NEAR(expected[i], out[i], 1e-6);
  }
}