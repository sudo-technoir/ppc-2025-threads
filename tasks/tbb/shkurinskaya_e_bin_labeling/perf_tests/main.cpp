#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

TEST(shkurinskaya_e_bin_labeling_omp, test_pipeline_run) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in(size, 1);
  std::vector<int> out(size);
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto task_tbb = std::make_shared<shkurinskaya_e_bin_labeling_tbb::TaskTBB>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(ans, out);
}

TEST(shkurinskaya_e_bin_labeling_omp, test_task_run) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  // Create data
  std::vector<int> in(size, 1);
  std::vector<int> out(size);
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_tbb = std::make_shared<ppc::core::TaskData>();
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_tbb->inputs_count.emplace_back(in.size());
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
  task_data_tbb->inputs_count.emplace_back(1);
  task_data_tbb->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_tbb->outputs_count.emplace_back(out.size());

  // Create Task
  auto task_tbb = std::make_shared<shkurinskaya_e_bin_labeling_tbb::TaskTBB>(task_data_tbb);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task_tbb);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  ASSERT_EQ(ans, out);
}
