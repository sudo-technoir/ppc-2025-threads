#pragma once

#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace shkurinskaya_e_bin_labeling_stl {

class TaskSTL : public ppc::core::Task {
 public:
  explicit TaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int width_ = 0;
  int height_ = 0;

  std::vector<int> input_;
  std::vector<int> res_;
  std::vector<int> parent_;
  std::vector<int> rank_;
  std::vector<int> label_;
  std::mutex uf_mutex_;

  int FindRoot(int index);
  void UnionSets(int a, int b);
  void InitializeUFRange(int row_start, int row_end);
  [[nodiscard]] void ProcessUnionRange(int row_start, int row_end);
  void CompressPathsRange(int row_start, int row_end);

  int NumThreads() const { return ppc::util::GetPPCNumThreads(); }
};

}  // namespace shkurinskaya_e_bin_labeling_stl
