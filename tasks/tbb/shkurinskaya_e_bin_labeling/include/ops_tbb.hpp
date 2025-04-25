#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_bin_labeling_tbb {

class TaskTBB : public ppc::core::Task {
 public:
  explicit TaskTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
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

  int FindRoot(int index);
  void UnionSets(int a, int b);
};

}  // namespace shkurinskaya_e_bin_labeling_tbb
