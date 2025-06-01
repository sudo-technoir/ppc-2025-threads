#pragma once

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include <cstddef>
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
  tbb::spin_mutex uf_mutex_;

  int FindRoot(int v);
  void UnionSets(int a, int b);
  [[nodiscard]] bool IsValidIndex(int i, int j) const;
  void ProcessUnion();
  void InitializeUF();
  void CollectPairs(std::vector<std::pair<int,int>>& all_pairs);
  void ExecuteUnions(const std::vector<std::pair<int,int>>& all_pairs);
  void CompressPaths();
};

}  // namespace shkurinskaya_e_bin_labeling_tbb
