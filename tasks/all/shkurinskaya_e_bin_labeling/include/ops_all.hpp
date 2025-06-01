#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkurinskaya_e_bin_labeling_all {

class TaskMPITBB : public ppc::core::Task {
 public:
  explicit TaskMPITBB(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  boost::mpi::communicator world_;
  std::vector<int> input_;
  std::vector<int> res_local_;
  std::vector<int> counts_, displs_;
  std::vector<int> input_global_;
  int width_, height_;
  int local_H_, local_offset_rows_;

  int FindRoot(int);
  void UnionSets(int, int);
  int FindRootGlobal(int v, std::vector<int> &parent_vec);
  bool IsValidIndex(int i, int j) const;
};

}  // namespace shkurinskaya_e_bin_labeling_all
