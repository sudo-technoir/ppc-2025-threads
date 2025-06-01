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
  boost::mpi::communicator world_, group_;
  std::vector<int> input_;
  std::vector<int> res_local_; 
  std::vector<int> counts_, displs_;
  int width_, height_;
  int local_H_, local_offset_rows_;

  int FindRoot(int);
  void UnionSets(int, int);
  void InitializeUFRange(int r0, int r1);
  void CompressPathsRange(int r0, int r1);

  void SolveBoundaryPairs();
  void ExchangeBoundaryRows(std::vector<int>& top_row, std::vector<int>& bottom_row); 
  void BuildBoundaryPairs(const std::vector<int>& recv_top, const std::vector<int>& recv_bottom);
};

}  // namespace shkurinskaya_e_bin_labeling_tbb
