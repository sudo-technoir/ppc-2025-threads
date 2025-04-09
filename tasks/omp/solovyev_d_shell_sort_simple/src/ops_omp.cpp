#include "omp/solovyev_d_shell_sort_simple/include/ops_omp.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

bool solovyev_d_shell_sort_simple_omp::TaskOMP::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  return true;
}

bool solovyev_d_shell_sort_simple_omp::TaskOMP::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool solovyev_d_shell_sort_simple_omp::TaskOMP::RunImpl() {
  for (int gap = (int)input_.size() / 2; gap > 0; gap /= 2) {
#pragma omp parallel for
    for (int i = 0; i < gap; i++) {
      for (int f = gap + i; f < (int)input_.size(); f = f + gap) {
        int val = input_[f];
        int j = f;
        while (j >= gap && input_[j - gap] > val) {
          input_[j] = input_[j - gap];
          j -= gap;
        }
        input_[j] = val;
      }
    }
  }
  return true;
}
bool solovyev_d_shell_sort_simple_omp::TaskOMP::PostProcessingImpl() {
  for (size_t i = 0; i < input_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = input_[i];
  }
  return true;
}
