#include "omp/shkurinskaya_e_bin_labeling/include/ops_omp.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace shkurinskaya_e_bin_labeling_omp {

void TaskOMP::ProcessUnion() {
  const int H = height_;
  const int W = width_;
  const int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

#pragma omp parallel for collapse(2) schedule(dynamic)
  for (int i = 0; i < H; ++i) {
    for (int j = 0; j < W; ++j) {
      int idx = i * W + j;
      if (input_[idx] != 1) continue;
      for (int d = 0; d < 8; ++d) {
        int ni = i + directions[d][0];
        int nj = j + directions[d][1];
        if (ni < 0 || ni >= H || nj < 0 || nj >= W) continue;
        int nidx = ni * W + nj;
        if (input_[nidx] == 1) {
          UnionSets(idx, nidx);
        }
      }
    }
  }
}

bool TaskOMP::PreProcessingImpl() {
  std::cout << "PreProcessingImpl: Initializing inputs and outputs...\n";
  auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  width_ = reinterpret_cast<int *>(task_data->inputs[2])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  const int N = task_data->inputs_count[0];

  input_.assign(tmp_ptr, tmp_ptr + N);
  res_.resize(N);
  parent_.resize(N);
  rank_.resize(N);
  label_.assign(N, 0);

  return true;
}

bool TaskOMP::ValidationImpl() {
  std::cout << "ValidationImpl: Validating input data...\n";
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskOMP::RunImpl() {
  std::cout << "[DEBUG] RunImpl: Starting processing...\n";
  const int N = height_ * width_;

// I этап: инициализация
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < N; ++i) {
    if (input_[i] == 1) {
      parent_[i] = i;
      rank_[i] = 0;
    } else {
      parent_[i] = -1;
    }
  }

  // II этап: объединение
  ProcessUnion();

// III этап: полное сжатие пути
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < N; ++i) {
    if (input_[i] == 1) {
      parent_[i] = FindRoot(i);
    }
  }

  std::cout << "[DEBUG] RunImpl: Processing completed\n";
  return true;
}

inline int TaskOMP::FindRoot(int i) {
  while (parent_[i] != i) {
    i = parent_[i];
  }
  return i;
}

void TaskOMP::UnionSets(int a, int b) {
  int rootA = FindRoot(a);
  int rootB = FindRoot(b);
  if (rootA == rootB) return;

#pragma omp critical
  {
    int rA = FindRoot(rootA);
    int rB = FindRoot(rootB);
    if (rA != rB) {
      // Union by rank
      if (rank_[rA] < rank_[rB]) {
        parent_[rA] = rB;
      } else if (rank_[rA] > rank_[rB]) {
        parent_[rB] = rA;
      } else {
        parent_[rB] = rA;
        ++rank_[rA];
      }
    }
  }
}

bool TaskOMP::PostProcessingImpl() {
  std::cout << "PostProcessingImpl: Starting post-processing...\n";
  int comp = 1;
  const int N = height_ * width_;

  std::fill(label_.begin(), label_.end(), 0);
  for (int idx = 0; idx < N; ++idx) {
    if (input_[idx] != 1) {
      res_[idx] = 0;
      continue;
    }
    int root = FindRoot(idx);
    if (label_[root] == 0) {
      label_[root] = comp++;
    }
    res_[idx] = label_[root];
  }
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}

bool TaskOMP::IsValidIndex(int i, int j) const { return (i >= 0 && i < height_ && j >= 0 && j < width_);  }

}  // namespace shkurinskaya_e_bin_labeling_omp
