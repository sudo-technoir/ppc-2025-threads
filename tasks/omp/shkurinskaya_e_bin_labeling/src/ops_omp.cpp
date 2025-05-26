#include "omp/shkurinskaya_e_bin_labeling/include/ops_omp.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace shkurinskaya_e_bin_labeling_omp {

void TaskOMP::ProcessUnion() {
  const int N = height_ * width_;
  const int W = width_;
  const int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

#pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < N; ++idx) {
    if (input_[idx] != 1) continue;
    int x = idx % W;
    int y = idx / W;
    for (int d = 0; d < 8; ++d) {
      int nx = x + directions[d][1];
      int ny = y + directions[d][0];
      if (!IsValidIndex(ny, nx)) continue;
      int nidx = ny * W + nx;
      if (input_[nidx] == 1) {
        UnionSets(idx, nidx);
      }
    }
  }
}

bool TaskOMP::PreProcessingImpl() {
  std::cout << "PreProcessingImpl: Initializing inputs and outputs...\n";
  auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];
  const int N = task_data->inputs_count[0];

  input_.assign(tmp_ptr, tmp_ptr + N);
  res_.resize(N);
  parent_.resize(N);
  rank_.resize(N);
  label_.assign(N, 0);
  locks_.resize(N);
#pragma omp parallel for
  for (int i = 0; i < size; ++i) {
    omp_init_lock(&locks_[i]);
  }
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
  if (rootA == rootB || rootA < 0 || rootB < 0) return;
  if (rootA > rootB) std::swap(rootA, rootB);

  omp_set_lock(&locks_[rootA]);
  omp_set_lock(&locks_[rootB]);

  int rA = FindRoot(rootA);
  int rB = FindRoot(rootB);
  if (rA == rB) {
    omp_unset_lock(&locks_[rootB]);
    omp_unset_lock(&locks_[rootA]);
    return;
  }

  if (rank_[rA] < rank_[rB]) {
    parent_[rA] = rB;
  } else if (rank_[rA] > rank_[rB]) {
    parent_[rB] = rA;
  } else {
    parent_[rB] = rA;
    rank_[rA]++;
  }

  omp_unset_lock(&locks_[rootB]);
  omp_unset_lock(&locks_[rootA]);
}

bool TaskOMP::PostProcessingImpl() {
  std::cout << "PostProcessingImpl: Starting post-processing...\n";
  int comp = 1;
  const int N = height_ * width_;

  for (int i = 0; i < N; ++i) {
    if (parent_[i] < 0) continue;
    // Находим корень (уже сжатый)
    int root = parent_[i];
    while (parent_[root] != root) {
      root = parent_[root];
    }
    if (label_[root] == 0) {
      label_[root] = comp++;
    }
    res_[i] = label_[root];
  }

  for (int i = 0; i < N; ++i) {
    omp_destroy_lock(&locks_[i]);
  }

  std::copy(res_.begin(), res_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}

bool TaskOMP::IsValidIndex(int i, int j) const { return (i >= 0 && i < height_ && j >= 0 && j < width_); }

}  // namespace shkurinskaya_e_bin_labeling_omp
