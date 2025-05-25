#include "omp/shkurinskaya_e_bin_labeling/include/ops_omp.hpp"

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

void shkurinskaya_e_bin_labeling_omp::TaskOMP::ProcessUnion() {
  const int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
#pragma omp parallel for schedule(dynamic)
  for (int idx = 0; idx < N; ++idx) {
    if (input_[idx] != 1) continue;
    int x = idx % W;
    int y = idx / W;
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        if (dx == 0 && dy == 0) continue;
        int nx = x + dx, ny = y + dy;
        if (!IsValidIndex(nx, ny)) continue;
        int nidx = ny * W + nx;
        if (input_[nidx] == 1) {
          UnionSets(idx, nidx);
        }
      }
    }
  }

  bool shkurinskaya_e_bin_labeling_omp::TaskOMP::IsValidIndex(int i, int j) const {
    return (i >= 0 && i < height_ && j >= 0 && j < width_);
  }

  void shkurinskaya_e_bin_labeling_omp::TaskOMP::UnionSets(int a, int b) {
    int rootA = FindRoot(a);
    int rootB = FindRoot(b);
    if (rootA == rootB) return;

#pragma omp critical
    {
      rootA = FindRoot(rootA);
      rootB = FindRoot(rootB);
      if (rootA == rootB) return;

      if (rank_[rootA] < rank_[rootB]) {
        parent_[rootA] = rootB;
      } else if (rank_[rootA] > rank_[rootB]) {
        parent_[rootB] = rootA;
      } else {
        parent_[rootB] = rootA;
        ++rank_[rootA];
      }
    }
  }

  inline int shkurinskaya_e_bin_labeling_omp::TaskOMP::FindRoot(int i) {
    if (parent_[i] != i) {
      parent_[i] = FindRoot(parent_[i]);
    }
    return parent_[i];
  }

  bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PreProcessingImpl() {
    // Init value for input and output
    std::cout << "PreProcessingImpl: Initializing inputs and outputs...\n";
    input_ = std::vector<int>(task_data->inputs_count[0]);
    auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
    height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];
    std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_.begin());
    // Init value for output
    res_.resize(task_data->inputs_count[0]);
    parent_.resize(task_data->inputs_count[0]);
    rank_.resize(task_data->inputs_count[0]);
    label_.resize(task_data->inputs_count[0]);
    return true;
  }

  bool shkurinskaya_e_bin_labeling_omp::TaskOMP::ValidationImpl() {
    std::cout << "ValidationImpl: Validating input data...\n";
    // Check count elements of output
    return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
           task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
  }

  bool shkurinskaya_e_bin_labeling_omp::TaskOMP::RunImpl() {
    std::cout << "[DEBUG] RunImpl: Starting processing...\n";

    // Первый этап
#pragma omp parallel for
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        int index = (i * width_) + j;
        if (input_[index] == 1) {
          parent_[index] = index;
          rank_[index] = 0;
        } else {
          parent_[index] = -1;
        }
      }
    }

    // Второй этап
    ProcessUnion();

// Третий этап
#pragma omp parallel for
    for (int idx = 0; idx < N; ++idx) {
      if (input_[idx] == 1) {
        parent_[idx] = FindRoot(idx);
      }
    }
    std::cout << "[DEBUG] RunImpl: Processing completed\n";
    return true;
  }

  bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PostProcessingImpl() {
    std::cout << "PostProcessingImpl: Starting post-processing...\n";
    // mark the parent_ with smallest label
    int comp = 1;
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        int index = (i * width_) + j;
        int root = index;
        if (parent_[root] == -1) {
          continue;
        }
        while (parent_[root] != root) {
          root = parent_[root];
        }
        if (label_[parent_[root]] == 0) {
          label_[parent_[root]] = comp++;
        }
        res_[index] = label_[parent_[root]];
      }
    }
    std::ranges::copy(res_.begin(), res_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
    return true;
  }
