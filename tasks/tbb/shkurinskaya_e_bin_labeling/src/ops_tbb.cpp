#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <climits>
#include <vector>

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_.begin());

  int size = width_ * height_;
  res_.resize(size);
  parent_.resize(size);
  rank_.resize(size);
  label_.resize(size);
  return true;
}

int shkurinskaya_e_bin_labeling_tbb::TaskTBB::FindRoot(int index) {
  while (parent_[index] != index) {
    parent_[index] = parent_[parent_[index]];
    index = parent_[index];
  }
  return index;
}

void shkurinskaya_e_bin_labeling_tbb::TaskTBB::UnionSets(int a, int b) {
  int root_a = FindRoot(a);
  int root_b = FindRoot(b);
  if (root_a == root_b) return;

  if (rank_[root_a] < rank_[root_b]) {
    parent_[root_a] = root_b;
  } else if (rank_[root_a] > rank_[root_b]) {
    parent_[root_b] = root_a;
  } else {
    parent_[root_b] = root_a;
    rank_[root_a]++;
  }
}

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::RunImpl() {
  const int directions[8][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}, {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

  // Init parents
  tbb::parallel_for(0, height_, [&](int i) {
    for (int j = 0; j < width_; ++j) {
      int idx = i * width_ + j;
      if (input_[idx] == 1) {
        parent_[idx] = idx;
        rank_[idx] = 0;
      } else {
        parent_[idx] = -1;
      }
    }
  });

  // Union-Find
  tbb::parallel_for(tbb::blocked_range2d<int>(0, height_, 0, width_), [&](const tbb::blocked_range2d<int> &r) {
    for (int i = r.rows().begin(); i < r.rows().end(); ++i) {
      for (int j = r.cols().begin(); j < r.cols().end(); ++j) {
        int index = i * width_ + j;
        if (input_[index] != 1) continue;
        for (int d = 0; d < 8; ++d) {
          int ni = i + directions[d][0];
          int nj = j + directions[d][1];
          if (ni >= 0 && ni < height_ && nj >= 0 && nj < width_) {
            int neighbor_index = ni * width_ + nj;
            if (input_[neighbor_index] == 1) {
              UnionSets(index, neighbor_index);
             }
                }
              }
            }
          }
        }
});

// Path compression
tbb::parallel_for(0, height_, [&](int i) {
  for (int j = 0; j < width_; ++j) {
    int idx = i * width_ + j;
    if (input_[idx] == 1) {
      parent_[idx] = FindRoot(idx);
    }
  }
});
return true;

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::PostProcessingImpl() {
  int comp = 1;
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int idx = i * width_ + j;
      if (parent_[idx] == -1) continue;
      int root = FindRoot(idx);
      if (label_[root] == 0) label_[root] = comp++;
      res_[idx] = label_[root];
    }
  }
  std::copy(res_.begin(), res_.end(), reinterpret_cast<int *>(task_data->outputs[0]));
  return true;
}
