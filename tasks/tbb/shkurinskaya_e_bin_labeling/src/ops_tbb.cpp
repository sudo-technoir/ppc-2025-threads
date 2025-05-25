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

void shkurinskaya_e_bin_labeling_tbb::TaskTBB::CompressPathsSequential_() {
  size_t total = static_cast<size_t>(width_) * height_;
  for (size_t i = 0; i < total; ++i)
    if (input_[i]) parent_[i] = FindRoot(static_cast<int>(i));
}

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::PreProcessingImpl() {
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto *tmp_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  width_ = reinterpret_cast<int *>(task_data->inputs[2])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
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
    index = parent_[index];
  }
  return index;
}

void shkurinskaya_e_bin_labeling_tbb::TaskTBB::UnionSets(int a, int b) {
  std::lock_guard<std::mutex> lock(union_mutex_);
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

bool TaskTBB::RunImpl() {
  const int H = height_, W = width_;
  const int N = H * W;

  // I. Инициализация множества
  tbb::parallel_for(0, H, [&](int i) {
    int base = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base + j;
      if (input_[idx] == 1) {
        parent_[idx] = idx;
        rank_[idx] = 0;
      } else {
        parent_[idx] = -1;
      }
    }
  });

  // II. Параллельное объединение по 8-ми соседям
  static const int dirs[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
  tbb::parallel_for(tbb::blocked_range2d<int>(0, H, 0, W), [&](auto const &br) {
    for (int i = br.rows().begin(); i != br.rows().end(); ++i) {
      int base = i * W;
      for (int j = br.cols().begin(); j != br.cols().end(); ++j) {
        int idx = base + j;
        if (input_[idx] != 1) continue;
        for (auto &d : dirs) {
          int ni = i + d[0], nj = j + d[1];
          if (ni < 0 || ni >= H || nj < 0 || nj >= W) continue;
          UnionSets(idx, ni * W + nj);
        }
      }
    }
  });

  // III. Сжатие путей (последовательное)
  CompressPathsSequential_();

  return true;
}

bool shkurinskaya_e_bin_labeling_tbb::TaskTBB::PostProcessingImpl() {
  const int N = height_ * width_;
  int comp = 1;
  std::fill(label_.begin(), label_.end(), 0);
  for (int idx = 0; idx < N; ++idx) {
    if (parent_[idx] < 0) {
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
