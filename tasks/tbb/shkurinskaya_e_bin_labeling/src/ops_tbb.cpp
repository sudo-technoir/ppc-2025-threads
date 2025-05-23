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

void shkurinskaya_e_bin_labeling_tbb::TaskTBB::ParallelCollectPairs_(
    tbb::concurrent_vector<std::pair<size_t, size_t>> &pairs) {
  auto range = tbb::blocked_range2d<int>(0, height_, 0, width_, 64, 64);

  tbb::parallel_for(range, [&, this](const tbb::blocked_range2d<int> &r) {
    for (int r0 = r.rows().begin(); r0 != r.rows().end(); ++r0)
      for (int c0 = r.cols().begin(); c0 != r.cols().end(); ++c0) {
        size_t idx = CoordToIndex(r0, c0);
        if (!input_[idx]) continue;

        for (auto [dr, dc] : directions) {
          int nr = r0 + dr, nc = c0 + dc;
          if (!IsValidCoord(nr, nc)) continue;

          size_t nidx = CoordToIndex(nr, nc);
          if (input_[nidx])
            pairs.push_back({idx, nidx});
        }
      }
  });
}

void shkurinskaya_e_bin_labeling_tbb::TaskTBB::CompressPathsSequential_() {
  for (size_t i = 0; i < total_; ++i)
    if (input_[i]) parent_[i] = FindRoot(i);
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

  tbb::concurrent_vector<std::pair<size_t, size_t>> pairs;
  ParallelCollectPairs_(pairs);

  for (auto& p : pairs) UnionSets(p.first, p.second);
  CompressPathsSequential_();
  return true;
}

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
