#include "omp/shkurinskaya_e_bin_labeling/include/ops_omp.hpp"

#include <algorithm>
#include <climits>
#include <vector>

static const int directions[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

void shkurinskaya_e_bin_labeling_omp::TaskOMP::ParallelCollectPairs_(std::vector<std::pair<size_t, size_t>>& pairs) {
  pairs.clear();
  pairs.reserve(static_cast<size_t>(width_) * height_);

#pragma omp parallel
  {
    std::vector<std::pair<size_t, size_t>> local;
    local.reserve(64);

#pragma omp for nowait schedule(static)
    for (int r = 0; r < height_; ++r) {
      for (int c = 0; c < width_; ++c) {
        int idx = r * width_ + c;
        if (input_[idx] == 0)
          continue;

        for (auto [dr, dc] : directions) {
          int nr = r + dr, nc = c + dc;
          if (!IsValidIndex(nr, nc))
            continue;

          int nidx = nr * width_ + nc;
          if (input_[nidx] == 1)
            local.emplace_back(static_cast<size_t>(idx),
                               static_cast<size_t>(nidx));
        }

        if (local.size() > 256) {
#pragma omp critical
          { pairs.insert(pairs.end(), local.begin(), local.end()); }
          local.clear();
        }
      }
    }

#pragma omp critical
    { pairs.insert(pairs.end(), local.begin(), local.end()); }
  }
}

void shkurinskaya_e_bin_labeling_omp::TaskOMP::ProcessUnion() {
  std::vector<std::pair<size_t, size_t>> pairs;
  ParallelCollectPairs_(pairs);
  for (auto& [a, b] : pairs) UnionSets(a, b);
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::IsValidIndex(int i, int j) const {
  return (i >= 0 && i < height_ && j >= 0 && j < width_);
}

void shkurinskaya_e_bin_labeling_omp::TaskOMP::UnionSets(int index_a, int index_b) {
  int root_a = FindRoot(index_a);
  int root_b = FindRoot(index_b);

  if (root_a != root_b) {
    if (rank_[root_a] < rank_[root_b]) {
      parent_[root_a] = root_b;
    } else if (rank_[root_a] > rank_[root_b]) {
      parent_[root_b] = root_a;
    } else {
      parent_[root_b] = root_a;
      rank_[root_a]++;
    }
  }
}

int shkurinskaya_e_bin_labeling_omp::TaskOMP::FindRoot(int index) {
  while (parent_[index] != index) {
    parent_[index] = parent_[parent_[index]];
    index = parent_[index];
  }
  return index;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PreProcessingImpl() {
  // Init value for input and output
  input_ = std::vector<int>(task_data->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  width_ = reinterpret_cast<int*>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int*>(task_data->inputs[2])[0];
  std::copy(tmp_ptr, tmp_ptr + task_data->inputs_count[0], input_.begin());
  // Init value for output
  res_.resize(task_data->inputs_count[0], 0);
  parent_.resize(task_data->inputs_count[0], 0);
  rank_.resize(task_data->inputs_count[0], 0);
  label_.resize(task_data->inputs_count[0], 0);
  return true;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::ValidationImpl() {
  // Check count elements of output
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::RunImpl() {
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
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int index = (i * width_) + j;
      if (input_[index] == 1) {
        while (parent_[index] != parent_[parent_[index]]) {
          parent_[index] = parent_[parent_[index]];
        }
      }
    }
  }
  return true;
}

bool shkurinskaya_e_bin_labeling_omp::TaskOMP::PostProcessingImpl() {
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
  std::ranges::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(task_data->outputs[0]));
  return true;
}
