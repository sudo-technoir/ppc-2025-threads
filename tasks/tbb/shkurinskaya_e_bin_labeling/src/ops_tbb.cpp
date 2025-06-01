#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/spin_mutex.h>

#include <cstddef>
#include <utility>

#include "core/util/include/util.hpp"
#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

namespace shkurinskaya_e_bin_labeling_tbb {

bool TaskTBB::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskTBB::PreProcessingImpl() {
  size_t total_size = task_data->inputs_count[0];
  input_.resize(total_size);
  int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  for (size_t i = 0; i < total_size; ++i) {
    input_[i] = ptr[i];
  }
  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];

  res_.assign(width_ * height_, 0);
  parent_.assign(width_ * height_, -1);
  rank_.assign(width_ * height_, 0);
  label_.assign(width_ * height_, 0);
  return true;
}

int TaskTBB::FindRoot(int v) {
  int u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0) {
      break;
    }
    int gp = parent_[p];
    if (gp < 0) {
      break;
    }
    if (p == gp) {
      break;
    }

    parent_[u] = gp;
    u = gp;
  }
  if (u < 0) {
    return -1;
  }
  int root = u;
  u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0 || p == root) {
      break;
    }
    parent_[u] = root;
    u = p;
  }
  return root;
}

void TaskTBB::UnionSets(int idx_a, int idx_b) {
  int root_a = FindRoot(idx_a);
  int root_b = FindRoot(idx_b);
  if (root_a == root_b || root_a < 0 || root_b < 0) {
    return;
  }

  tbb::spin_mutex::scoped_lock lock(uf_mutex_);

  root_a = FindRoot(root_a);
  root_b = FindRoot(root_b);
  if (root_a == root_b) {
    return;
  }

  if (rank_[root_a] < rank_[root_b]) {
    std::swap(root_a, root_b);
  }
  parent_[root_b] = root_a;
  if (rank_[root_a] == rank_[root_b]) {
    rank_[root_a]++;
  }
}

int TaskTBB::FindRootGlobal(int v, std::vector<int> &parent_vec) {
  int x = v;
  while (parent_vec[x] >= 0 && parent_vec[x] != x) {
    x = parent_vec[x];
  }
  if (x < 0) {
    return -1;
  }
  int root = x;
  x = v;
  while (parent_vec[x] != root) {
    int p = parent_vec[x];
    parent_vec[x] = root;
    x = p;
  }
  return root;
}

bool TaskTBB::RunImpl() {
  InitializeUF();

  std::vector<std::pair<int, int>> all_pairs;
  CollectPairs(all_pairs);

  ExecuteUnions(all_pairs);
  CompressPaths();

  return true;
}

void TaskTBB::InitializeUF() {
  const int w = width_, h = height_;
  tbb::parallel_for(tbb::blocked_range<int>(0, h), [&](const tbb::blocked_range<int> &rows) {
    for (int i = rows.begin(); i < rows.end(); ++i) {
      int base_idx = i * w;
      for (int j = 0; j < w; ++j) {
        int idx = base_idx + j;
        if (input_[idx] == 1) {
          parent_[idx] = idx;
          rank_[idx] = 0;
        } else {
          parent_[idx] = -1;
        }
      }
    }
  });
}

void TaskTBB::CollectPairs(std::vector<std::pair<int, int>> &all_pairs) {
  const int w = width_, h = height_;
  all_pairs.reserve(h * w / 2);
  for (int i = 0; i < h; ++i) {
    int base = i * w;
    for (int j = 0; j < w; ++j) {
      int idx = base + j;
      if (input_[idx] != 1) {
        continue;
      }
      if (j + 1 < w && input_[idx + 1] == 1) {
        all_pairs.emplace_back(idx, idx + 1);
      }
      if (i + 1 < h && input_[(i + 1) * w + j] == 1) {
        all_pairs.emplace_back(idx, (i + 1) * w + j);
      }
      if (i + 1 < h && j + 1 < w && input_[(i + 1) * w + (j + 1)] == 1) {
        all_pairs.emplace_back(idx, (i + 1) * w + (j + 1));
      }
      if (i + 1 < h && j > 0 && input_[(i + 1) * w + (j - 1)] == 1) {
        all_pairs.emplace_back(idx, (i + 1) * w + (j - 1));
      }
    }
  }
}

void TaskTBB::ExecuteUnions(const std::vector<std::pair<int, int>> &all_pairs) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, all_pairs.size()), [&](const tbb::blocked_range<size_t> &range) {
    for (size_t z = range.begin(); z < range.end(); ++z) {
      const auto &pr = all_pairs[z];
      UnionSets(pr.first, pr.second);
    }
  });
}

void TaskTBB::CompressPaths() {
  const int w = width_, h = height_;
  tbb::parallel_for(tbb::blocked_range<int>(0, h), [&](const tbb::blocked_range<int> &rows) {
    for (int i = rows.begin(); i < rows.end(); ++i) {
      int base_idx = i * w;
      for (int j = 0; j < w; ++j) {
        int idx = base_idx + j;
        if (input_[idx] == 1) {
          while (true) {
            int p = parent_[idx];
            int gp = (p >= 0 ? parent_[p] : -1);
            if (p < 0 || gp < 0 || p == gp) {
              break;
            }
            parent_[idx] = gp;
          }
        }
      }
    }
  });
}

bool TaskTBB::PostProcessingImpl() {
  const int w = width_;
  const int h = height_;

  std::ranges::fill(label_.begin(), label_.end(), 0);

  int comp = 1;
  for (int i = 0; i < h; ++i) {
    int base = i * w;
    for (int j = 0; j < w; ++j) {
      int idx = base + j;
      if (parent_[idx] < 0) {
        res_[idx] = 0;
        continue;
      }
      int root = idx;
      while (parent_[root] != root) {
        root = parent_[root];
      }
      if (label_[root] == 0) {
        label_[root] = comp++;
      }
      res_[idx] = label_[root];
    }
  }

  int *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
  for (size_t i = 0; i < res_.size(); ++i) {
    out_ptr[i] = res_[i];
  }
  return true;
}

bool TaskTBB::IsValidIndex(int i, int j) const { return (i >= 0 && i < height_ && j >= 0 && j < width_); }

}  // namespace shkurinskaya_e_bin_labeling_tbb
