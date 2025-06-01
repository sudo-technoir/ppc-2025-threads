#include "stl/shkurinskaya_e_bin_labeling/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace shkurinskaya_e_bin_labeling_stl {

bool TaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskSTL::PreProcessingImpl() {
  const int total_size = task_data->inputs_count[0];
  int *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + total_size);

  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];

  res_.resize(total_size);
  parent_.resize(total_size);
  rank_.resize(total_size);
  label_.resize(total_size);

  return true;
}

int TaskSTL::FindRoot(int v) {
  int u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0 || p == u) break;
    u = p;
  }
  if (u < 0) return -1;
  int root = u;
  u = v;
  while (true) {
    int p = parent_[u];
    if (p < 0 || p == root) break;
    parent_[u] = root;
    u = p;
  }
  return root;
}

void TaskSTL::UnionSets(int idx_a, int idx_b) {
  int rootA = FindRoot(idx_a);
  int rootB = FindRoot(idx_b);
  if (rootA == rootB || rootA < 0 || rootB < 0) return;

  std::lock_guard<std::mutex> lock(uf_mutex_);

  rootA = FindRoot(rootA);
  rootB = FindRoot(rootB);
  if (rootA == rootB) return;

  if (rank_[rootA] < rank_[rootB]) {
    std::swap(rootA, rootB);
  }
  parent_[rootB] = rootA;
  if (rank_[rootA] == rank_[rootB]) {
    rank_[rootA]++;
  }
}

void TaskSTL::InitializeUFRange(int row_start, int row_end) {
  const int W = width_;
  for (int i = row_start; i < row_end; ++i) {
    int base_idx = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base_idx + j;
      if (input_[idx] == 1) {
        parent_[idx] = idx;
        rank_[idx] = 0;
      } else {
        parent_[idx] = -1;
      }
    }
  }
}

void TaskSTL::CompressPathsRange(int row_start, int row_end) {
  const int W = width_;
  for (int i = row_start; i < row_end; ++i) {
    int base_idx = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base_idx + j;
      if (input_[idx] == 1) {
        while (true) {
          int p = parent_[idx];
          if (p < 0) break;
          int gp = parent_[p];
          if (gp < 0 || p == gp) break;
          parent_[idx] = gp;
        }
      }
    }
  }
}

bool TaskSTL::RunImpl() {
  const int W = width_;
  const int H = height_;

  // 1) Parallel InitializeUFRange по строкам
  {
    int num_threads = NumThreads() > 0 ? NumThreads() : 1;
    int Trows = std::min(num_threads, H);
    if (Trows <= 0) Trows = 1;

    auto compute_row_range = [&](int t) {
      int start = (H * t) / Trows;
      int end = (H * (t + 1)) / Trows;
      return std::pair<int, int>(start, end);
    };

    std::vector<std::thread> threads;
    threads.reserve(Trows);
    for (int t = 0; t < Trows; ++t) {
      auto tmp = compute_row_range(t);
      int r0 = tmp.first;
      int r1 = tmp.second;
      threads.emplace_back(&TaskSTL::InitializeUFRange, this, r0, r1);
    }
    for (auto &th : threads) th.join();
  }
  // 2) Собираем список всех пар allPairs (single‐thread)
  std::vector<std::pair<int, int>> allPairs;
  allPairs.reserve((size_t)H * W / 2);
  for (int i = 0; i < H; ++i) {
    int base_idx = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base_idx + j;
      if (input_[idx] != 1) continue;
      if (j + 1 < W && input_[idx + 1] == 1) allPairs.emplace_back(idx, idx + 1);
      if (i + 1 < H) {
        int idx_down = (i + 1) * W + j;
        if (input_[idx_down] == 1) allPairs.emplace_back(idx, idx_down);
      }
      if (i + 1 < H && j + 1 < W) {
        int idx_dr = (i + 1) * W + (j + 1);
        if (input_[idx_dr] == 1) allPairs.emplace_back(idx, idx_dr);
      }
      if (i + 1 < H && j > 0) {
        int idx_dl = (i + 1) * W + (j - 1);
        if (input_[idx_dl] == 1) allPairs.emplace_back(idx, idx_dl);
      }
    }
  }

  // 3) Parallel UnionSets по allPairs
  {
    size_t M = allPairs.size();
    int num_threads = NumThreads() > 0 ? NumThreads() : 1;
    int Tpairs = std::min<int>(num_threads, (int)M);
    if (Tpairs <= 0) Tpairs = 1;

    auto compute_pair_range = [&](int t) {
      size_t start = (M * t) / Tpairs;
      size_t end = (M * (t + 1)) / Tpairs;
      return std::pair<size_t, size_t>(start, end);
    };

    std::vector<std::thread> threads;
    threads.reserve(Tpairs);
    for (int t = 0; t < Tpairs; ++t) {
      auto tmp = compute_pair_range(t);
      size_t b = tmp.first;
      size_t e = tmp.second;
      threads.emplace_back([&, b, e]() {
        for (size_t z = b; z < e; ++z) {
          const auto &pr = allPairs[z];
          UnionSets(pr.first, pr.second);
        }
      });
    }
    for (auto &th : threads) th.join();
  }

  // 4) Parallel CompressPathsRange по строкам
  {
    int num_threads = NumThreads() > 0 ? NumThreads() : 1;
    int Trows = std::min(num_threads, H);
    if (Trows <= 0) Trows = 1;
  
    auto compute_row_range = [&](int t) {
      int start = (H * t) / Trows;
      int end = (H * (t + 1)) / Trows;
      return std::pair<int, int>(start, end);
    };

    std::vector<std::thread> threads;
    threads.reserve(Trows);
    for (int t = 0; t < Trows; ++t) {
      auto tmp = compute_row_range(t);
      int r0 = tmp.first;
      int r1 = tmp.second;
      threads.emplace_back(&TaskSTL::CompressPathsRange, this, r0, r1);
    }
    for (auto &th : threads) th.join();
  }
  return true;
}

bool TaskSTL::PostProcessingImpl() {
  const int W = width_;
  const int H = height_;

  std::fill(label_.begin(), label_.end(), 0);
  int comp = 1;

  for (int i = 0; i < H; ++i) {
    int base_idx = i * W;
    for (int j = 0; j < W; ++j) {
      int idx = base_idx + j;
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
  std::copy(res_.begin(), res_.end(), out_ptr);

  return true;
}

}  // namespace shkurinskaya_e_bin_labeling_stl
