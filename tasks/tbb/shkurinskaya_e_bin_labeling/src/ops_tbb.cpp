#include "tbb/shkurinskaya_e_bin_labeling/include/ops_tbb.hpp"

#include <oneapi/tbb/blocked_range2d.h>
#include <oneapi/tbb/parallel_for.h>

#include <algorithm>
#include <climits>
#include <vector>
namespace shkurinskaya_e_bin_labeling_tbb {

bool TaskTBB::ValidationImpl() {

  return task_data->inputs_count[0] > 1 &&
         task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskTBB::PreProcessingImpl() {

  const int total_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + total_size);

  width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
  height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];

  res_.resize(total_size);
  parent_.resize(total_size);
  rank_.resize(total_size);
  label_.resize(total_size);
  return true;
}

bool TaskTBB::RunImpl() {
  const int W = width_;
  const int H = height_;
  const int N = W * H;

  tbb::parallel_for(tbb::blocked_range<int>(0, H),
                    [&](const tbb::blocked_range<int> &rows) {
                      for (int i = rows.begin(); i < rows.end(); ++i) {
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
                      }
                    });

  ProcessUnion();

  tbb::parallel_for(tbb::blocked_range<int>(0, H),
                    [&](const tbb::blocked_range<int> &rows) {
                      for (int i = rows.begin(); i < rows.end(); ++i) {
                        int base = i * W;
                        for (int j = 0; j < W; ++j) {
                          int idx = base + j;
                          if (input_[idx] == 1) {
                            while (true) {
                              int p = parent_[idx];
                              if (p < 0)
                                break;
                              int gp = parent_[p];
                              if (gp < 0)
                                break;
                              if (p == gp)
                                break;
                              parent_[idx] = gp;
                            }
                          }
                        }
                      }
                    });

  return true;
}

void TaskTBB::ProcessUnion() {
  const int W = width_;
  const int H = height_;
  static constexpr int dirs[8][2] = {{-1, 0},  {1, 0},  {0, -1}, {0, 1},
                                     {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};

  tbb::parallel_for(tbb::blocked_range<int>(0, H),
                    [&](const tbb::blocked_range<int> &rows) {
                      for (int i = rows.begin(); i < rows.end(); ++i) {
                        int base = i * W;
                        for (int j = 0; j < W; ++j) {
                          int idx = base + j;
                          if (input_[idx] != 1)
                            continue;
                          for (int d = 0; d < 8; ++d) {
                            int ni = i + dirs[d][0];
                            int nj = j + dirs[d][1];
                            if (!IsValidIndex(ni, nj))
                              continue;
                            int nidx = ni * W + nj;
                            if (input_[nidx] == 1) {
                              UnionSets(idx, nidx);
                            }
                          }
                        }
                      }
                    });
}

void TaskTBB::UnionSets(int idx_a, int idx_b) {
  int rootA = FindRoot(idx_a);
  int rootB = FindRoot(idx_b);
  if (rootA == rootB || rootA < 0 || rootB < 0)
    return;

  tbb::spin_mutex::scoped_lock lock(uf_mutex_);

  rootA = FindRoot(rootA);
  rootB = FindRoot(rootB);
  if (rootA == rootB)
    return;

  if (rank_[rootA] < rank_[rootB]) {
    std::swap(rootA, rootB);
  }
  parent_[rootB] = rootA;
  if (rank_[rootA] == rank_[rootB]) {
    rank_[rootA]++;
  }
}

int TaskTBB::FindRoot(int v) {
  int p = parent_[v];
  if (p < 0)
    return -1;
  if (p == v)
    return v;
  int root = FindRoot(p);
  parent_[v] = root;
  return root;
}

bool TaskTBB::IsValidIndex(int i, int j) const {
  return (i >= 0 && i < height_ && j >= 0 && j < width_);
}

bool TaskTBB::PostProcessingImpl() {
  const int W = width_;
  const int H = height_;
  const int N = W * H;

  std::fill(label_.begin(), label_.end(), 0);

  int comp = 1;

  for (int i = 0; i < H; ++i) {
    int base = i * W;
    for (int j = 0; j < W; ++j) {
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
  std::ranges::copy(res_.begin(), res_.end(), out_ptr);
  return true;
}

} // namespace shkurinskaya_e_bin_labeling_tbb
