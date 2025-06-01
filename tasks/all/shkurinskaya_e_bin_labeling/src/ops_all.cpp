#include "all/shkurinskaya_e_bin_labeling/include/ops_all.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace shkurinskaya_e_bin_labeling_all {

bool TaskMPITBB::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] == task_data->inputs_count[0] &&
         task_data->inputs_count[1] == 1 && task_data->inputs_count[2] == 1;
}

bool TaskMPITBB::PreProcessingImpl() {
  int rank = world_.rank();

  if (rank == 0) {
    int total_size = task_data->inputs_count[0];
    int *ptr = reinterpret_cast<int *>(task_data->inputs[0]);
    input_global_.assign(ptr, ptr + total_size);

    width_ = reinterpret_cast<int *>(task_data->inputs[1])[0];
    height_ = reinterpret_cast<int *>(task_data->inputs[2])[0];

    boost::mpi::broadcast(world_, width_, 0);
    boost::mpi::broadcast(world_, height_, 0);
  } else {
    boost::mpi::broadcast(world_, width_, 0);
    boost::mpi::broadcast(world_, height_, 0);
  }

  // 2) Разбиваем по строкам
  int num_procs = world_.size();
  counts_.assign(num_procs, height_ / num_procs);
  for (int i = 0; i < height_ % num_procs; ++i) {
    counts_[i]++;
  }
  displs_.assign(num_procs, 0);
  for (int i = 1; i < num_procs; ++i) {
    displs_[i] = displs_[i - 1] + counts_[i - 1];
  }
  // counts_[i] = число строк, отданных rank i
  local_H_ = counts_[rank];
  local_offset_rows_ = displs_[rank];

  std::vector<int> counts_with_pixels(num_procs);
  std::vector<int> displs_with_pixels(num_procs);
  for (int i = 0; i < num_procs; ++i) {
    counts_with_pixels[i] = counts_[i] * width_;
    displs_with_pixels[i] = displs_[i] * width_;
  }

  // 4) Scatterv: отправляем rank’ам нужные куски
  std::vector<int> local_flat(local_H_ * width_);
  boost::mpi::scatterv(world_,
                       input_global_.data(),   // у rank0 — полный массив
                       counts_with_pixels,     // сколько пикселей на каждый rank
                       displs_with_pixels,     // сдвиг в пикселях
                       local_flat.data(),      // буфер, куда записать локально
                       local_H_ * width_, 0);  // rank0 — root communicator

  // 5) Сохраняем локальный кусок в input_
  input_ = std::move(local_flat);

  // Готовим под local_H_ * width_:
  res_local_.resize(local_H_ * width_);
  parent_.resize(local_H_ * width_);
  rank_.resize(local_H_ * width_);
  label_.resize(local_H_ * width_);

  return true;
}

bool TaskMPITBB::RunImpl() {
  int rank = world_.rank();
  int num_procs = world_.size();

  tbb::parallel_for(0, local_H_, [&](int i) {
    int base = i * width_;
    for (int j = 0; j < width_; ++j) {
      int idx = base + j;
      if (input_[idx] == 1) {
        parent_[idx] = idx;
        rank_[idx] = 0;
      } else {
        parent_[idx] = -1;
      }
    }
  });

  // 1.2) Собираем localPairs (локальные пары 1–1)
  std::vector<std::pair<int, int>> localPairs;
  localPairs.reserve(local_H_ * width_ / 2);
  for (int i = 0; i < local_H_; ++i) {
    int base = i * width_;
    for (int j = 0; j < width_; ++j) {
      int idx = base + j;
      if (input_[idx] != 1) continue;
      // вправо
      if (j + 1 < width_ && input_[idx + 1] == 1) {
        localPairs.emplace_back(idx, idx + 1);
      }
      // вниз (если внутри локальной полосы)
      if (i + 1 < local_H_ && input_[(i + 1) * width_ + j] == 1) {
        localPairs.emplace_back(idx, (i + 1) * width_ + j);
      }
      // диагонали вниз-вправо и вниз-влево (если i+1 < local_H_)
      if (i + 1 < local_H_ && j + 1 < width_ && input_[(i + 1) * width_ + (j + 1)] == 1) {
        localPairs.emplace_back(idx, (i + 1) * width_ + (j + 1));
      }
      if (i + 1 < local_H_ && j - 1 >= 0 && input_[(i + 1) * width_ + (j - 1)] == 1) {
        localPairs.emplace_back(idx, (i + 1) * width_ + (j - 1));
      }
    }
  }

  // 1.3) Параллельный UnionSets для localPairs
  tbb::parallel_for(size_t(0), localPairs.size(), [&](size_t z) {
    auto [a, b] = localPairs[z];
    UnionSets(a, b);
  });

  // 1.4) Параллельный path‐compression
  tbb::parallel_for(0, local_H_, [&](int i) {
    int base = i * width_;
    for (int j = 0; j < width_; ++j) {
      int idx = base + j;
      if (input_[idx] == 1) {
        while (true) {
          int p = parent_[idx];
          if (p < 0 || parent_[p] == p) break;
          parent_[idx] = parent_[p];
        }
      }
    }
  });

  std::vector<int> top_row(width_, 0), bottom_row(width_, 0);
  if (local_H_ > 0) {
    std::copy(input_.begin() + (local_H_ - 1) * width_, input_.begin() + local_H_ * width_, top_row.begin());
  }
  if (local_H_ > 0) {
    std::copy(input_.begin(), input_.begin() + width_, bottom_row.begin());
  }

  std::vector<int> recv_top(width_, 0), recv_bottom(width_, 0);
  int prev = (rank - 1 + num_procs) % num_procs;
  int next = (rank + 1) % num_procs;

  boost::mpi::request req_top = world_.irecv(prev, 0, recv_top.data(), width_);
  world_.send(next, 0, top_row.data(), width_);
  req_top.wait();

  boost::mpi::request req_bottom = world_.irecv(next, 1, recv_bottom.data(), width_);
  world_.send(prev, 1, bottom_row.data(), width_);
  req_bottom.wait();

  std::vector<std::pair<int, int>> local_boundary_pairs;
  if (rank < num_procs - 1) {
    for (int j = 0; j < width_; ++j) {
      if (top_row[j] == 1 && recv_bottom[j] == 1) {
        int local_idx = (local_H_ - 1) * width_ + j;
        int global_idx = local_offset_rows_ * width_ + local_idx;
        int neighbor_global_idx = (local_offset_rows_ + local_H_) * width_ + j;
        local_boundary_pairs.emplace_back(global_idx, neighbor_global_idx);
      }
    }
  }

  int local_pairs_count = static_cast<int>(local_boundary_pairs.size());
  std::vector<int> boundary_counts(num_procs);
  boost::mpi::all_gather(world_, local_pairs_count, boundary_counts);

  std::vector<int> boundary_displs(num_procs, 0);
  for (int i = 1; i < num_procs; ++i) {
    boundary_displs[i] = boundary_displs[i - 1] + boundary_counts[i - 1];
  }

  std::vector<std::pair<int, int>> allBoundaryPairs;
  if (rank == 0) {
    int total_boundary = boundary_displs[num_procs - 1] + boundary_counts[num_procs - 1];
    allBoundaryPairs.resize(total_boundary);
  }

  boost::mpi::gatherv(world_, local_boundary_pairs.data(), local_pairs_count,
                      rank == 0 ? allBoundaryPairs.data() : nullptr, boundary_counts, boundary_displs, 0);

  int total_nodes = width_ * height_;
  std::vector<int> global_map;

  if (rank == 0) {
    std::vector<int> global_parent(total_nodes, -1);
    std::vector<int> global_rank(total_nodes, 0);
    for (int u = 0; u < total_nodes; ++u) {
      if (input_global_[u] == 1) {
        global_parent[u] = u;
        global_rank[u] = 0;
      }
    }

    for (auto &pr : allBoundaryPairs) {
      int u = pr.first;
      int v = pr.second;
      int ru = FindRootGlobal(u, global_parent);
      int rv = FindRootGlobal(v, global_parent);
      if (ru >= 0 && rv >= 0 && ru != rv) {
        if (global_rank[ru] < global_rank[rv]) std::swap(ru, rv);
        global_parent[rv] = ru;
        if (global_rank[ru] == global_rank[rv]) global_rank[ru]++;
      }
    }

    global_map.resize(total_nodes);
    for (int u = 0; u < total_nodes; ++u) {
      int ru = FindRootGlobal(u, global_parent);
      global_map[u] = ru;
    }
  }

  if (rank != 0) {
    global_map.resize(total_nodes);
  }

  boost::mpi::broadcast(world_, global_map, 0);
  tbb::parallel_for(0, local_H_, [&](int i) {
    int base = i * width_;
    for (int j = 0; j < width_; ++j) {
      int idx = base + j;
      if (input_[idx] == 1) {
        int local_root = parent_[idx];
        int global_root = local_offset_rows_ * width_ + local_root;
        int final_root = global_map[global_root];
        res_local_[idx] = final_root;
      } else {
        res_local_[idx] = 0;
      }
    }
  });

  return true;
}

bool TaskMPITBB::PostProcessingImpl() {
  int rank = world_.rank();
  int num_procs = world_.size();

  std::vector<int> res_global;
  if (rank == 0) {
    res_global.resize(width_ * height_);
  }

  std::vector<int> counts_with_pixels(num_procs);
  std::vector<int> displs_with_pixels(num_procs);
  for (int i = 0; i < num_procs; ++i) {
    counts_with_pixels[i] = counts_[i] * width_;
    displs_with_pixels[i] = displs_[i] * width_;
  }
  boost::mpi::gatherv(world_, res_local_.data(), static_cast<int>(local_H_ * width_),
                      rank == 0 ? res_global.data() : nullptr, counts_with_pixels, displs_with_pixels, 0);

  if (rank == 0) {
    int *out_ptr = reinterpret_cast<int *>(task_data->outputs[0]);
    std::ranges::copy(res_global.begin(), res_global.end(), out_ptr);
  }
  return true;
}

void TaskMPITBB::UnionSets(int idx_a, int idx_b) {
  int rootA = FindRoot(idx_a);
  int rootB = FindRoot(idx_b);
  if (rootA == rootB || rootA < 0 || rootB < 0) return;

  tbb::spin_mutex::scoped_lock lock(uf_mutex_);

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

int TaskMPITBB::FindRoot(int v) {
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

int TaskMPITBB::FindRootGlobal(int v, std::vector<int> &parent_vec) {
  int x = v;
  // шаг 1: найти корень
  while (parent_vec[x] >= 0 && parent_vec[x] != x) {
    x = parent_vec[x];
  }
  if (x < 0) return -1;
  int root = x;
  // шаг 2: сжать путь
  x = v;
  while (parent_vec[x] != root) {
    int p = parent_vec[x];
    parent_vec[x] = root;
    x = p;
  }
  return root;
}

bool TaskMPITBB::IsValidIndex(int i, int j) const { return (i >= 0 && i < height_ && j >= 0 && j < width_); }

}  // namespace shkurinskaya_e_bin_labeling_all
