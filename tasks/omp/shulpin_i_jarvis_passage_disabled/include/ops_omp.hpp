#pragma once

#include <omp.h>

#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shulpin_i_jarvis_omp {

struct Point {
  double x, y;
  Point() : x(0), y(0) {}
  Point(double x_coordinate, double y_coordinate) : x(x_coordinate), y(y_coordinate) {}
};

class JarvisSequential : public ppc::core::Task {
 public:
  explicit JarvisSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MakeJarvisPassage(std::vector<shulpin_i_jarvis_omp::Point>& input,
                                std::vector<shulpin_i_jarvis_omp::Point>& output);

 private:
  std::vector<shulpin_i_jarvis_omp::Point> input_seq_, output_seq_;
};

class JarvisOMPParallel : public ppc::core::Task {
 public:
  explicit JarvisOMPParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  static void MakeJarvisPassageOMP(std::vector<shulpin_i_jarvis_omp::Point>& input,
                                   std::vector<shulpin_i_jarvis_omp::Point>& output);

 private:
  std::vector<shulpin_i_jarvis_omp::Point> input_omp_, output_omp_;
};
}  // namespace shulpin_i_jarvis_omp
