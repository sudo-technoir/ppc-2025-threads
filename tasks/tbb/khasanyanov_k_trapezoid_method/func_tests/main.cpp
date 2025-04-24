#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "../include/integrate_tbb.hpp"
#include "../include/integrator.hpp"
#include "core/task/include/task.hpp"

using namespace khasanyanov_k_trapezoid_method_tbb;

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_linear_function) {
  auto f = [](const std::vector<double>& x) -> double { return (5 * x[0]) + (2 * x[1]) - (3 * x[2]); };

  IntegrationBounds bounds = {{-3, 1.0}, {0.0, 2.0}, {0.5, 1.0}};

  double precision = 1e-6;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(-21.0, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_quad_function) {
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1]) - (6.5 * x[2]); };

  IntegrationBounds bounds = {{-3.0, -2.0}, {0.0, 2.0}, {0.5, 1.0}};

  double precision = 0.01;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(3.4583, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_mixed_function) {
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1] * x[0]) + x[2]; };

  IntegrationBounds bounds = {{-2.5, 0.0}, {0.0, 3.0}, {2.0, 2.5}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(2.1875, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_trigonometric_function) {
  auto f = [](const std::vector<double>& x) -> double { return sin(x[0]) - x[1]; };

  IntegrationBounds bounds = {{0.0, 1.0}, {0.0, 2.0}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(-1.08060, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_long_function) {
  auto f = [](const std::vector<double>& x) -> double {
    return x[0] + (x[1] / 2.0) - (x[2] / 3.0) + (x[3] / 4.0) - (x[4] / 5.0);
  };

  IntegrationBounds bounds = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);

  ASSERT_NEAR(0.60833, result, precision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_wrong_bounds) {
  auto f = [](const std::vector<double>& x) -> double {
    return x[0] + (x[1] / 2.0) - (x[2] / 3.0) + (x[3] / 4.0) - (x[4] / 5.0);
  };

  IntegrationBounds bounds = {{0.0, 1.0}, {2.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};

  double precision = 0.001;
  ASSERT_ANY_THROW(Integrator<kSequential>{}(f, bounds, precision));
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrator_constant_function) {
  auto f = [](const std::vector<double>& x) -> double { return 45; };

  IntegrationBounds bounds = {{2.0, 3.0}};

  double precision = 0.001;
  double result = Integrator<kSequential>{}(f, bounds, precision);
  ASSERT_NEAR(45.0, result, precision);
}

//-----------------------------------------------------------------------------------------------------------------------------------------//

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrate_1) {
  constexpr double kPrecision = 1e-6;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (5 * x[0]) + (2 * x[1]) - (3 * x[2]); };

  IntegrationBounds bounds = {{-3, 1.0}, {0.0, 2.0}, {0.5, 1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodTBB::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodTBB task(task_data_seq);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_NEAR(-21.0, result, kPrecision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrate_2) {
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1]) - (6.5 * x[2]); };

  IntegrationBounds bounds = {{-3, -2.0}, {0.0, 2.0}, {0.5, 1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodTBB::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodTBB task(task_data_seq);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_NEAR(3.4583, result, kPrecision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrate_3) {
  constexpr double kPrecision = 0.001;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return sin(x[0]) - x[1]; };

  IntegrationBounds bounds = {{0.0, 1.0}, {0.0, 2.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodTBB::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodTBB task(task_data_seq);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_NEAR(-1.08060, result, kPrecision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_integrate_4) {
  constexpr double kPrecision = 0.001;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (7.4 * x[0]) - (x[1] * x[1]); };

  IntegrationBounds bounds = {{-50.0, -47.0}, {-2.0, -1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodTBB::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodTBB task(task_data_seq);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  ASSERT_NEAR(-1083.7, result, kPrecision);
}

TEST(khasanyanov_k_trapezoid_method_tbb, test_invalid_input) {
  constexpr double kPrecision = 0.001;
  auto f = [](const std::vector<double>& x) -> double { return sin(x[0]) - x[1]; };

  IntegrationBounds bounds;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodTBB::CreateTaskData(task_data_seq, context, nullptr);
  TrapezoidalMethodTBB task(task_data_seq);

  ASSERT_FALSE(task.Validation());
}
