#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "all/shkurinskaya_e_bin_labeling/include/ops_all.hpp"
#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

TEST(shkurinskaya_e_bin_labeling_all, empty_input) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  boost::mpi::communicator world;

  // Create data
  std::vector<int> in;
  std::vector<int> out(size);
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(task_all.Validation(), false);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, empty_output) {
  int height = 5000;
  int width = 5000;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out;
  std::vector<int> ans(size, 1);
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  if (world.rank() == 0) {
    ASSERT_EQ(task_all.Validation(), false);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, test_diag_object) {
  int height = 100;
  int width = 100;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int i = 0; i < height; ++i) {
    in[(i * width) + i] = 1;
    ans[(i * width) + i] = 1;
  }
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans, out);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, test_two_components) {
  int height = 100;
  int width = 100;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  in[0] = 1;
  in[9999] = 1;
  ans[0] = 1;
  ans[9999] = 2;
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans, out);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, test_horizontal_stripe) {
  int height = 100;
  int width = 100;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < width; ++j) {
    in[j] = 1;
    ans[j] = 1;
  }
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans, out);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, test_vertical_stripe) {
  int height = 100;
  int width = 100;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < height; ++j) {
    in[j * width] = 1;
    ans[j * width] = 1;
  }
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans, out);
  }
}

TEST(shkurinskaya_e_bin_labeling_all, test_horizontal_stripe_dif_size) {
  int height = 50;
  int width = 100;
  int size = width * height;
  boost::mpi::communicator world;
  // Create data
  std::vector<int> in(size);
  std::vector<int> out(size);
  std::vector<int> ans(size);
  for (int j = 0; j < width; ++j) {
    in[j] = 1;
    ans[j] = 1;
  }
  // Create TaskData
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&height));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(&width));
    task_data_all->inputs_count.emplace_back(1);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }
  // Create Task
  shkurinskaya_e_bin_labeling_all::TaskMPITBB task_all(task_data_all);
  ASSERT_EQ(task_all.Validation(), true);
  task_all.PreProcessing();
  task_all.Run();
  task_all.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans, out);
  }
}
