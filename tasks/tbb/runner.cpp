#include <gtest/gtest.h>
#include <tbb/global_control.h>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/global_control.h"

int main(int argc, char** argv) {
  // Limit the number of threads in TBB
  tbb::global_control control(tbb::global_control::max_allowed_parallelism, ppc::util::GetPPCNumThreads());

  try {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    std::cerr << "Caught std::exception: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Caught unknown exception." << std::endl;
    return EXIT_FAILURE;
  }
}
