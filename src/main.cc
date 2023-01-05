#include <cuda_runtime.h>
#include <cupti.h>
#include <fmt/ostream.h>
#include <cxxopts/cxxopts.hpp>
#include <glog/logging.h>

#include "fijit.h"
#include "backtrace.h"

using namespace std;

int main(int argc, char *argv[]) {
  // Setup logging
  FLAGS_logtostderr = 1;
  FLAGS_colorlogtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  // Setup CLI parsing
  cxxopts::Options options("fijit-sys", "FIJIT Inference Engine");
  options.positional_help("[optional args]").show_positional_help();
  // clang-format off
  options.add_options()
      ("backtrace", "Print backtrace on crash")
      ("h, help", "Print help");
  // clang-format on
  auto result = options.parse(argc, argv);

  if (result.count("help")) {
    std::cout << options.help({"", "Group"}) << std::endl;
    exit(0);
  }

  if (result.count("backtrace")) {
    LOG(INFO) << "Installed backtrace";
    std::set_terminate([]() { backtrace(); });
  }

  Fijit fijit;
  fijit.run();
}
