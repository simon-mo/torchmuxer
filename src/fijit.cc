#include <glog/logging.h>
#include <fmt/format.h>

#include "fijit.h"
#include "common.h"
#include "cupti.h"

Fijit::Fijit() {
  // Initialize CUDA Context
  LOG(INFO) << "initializing cuda context";
  cudaCtx = cuda_init();
}

void Fijit::run() {
  int count = -1;
  cudaGetDeviceCount(&count);
  LOG(INFO) << fmt::format("Total number of device {}", count);
  // CHECK_CUPTI(cuptiSubscribe());
}
