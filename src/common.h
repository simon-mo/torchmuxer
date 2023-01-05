#ifndef FIJIT_SYS_COMMON_CUH_H
#define FIJIT_SYS_COMMON_CUH_H

#include "cuda.h"
#include "cupti.h"
#include <iostream>
#include <vector>

#define STR(x) #x

#define PRINT_ERR(expression, err_str)                                         \
  std::cerr << "Error on line " << STR(expression) << ": " << err_str << "\n"  \
            << __FILE__ << ":" << __LINE__ << std::endl;

#define CHECK_CUDA(expression)                                                 \
  {                                                                            \
    cudaError_t status = (expression);                                         \
    if (status != cudaSuccess) {                                               \
      const char *err_str = cudaGetErrorString(status);                        \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CHECK_CUDEVICE(expression)                                             \
  {                                                                            \
    CUresult status = (expression);                                            \
    if (status != CUDA_SUCCESS) {                                              \
      const char *err_str;                                                     \
      cuGetErrorString(status, &err_str);                                      \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }

#define CUDevicePtrConstCast(expression) (const void *)(uintptr_t)(expression)

#define CUDevicePtrCast(expression) (void *)(uintptr_t)(expression)

CUcontext cuda_init(void) {
      CUdevice cuDevice;
  CUcontext cuContext;
  CHECK_CUDEVICE(cuInit(0));
  CHECK_CUDEVICE(cuDeviceGet(&cuDevice, 0));
  CHECK_CUDEVICE(cuCtxCreate(&cuContext, 0, cuDevice));
  return cuContext;
}


#define CHECK_CUPTI(expression)                                                \
  {                                                                            \
    CUptiResult status = (expression);                                         \
    if (status != CUPTI_SUCCESS) {                                             \
      const char *err_str;                                                     \
      cuptiGetResultString(status, &err_str);                                  \
      PRINT_ERR(expression, err_str)                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  }


#endif // FIJIT_SYS_COMMON_CUH_H