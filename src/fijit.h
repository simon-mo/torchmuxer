#ifndef FIJIT_SYS_FIJIT_H
#define FIJIT_SYS_FIJIT_H

#include <cuda.h>
#include <cuda_runtime.h>


using namespace std;

class Fijit {
public:
  Fijit();
  ~Fijit() = default;

  void run();

private:
  CUcontext cudaCtx;
};

#endif // FIJIT_SYS_FIJIT_H
