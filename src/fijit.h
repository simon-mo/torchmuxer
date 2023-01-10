#ifndef FIJIT_SYS_FIJIT_H
#define FIJIT_SYS_FIJIT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>


using namespace std;

class Fijit {
public:
  Fijit();
  ~Fijit();

  void run();

private:
  CUcontext cudaCtx;
  CUpti_SubscriberHandle subscriber;

  static void ActivtyAPIRequestBufferCallback(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
  static void ActivityAPICompleteBufferCallback(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
  static void CallbackAPICallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const void *cbdata);
};

#endif // FIJIT_SYS_FIJIT_H
