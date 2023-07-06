#ifndef FIJIT_SYS_FIJIT_H
#define FIJIT_SYS_FIJIT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>

#include <vector>
#include <string>


class Fijit {
public:
  Fijit(bool enable_activity_api, bool enable_callback_api);
  ~Fijit();

  void run();
  std::vector<std::string> get_kernel_records();

private:
  bool enable_activity_api_;
  bool enable_callback_api_;

  std::vector<CUpti_ActivityKernel7> kernel_records_;

  CUcontext cudaCtx;
  CUpti_SubscriberHandle subscriber;

  static void ActivtyAPIRequestBufferCallback(uint8_t **buffer, size_t *size, size_t *maxNumRecords);
  static void ActivityAPICompleteBufferCallback(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);
  static void CallbackAPICallback(void *userdata, CUpti_CallbackDomain domain,
                                  CUpti_CallbackId cbid, const void *cbdata);
};

#endif // FIJIT_SYS_FIJIT_H
