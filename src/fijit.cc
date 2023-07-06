#include <fmt/format.h>
#include <set>
#include <string>

#include "common.h"
#include "constants.h"
#include "cupti.h"
#include "fijit.h"
// #include "torch/torch.h"
// #include "disable-torch-log.h.inc"
#include <glog/logging.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#define CUPTI_CHECK_ERROR(call)                                                \
  {                                                                            \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *error_str;                                                   \
      cuptiGetResultString(_status, &error_str);                               \
      LOG(FATAL) << fmt::format(                                               \
          "[cupti] {}:{}: function {} fialed with error {}.");                 \
    }                                                                          \
  }

// singleton to get and set the global fijit pointer
class FijitContainer {
public:
   Fijit *get() {
    return fijit_;
  }
   void set(Fijit *fijit) {
    fijit_ = fijit;
  }
private:
   Fijit *fijit_ = nullptr;
};

FijitContainer* get_fijit_container() {
  static FijitContainer fijit_container;
  return &fijit_container;
}


Fijit::Fijit(bool enable_activity_api, bool enable_callback_api)
    : enable_activity_api_{enable_activity_api}, enable_callback_api_{
                                                     enable_callback_api} {
  get_fijit_container()->set(this);
                                                     }

Fijit::~Fijit() { CUPTI_CHECK_ERROR(cuptiActivityFlushAll(1)); }

static uint64_t startTimestamp;

void Fijit::run() {
  // torch::Tensor tensor = torch::rand({2, 3});
  // LOG(INFO) << tensor;
  // LOG(INFO) << "Created tensor";

  if (enable_activity_api_) {
    LOG(INFO) << "initializing cupti activity trace";

    // TODO: enable more activities!
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
    CUPTI_CHECK_ERROR(
        cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    // CUPTI_CHECK_ERROR(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

    // TODO: customize the buffer size and limit with
    // cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT,
    CUPTI_CHECK_ERROR(cuptiActivityRegisterCallbacks(
        ActivtyAPIRequestBufferCallback, ActivityAPICompleteBufferCallback));
  }

  if (enable_callback_api_) {
    // TODO: subscribe to more API
    cuptiSubscribe(&subscriber, CallbackAPICallback, /*user_data*/ nullptr);
    cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
  }

  // Initialize CUDA Context
  // LOG(INFO) << "initializing cuda context";
  // cudaCtx = cuda_init();

  // int count = -1;
  // cudaGetDeviceCount(&count);
  // LOG(INFO) << fmt::format("Total number of device {}", count);

  CUPTI_CHECK_ERROR(cuptiGetTimestamp(&startTimestamp));
  LOG(INFO) << fmt::format("Timestamp {}", startTimestamp);
}

#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t)(buffer) & ((align)-1))                                         \
       ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1)))            \
       : (buffer))

void Fijit::ActivtyAPIRequestBufferCallback(uint8_t **buffer, size_t *size,
                                            size_t *maxNumRecords) {
  // TODO: optimize (for latency) this with pre-allocated memory pool.
  uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(EXIT_FAILURE);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

static const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

const char *getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

uint32_t getActivityObjectKindId(CUpti_ActivityObjectKind kind,
                                 CUpti_ActivityObjectKindId *id) {
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

static const char *getComputeApiKindString(CUpti_ActivityComputeApiKind kind) {
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

static void printActivity(CUpti_Activity *record) {
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE: {
    CUpti_ActivityDevice4 *device = (CUpti_ActivityDevice4 *)record;
    printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u "
           "GB/s, size %u MB), "
           "multiprocessors %u, clock %u MHz\n",
           device->name, device->id, device->computeCapabilityMajor,
           device->computeCapabilityMinor,
           (unsigned int)(device->globalMemoryBandwidth / 1024 / 1024),
           (unsigned int)(device->globalMemorySize / 1024 / 1024),
           device->numMultiprocessors,
           (unsigned int)(device->coreClockRate / 1000));
    break;
  }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
    CUpti_ActivityDeviceAttribute *attribute =
        (CUpti_ActivityDeviceAttribute *)record;
    printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
           attribute->attribute.cupti, attribute->deviceId,
           (unsigned long long)attribute->value.vUint64);
    break;
  }
  case CUPTI_ACTIVITY_KIND_CONTEXT: {
    CUpti_ActivityContext *context = (CUpti_ActivityContext *)record;
    printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
           context->contextId, context->deviceId,
           getComputeApiKindString(
               (CUpti_ActivityComputeApiKind)context->computeApiKind),
           (int)context->nullStreamId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    CUpti_ActivityMemcpy5 *memcpy = (CUpti_ActivityMemcpy5 *)record;
    printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, size "
           "%llu, correlation %u\n",
           getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
           (unsigned long long)(memcpy->start - startTimestamp),
           (unsigned long long)(memcpy->end - startTimestamp), memcpy->deviceId,
           memcpy->contextId, memcpy->streamId,
           (unsigned long long)memcpy->bytes, memcpy->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMSET: {
    CUpti_ActivityMemset4 *memset = (CUpti_ActivityMemset4 *)record;
    printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, "
           "correlation %u\n",
           memset->value, (unsigned long long)(memset->start - startTimestamp),
           (unsigned long long)(memset->end - startTimestamp), memset->deviceId,
           memset->contextId, memset->streamId, memset->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    const char *kindString =
        (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
    CUpti_ActivityKernel7 *kernel = (CUpti_ActivityKernel7 *)record;
    printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, "
           "correlation %u\n",
           kindString, kernel->name,
           (unsigned long long)(kernel->start - startTimestamp),
           (unsigned long long)(kernel->end - startTimestamp), kernel->deviceId,
           kernel->contextId, kernel->streamId, kernel->correlationId);
    printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, "
           "dynamic %u)\n",
           kernel->gridX, kernel->gridY, kernel->gridZ, kernel->blockX,
           kernel->blockY, kernel->blockZ, kernel->staticSharedMemory,
           kernel->dynamicSharedMemory);
    break;
  }
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation "
           "%u\n",
           api->cbid, (unsigned long long)(api->start - startTimestamp),
           (unsigned long long)(api->end - startTimestamp), api->processId,
           api->threadId, api->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation "
           "%u\n",
           api->cbid, (unsigned long long)(api->start - startTimestamp),
           (unsigned long long)(api->end - startTimestamp), api->processId,
           api->threadId, api->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_NAME: {
    CUpti_ActivityName *name = (CUpti_ActivityName *)record;
    switch (name->objectKind) {
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      printf("NAME  %s %u %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE,
                                     &name->objectId),
             name->name);
      break;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      printf("NAME %s %u %s %u %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT,
                                     &name->objectId),
             getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
             getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE,
                                     &name->objectId),
             name->name);
      break;
    default:
      printf("NAME %s id %u, name %s\n",
             getActivityObjectKindString(name->objectKind),
             getActivityObjectKindId(name->objectKind, &name->objectId),
             name->name);
      break;
    }
    break;
  }
  case CUPTI_ACTIVITY_KIND_MARKER: {
    CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *)record;
    printf("MARKER id %u [ %llu ], name %s, domain %s\n", marker->id,
           (unsigned long long)marker->timestamp, marker->name, marker->domain);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
    CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *)record;
    printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
           marker->id, marker->color, marker->category,
           (unsigned long long)marker->payload.metricValueUint64,
           marker->payload.metricValueDouble);
    break;
  }
  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *)record;
    printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
           getActivityOverheadKindString(overhead->overheadKind),
           (unsigned long long)overhead->start - startTimestamp,
           (unsigned long long)overhead->end - startTimestamp,
           getActivityObjectKindString(overhead->objectKind),
           getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
    break;
  }
  default:
    printf("  <unknown>\n");
    break;
  }
}

// implement get_kernel_records by returning a list of kernel records in json format
std::vector<std::string> Fijit::get_kernel_records() {
  std::vector<std::string> kernel_records;
  for (auto &kernel : kernel_records_) {
    json kernel_record = {
        {"name", std::string(kernel.name)},
        {"start", uint64_t(kernel.start)},
        {"end", uint64_t(kernel.end)},
        {"stream", uint32_t(kernel.streamId)},
    };
    kernel_records.push_back(kernel_record.dump());
  }
  return kernel_records;
}


void Fijit::ActivityAPICompleteBufferCallback(CUcontext ctx, uint32_t streamId,
                                              uint8_t *buffer, size_t size,
                                              size_t validSize) {
  // TODO: offload these to separate thread so these can be completed ASAP.
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        // LOG(INFO) << fmt::format("activity detected: {}",
        //                          ACTIVITY_KIND_ENUM_MAP[record->kind]);
        CUpti_ActivityKernel7 *kernel = (CUpti_ActivityKernel7 *)record;
        get_fijit_container()->get()->kernel_records_.push_back(*kernel);
        // printActivity(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CHECK_ERROR(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CHECK_ERROR(
        cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(INFO) << fmt::format("Dropped {} activity records",
                               (unsigned int)dropped);
    }
  }

  free(buffer);
}

void Fijit::CallbackAPICallback(void *userdata, CUpti_CallbackDomain domain,
                                CUpti_CallbackId cbid, const void *cbdata) {
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
    LOG(INFO) << fmt::format("[callback api] domain {} callback id {}", domain,
                             CALLBACK_API_DRIVER_ID_MAP[cbid]);
  }
}