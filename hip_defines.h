#ifndef HIP_DEFINES_H
#define HIP_DEFINES_H

#if defined(USE_HIP)

#include <hip/hip_runtime.h>

#ifndef cudaError_t
#define cudaError_t hipError_t
#endif

#ifndef cudaSuccess
#define cudaSuccess hipSuccess
#endif

#ifndef cudaGetErrorString
#define cudaGetErrorString hipGetErrorString
#endif

#ifndef cudaStream_t
#define cudaStream_t hipStream_t
#endif

#ifndef cudaStreamCreate
#define cudaStreamCreate hipStreamCreate
#endif

#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize hipStreamSynchronize
#endif

#ifndef cudaStreamDestroy
#define cudaStreamDestroy hipStreamDestroy
#endif

#ifndef cudaGraph_t
#define cudaGraph_t hipGraph_t
#endif

#ifndef cudaGraphCreate
#define cudaGraphCreate hipGraphCreate
#endif

#ifndef cudaGraphInstantiate
#define cudaGraphInstantiate hipGraphInstantiate
#endif

#ifndef cudaGraphLaunch
#define cudaGraphLaunch hipGraphLaunch
#endif

#ifndef cudaGraphNode_t
#define cudaGraphNode_t hipGraphNode_t
#endif

#ifndef cudaGraphAddKernelNode
#define cudaGraphAddKernelNode hipGraphAddKernelNode
#endif

#ifndef cudaGraphDestroy
#define cudaGraphDestroy hipGraphDestroy
#endif

#ifndef cudaGraphExec_t
#define cudaGraphExec_t hipGraphExec_t
#endif

#ifndef cudaGraphExecDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#endif

#ifndef cudaFree
#define cudaFree hipFree
#endif

#ifndef cudaFreeHost
#define cudaFreeHost hipFreeHost
#endif

#ifndef cudaMalloc
#define cudaMalloc hipMalloc
#endif

#ifndef cudaMallocHost
#define cudaMallocHost hipMallocHost
#endif

#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync hipMemcpyAsync
#endif

#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#endif

#ifndef cudaMemset
#define cudaMemset hipMemset
#endif

#ifndef cudaMemsetAsync
#define cudaMemsetAsync hipMemsetAsync
#endif

#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#endif

#ifndef cudaKernelNodeParams
#define cudaKernelNodeParams hipKernelNodeParams
#endif

#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#endif

#ifndef cudaGetDevice
#define cudaGetDevice hipGetDevice
#endif

#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#endif

#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#endif

#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties hipGetDeviceProperties
#endif

#ifndef cudaDeviceProp
#define cudaDeviceProp hipDeviceProp_t
#endif

#ifndef cudaDeviceGetPCIBusId
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#endif

#endif // USE_HIP

#endif // HIP_DEFINES_H
