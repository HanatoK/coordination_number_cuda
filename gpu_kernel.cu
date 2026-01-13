#include "gpu_kernel.h"
#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#if defined(USE_CUDA)
#include <cub/block/block_reduce.cuh>
#include <cuda_pipeline.h>
#elif defined(USE_HIP)
#include <hipcub/block/block_reduce.hpp>
#endif

template <int N, int M, int block_size, int group2BatchSize, int numGroup2BatchesPerBlock>
__global__ void computeCoordinationNumberTwoGroupsCUDAKernel1(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const double* __restrict pos2x,
  const double* __restrict pos2y,
  const double* __restrict pos2z,
  const unsigned int numAtoms1,
  const unsigned int numAtoms2,
  const double inv_r0,
  double* __restrict energy,
  double* __restrict gx1,
  double* __restrict gy1,
  double* __restrict gz1,
  double* __restrict gx2,
  double* __restrict gy2,
  double* __restrict gz2) {
  // TODO: Figure out a way to remove this limitation
  static_assert(block_size == group2BatchSize * numGroup2BatchesPerBlock,
                "block_size != group2BatchSize * numGroup2BatchesPerBlock");
  // Shared memory buffers for atoms in group2
  __shared__ double3 shPosition[group2BatchSize];
  __shared__ double3 shJGrad[numGroup2BatchesPerBlock][group2BatchSize];
  __shared__ bool shJMask[group2BatchSize];
  // Total energy
  double ei = 0;
  // Number of blocks required to iterate over group1
  const unsigned int numBlocksInGroup1 = (numAtoms1 + block_size - 1) / block_size;
  // Number of blocks required to iterate over group2
  const unsigned int numBatchesInGroup2 = (numAtoms2 + group2BatchSize - 1) / group2BatchSize;
  const unsigned int group2WorkSize = numBatchesInGroup2 * group2BatchSize;
  const unsigned int group2BatchID = threadIdx.x / group2BatchSize;
  const unsigned int group2LaneID = threadIdx.x % group2BatchSize;
  for (unsigned int i = blockIdx.x; i < numBlocksInGroup1; i += gridDim.x) {
    const unsigned int tid = i * blockDim.x + threadIdx.x;
    // Load the atom i from group1
    const bool mask_i = tid < numAtoms1;
    const double x1 = mask_i ? pos1x[tid] : 0;
    const double y1 = mask_i ? pos1y[tid] : 0;
    const double z1 = mask_i ? pos1z[tid] : 0;
    double3 iGrad{0, 0, 0};
    // Load atom j from group2
    for (unsigned int k = 0; k < group2WorkSize; k += group2BatchSize) {
      const unsigned int j = k + group2LaneID;
      if (group2BatchID == 0) {
        const bool mask_j = j < numAtoms2;
        if (mask_j) {
#if defined(USE_CUDA)
          __pipeline_memcpy_async(&shPosition[group2LaneID].x, &pos2x[j], sizeof(double));
          __pipeline_memcpy_async(&shPosition[group2LaneID].y, &pos2y[j], sizeof(double));
          __pipeline_memcpy_async(&shPosition[group2LaneID].z, &pos2z[j], sizeof(double));
          __pipeline_commit();
#elif defined(USE_HIP)
          shPosition[group2LaneID].x = pos2x[j];
          shPosition[group2LaneID].y = pos2y[j];
          shPosition[group2LaneID].z = pos2z[j];
#endif
        }
        shJMask[group2LaneID] = mask_j;
      }
      shJGrad[group2BatchID][group2LaneID].x = 0;
      shJGrad[group2BatchID][group2LaneID].y = 0;
      shJGrad[group2BatchID][group2LaneID].z = 0;
#if defined(USE_CUDA)
      __pipeline_wait_prior(0);
#endif
      __syncthreads();
      for (unsigned int t = 0; t < group2BatchSize; ++t) {
        // Since we need to store the interaction gradient into the
        // shared memory buffer, we need to avoid accumulation of
        // gradient into the same shared memory position from different
        // threads. (see also https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
        // const unsigned int jid = (t + threadIdx.x) % block_size;
        // Another swizzling method using XOR from possibly CuTe
        // (see also https://zhuanlan.zhihu.com/p/1941306442683515068)
        const unsigned int jid = t ^ group2LaneID;
        const bool mask_jid = shJMask[jid];
        if (mask_i && mask_jid) {
          const double x2 = shPosition[jid].x;
          const double y2 = shPosition[jid].y;
          const double z2 = shPosition[jid].z;
          coordnum<N, M>(
            x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei,
            iGrad.x, iGrad.y, iGrad.z,
            shJGrad[group2BatchID][jid].x,
            shJGrad[group2BatchID][jid].y,
            shJGrad[group2BatchID][jid].z);
        }
        __syncthreads();
      }
      // Reduction over the shared memory
      #pragma unroll
      for (unsigned int l = numGroup2BatchesPerBlock / 2; l > 0; l >>= 1) {
        if (group2BatchID < l) {
          shJGrad[group2BatchID][group2LaneID].x += shJGrad[group2BatchID + l][group2LaneID].x;
          shJGrad[group2BatchID][group2LaneID].y += shJGrad[group2BatchID + l][group2LaneID].y;
          shJGrad[group2BatchID][group2LaneID].z += shJGrad[group2BatchID + l][group2LaneID].z;
        }
        __syncthreads();
      }
      if (group2BatchID == 0) {
        if (shJMask[group2LaneID]) {
          atomicAdd(&gx2[j], shJGrad[0][group2LaneID].x);
          atomicAdd(&gy2[j], shJGrad[0][group2LaneID].y);
          atomicAdd(&gz2[j], shJGrad[0][group2LaneID].z);
        }
      }
      __syncthreads();
    }
    if (mask_i) {
      // Save the i-gradients to group1
      atomicAdd(&gx1[tid], iGrad.x);
      atomicAdd(&gy1[tid], iGrad.y);
      atomicAdd(&gz1[tid], iGrad.z);
    }
  }
  // Reduction for energy
  __syncthreads();
#if defined(USE_CUDA)
  typedef cub::BlockReduce<double, block_size> BlockReduce;
#elif defined(USE_HIP)
  typedef hipcub::BlockReduce<double, block_size> BlockReduce;
#endif
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const double total_e = BlockReduce(temp_storage).Sum(ei); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(energy, total_e);
  }
}

void computeCoordinationNumberTwoGroupsCUDA(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupGradientsCUDA& gradient1,
  AtomGroupGradientsCUDA& gradient2,
  double inv_r0,
  double* d_energy,
  cudaGraph_t& graph,
  cudaStream_t stream) {
  unsigned int numAtoms1 = group1.getNumAtoms();
  unsigned int numAtoms2 = group2.getNumAtoms();
  if (numAtoms2 > numAtoms1) {
    computeCoordinationNumberTwoGroupsCUDA(
      group2, group1, gradient2, gradient1, inv_r0, d_energy, graph, stream);
    return;
  }
  const double* pos1x = group1.getDeviceX();
  const double* pos1y = group1.getDeviceY();
  const double* pos1z = group1.getDeviceZ();
  const double* pos2x = group2.getDeviceX();
  const double* pos2y = group2.getDeviceY();
  const double* pos2z = group2.getDeviceZ();
  double* g1x = gradient1.getDeviceX();
  double* g1y = gradient1.getDeviceY();
  double* g1z = gradient1.getDeviceZ();
  double* g2x = gradient2.getDeviceX();
  double* g2y = gradient2.getDeviceY();
  double* g2z = gradient2.getDeviceZ();
  void* args[] = {
    &pos1x, &pos1y, &pos1z,
    &pos2x, &pos2y, &pos2z,
    &numAtoms1, &numAtoms2,
    &inv_r0, &d_energy,
    &g1x, &g1y, &g1z,
    &g2x, &g2y, &g2z};
  const unsigned int minNumAtoms = std::min(numAtoms1, numAtoms2);
  // I'm not sure if it's necessary to limit the number of blocks
  // constexpr const unsigned int maxNumBlocks = 65536;
  constexpr unsigned int const block_size = 128;
  cudaKernelNodeParams kernelNodeParams = {0};
  kernelNodeParams.blockDim       = dim3(block_size, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams   = args;
  kernelNodeParams.extra          = NULL;
  if (minNumAtoms > 64) {
    constexpr unsigned int const group2WorkSize = 64;
    kernelNodeParams.func =
      (void*)computeCoordinationNumberTwoGroupsCUDAKernel1<6, 12, block_size, group2WorkSize, block_size / group2WorkSize>;
  } else if (minNumAtoms > 16) {
    constexpr unsigned int const group2WorkSize = 16;
    kernelNodeParams.func =
      (void*)computeCoordinationNumberTwoGroupsCUDAKernel1<6, 12, block_size, group2WorkSize, block_size / group2WorkSize>;
  } else {
    constexpr unsigned int const group2WorkSize = 8;
    kernelNodeParams.func =
      (void*)computeCoordinationNumberTwoGroupsCUDAKernel1<6, 12, block_size, group2WorkSize, block_size / group2WorkSize>;
  }
  // Occupancy calculator
  int num_blocks_occ;
  int deviceID = 0;
  int multiProcessorCount;
  checkGPUError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_occ, kernelNodeParams.func, block_size, 0));
  checkGPUError(cudaGetDevice(&deviceID));
  checkGPUError(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceID));
  cudaDeviceProp props = {0};
  checkGPUError(cudaGetDeviceProperties(&props, deviceID));
  char busID[256];
  checkGPUError(cudaDeviceGetPCIBusId(busID, 256, deviceID));
  std::cout << "GPU Name: " << props.name << ", PCI Bus ID: " << busID;
#if defined(USE_HIP)
  std::cout << ", GCN Arch Name: " << props.gcnArchName;
#endif
  std::cout << std::endl;
  // From CUDA samples
  const unsigned int maxNumBlocks = num_blocks_occ * multiProcessorCount;
  const unsigned int num_blocks = std::min(maxNumBlocks, (numAtoms1 + block_size - 1) / block_size);
  kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);

  cudaGraphNode_t node;
  checkGPUError(cudaGraphAddKernelNode(
    &node, graph, NULL,
    0, &kernelNodeParams));
}

computeCoordinationNumberSelfGroupCUDAObject::~computeCoordinationNumberSelfGroupCUDAObject() {
  if (d_tilesList) {
    cudaFree(d_tilesList);
    d_tilesList = nullptr;
  }
  if (d_tilesListStart) {
    cudaFree(d_tilesListStart);
    d_tilesListStart = nullptr;
  }
  if (d_tilesListSizes) {
    cudaFree(d_tilesListSizes);
    d_tilesListSizes = nullptr;
  }
  if (d_pairlist) {
    cudaFree(d_pairlist);
    d_pairlist = nullptr;
  }
}

void computeCoordinationNumberSelfGroupCUDAObject::initialize(unsigned int numAtoms) {
  if (initialized) {
    if (d_tilesList) checkGPUError(cudaFree(d_tilesList));
    if (d_tilesListSizes) checkGPUError(cudaFree(d_tilesListSizes));
    if (d_tilesListStart) checkGPUError(cudaFree(d_tilesListStart));
    if (d_pairlist) checkGPUError(cudaFree(d_pairlist));
  }
  prepareTilesList(numAtoms);
  const size_t pairlist_size = sizeof(bool) * size_t(numAtoms - 1) * size_t(numAtoms - 1);
  checkGPUError(cudaMalloc(&d_pairlist, pairlist_size));
  checkGPUError(cudaMemset(d_pairlist, 1, pairlist_size));
  initialized = true;
}

void computeCoordinationNumberSelfGroupCUDAObject::prepareTilesList(unsigned int numAtoms) {
  const unsigned int blockSize = self_group_block_size;
  const unsigned int numTiles = (numAtoms + blockSize - 1) / blockSize;
  std::vector<std::vector<unsigned int>> tilesList(numTiles);
  const unsigned int tileSize = numTiles / 2;
  for (unsigned int i = 0; i < numTiles - 1; ++i) {
    for (unsigned int j = i + 1; j < numTiles; ++j) {
      if (tilesList[i].size() < tileSize) {
        tilesList[i].push_back(j);
      } else {
        tilesList[j].push_back(i);
      }
    }
  }
  // Flattened list
  std::vector<unsigned int> tilesListFlattened;
  for (const auto& list: tilesList) {
    tilesListFlattened.insert(tilesListFlattened.end(), list.begin(), list.end());
  }
  // The size of each tile list
  std::vector<unsigned int> tilesListSizes(tilesList.size());
  std::transform(
    tilesList.begin(), tilesList.end(), tilesListSizes.begin(),
    [](const auto& tile){return tile.size();});
  // The start index of each tile list
  std::vector<unsigned int> tilesListStart(tilesList.size(), 0);
  std::exclusive_scan(tilesListSizes.begin(), tilesListSizes.end(), tilesListStart.begin(), (unsigned int)0);
  // Copy the data to GPU
  checkGPUError(cudaMalloc(&d_tilesList, tilesListFlattened.size() * sizeof(unsigned int)));
  checkGPUError(cudaMemcpy(d_tilesList, tilesListFlattened.data(), tilesListFlattened.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  checkGPUError(cudaMalloc(&d_tilesListSizes, tilesListSizes.size() * sizeof(unsigned int)));
  checkGPUError(cudaMemcpy(d_tilesListSizes, tilesListSizes.data(), tilesListSizes.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
  checkGPUError(cudaMalloc(&d_tilesListStart, tilesListStart.size() * sizeof(unsigned int)));
  checkGPUError(cudaMemcpy(d_tilesListStart, tilesListStart.data(), tilesListStart.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

template <unsigned int N, unsigned int M, unsigned int block_size, bool use_pairlist, bool rebuild_pairlist>
__global__ void computeCoordinationNumberSelfGroupCUDAKernel1(
  const double* __restrict pos1x,
  const double* __restrict pos1y,
  const double* __restrict pos1z,
  const unsigned int numAtoms1,
  const double inv_r0,
  double* __restrict energy,
  double* __restrict gx1,
  double* __restrict gy1,
  double* __restrict gz1,
  const unsigned int* tilesList,
  const unsigned int* tilesListStart,
  const unsigned int* tilesListSizes,
  const double pairlist_tol,
  bool* pairlist) {
  __shared__ double3 shPosition[block_size];
  __shared__ double3 shJGrad[block_size];
  __shared__ bool mask[block_size];
  __shared__ unsigned int globalJIDs[block_size];
  // __shared__ bool blockPairlist[block_size];
  double ei = 0;
  static constexpr const unsigned int half_block_size = block_size / 2;
  // Number of blocks required to iterate over group1
  const unsigned int numBlocksInGroup1 = (numAtoms1 + block_size - 1) / block_size;
  for (unsigned int i = blockIdx.x; i < numBlocksInGroup1; i += gridDim.x) {
    const unsigned int tid = i * blockDim.x + threadIdx.x;
    const bool mask_i = tid < numAtoms1;
    const double x1 = mask_i ? pos1x[tid] : 0;
    const double y1 = mask_i ? pos1y[tid] : 0;
    const double z1 = mask_i ? pos1z[tid] : 0;
    unsigned int pair_id_i;
    if (use_pairlist) {
      pair_id_i = tid * (2 * numAtoms1 - 1 - tid) / 2;
      globalJIDs[threadIdx.x] = tid;
      // blockPairlist[threadIdx.x] = mask_i ? pairlist[tid] :
    }
    double3 iGrad{0, 0, 0};
    // Self tile
    mask[threadIdx.x] = mask_i;
    shPosition[threadIdx.x].x = x1;
    shPosition[threadIdx.x].y = y1;
    shPosition[threadIdx.x].z = z1;
    shJGrad[threadIdx.x].x = 0;
    shJGrad[threadIdx.x].y = 0;
    shJGrad[threadIdx.x].z = 0;
    __syncthreads();
    for (unsigned int t = 1; t < half_block_size; ++t) {
      // NAMD/OpenMM style swizzling
      const unsigned int jid = (t + threadIdx.x) & (block_size - 1);
      unsigned int pairlistID;
      bool pairlist_elem;
      if (use_pairlist) {
        const unsigned int jid_global = globalJIDs[jid];
        pairlistID = pair_id_i + (jid_global - tid - 1);
      }
      if (mask_i && mask[jid]) {
        if (use_pairlist && !rebuild_pairlist) {
          pairlist_elem = pairlist[pairlistID];
        }
        const double x2 = shPosition[jid].x;
        const double y2 = shPosition[jid].y;
        const double z2 = shPosition[jid].z;
        coordnum_pairlist<N, M, use_pairlist, rebuild_pairlist>(
          x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei,
          iGrad.x, iGrad.y, iGrad.z,
          shJGrad[jid].x,
          shJGrad[jid].y,
          shJGrad[jid].z,
          pairlist_tol,
          &pairlist_elem);
        if (use_pairlist && rebuild_pairlist) {
          pairlist[pairlistID] = pairlist_elem;
        }
      }
      __syncthreads();
    }

    // Last loop: t == block_size / 2
    {
      // NAMD/OpenMM style swizzling
      const unsigned int jid = (half_block_size + threadIdx.x) & (block_size - 1);
      unsigned int pairlistID;
      bool pairlist_elem;
      if (jid > threadIdx.x) {
        if (use_pairlist) {
          const unsigned int jid_global = globalJIDs[jid];
          pairlistID = pair_id_i + (jid_global - tid - 1);
        }
        if (mask_i && mask[jid]) {
          if (use_pairlist && !rebuild_pairlist) {
            pairlist_elem = pairlist[pairlistID];
          }
          const double x2 = shPosition[jid].x;
          const double y2 = shPosition[jid].y;
          const double z2 = shPosition[jid].z;
          coordnum_pairlist<N, M, use_pairlist, rebuild_pairlist>(
            x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei,
            iGrad.x, iGrad.y, iGrad.z,
            shJGrad[jid].x,
            shJGrad[jid].y,
            shJGrad[jid].z,
            pairlist_tol,
            &pairlist_elem);
          if (use_pairlist && rebuild_pairlist) {
            pairlist[pairlistID] = pairlist_elem;
          }
        }
      }
      __syncthreads();
    }

    if (mask_i) {
      atomicAdd(&gx1[tid], shJGrad[threadIdx.x].x);
      atomicAdd(&gy1[tid], shJGrad[threadIdx.x].y);
      atomicAdd(&gz1[tid], shJGrad[threadIdx.x].z);
    }
    __syncthreads();

    // Iterate over other tiles
    const unsigned int jBlockStart = tilesListStart[i];
    const unsigned int numJBlocks = tilesListSizes[i];
    const unsigned int jBlockEnd = jBlockStart + numJBlocks;
    for (unsigned int l = jBlockStart; l < jBlockEnd; ++l) {
      const unsigned int jBlockIndex = tilesList[l];
      // Fetch atom j from i-tile
      const unsigned int jid_global = jBlockIndex * blockDim.x + threadIdx.x;
      const bool mask_j = jid_global < numAtoms1;
      if (mask_j) {
        shPosition[threadIdx.x].x = pos1x[jid_global];
        shPosition[threadIdx.x].y = pos1y[jid_global];
        shPosition[threadIdx.x].z = pos1z[jid_global];
      }
      mask[threadIdx.x] = mask_j;
      // Reset the gradients
      shJGrad[threadIdx.x].x = 0;
      shJGrad[threadIdx.x].y = 0;
      shJGrad[threadIdx.x].z = 0;
      if (use_pairlist) {
        globalJIDs[threadIdx.x] = jid_global;
      }
      __syncthreads();
      for (unsigned int t = 0; t < block_size; ++t) {
        const unsigned int jid = t ^ threadIdx.x;
        unsigned int pairlistID;
        bool pairlist_elem;
        if (use_pairlist) {
          pairlistID = pair_id_i + (globalJIDs[jid] - tid - 1);
        }
        if (mask_i && mask[jid]) {
          if (use_pairlist && !rebuild_pairlist) {
            pairlist_elem = pairlist[pairlistID];
          }
          const double x2 = shPosition[jid].x;
          const double y2 = shPosition[jid].y;
          const double z2 = shPosition[jid].z;
          coordnum_pairlist<N, M, use_pairlist, rebuild_pairlist>(
            x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei,
            iGrad.x, iGrad.y, iGrad.z,
            shJGrad[jid].x,
            shJGrad[jid].y,
            shJGrad[jid].z,
            pairlist_tol,
            &pairlist_elem);
          if (use_pairlist && rebuild_pairlist) {
            pairlist[pairlistID] = pairlist_elem;
          }
        }
        __syncthreads();
      }
      if (mask_j) {
        atomicAdd(&gx1[jid_global], shJGrad[threadIdx.x].x);
        atomicAdd(&gy1[jid_global], shJGrad[threadIdx.x].y);
        atomicAdd(&gz1[jid_global], shJGrad[threadIdx.x].z);
      }
    }
    if (mask_i) {
      atomicAdd(&gx1[tid], iGrad.x);
      atomicAdd(&gy1[tid], iGrad.y);
      atomicAdd(&gz1[tid], iGrad.z);
    }
  }
  // Reduction for energy
#if defined(USE_CUDA)
  typedef cub::BlockReduce<double, block_size> BlockReduce;
#elif defined(USE_HIP)
  typedef hipcub::BlockReduce<double, block_size> BlockReduce;
#endif
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const double total_e = BlockReduce(temp_storage).Sum(ei); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(energy, total_e);
  }
}

void computeCoordinationNumberSelfGroupCUDAObject::computeCoordinationNumberSelfGroupCUDA(
  const AtomGroupPositionsCUDA& group1,
  AtomGroupGradientsCUDA& gradient1,
  double inv_r0,
  double* d_energy,
  cudaGraph_t& graph,
  cudaStream_t stream) {
  unsigned int numAtoms1 = group1.getNumAtoms();
  if (numAtoms1 < 2) return;
  const double* pos1x = group1.getDeviceX();
  const double* pos1y = group1.getDeviceY();
  const double* pos1z = group1.getDeviceZ();
  double* g1x = gradient1.getDeviceX();
  double* g1y = gradient1.getDeviceY();
  double* g1z = gradient1.getDeviceZ();
  double pairlist_tol = 0.1;
  void* args[] = {
    &pos1x, &pos1y, &pos1z,
    &numAtoms1,
    &inv_r0, &d_energy,
    &g1x, &g1y, &g1z,
    &d_tilesList,
    &d_tilesListStart,
    &d_tilesListSizes,
    &pairlist_tol,
    &d_pairlist};
  cudaKernelNodeParams kernelNodeParams = {0};
  kernelNodeParams.blockDim       = dim3(self_group_block_size, 1, 1);
  kernelNodeParams.sharedMemBytes = 0;
  kernelNodeParams.kernelParams   = args;
  kernelNodeParams.extra          = NULL;
  kernelNodeParams.func           =
    (void*)computeCoordinationNumberSelfGroupCUDAKernel1<6, 12, self_group_block_size, false, false>;
  int deviceID = 0;
  int num_blocks_occ;
  int multiProcessorCount;
  checkGPUError(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_occ, kernelNodeParams.func, self_group_block_size, 0));
  checkGPUError(cudaGetDevice(&deviceID));
  checkGPUError(cudaDeviceGetAttribute(&multiProcessorCount, cudaDevAttrMultiProcessorCount, deviceID));
  const unsigned int maxNumBlocks = num_blocks_occ * multiProcessorCount;
  const unsigned int num_blocks = std::min(maxNumBlocks, (numAtoms1 + self_group_block_size - 1) / self_group_block_size);
  kernelNodeParams.gridDim        = dim3(num_blocks, 1, 1);
  cudaGraphNode_t node;
  checkGPUError(cudaGraphAddKernelNode(
    &node, graph, NULL,
    0, &kernelNodeParams));
}
