#include "gpu_kernel.h"
#include <cub/block/block_reduce.cuh>

template <int N, int M, int block_size>
__global__ void computeCoordinationNumberCUDAKernel1(
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
  double* __restrict fx1,
  double* __restrict fy1,
  double* __restrict fz1,
  double* __restrict fx2,
  double* __restrict fy2,
  double* __restrict fz2) {
  // Shared memory buffers for atoms in group2
  __shared__ double3 shPosition[block_size];
  __shared__ double3 shJForce[block_size];
  __shared__ bool shJMask[block_size];
  // Total energy
  double ei = 0;
  // Number of blocks required to iterate over group1
  const unsigned int numBlocksInGroup1 = (numAtoms1 + block_size - 1) / block_size;
  // Number of blocks required to iterate over group2
  const unsigned int numBlocksInGroup2 = (numAtoms2 + block_size - 1) / block_size;
  for (unsigned int i = blockIdx.x; i < numBlocksInGroup1; i += gridDim.x) {
    unsigned int tid = i * blockDim.x + threadIdx.x;
    // Load the atom i from group1
    const bool mask_i = tid < numAtoms1;
    const double x1 = mask_i ? pos1x[tid] : 0;
    const double y1 = mask_i ? pos1y[tid] : 0;
    const double z1 = mask_i ? pos1z[tid] : 0;
    double3 iForce{0, 0, 0};
    for (unsigned int k = 0; k < numBlocksInGroup2; ++k) {
      const unsigned int j = k * blockDim.x + threadIdx.x;
      // Load atoms from group2 into the shared memory
      const bool mask_j = j < numAtoms2;
      shJMask[threadIdx.x] = mask_j ? true : false;
      shPosition[threadIdx.x].x = mask_j ? pos2x[j] : 0;
      shPosition[threadIdx.x].y = mask_j ? pos2y[j] : 0;
      shPosition[threadIdx.x].z = mask_j ? pos2z[j] : 0;
      shJForce[threadIdx.x].x = 0;
      shJForce[threadIdx.x].y = 0;
      shJForce[threadIdx.x].z = 0;
      __syncthreads();
      for (unsigned int t = 0; t < block_size; ++t) {
        // Since we need to store the interaction force into the
        // shared memory buffer, we need to avoid accumulation of
        // force into the same shared memory position from different
        // threads. (see also https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
        // const unsigned int jid = (t + threadIdx.x) % block_size;
        // Another swizzling method using XOR from possibly CuTe
        // (see also https://zhuanlan.zhihu.com/p/1941306442683515068)
        const unsigned int jid = t ^ threadIdx.x;
        const bool mask_jid = shJMask[jid];
        if (mask_i && mask_jid) {
          const double x2 = shPosition[jid].x;
          const double y2 = shPosition[jid].y;
          const double z2 = shPosition[jid].z;
          coordnum<N, M>(
            x1, x2, y1, y2, z1, z2, inv_r0, ei,
            iForce.x, iForce.y, iForce.z,
            shJForce[jid].x,
            shJForce[jid].y,
            shJForce[jid].z);
        }
        __syncthreads();
      }
      if (mask_j) {
        // Save the j-forces to group2
        atomicAdd(&fx2[j], shJForce[threadIdx.x].x);
        atomicAdd(&fy2[j], shJForce[threadIdx.x].y);
        atomicAdd(&fz2[j], shJForce[threadIdx.x].z);
      }
    }
    if (mask_i) {
      // Save the i-forces to group1
      atomicAdd(&fx1[tid], iForce.x);
      atomicAdd(&fy1[tid], iForce.y);
      atomicAdd(&fz1[tid], iForce.z);
    }
  }
  // Reduction for energy
  __syncthreads();
  typedef cub::BlockReduce<double, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  const double total_e = BlockReduce(temp_storage).Sum(ei); __syncthreads();
  if (threadIdx.x == 0) {
    atomicAdd(energy, total_e);
  }
}

void computeCoordinationNumberCUDA(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupForcesCUDA& force1,
  AtomGroupForcesCUDA& force2,
  double inv_r0,
  double* d_energy,
  cudaStream_t stream) {
  unsigned int numAtoms1 = group1.getNumAtoms();
  unsigned int numAtoms2 = group2.getNumAtoms();
  if (numAtoms2 > numAtoms1) {
    computeCoordinationNumberCUDA(
      group2, group1, force2, force1, inv_r0, d_energy, stream);
    return;
  }
  const double* pos1x = group1.getDeviceX();
  const double* pos1y = group1.getDeviceY();
  const double* pos1z = group1.getDeviceZ();
  const double* pos2x = group2.getDeviceX();
  const double* pos2y = group2.getDeviceY();
  const double* pos2z = group2.getDeviceZ();
  double* f1x = force1.getDeviceX();
  double* f1y = force1.getDeviceY();
  double* f1z = force1.getDeviceZ();
  double* f2x = force2.getDeviceX();
  double* f2y = force2.getDeviceY();
  double* f2z = force2.getDeviceZ();
  void* args[] = {
    &pos1x, &pos1y, &pos1z,
    &pos2x, &pos2y, &pos2z,
    &numAtoms1, &numAtoms2,
    &inv_r0, &d_energy,
    &f1x, &f1y, &f1z,
    &f2x, &f2y, &f2z};
  const unsigned int maxNumAtoms = std::max(numAtoms1, numAtoms2);
  // I'm not sure if it's necessary to limit the number of blocks
  constexpr const unsigned int maxNumBlocks = 65536;
  if (maxNumAtoms > 1024) {
    constexpr unsigned int const block_size = 256;
    unsigned int num_blocks = (numAtoms1 + block_size - 1) / block_size;
    num_blocks = num_blocks > maxNumBlocks ? maxNumBlocks : num_blocks;
    checkGPUError(cudaLaunchKernel(
      (void*)(computeCoordinationNumberCUDAKernel1<6, 12, block_size>),
      num_blocks, block_size, args, 0, stream));
  } else if (maxNumAtoms > 256) {
    constexpr unsigned int const block_size = 64;
    unsigned int num_blocks = (numAtoms1 + block_size - 1) / block_size;
    num_blocks = num_blocks > maxNumBlocks ? maxNumBlocks : num_blocks;
    checkGPUError(cudaLaunchKernel(
      (void*)(computeCoordinationNumberCUDAKernel1<6, 12, block_size>),
      num_blocks, block_size, args, 0, stream));
  } else {
    constexpr unsigned int const block_size = 32;
    unsigned int num_blocks = (numAtoms1 + block_size - 1) / block_size;
    num_blocks = num_blocks > maxNumBlocks ? maxNumBlocks : num_blocks;
    checkGPUError(cudaLaunchKernel(
      (void*)(computeCoordinationNumberCUDAKernel1<6, 12, block_size>),
      num_blocks, block_size, args, 0, stream));
  }
}
