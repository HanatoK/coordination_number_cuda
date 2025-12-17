#include "common.h"

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

void computeCoordinationNumberCUDA(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupForcesCUDA& force1,
  AtomGroupForcesCUDA& force2,
  double inv_r0,
  double* d_energy,
  cudaGraph_t& graph,
  cudaStream_t stream);

#endif // GPU_KERNEL_H
