#include "common.h"

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

void computeCoordinationNumberTwoGroupsCUDA(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupGradientsCUDA& force1,
  AtomGroupGradientsCUDA& force2,
  double inv_r0,
  double* d_energy,
  cudaGraph_t& graph,
  cudaStream_t stream);

#endif // GPU_KERNEL_H
