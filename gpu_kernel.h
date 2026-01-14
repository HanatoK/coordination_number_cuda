#include "common.h"

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

void computeCoordinationNumberTwoGroupsCUDA(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupGradientsCUDA& gradient1,
  AtomGroupGradientsCUDA& gradient2,
  double inv_r0,
  double* d_energy,
  cudaGraph_t& graph,
  cudaStream_t stream);

class computeCoordinationNumberSelfGroupCUDAObject {
public:
  computeCoordinationNumberSelfGroupCUDAObject() {}
  ~computeCoordinationNumberSelfGroupCUDAObject();
  void initialize(unsigned int numAtoms, bool usePairlist = false, double pairlistTolerance = 0);
  void computeCoordinationNumberSelfGroupCUDA(
    const AtomGroupPositionsCUDA& group1,
    AtomGroupGradientsCUDA& gradient1,
    double inv_r0,
    double* d_energy,
    bool rebuild_pairlist,
    cudaGraph_t& graph,
    cudaStream_t stream);
  // For debug
  std::vector<char> pairlistToHost() const;
private:
  void prepareTilesList(unsigned int numAtoms);
private:
  static constexpr const unsigned int self_group_block_size = 32;
  unsigned int* d_tilesList = nullptr;
  unsigned int* d_tilesListStart = nullptr;
  unsigned int* d_tilesListSizes = nullptr;
  bool initialized = false;
  bool m_usePairlist = false;
  size_t pairlistSize = 0;
  bool* d_pairlist = nullptr;
  double m_pairlistTolerance = 0;
};

#endif // GPU_KERNEL_H
