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
  double* h_energy,
  unsigned int* d_tbcount,
  cudaGraph_t& graph);

void computeCoordinationNumberTwoGroupsCUDAPairlist(
  const AtomGroupPositionsCUDA& group1,
  const AtomGroupPositionsCUDA& group2,
  AtomGroupGradientsCUDA& gradient1,
  AtomGroupGradientsCUDA& gradient2,
  double inv_r0,
  double* d_energy,
  double* h_energy,
  unsigned int* d_tbcount,
  cudaGraph_t& graph,
  bool* pairlist,
  double pairlist_tol,
  bool rebuild_pairlist,
  cudaGraphNode_t& node,
  const std::vector<cudaGraphNode_t>& dependencies);

class ComputeCoordinationNumberSelfGroupCUDA {
public:
  ComputeCoordinationNumberSelfGroupCUDA() {}
  ~ComputeCoordinationNumberSelfGroupCUDA();
  void initialize(unsigned int numAtoms, bool usePairlist = false, double pairlistTolerance = 0);
  void addComputeToGraph(
    const AtomGroupPositionsCUDA& group1,
    AtomGroupGradientsCUDA& gradient1,
    double inv_r0,
    double* energy_out,
    bool rebuild_pairlist,
    cudaGraphNode_t& node,
    const std::vector<cudaGraphNode_t>& dependencies,
    cudaGraph_t& graph,
    cudaStream_t stream);
  // For debug
  std::vector<char> pairlistToHost() const;
private:
  void prepareTilesList(unsigned int numAtoms);
private:
  static constexpr const unsigned int self_group_block_size = 128;
  unsigned int* d_tilesList = nullptr;
  unsigned int* d_tilesListStart = nullptr;
  unsigned int* d_tilesListSizes = nullptr;
  bool initialized = false;
  bool m_usePairlist = false;
  size_t pairlistSize = 0;
  bool* d_pairlist = nullptr;
  double m_pairlistTolerance = 0;
  unsigned int* d_tbcount = nullptr;
  double* d_energy_tmp = nullptr;
};

#endif // GPU_KERNEL_H
