#include "common.h"

#ifndef GPU_KERNEL_H
#define GPU_KERNEL_H

class ComputeCoordinationNumberTwoGroups {
public:
  ComputeCoordinationNumberTwoGroups() {}
  ~ComputeCoordinationNumberTwoGroups();
  void initialize(
    unsigned int numAtoms1,
    unsigned int numAtoms2,
    bool usePairlist = false,
    double pairlistTolerance = 0);
  void addComputeToGraph(
    const AtomGroupPositionsCUDA& group1,
    const AtomGroupPositionsCUDA& group2,
    AtomGroupGradientsCUDA& gradient1,
    AtomGroupGradientsCUDA& gradient2,
    double inv_r0,
    double* energy_out,
    bool rebuild_pairlist,
    cudaGraphNode_t& node,
    const std::vector<cudaGraphNode_t>& dependencies,
    cudaGraph_t& graph);
  // For debug
  std::vector<char> pairlistToHost() const;
private:
  static constexpr const unsigned int default_block_size = 128;
  bool initialized = false;
  bool m_usePairlist = false;
  size_t pairlistSize = 0;
  bool* d_pairlist = nullptr;
  double m_pairlistTolerance = 0;
  unsigned int* d_tbcount = nullptr;
  double* d_energy_tmp = nullptr;
  bool pairlistTransposed = false;
  unsigned int m_numAtoms1 = 0;
  unsigned int m_numAtoms2 = 0;
};

class ComputeCoordinationNumberTwoGroups2 {
public:
  ComputeCoordinationNumberTwoGroups2() {}
  ~ComputeCoordinationNumberTwoGroups2();
  void initialize(
    unsigned int numAtoms1,
    unsigned int numAtoms2,
    bool enablePairlist = false,
    double pairlistTolerance = 0);
  void addComputeToGraph(
    const AtomGroupPositionsCUDA& group1,
    const AtomGroupPositionsCUDA& group2,
    AtomGroupGradientsCUDA& gradient1,
    AtomGroupGradientsCUDA& gradient2,
    double inv_r0,
    double* energy_out,
    cudaGraphNode_t& node,
    const std::vector<cudaGraphNode_t>& dependencies,
    cudaGraph_t& graph);
  // For debug
  std::vector<char> pairlistToHost() const;
  void addUpdateFlagsToGraph(
    const bool use_pairlist_in,
    const bool rebuild_pairlist_in,
    cudaGraphNode_t& node_use_pairlist,
    cudaGraphNode_t& node_rebuild_pairlist,
    const std::vector<cudaGraphNode_t>& dependencies,
    cudaGraph_t& graph);
private:
  static constexpr const unsigned int default_block_size = 128;
  bool initialized = false;
  bool m_enablePairlist = false;
  size_t pairlistSize = 0;
  bool* d_pairlist = nullptr;
  double m_pairlistTolerance = 0;
  unsigned int* d_tbcount = nullptr;
  double* d_energy_tmp = nullptr;
  bool pairlistTransposed = false;
  unsigned int m_numAtoms1 = 0;
  unsigned int m_numAtoms2 = 0;
  bool* d_use_pairlist = nullptr;
  bool* d_rebuild_pairlist = nullptr;
};

class ComputeCoordinationNumberSelfGroupCUDA {
public:
  ComputeCoordinationNumberSelfGroupCUDA(unsigned int self_group_block_size_in = 128, unsigned int tile_size_in = 16);
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
    cudaGraph_t& graph);
  // For debug
  std::vector<char> pairlistToHost() const;
  unsigned int getBlockSize() const {return selfGroupBlockSize;}
private:
  void prepareTilesList(unsigned int numAtoms);
private:
  // static constexpr const unsigned int self_group_block_size = 128;
  inline static constexpr unsigned int selfGroupBlockSizeSupported[] = {32, 64, 128, 256};
  inline static constexpr unsigned int tileSizeSupported[] = {8, 16, 32};
  unsigned int selfGroupBlockSize;
  unsigned int tileSize;
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
