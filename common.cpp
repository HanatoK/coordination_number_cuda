#include "common.h"
#include <fmt/format.h>
#include <fstream>
#include <iostream>

AtomGroupPositions generateRandomAtomGroupPositions(int seed, size_t numAtoms, 
                                                     double xMin, double xMax, 
                                                     double yMin, double yMax, 
                                                     double zMin, double zMax) {
  AtomGroupPositions positions;
  positions.x.resize(numAtoms);
  positions.y.resize(numAtoms);
  positions.z.resize(numAtoms);

  std::mt19937 gen(seed); // Fixed seed for reproducibility
  std::uniform_real_distribution<> disX(xMin, xMax);
  std::uniform_real_distribution<> disY(yMin, yMax);
  std::uniform_real_distribution<> disZ(zMin, zMax);

  for (size_t i = 0; i < numAtoms; ++i) {
    positions.x[i] = disX(gen);
    positions.y[i] = disY(gen);
    positions.z[i] = disZ(gen);
  }

  return positions;
}


void computeCoordinationNumberTwoGroups(
  const AtomGroupPositions& pos1,
  const AtomGroupPositions& pos2,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1,
  AtomGroupGradients& gradients2) {
  const size_t numAtoms1 = pos1.x.size();
  const size_t numAtoms2 = pos2.x.size();
  for (size_t i = 0; i < numAtoms1; ++i) {
    double gx1 = 0.0, gy1 = 0.0, gz1 = 0.0;
    const double x1 = pos1.x[i];
    const double y1 = pos1.y[i];
    const double z1 = pos1.z[i];
    double ei = 0.0;
    for (size_t j = 0; j < numAtoms2; ++j) {
      const double x2 = pos2.x[j];
      const double y2 = pos2.y[j];
      const double z2 = pos2.z[j];
      coordnum<6, 12>(x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
                      gradients2.gx[j], gradients2.gy[j], gradients2.gz[j]);
    }
    energy += ei;
    gradients1.gx[i] += gx1;
    gradients1.gy[i] += gy1;
    gradients1.gz[i] += gz1;
  }
}

void computeCoordinationNumberSelfGroup(
  const AtomGroupPositions& __restrict pos1,
  double inv_r0,
  double& __restrict energy,
  AtomGroupGradients& __restrict gradients1) {
  const size_t numAtoms1 = pos1.x.size();
  if (numAtoms1 < 2) return;
  for (size_t i = 0; i < numAtoms1 - 1; ++i) {
    double gx1 = 0.0, gy1 = 0.0, gz1 = 0.0;
    const double x1 = pos1.x[i];
    const double y1 = pos1.y[i];
    const double z1 = pos1.z[i];
    double ei = 0.0;
    // #pragma omp simd
    for (size_t j = i + 1; j < numAtoms1; ++j) {
      const double x2 = pos1.x[j];
      const double y2 = pos1.y[j];
      const double z2 = pos1.z[j];
      // std::cout << fmt::format("(CPU) x1 = {}, x2 = {}\n", x1, x2);
      coordnum<6, 12>(x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
                      gradients1.gx[j], gradients1.gy[j], gradients1.gz[j]);
    }
    energy += ei;
    gradients1.gx[i] += gx1;
    gradients1.gy[i] += gy1;
    gradients1.gz[i] += gz1;
  }
}

void computeCoordinationNumberSelfGroup2(
  const AtomGroupPositions& __restrict pos1,
  double inv_r0,
  double& __restrict energy,
  AtomGroupGradients& __restrict gradients1) {
  const size_t numAtoms1 = pos1.x.size();
  if (numAtoms1 < 2) return;
  constexpr const size_t block_size = vector_ext::vsize;
  const size_t num_blocks = (numAtoms1 + block_size - 1) / block_size;
  // fmt::print("num_blocks = {}\n", num_blocks);
  vector_ext::v4si mask_i, mask_j, tids, jids, mask;
  vector_ext::v4sd posx_i, posy_i, posz_i;
  vector_ext::v4sd posx_j, posy_j, posz_j;
  for (size_t iblock_index = 0; iblock_index < num_blocks; ++iblock_index) {
    double gx_i[block_size] = {0};
    double gy_i[block_size] = {0};
    double gz_i[block_size] = {0};
    double e = 0;
    // #pragma unroll 4
    for (size_t thread_index = 0; thread_index < block_size; ++thread_index) {
      const size_t tid = iblock_index * block_size + thread_index;
      mask_i[thread_index] = int(tid < numAtoms1);
      posx_i[thread_index] = tid < numAtoms1 ? pos1.x[tid] : 0.0;
      posy_i[thread_index] = tid < numAtoms1 ? pos1.y[tid] : 0.0;
      posz_i[thread_index] = tid < numAtoms1 ? pos1.z[tid] : 0.0;
      tids[thread_index] = tid;
    }
    posx_j = posx_i;
    posy_j = posy_i;
    posz_j = posz_i;
    mask_j = mask_i;
    jids = tids;
    // Self block
    #pragma unroll 7
    for (size_t thread_index = 1; thread_index < block_size; ++thread_index) {
      posx_j = __builtin_shufflevector(posx_j, posx_j, 1, 2, 3, 0);
      posy_j = __builtin_shufflevector(posy_j, posy_j, 1, 2, 3, 0);
      posz_j = __builtin_shufflevector(posz_j, posz_j, 1, 2, 3, 0);
      mask_j = __builtin_shufflevector(mask_j, mask_j, 1, 2, 3, 0);
      jids = __builtin_shufflevector(jids, jids, 1, 2, 3, 0);
      mask = mask_i & mask_j;
      for (size_t m = 0; m < thread_index; ++m) {
        mask[block_size - thread_index + m] = 0;
      }
      vector_ext::coordnum<6, 12>(
        mask, posx_i, posx_j, posy_i, posy_j, posz_i, posz_j,
        inv_r0, inv_r0, inv_r0, e, jids, gx_i, gy_i, gz_i,
        gradients1.gx.data(), gradients1.gy.data(), gradients1.gz.data());
    }
    // Iterate over j-blocks
    // #pragma unroll 4
    for (size_t jblock_index = iblock_index + 1; jblock_index < num_blocks; ++jblock_index) {
      #pragma unroll 8
      for (size_t thread_index = 0; thread_index < block_size; ++thread_index) {
        const size_t jid = jblock_index * block_size + thread_index;
        mask_j[thread_index] = int(jid < numAtoms1);
        posx_j[thread_index] = jid < numAtoms1 ? pos1.x[jid] : 0.0;
        posy_j[thread_index] = jid < numAtoms1 ? pos1.y[jid] : 0.0;
        posz_j[thread_index] = jid < numAtoms1 ? pos1.z[jid] : 0.0;
        jids[thread_index] = jid;
      }
      // First round: without reshuffling
      {
        mask = mask_i & mask_j;
        vector_ext::coordnum<6, 12>(
          mask, posx_i, posx_j, posy_i, posy_j, posz_i, posz_j,
          inv_r0, inv_r0, inv_r0, e, jids, gx_i, gy_i, gz_i,
          gradients1.gx.data(), gradients1.gy.data(), gradients1.gz.data());
      }
      #pragma unroll 7
      for (size_t thread_index = 1; thread_index < block_size; ++thread_index) {
        posx_j = __builtin_shufflevector(posx_j, posx_j, 1, 2, 3, 0);
        posy_j = __builtin_shufflevector(posy_j, posy_j, 1, 2, 3, 0);
        posz_j = __builtin_shufflevector(posz_j, posz_j, 1, 2, 3, 0);
        mask_j = __builtin_shufflevector(mask_j, mask_j, 1, 2, 3, 0);
        jids = __builtin_shufflevector(jids, jids, 1, 2, 3, 0);
        mask = mask_i & mask_j;
        vector_ext::coordnum<6, 12>(
          mask, posx_i, posx_j, posy_i, posy_j, posz_i, posz_j,
          inv_r0, inv_r0, inv_r0, e, jids, gx_i, gy_i, gz_i,
          gradients1.gx.data(), gradients1.gy.data(), gradients1.gz.data());
      }
    }
    #pragma unroll 8
    for (size_t thread_index = 0; thread_index < block_size; ++thread_index) {
      const size_t tid = iblock_index * block_size + thread_index;
      if (tid < numAtoms1) {
        gradients1.gx[tid] += gx_i[thread_index];
        gradients1.gy[tid] += gy_i[thread_index];
        gradients1.gz[tid] += gz_i[thread_index];
      }
    }
    energy += e;
  }
}

void computeCoordinationNumberSelfGroupWithPairlist(
  const AtomGroupPositions& pos1,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1,
  bool rebuildPairlist,
  bool* pairlist,
  double pairlistTolerance) {
  const size_t numAtoms1 = pos1.x.size();
  if (numAtoms1 < 2) return;
  bool* pairlist_ptr = pairlist;
  for (size_t i = 0; i < numAtoms1 - 1; ++i) {
    double gx1 = 0.0, gy1 = 0.0, gz1 = 0.0;
    const double x1 = pos1.x[i];
    const double y1 = pos1.y[i];
    const double z1 = pos1.z[i];
    double ei = 0.0;
    // const size_t pair_id_i = i * (2 * numAtoms1 - 1 - i) / 2;
    for (size_t j = i + 1; j < numAtoms1; ++j) {
      // const size_t pair_id_j = j - i - 1;
      const double x2 = pos1.x[j];
      const double y2 = pos1.y[j];
      const double z2 = pos1.z[j];
      // std::cout << fmt::format("(CPU) x1 = {}, x2 = {}\n", x1, x2);
      if (rebuildPairlist) {
        coordnum_pairlist<6, 12, true, true>(
          x1, x2, y1, y2, z1, z2,
          inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
          gradients1.gx[j], gradients1.gy[j], gradients1.gz[j],
          pairlistTolerance, pairlist_ptr);
      } else {
        coordnum_pairlist<6, 12, true, false>(
          x1, x2, y1, y2, z1, z2,
          inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
          gradients1.gx[j], gradients1.gy[j], gradients1.gz[j],
          pairlistTolerance, pairlist_ptr);
      }
      // if (pair_id_i + pair_id_j != int(pairlist_ptr - pairlist)) {
      //   std::cout << fmt::format("k = {}, real = {}\n", pair_id_i + pair_id_j, int(pairlist_ptr - pairlist));
      // }
      // std::cout << fmt::format("(CPU) iid = {}, jid = {}, pairlistID = {}\n", i, j, int(pairlist_ptr - pairlist));
      pairlist_ptr++;
    }
    energy += ei;
    gradients1.gx[i] += gx1;
    gradients1.gy[i] += gy1;
    gradients1.gz[i] += gz1;
  }
}

void writeToFile(
  const std::vector<double>& x,
  const std::vector<double>& y,
  const std::vector<double>& z,
  const std::string& filename) {
  std::ofstream ofs(filename);
  const size_t sz = x.size();
  for (size_t i = 0; i < sz; ++i) {
    ofs << fmt::format("{:15.7e} {:15.7e} {:15.7e}\n", x[i], y[i], z[i]);
  }
}

int gpuAssert(cudaError_t code, const char *file, int line)
{
  if (code != cudaSuccess) {
    std::string error =
      std::string("GPUassert: ") +
      cudaGetErrorString(code) + " " + file + ", line " + std::to_string(line);
    std::cerr << error << std::endl;
    throw(error);
    // return cvm::error(error, COLVARS_ERROR);
  }
  return 0;
}
