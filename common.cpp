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
      coordnum_pairlist<6, 12, false, false>(
        x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
        gradients1.gx[j], gradients1.gy[j], gradients1.gz[j], 0, nullptr);
    }
    energy += ei;
    gradients1.gx[i] += gx1;
    gradients1.gy[i] += gy1;
    gradients1.gz[i] += gz1;
  }
}

void computeCoordinationNumberSelfGroupInterpolate(
  const AtomGroupPositions& __restrict pos1,
  double inv_r0,
  double& __restrict energy,
  AtomGroupGradients& __restrict gradients1,
  const SplineInterpolate& spline) {
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
      // coordnum_pairlist<6, 12, false, false>(
      //   x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
      //   gradients1.gx[j], gradients1.gy[j], gradients1.gz[j], 0, nullptr);
      coordnum_pairlist_interp<false, false>(
        x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, ei, gx1, gy1, gz1,
        gradients1.gx[j], gradients1.gy[j], gradients1.gz[j], 0, nullptr, spline);
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

SplineInterpolate::SplineInterpolate(
  const std::vector<double>& X,
  const std::vector<double>& Y,
  boundary_condition bc):
  m_bc(bc), m_X(X), m_Y(Y) {
  calcFactors();
}

void SplineInterpolate::calcFactors() {
  // solve the equations:
  // \Delta X_i C_i + 2(\Delta X_{i+1} + \Delta X_{i})C_{i+1}+\Delta X_{i+1} C_{i+2} = 3(\Delta Y_{i+1} - \Delta Y_{i})
  const size_t num_points = m_X.size();
  const size_t N = num_points - 1;
  std::vector<double> dY(N);
  std::vector<double> dX(N);
  m_A.resize(N);
  m_B.resize(N);
  m_C.resize(N);
  m_D.resize(N);
  for (size_t i = 0; i < N; ++i) {
    dX[i] = m_X[i+1] - m_X[i];
    dY[i] = m_Y[i+1] - m_Y[i];
  }
  Matrix tmp_mat(N+1, N+1);
  Matrix tmp_vec(N+1, 1);
  for (size_t i = 1; i < N; ++i) {
    tmp_mat(i, i-1) = dX[i-1];
    tmp_mat(i, i) = 2.0 * (dX[i-1] + dX[i]);
    tmp_mat(i, i+1) = dX[i];
    tmp_vec(i, 0) = 3.0 * (dY[i] / dX[i] - dY[i-1] / dX[i-1]);
  }
  if (m_bc == boundary_condition::natural) {
    // std::cout << "Using natural boundary";
    // natural boundary:
    // S_{0}'' (X_0) = 0
    tmp_mat(0, 0) = 1.0;
    tmp_vec(0, 0) = 0.0;
    // although we have only N splines, assuming there is an extra spline,
    // then S_{N}'' (X_N) = 0
    tmp_mat(N, N) = 1.0;
    tmp_vec(N, 0) = 0.0;
  } else if (m_bc == boundary_condition::not_a_knot) {
    // std::cout << "Using not-a-knot boundary";
    // S_{0}''' (X_1) = S_{1}''' (X_1)
    tmp_mat(0, 0) = -dX[1];
    tmp_mat(0, 1) = dX[0] + dX[1];
    tmp_mat(0, 2) = -dX[0];
    tmp_vec(0, 0) = 0.0;
    tmp_mat(N, N-2) = -dX[N-2];
    tmp_mat(N, N-1) = dX[N-2] + dX[N-1];
    tmp_mat(N, N) = -dX[N-1];
    tmp_vec(N, N) = 0.0;
  }
  const Matrix tmp_C = GaussianElimination(tmp_mat, tmp_vec);
  for (size_t i = 0; i < N; ++i) {
    m_C[i] = tmp_C(i, 0);
    m_D[i] = (tmp_C(i+1, 0) - tmp_C(i, 0)) / (3.0 * dX[i]);
    m_B[i] = dY[i] / dX[i] - dX[i] * (2.0 * tmp_C(i, 0) + tmp_C(i+1, 0)) / 3.0;
    m_A[i] = m_Y[i];
  }
}

SplineInterpolate::Matrix SplineInterpolate::GaussianElimination(SplineInterpolate::Matrix& matA, SplineInterpolate::Matrix& matB) {
  // assume matA is always a square matrix
  const size_t N = matA.numRows();
  // B has M columns
  const size_t M = matB.numColumns();
  // std::cerr << "N = " << N << " ; M = " << M << std::endl;
  // row index of column pivots
  std::vector<size_t> pivot_indices(N, 0);
  // bookkeep the used rows
  std::vector<bool> used_rows(N, false);
  // iterate over columns
  for (size_t j = 0; j < N; ++j) {
    // iterate over rows and find the pivot
    bool firsttime = true;
    double pivot = 0.0;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        // find column pivot in the remaining rows
        if (firsttime) {
          pivot = matA(k, j);
          pivot_indices[j] = k;
          firsttime = false;
        } else {
          if (std::abs(matA(k, j)) > std::abs(pivot)) {
            pivot = matA(k, j);
            pivot_indices[j] = k;
          }
        }
      }
    }
    used_rows[pivot_indices[j]] = true;
    for (size_t k = 0; k < N; ++k) {
      if (used_rows[k] == false) {
        const double factor = matA(k, j) / pivot;
#ifdef DEBUG
        std::cout << "k = " << k << " ; factor = " << factor << std::endl;
#endif
        for (size_t i = j; i < N; ++i) {
          matA(k, i) = matA(k, i) - matA(pivot_indices[j], i) * factor;
        }
        for (size_t i = 0; i < M; ++i) {
          matB(k, i) = matB(k, i) - matB(pivot_indices[j], i) * factor;
        }
      }
    }
#ifdef DEBUG
    std::cout << "Matrix A:\n";
    matA.print(std::cout) << '\n';
#endif
  }
#ifdef DEBUG
  std::cout << "pivot_indices:\n";
  for (const auto& i : pivot_indices) {
    std::cout << i << std::endl;
  }
#endif
  // solve X, backsubstitution
  SplineInterpolate::Matrix matX(N, M);
  // boundary check
  if (N > 0) {
    for (int64_t j = N - 1; j >= 0; --j) {
      // first, we need to find which row has the last pivot
      const size_t l = pivot_indices[j];
      for (size_t i = 0; i < M; ++i) {
        if (j == int64_t(N - 1)) {
          matX(j, i) = matB(l, i) / matA(l, j);
        } else if (j == int64_t(N - 2)) {
          matX(j, i) = (matB(l, i) - matX(j+1, i) * matA(l, j+1)) / matA(l, j);
        } else {
          double sum = 0.0;
          for (size_t k = 1; k < N - j; ++k) {
            sum += matX(j+k, i) * matA(l, j+k);
          }
          matX(j, i) = (matB(pivot_indices[j], i) - sum) / matA(l, j);
        }
      }
    }
  }
  return matX;
}

SplineInterpolate genCoordNumInterp(int n, int m, double r2_min, double r2_max, size_t points) {
  std::vector<double> X(points);
  std::vector<double> Y(points);
  const double step = (r2_max - r2_min) / (points - 1);
  for (size_t i = 0; i < points; ++i) {
    const double r2 = r2_min + i * step;
    X[i] = r2;
    const double xn = 1.0 - integer_power(std::sqrt(r2), n);
    const double xd = 1.0 - integer_power(std::sqrt(r2), m);
    if (std::abs(xd) < 1.0e-7) {
      Y[i] = 0.5;
    } else {
      Y[i] = xn / xd;
    }
  }
  SplineInterpolate spline(X, Y);
  return spline;
}
