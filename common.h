#include <cstddef>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fmt/printf.h>

#if defined(USE_CUDA)
#include <cuda_runtime.h>
#elif defined(USE_HIP)
#include <hip/hip_runtime.h>
#include "hip_defines.h"
#endif

#ifndef COMMON_H
#define COMMON_H

/**
 * @brief Check for CUDA errors and report them
 *
 * @param code The CUDA error code to check
 * @param file The source file where the error occurred
 * @param line The line number in the source file
 * @return 0 if no error, otherwise throw
 */
int gpuAssert(cudaError_t code, const char *file, int line);

#define checkGPUError(ans) gpuAssert((ans), __FILE__, __LINE__);

struct AtomGroupPositions {
  std::vector<double> x;
  std::vector<double> y;
  std::vector<double> z;
};

struct AtomGroupGradients {
  std::vector<double> gx;
  std::vector<double> gy;
  std::vector<double> gz;
};

void writeToFile(
  const std::vector<double>& x,
  const std::vector<double>& y,
  const std::vector<double>& z,
  const std::string& filename);

class AtomGroupPositionsCUDA {
private:
  double *d_x;
  double *d_y;
  double *d_z;
  size_t numAtoms;
  // size_t atomStorageSize;
  cudaStream_t stream;
public:
  AtomGroupPositionsCUDA(): d_x(nullptr), d_y(nullptr), d_z(nullptr), numAtoms(0), /*atomStorageSize(0),*/ stream(0) {}
  ~AtomGroupPositionsCUDA() {
    if (d_x) checkGPUError(cudaFree(d_x));
    if (d_y) checkGPUError(cudaFree(d_y));
    if (d_z) checkGPUError(cudaFree(d_z));
  }
  AtomGroupPositionsCUDA(const AtomGroupPositions& hostPositions, cudaStream_t stream_in, size_t clusterSize = 1) {
    numAtoms = hostPositions.x.size();
    stream = stream_in;
    checkGPUError(cudaMalloc(&d_x, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_y, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_z, numAtoms * sizeof(double)));
    checkGPUError(cudaMemcpyAsync(d_x, hostPositions.x.data(), numAtoms * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkGPUError(cudaMemcpyAsync(d_y, hostPositions.y.data(), numAtoms * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkGPUError(cudaMemcpyAsync(d_z, hostPositions.z.data(), numAtoms * sizeof(double), cudaMemcpyHostToDevice, stream));
  }
  const double* getDeviceX() const { return d_x; }
  const double* getDeviceY() const { return d_y; }
  const double* getDeviceZ() const { return d_z; }
  size_t getNumAtoms() const { return numAtoms; }
};

class AtomGroupGradientsCUDA {
private:
  double *d_gx;
  double *d_gy;
  double *d_gz;
  size_t numAtoms;
  // size_t atomStorageSize;
  cudaStream_t stream;
public:
  AtomGroupGradientsCUDA(): d_gx(nullptr), d_gy(nullptr), d_gz(nullptr), numAtoms(0), /*atomStorageSize(0),*/ stream(0) {}
  ~AtomGroupGradientsCUDA() {
    if (d_gx) checkGPUError(cudaFree(d_gx));
    if (d_gy) checkGPUError(cudaFree(d_gy));
    if (d_gz) checkGPUError(cudaFree(d_gz));
  }
  void initialize(size_t numAtomsInput, cudaStream_t stream_in, size_t clusterSize = 1) {
    numAtoms = numAtomsInput;
    stream = stream_in;
    checkGPUError(cudaMalloc(&d_gx, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_gy, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_gz, numAtoms * sizeof(double)));
    checkGPUError(cudaMemsetAsync(d_gx, 0, numAtoms * sizeof(double), stream));
    checkGPUError(cudaMemsetAsync(d_gy, 0, numAtoms * sizeof(double), stream));
    checkGPUError(cudaMemsetAsync(d_gz, 0, numAtoms * sizeof(double), stream));
  }
  const double* getDeviceX() const { return d_gx; }
  const double* getDeviceY() const { return d_gy; }
  const double* getDeviceZ() const { return d_gz; }
  double* getDeviceX() { return d_gx; }
  double* getDeviceY() { return d_gy; }
  double* getDeviceZ() { return d_gz; }
  AtomGroupGradients toHost() const {
    AtomGroupGradients result;
    result.gx.resize(numAtoms);
    result.gy.resize(numAtoms);
    result.gz.resize(numAtoms);
    const size_t copySize = numAtoms * sizeof(double);
    checkGPUError(cudaMemcpyAsync(result.gx.data(), d_gx, copySize, cudaMemcpyDeviceToHost, stream));
    checkGPUError(cudaMemcpyAsync(result.gy.data(), d_gy, copySize, cudaMemcpyDeviceToHost, stream));
    checkGPUError(cudaMemcpyAsync(result.gz.data(), d_gz, copySize, cudaMemcpyDeviceToHost, stream));
    return result;
  }
};

AtomGroupPositions generateRandomAtomGroupPositions(int seed, size_t numAtoms, 
                                                    double xMin, double xMax, 
                                                    double yMin, double yMax, 
                                                    double zMin, double zMax);

void computeCoordinationNumberTwoGroups(
  const AtomGroupPositions& pos1,
  const AtomGroupPositions& pos2,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1,
  AtomGroupGradients& gradients2);

void computeCoordinationNumberTwoGroupsWithPairlist(
  const AtomGroupPositions& pos1,
  const AtomGroupPositions& pos2,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1,
  AtomGroupGradients& gradients2,
  bool rebuildPairlist,
  bool* pairlist,
  double pairlistTolerance = 0);

void computeCoordinationNumberSelfGroup(
  const AtomGroupPositions& pos1,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1);

void computeCoordinationNumberSelfGroup2(
  const AtomGroupPositions& __restrict pos1,
  double inv_r0,
  double& __restrict energy,
  AtomGroupGradients& __restrict gradients1);

void computeCoordinationNumberSelfGroupWithPairlist(
  const AtomGroupPositions& pos1,
  double inv_r0,
  double& energy,
  AtomGroupGradients& gradients1,
  bool rebuildPairlist,
  bool* pairlist,
  double pairlistTolerance = 0);

inline __host__ __device__ double integer_power(double const& __restrict x, int const n) {
  double yy, ww;
  if (x == 0.0) return 0.0;
  int nn = (n > 0) ? n : -n;
  ww = x;
  for (yy = 1.0; nn != 0; nn >>= 1, ww *=ww) {
    if (nn & 1) yy *= ww;
    // yy *= (nn & 1) * ww;
  }
  return (n > 0) ? yy : 1.0/yy;
}

template <int n>
inline __host__ __device__ double integer_power(double const& __restrict x) {
  double yy, ww;
  if (x == 0.0) return 0.0;
  int nn = (n > 0) ? n : -n;
  ww = x;
  for (yy = 1.0; nn != 0; nn >>= 1, ww *=ww) {
    if (nn & 1) yy *= ww;
    // yy *= (nn & 1) * ww;
  }
  return (n > 0) ? yy : 1.0/yy;
}

template <int N, int M>                               
inline void __host__ __device__ coordnum(
  double x1, double x2,
  double y1, double y2,
  double z1, double z2,
  double inv_r0_x,
  double inv_r0_y,
  double inv_r0_z,
  double& __restrict energy,
  double& __restrict gx1, double& __restrict gy1, double& __restrict gz1,
  double& __restrict gx2, double& __restrict gy2, double& __restrict gz2) {
  const double dx = (x2 - x1) * inv_r0_x;
  const double dy = (y2 - y1) * inv_r0_y;
  const double dz = (z2 - z1) * inv_r0_z;
  const double r2 = dx * dx + dy * dy + dz * dz;
  int const en2 = N/2;
  int const ed2 = M/2;
  const double xn = integer_power<N/2>(r2);
  const double xd = integer_power<M/2>(r2);
  const double func = (1.0-xn)/(1.0-xd);
  energy += func < 0 ? 0.0 : func;
  if (func > 0.0) {
    const double dfunc_dr2 = func * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
    const double dr2_dx = 2.0 * dx * inv_r0_x;
    const double dr2_dy = 2.0 * dy * inv_r0_y;
    const double dr2_dz = 2.0 * dz * inv_r0_z;
    gx1 += -dfunc_dr2 * dr2_dx;
    gy1 += -dfunc_dr2 * dr2_dy;
    gz1 += -dfunc_dr2 * dr2_dz;
    gx2 += dfunc_dr2 * dr2_dx;
    gy2 += dfunc_dr2 * dr2_dy;
    gz2 += dfunc_dr2 * dr2_dz;
  }
}

template <int N, int M, bool use_pairlist, bool rebuild_pairlist>
inline void __host__ __device__ coordnum_pairlist(
  double x1, double x2,
  double y1, double y2,
  double z1, double z2,
  double inv_r0_x,
  double inv_r0_y,
  double inv_r0_z,
  double& energy,
  double& gx1, double& gy1, double& gz1,
  double& gx2, double& gy2, double& gz2,
  double pairlist_tol,
  bool* pairlist_elem) {
  static_assert(N % 2 == 0, "");
  constexpr bool m_is_2n = (M == 2 * N);
  if constexpr (use_pairlist && !rebuild_pairlist) {
    bool const within = *pairlist_elem;
    if (!within) {
      return;
    }
  }
  const double dx = (x2 - x1) * inv_r0_x;
  const double dy = (y2 - y1) * inv_r0_y;
  const double dz = (z2 - z1) * inv_r0_z;
  const double r2 = dx * dx + dy * dy + dz * dz;
  double func, inv_one_pairlist_tol, func_no_pairlist, xn, xd;
  constexpr int const en2 = N/2;
  constexpr int const ed2 = M/2;
  if constexpr (m_is_2n) {
    xn = integer_power<N/2>(r2);
    func_no_pairlist = 1.0 / (1.0 + xn);
  } else {
    xn = integer_power<N/2>(r2);
    xd = integer_power<M/2>(r2);
    func_no_pairlist = (1.0-xn)/(1.0-xd);
  }
  if constexpr (use_pairlist) {
    inv_one_pairlist_tol = 1 / (1.0-pairlist_tol);
    func = (func_no_pairlist - pairlist_tol) * inv_one_pairlist_tol;
  } else {
    func = func_no_pairlist;
  }
  if constexpr (use_pairlist && rebuild_pairlist) {
    *pairlist_elem = (func > (-pairlist_tol * 0.5)) ? true : false;
  }
  energy += func < 0 ? 0.0 : func;
  if (func > 0.0) {
    double dfunc_dr2;
    if constexpr (m_is_2n) {
      if constexpr (use_pairlist) {
        dfunc_dr2 = -0.5 * (func_no_pairlist * func_no_pairlist) * N * xn / r2 * (inv_one_pairlist_tol);
      } else {
        dfunc_dr2 = -0.5 * (func_no_pairlist * func_no_pairlist) * N * xn / r2;
      }
    } else {
      if constexpr (use_pairlist) {
        dfunc_dr2 = func_no_pairlist * inv_one_pairlist_tol * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
      } else {
        dfunc_dr2 = func * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
      }
    }
    const double dr2_dx = 2.0 * dx * inv_r0_x;
    const double dr2_dy = 2.0 * dy * inv_r0_y;
    const double dr2_dz = 2.0 * dz * inv_r0_z;
    gx1 += -dfunc_dr2 * dr2_dx;
    gy1 += -dfunc_dr2 * dr2_dy;
    gz1 += -dfunc_dr2 * dr2_dz;
    gx2 +=  dfunc_dr2 * dr2_dx;
    gy2 +=  dfunc_dr2 * dr2_dy;
    gz2 +=  dfunc_dr2 * dr2_dz;
  }
}

namespace vector_ext {
constexpr const int vsize = 4;
typedef double v4sd __attribute__ ((vector_size(sizeof(double) * vsize)));
typedef int v4si __attribute__ ((vector_size(sizeof(int) * vsize)));
// v4si shuffle_mask[vsize] = {
//   {0, 1, 2, 3},
//   {1, 2, 3, 0},
//   {2, 3, 0, 1},
//   {3, 0, 1, 2}};
template <int N, int M>
inline void coordnum(
  v4si mask,
  v4sd x1, v4sd x2,
  v4sd y1, v4sd y2,
  v4sd z1, v4sd z2,
  const double inv_r0_x,
  const double inv_r0_y,
  const double inv_r0_z,
  double& __restrict energy,
  const v4si& __restrict jid,
  double* __restrict gx1, double* __restrict gy1, double* __restrict gz1,
  double* __restrict gx2, double* __restrict gy2, double* __restrict gz2) {
  const v4sd dx = (x2 - x1) * inv_r0_x;
  const v4sd dy = (y2 - y1) * inv_r0_y;
  const v4sd dz = (z2 - z1) * inv_r0_z;
  const v4sd r2 = dx * dx + dy * dy + dz * dz;
  constexpr int const en2 = N/2;
  constexpr int const ed2 = M/2;
  v4sd xn;
  v4sd xd;
  for (int i = 0; i < vsize; ++i) {
    xn[i] = 1.0;
    xd[i] = 1.0;
  }
  for (int i = 0; i < en2; ++i) {
    xn *= r2;
  }
  for (int i = 0; i < ed2; ++i) {
    xd *= r2;
  }
  v4sd func = (1.0-xn)/(1.0-xd);
  func = func < 0.0 ? 0.0 : func;
  const v4sd dfunc_dr2 = func * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
  const v4sd dr2_dx = 2.0 * dfunc_dr2 * dx * inv_r0_x;
  const v4sd dr2_dy = 2.0 * dfunc_dr2 * dy * inv_r0_y;
  const v4sd dr2_dz = 2.0 * dfunc_dr2 * dz * inv_r0_z;
  for (int i = 0; i < vsize; ++i) {
    const bool m = mask[i];
    energy += m ? func[i] : 0.0;
    if (m) {
      gx1[i] += -dr2_dx[i];
      gy1[i] += -dr2_dy[i];
      gz1[i] += -dr2_dz[i];
      // fmt::println("r2[{}] = {}", i, double(r2[i]));
      gx2[jid[i]] += dr2_dx[i];
      gy2[jid[i]] += dr2_dy[i];
      gz2[jid[i]] += dr2_dz[i];
    }
  }
}
}

class SplineInterpolate {
public:
  enum class boundary_condition {natural, not_a_knot};
  SplineInterpolate(const std::vector<double>& X,
                    const std::vector<double>& Y,
                    boundary_condition bc = boundary_condition::natural);
  class Matrix {
  private:
    std::vector<double> m_data;
    size_t m_rows;
    size_t m_cols;
  public:
    Matrix(size_t rows, size_t cols): m_data(rows * cols, 0), m_rows(rows), m_cols(cols) {}
    double& operator()(size_t i, size_t j) {return m_data[i * m_cols + j];}
    const double& operator()(size_t i, size_t j) const {return m_data[i * m_cols + j];}
    size_t numRows() const {return m_rows;}
    size_t numColumns() const {return m_cols;}
  };
  inline int fastIndex(const double x) const {
    // assume the steps are the same
    const double step = m_X[1] - m_X[0];
    // assume m_X is sorted to be monotonically increasing
    const int index = std::min(
      static_cast<int>(std::floor((x - m_X[0]) / step)),
      static_cast<int>(m_X.size() - 2));
    return index > 0 ? index : 0;
  }
  inline double evaluate(const double x) const {
    const int ix = fastIndex(x);
    const double dx = x - m_X[ix];
    const double dx2 = dx * dx;
    const double dx3 = dx2 * dx;
    const double interp_y = m_A[ix] + m_B[ix] * dx + m_C[ix] * dx2 + m_D[ix] * dx3;
    return interp_y;
  }
  inline double evaluateDerivative(const double x) const {
    const int ix = fastIndex(x);
    const double dx = x - m_X[ix];
    const double dx2 = dx * dx;
    const double dydx = m_B[ix] + m_C[ix] * dx * 2.0 + m_D[ix] * dx2 * 3.0;
    return dydx;
  }
  Matrix GaussianElimination(Matrix& matA, Matrix& matB);
private:
  void calcFactors();
  boundary_condition m_bc;
  std::vector<double> m_X;
  std::vector<double> m_Y;
  std::vector<double> m_A;
  std::vector<double> m_B;
  std::vector<double> m_C;
  std::vector<double> m_D;
};

template <bool use_pairlist, bool rebuild_pairlist>
inline void __host__ __device__ coordnum_pairlist_interp(
  double x1, double x2,
  double y1, double y2,
  double z1, double z2,
  double inv_r0_x,
  double inv_r0_y,
  double inv_r0_z,
  double& energy,
  double& gx1, double& gy1, double& gz1,
  double& gx2, double& gy2, double& gz2,
  double pairlist_tol,
  bool* pairlist_elem,
  const SplineInterpolate& spline) {
  // static_assert(N % 2 == 0, "");
  // constexpr bool m_is_2n = (M == 2 * N);
  if constexpr (use_pairlist && !rebuild_pairlist) {
    bool const within = *pairlist_elem;
    if (!within) {
      return;
    }
  }
  const double dx = (x2 - x1) * inv_r0_x;
  const double dy = (y2 - y1) * inv_r0_y;
  const double dz = (z2 - z1) * inv_r0_z;
  const double r2 = dx * dx + dy * dy + dz * dz;
  double func, inv_one_pairlist_tol, func_no_pairlist/*, xn, xd*/;
  // constexpr int const en2 = N/2;
  // constexpr int const ed2 = M/2;
  // if constexpr (m_is_2n) {
  //   xn = integer_power<N/2>(r2);
  //   func_no_pairlist = 1.0 / (1.0 + xn);
  // } else {
  //   xn = integer_power<N/2>(r2);
  //   xd = integer_power<M/2>(r2);
  //   func_no_pairlist = (1.0-xn)/(1.0-xd);
  // }
  func_no_pairlist = spline.evaluate(r2);
  if constexpr (use_pairlist) {
    inv_one_pairlist_tol = 1 / (1.0-pairlist_tol);
    func = (func_no_pairlist - pairlist_tol) * inv_one_pairlist_tol;
  } else {
    func = func_no_pairlist;
  }
  if constexpr (use_pairlist && rebuild_pairlist) {
    *pairlist_elem = (func > (-pairlist_tol * 0.5)) ? true : false;
  }
  energy += func < 0 ? 0.0 : func;
  if (func > 0.0) {
    double dfunc_dr2 = spline.evaluateDerivative(r2);
    if constexpr (use_pairlist) {
      dfunc_dr2 *= inv_one_pairlist_tol;
    }
    // if constexpr (m_is_2n) {
    //   if (use_pairlist) {
    //     dfunc_dr2 = -0.5 * (func_no_pairlist * func_no_pairlist) * N * xn / r2 * (inv_one_pairlist_tol);
    //   } else {
    //     dfunc_dr2 = -0.5 * (func_no_pairlist * func_no_pairlist) * N * xn / r2;
    //   }
    // } else {
    //   if (use_pairlist) {
    //     dfunc_dr2 = func_no_pairlist * inv_one_pairlist_tol * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
    //   } else {
    //     dfunc_dr2 = func * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
    //   }
    // }
    const double dr2_dx = 2.0 * dx * inv_r0_x;
    const double dr2_dy = 2.0 * dy * inv_r0_y;
    const double dr2_dz = 2.0 * dz * inv_r0_z;
    gx1 += -dfunc_dr2 * dr2_dx;
    gy1 += -dfunc_dr2 * dr2_dy;
    gz1 += -dfunc_dr2 * dr2_dz;
    gx2 +=  dfunc_dr2 * dr2_dx;
    gy2 +=  dfunc_dr2 * dr2_dy;
    gz2 +=  dfunc_dr2 * dr2_dz;
  }
}

void computeCoordinationNumberSelfGroupInterpolate(
  const AtomGroupPositions& __restrict pos1,
  double inv_r0,
  double& __restrict energy,
  AtomGroupGradients& __restrict gradients1,
  const SplineInterpolate& spline);

SplineInterpolate genCoordNumInterp(int n = 6, int m = 12, double r2_min = 0.0, double r2_max = 20.0, size_t points = 2000);

#endif // COMMON_H
