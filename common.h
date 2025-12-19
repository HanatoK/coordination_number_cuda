#include <cstddef>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

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

struct AtomGroupForces {
  std::vector<double> fx;
  std::vector<double> fy;
  std::vector<double> fz;
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
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    if (d_z) cudaFree(d_z);
  }
  AtomGroupPositionsCUDA(const AtomGroupPositions& hostPositions, cudaStream_t stream_in, size_t clusterSize = 1) {
    numAtoms = hostPositions.x.size();
    stream = stream_in;
    // const size_t numClusters = size_t(numAtoms / clusterSize) + ((numAtoms % clusterSize) ? 1 : 0);
    // atomStorageSize = numClusters * clusterSize;
    // checkGPUError(cudaMalloc(&d_x, atomStorageSize * sizeof(double)));
    // checkGPUError(cudaMalloc(&d_y, atomStorageSize * sizeof(double)));
    // checkGPUError(cudaMalloc(&d_z, atomStorageSize * sizeof(double)));
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

class AtomGroupForcesCUDA {
private:
  double *d_fx;
  double *d_fy;
  double *d_fz;
  size_t numAtoms;
  // size_t atomStorageSize;
  cudaStream_t stream;
public:
  AtomGroupForcesCUDA(): d_fx(nullptr), d_fy(nullptr), d_fz(nullptr), numAtoms(0), /*atomStorageSize(0),*/ stream(0) {}
  ~AtomGroupForcesCUDA() {
    if (d_fx) cudaFree(d_fx);
    if (d_fy) cudaFree(d_fy);
    if (d_fz) cudaFree(d_fz);
  }
  void initialize(size_t numAtomsInput, cudaStream_t stream_in, size_t clusterSize = 1) {
    numAtoms = numAtomsInput;
    stream = stream_in;
    // const size_t numClusters = size_t(numAtoms / clusterSize) + ((numAtoms % clusterSize) ? 1 : 0);
    // // atomStorageSize = numClusters * clusterSize;
    // checkGPUError(cudaMalloc(&d_fx, atomStorageSize * sizeof(double)));
    // checkGPUError(cudaMalloc(&d_fy, atomStorageSize * sizeof(double)));
    // checkGPUError(cudaMalloc(&d_fz, atomStorageSize * sizeof(double)));
    // checkGPUError(cudaMemsetAsync(d_fx, 0, atomStorageSize * sizeof(double), stream));
    // checkGPUError(cudaMemsetAsync(d_fy, 0, atomStorageSize * sizeof(double), stream));
    // checkGPUError(cudaMemsetAsync(d_fz, 0, atomStorageSize * sizeof(double), stream));
    checkGPUError(cudaMalloc(&d_fx, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_fy, numAtoms * sizeof(double)));
    checkGPUError(cudaMalloc(&d_fz, numAtoms * sizeof(double)));
    checkGPUError(cudaMemsetAsync(d_fx, 0, numAtoms * sizeof(double), stream));
    checkGPUError(cudaMemsetAsync(d_fy, 0, numAtoms * sizeof(double), stream));
    checkGPUError(cudaMemsetAsync(d_fz, 0, numAtoms * sizeof(double), stream));
  }
  double* getDeviceX() const { return d_fx; }
  double* getDeviceY() const { return d_fy; }
  double* getDeviceZ() const { return d_fz; }
  AtomGroupForces toHost() const {
    AtomGroupForces result;
    result.fx.resize(numAtoms);
    result.fy.resize(numAtoms);
    result.fz.resize(numAtoms);
    const size_t copySize = numAtoms * sizeof(double);
    checkGPUError(cudaMemcpyAsync(result.fx.data(), d_fx, copySize, cudaMemcpyDeviceToHost, stream));
    checkGPUError(cudaMemcpyAsync(result.fy.data(), d_fy, copySize, cudaMemcpyDeviceToHost, stream));
    checkGPUError(cudaMemcpyAsync(result.fz.data(), d_fz, copySize, cudaMemcpyDeviceToHost, stream));
    return result;
  }
};

AtomGroupPositions generateRandomAtomGroupPositions(int seed, size_t numAtoms, 
                                                    double xMin, double xMax, 
                                                    double yMin, double yMax, 
                                                    double zMin, double zMax);

void computeCoordinationNumber(const AtomGroupPositions& pos1, 
                               const AtomGroupPositions& pos2,
                               double inv_r0,
                               double& energy, 
                               AtomGroupForces& forces1, 
                               AtomGroupForces& forces2);

inline __host__ __device__ double integer_power(double const& x, int const n) {
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

inline __host__ __device__ double integer_power_2(double const& x, int const n, double& z) {
  double yy, ww, zz;
  if (x == 0.0) return 0.0;
  int nn = (n > 0) ? n : -n;
  int nnn = (n - 1 > 0) ? n - 1 : -(n + 1);
  ww = x;
  for (yy = 1.0, zz = 1.0; nn != 0; nn >>= 1, nnn >>= 1, ww *=ww) {
    if (nn & 1) yy *= ww;
    if ((nnn != 0) && (nnn & 1)) zz *= ww;
  }
  z = n * ((n > 0) ? zz : 1.0/zz);
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
  double& energy,
  double& fx1, double& fy1, double& fz1,
  double& fx2, double& fy2, double& fz2) {
  const double dx = (x2 - x1) * inv_r0_x;
  const double dy = (y2 - y1) * inv_r0_y;
  const double dz = (z2 - z1) * inv_r0_z;
  const double r2 = dx * dx + dy * dy + dz * dz;
  int const en2 = N/2;
  int const ed2 = M/2;
  double nxn_1, dxd_1;
  double const xn = 1.0 - integer_power_2(r2, en2, nxn_1);
  double const xd_inv = 1.0 / (1.0 - integer_power_2(r2, ed2, dxd_1));
  double const func = xn * xd_inv;
  energy += func < 0 ? 0.0 : func;
  if (func > 0.0) {
    // Compute forces: the negative of the gradients
    const double dfunc_dr2 = (nxn_1 - dxd_1 * func) * xd_inv;
    // const double dfunc_dr2 = func * ((ed2 * xd) / ((1.0 - xd) * r2) - (en2 * xn / ((1.0 - xn) * r2)));
    const double dr2_dx = 2.0 * dx * inv_r0_x;
    const double dr2_dy = 2.0 * dy * inv_r0_y;
    const double dr2_dz = 2.0 * dz * inv_r0_z;
    fx1 += dfunc_dr2 * dr2_dx;
    fy1 += dfunc_dr2 * dr2_dy;
    fz1 += dfunc_dr2 * dr2_dz;
    fx2 += -dfunc_dr2 * dr2_dx;
    fy2 += -dfunc_dr2 * dr2_dy;
    fz2 += -dfunc_dr2 * dr2_dz;
  }
}

#endif // COMMON_H
