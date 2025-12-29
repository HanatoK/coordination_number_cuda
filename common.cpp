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
