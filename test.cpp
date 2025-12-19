#include "common.h"
#include <iostream>
#include <fmt/format.h>
#include <chrono>
#include "gpu_kernel.h"

struct calculationResult {
  AtomGroupForces forces1;
  AtomGroupForces forces2;
  double energy;
};

void testNumericalGradient() {
  // Generate two random positions
  const double x1 = 1.0;
  const double y1 = 2.0;
  const double z1 = 3.0;
  const double x2 = 1.5;
  const double y2 = 2.5;
  const double z2 = 3.5;
  const double cutoffDistance = 2.0;
  const double inv_r0 = 1.0 / cutoffDistance;
  double energy = 0.0;
  double fx1 = 0.0, fy1 = 0.0, fz1 = 0.0;
  double fx2 = 0.0, fy2 = 0.0, fz2 = 0.0;
  coordnum<6, 12>(x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  std::cout << fmt::format("Analytical gradients:\n");
  std::cout << fmt::format("Atom 1: dx = {}, dy = {}, dz = {}\n", fx1, fy1, fz1);
  std::cout << fmt::format("Atom 2: dx = {}, dy = {}, dz = {}\n", fx2, fy2, fz2);
  std::cout << fmt::format("Energy: {}\n", energy);
  // Numerical gradient
  const double delta = 1e-5;
  double energy_prev = 0, energy_next = 0;
  // For position x1
  coordnum<6, 12>(x1 - delta, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1 + delta, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fx1 = (energy_next - energy_prev) / (2 * delta);
  // For position y1
  energy_prev = 0;
  energy_next = 0;
  coordnum<6, 12>(x1, x2, y1 - delta, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1, x2, y1 + delta, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fy1 = (energy_next - energy_prev) / (2 * delta);
  // For position z1
  energy_prev = 0;
  energy_next = 0;
  coordnum<6, 12>(x1, x2, y1, y2, z1 - delta, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1, x2, y1, y2, z1 + delta, z2, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fz1 = (energy_next - energy_prev) / (2 * delta);
  std::cout << fmt::format("Numerical gradients:\n");
  std::cout << fmt::format("Atom 1: dx = {}, dy = {}, dz = {}\n", numerical_fx1, numerical_fy1, numerical_fz1);
  // For position x2
  energy_prev = 0;
  energy_next = 0;
  coordnum<6, 12>(x1, x2 - delta, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1, x2 + delta, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fx2 = (energy_next - energy_prev) / (2 * delta);
  // For position y2
  energy_prev = 0;
  energy_next = 0;
  coordnum<6, 12>(x1, x2, y1, y2 - delta, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1, x2, y1, y2 + delta, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fy2 = (energy_next - energy_prev) / (2 * delta);
  // For position z2
  energy_prev = 0;
  energy_next = 0;
  coordnum<6, 12>(x1, x2, y1, y2, z1, z2 - delta, inv_r0, inv_r0, inv_r0, energy_prev,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  coordnum<6, 12>(x1, x2, y1, y2, z1, z2 + delta, inv_r0, inv_r0, inv_r0, energy_next,
                  fx1, fy1, fz1, fx2, fy2, fz2);
  double numerical_fz2 = (energy_next - energy_prev) / (2 * delta);
  std::cout << fmt::format("Atom 2: dx = {}, dy = {}, dz = {}\n", numerical_fx2, numerical_fy2, numerical_fz2);
}

calculationResult testCoordinationNumber(const AtomGroupPositions& pos1, const AtomGroupPositions& pos2, double cutoffDistance) {
  const size_t group1_size = pos1.x.size();
  const size_t group2_size = pos2.x.size();

  AtomGroupForces forces1;
  forces1.fx.resize(group1_size, 0.0);
  forces1.fy.resize(group1_size, 0.0);
  forces1.fz.resize(group1_size, 0.0);

  AtomGroupForces forces2;
  forces2.fx.resize(group2_size, 0.0);
  forces2.fy.resize(group2_size, 0.0);
  forces2.fz.resize(group2_size, 0.0);
  
  double energy = 0.0;

  const auto start = std::chrono::high_resolution_clock::now();
  computeCoordinationNumber(pos1, pos2, 1.0 / cutoffDistance, energy, forces1, forces2);
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;
  
  std::cout << fmt::format("Coordination number: {:15.7e}, time (CPU) = {:10.5f} ms\n", energy, fp_ms.count());

  // Save positions and forces to files for further analysis if needed
  // writeToFile(pos1.x, pos1.y, pos1.z, "positions1.txt");
  // writeToFile(pos2.x, pos2.y, pos2.z, "positions2.txt");
  // writeToFile(forces1.fx, forces1.fy, forces1.fz, "gradients1.txt");
  // writeToFile(forces2.fx, forces2.fy, forces2.fz, "gradients2.txt");
  return calculationResult{forces1, forces2, energy};
}

calculationResult testCoordinationNumberCUDA(const AtomGroupPositions& pos1, const AtomGroupPositions& pos2, double cutoffDistance) {
  cudaStream_t stream;
  checkGPUError(cudaStreamCreate(&stream));
  AtomGroupPositionsCUDA cudaPos1(pos1, stream);
  AtomGroupPositionsCUDA cudaPos2(pos2, stream);
  AtomGroupForcesCUDA cudaForce1;
  AtomGroupForcesCUDA cudaForce2;
  cudaForce1.initialize(cudaPos1.getNumAtoms(), stream);
  cudaForce2.initialize(cudaPos2.getNumAtoms(), stream);
  double* d_energy;
  checkGPUError(cudaMalloc(&d_energy, sizeof(double)));
  checkGPUError(cudaMemset(d_energy, 0, sizeof(double)));
  double* h_energy;
  checkGPUError(cudaMallocHost(&h_energy, sizeof(double)));
  cudaGraph_t graph;
  checkGPUError(cudaGraphCreate(&graph, 0));
  computeCoordinationNumberCUDA(
    cudaPos1, cudaPos2, cudaForce1, cudaForce2,
    1.0 / cutoffDistance, d_energy, graph, stream);
  cudaGraphExec_t graph_exec;
  checkGPUError(cudaGraphInstantiate(&graph_exec, graph));
  checkGPUError(cudaStreamSynchronize(stream));

  const auto start = std::chrono::high_resolution_clock::now();
  // computeCoordinationNumberCUDA(
  //   cudaPos1, cudaPos2, cudaForce1, cudaForce2,
  //   1.0 / cutoffDistance, d_energy, stream);
  checkGPUError(cudaGraphLaunch(graph_exec, stream));
  checkGPUError(cudaStreamSynchronize(stream));
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;

  checkGPUError(cudaMemcpyAsync(h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost, stream));
  checkGPUError(cudaStreamSynchronize(stream));

  std::cout << fmt::format("Coordination number: {:15.7e}, time (GPU) = {:10.5f} ms\n", *h_energy, fp_ms.count());
  const double energy = *h_energy;
  const auto hostForce1 = cudaForce1.toHost();
  const auto hostForce2 = cudaForce2.toHost();
  // writeToFile(hostForce1.fx,
  //             hostForce1.fy,
  //             hostForce1.fz,
  //             "gradients1_cuda.txt");
  // writeToFile(hostForce2.fx,
  //             hostForce2.fy,
  //             hostForce2.fz,
  //             "gradients2_cuda.txt");

  checkGPUError(cudaFree(d_energy));
  checkGPUError(cudaFreeHost(h_energy));
  checkGPUError(cudaGraphExecDestroy(graph_exec));
  checkGPUError(cudaGraphDestroy(graph));
  checkGPUError(cudaStreamDestroy(stream));
  return calculationResult{hostForce1, hostForce2, energy};
}

void compareResults(const calculationResult& cpuResult, const calculationResult& cudaResult) {
  double maxRelErrorForces1x = 0;
  double maxRelErrorForces1y = 0;
  double maxRelErrorForces1z = 0;
  const size_t numForces1 = cpuResult.forces1.fx.size();
  for (size_t i = 0; i < numForces1; ++i) {
    const double diff_x = std::abs(cpuResult.forces1.fx[i] - cudaResult.forces1.fx[i]) / std::abs(cpuResult.forces1.fx[i]);
    const double diff_y = std::abs(cpuResult.forces1.fy[i] - cudaResult.forces1.fy[i]) / std::abs(cpuResult.forces1.fy[i]);
    const double diff_z = std::abs(cpuResult.forces1.fz[i] - cudaResult.forces1.fz[i]) / std::abs(cpuResult.forces1.fz[i]);
    maxRelErrorForces1x = std::max(diff_x, maxRelErrorForces1x);
    maxRelErrorForces1y = std::max(diff_y, maxRelErrorForces1y);
    maxRelErrorForces1z = std::max(diff_z, maxRelErrorForces1z);
  }
  std::cout << fmt::format("Max relative error of gradients of group1: {:15.7e} {:15.7e} {:15.7e}\n",
                           maxRelErrorForces1x, maxRelErrorForces1y, maxRelErrorForces1z);

  double maxRelErrorForces2x = 0;
  double maxRelErrorForces2y = 0;
  double maxRelErrorForces2z = 0;
  const size_t numForces2 = cpuResult.forces2.fx.size();
  for (size_t i = 0; i < numForces2; ++i) {
    const double diff_x = std::abs(cpuResult.forces2.fx[i] - cudaResult.forces2.fx[i]) / std::abs(cpuResult.forces2.fx[i]);
    const double diff_y = std::abs(cpuResult.forces2.fy[i] - cudaResult.forces2.fy[i]) / std::abs(cpuResult.forces2.fy[i]);
    const double diff_z = std::abs(cpuResult.forces2.fz[i] - cudaResult.forces2.fz[i]) / std::abs(cpuResult.forces2.fz[i]);
    maxRelErrorForces2x = std::max(diff_x, maxRelErrorForces2x);
    maxRelErrorForces2y = std::max(diff_y, maxRelErrorForces2y);
    maxRelErrorForces2z = std::max(diff_z, maxRelErrorForces2z);
  }
  std::cout << fmt::format("Max relative error of gradients of group2: {:15.7e} {:15.7e} {:15.7e}\n",
                           maxRelErrorForces2x, maxRelErrorForces2y, maxRelErrorForces2z);

  const double relErrorE = std::abs(cpuResult.energy - cudaResult.energy) / std::abs(cpuResult.energy);
  std::cout << fmt::format("Relative error of coordination number: {:15.7e}\n", relErrorE);
}

int main(int argc, char* argv[]) {
  testNumericalGradient();
  unsigned int group1_size = 10000;
  unsigned int group2_size = 2005;
  if (argc > 1) group1_size = std::stoull(argv[1]);
  if (argc > 2) group2_size = std::stoull(argv[2]);
  AtomGroupPositions pos1 = generateRandomAtomGroupPositions(123, group1_size, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
  AtomGroupPositions pos2 = generateRandomAtomGroupPositions(456, group2_size, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
  double cutoffDistance = 6.0;
  const auto cpuResult = testCoordinationNumber(pos1, pos2, cutoffDistance);
  const auto gpuResult = testCoordinationNumberCUDA(pos1, pos2, cutoffDistance);
  compareResults(cpuResult, gpuResult);
  return 0;
}
