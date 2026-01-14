#include "common.h"
#include <iostream>
#include <fmt/format.h>
#include <chrono>
#include "gpu_kernel.h"
#include <CLI/CLI.hpp>

struct calculationResult {
  AtomGroupGradients gradients1;
  AtomGroupGradients gradients2;
  double energy;
  std::vector<char> pairlist;
};

void testNumericalGradient() {
  std::random_device rd;
  std::mt19937 gen(rd());
  const size_t num_tests = 10;
  const double cutoffDistance = 6.0;
  const double inv_r0 = 1.0 / cutoffDistance;
  std::normal_distribution<double> dis(cutoffDistance, cutoffDistance / 3.0);
  // Numerical gradient
  constexpr const size_t num_deltas = 4;
  for (size_t i = 0; i < num_tests; ++i) {
    double delta = 1e-2;
    const double x1 = dis(gen);
    const double y1 = dis(gen);
    const double z1 = dis(gen);
    const double x2 = dis(gen);
    const double y2 = dis(gen);
    const double z2 = dis(gen);
    double energy = 0.0;
    double fx1 = 0.0, fy1 = 0.0, fz1 = 0.0;
    double fx2 = 0.0, fy2 = 0.0, fz2 = 0.0;
    coordnum<6, 12>(x1, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy,
                    fx1, fy1, fz1, fx2, fy2, fz2);
    std::cout << fmt::format("Group1 = ({:12.7e}, {:12.7e}, {:12.7e}), group2 = ({:12.7e}, {:12.7e}, {:12.7e}), coordnum = {:12.7e}\n", x1, y1, z1, x2, y2, z2, energy);
    for (size_t j = 0; j < num_deltas; ++j) {
      double energy_prev = 0, energy_next = 0;
      double tmp_fx1 = 0;
      double tmp_fy1 = 0;
      double tmp_fz1 = 0;
      double tmp_fx2 = 0;
      double tmp_fy2 = 0;
      double tmp_fz2 = 0;
      // For position x1
      coordnum<6, 12>(x1 - delta, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1 + delta, x2, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fx1 = (energy_next - energy_prev) / (2 * delta);
      // For position y1
      energy_prev = 0;
      energy_next = 0;
      coordnum<6, 12>(x1, x2, y1 - delta, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1, x2, y1 + delta, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fy1 = (energy_next - energy_prev) / (2 * delta);
      // For position z1
      energy_prev = 0;
      energy_next = 0;
      coordnum<6, 12>(x1, x2, y1, y2, z1 - delta, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1, x2, y1, y2, z1 + delta, z2, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fz1 = (energy_next - energy_prev) / (2 * delta);
      // For position x2
      energy_prev = 0;
      energy_next = 0;
      coordnum<6, 12>(x1, x2 - delta, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1, x2 + delta, y1, y2, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fx2 = (energy_next - energy_prev) / (2 * delta);
      // For position y2
      energy_prev = 0;
      energy_next = 0;
      coordnum<6, 12>(x1, x2, y1, y2 - delta, z1, z2, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1, x2, y1, y2 + delta, z1, z2, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fy2 = (energy_next - energy_prev) / (2 * delta);
      // For position z2
      energy_prev = 0;
      energy_next = 0;
      coordnum<6, 12>(x1, x2, y1, y2, z1, z2 - delta, inv_r0, inv_r0, inv_r0, energy_prev,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      coordnum<6, 12>(x1, x2, y1, y2, z1, z2 + delta, inv_r0, inv_r0, inv_r0, energy_next,
                      tmp_fx1, tmp_fy1, tmp_fz1, tmp_fx2, tmp_fy2, tmp_fz2);
      double numerical_fz2 = (energy_next - energy_prev) / (2 * delta);

      // Compute the error for gradient1
      const double diff1 =
        (fx1 - numerical_fx1) * (fx1 - numerical_fx1) +
        (fy1 - numerical_fy1) * (fy1 - numerical_fy1) +
        (fz1 - numerical_fz1) * (fz1 - numerical_fz1);
      const double denom1 =
        std::sqrt(fx1 * fx1 + fy1 * fy1 + fz1 * fz1) +
        std::sqrt(numerical_fx1 * numerical_fx1 + numerical_fy1 * numerical_fy1 + numerical_fz1 * numerical_fz1);
      const double error1 = std::sqrt(diff1) / denom1;

      // Compute the error for gradient2
      const double diff2 =
        (fx2 - numerical_fx2) * (fx2 - numerical_fx2) +
        (fy2 - numerical_fy2) * (fy2 - numerical_fy2) +
        (fz2 - numerical_fz2) * (fz2 - numerical_fz2);
      const double denom2 =
        std::sqrt(fx2 * fx2 + fy2 * fy2 + fz2 * fz2) +
        std::sqrt(numerical_fx2 * numerical_fx2 + numerical_fy2 * numerical_fy2 + numerical_fz2 * numerical_fz2);
      const double error2 = std::sqrt(diff2) / denom2;

      std::cout << fmt::format("  Delta = {:12.5e}, error1 = {:17.10e}, error2 = {:17.10e}\n", delta, error1, error2);
      delta /= 10.0;
    }
  }
}

calculationResult testCoordinationNumber(const AtomGroupPositions& pos1, const AtomGroupPositions& pos2, double cutoffDistance) {
  const size_t group1_size = pos1.x.size();
  const size_t group2_size = pos2.x.size();

  AtomGroupGradients gradients1;
  gradients1.gx.resize(group1_size, 0.0);
  gradients1.gy.resize(group1_size, 0.0);
  gradients1.gz.resize(group1_size, 0.0);

  AtomGroupGradients gradients2;
  gradients2.gx.resize(group2_size, 0.0);
  gradients2.gy.resize(group2_size, 0.0);
  gradients2.gz.resize(group2_size, 0.0);
  
  double energy = 0.0;

  const auto start = std::chrono::high_resolution_clock::now();
  computeCoordinationNumberTwoGroups(pos1, pos2, 1.0 / cutoffDistance, energy, gradients1, gradients2);
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;
  
  std::cout << fmt::format("Coordination number: {:15.7e}, time (CPU) = {:10.5f} ms\n", energy, fp_ms.count());

  // Save positions and gradients to files for further analysis if needed
  // writeToFile(pos1.x, pos1.y, pos1.z, "positions1.txt");
  // writeToFile(pos2.x, pos2.y, pos2.z, "positions2.txt");
  // writeToFile(gradients1.fx, gradients1.fy, gradients1.fz, "gradients1.txt");
  // writeToFile(gradients2.fx, gradients2.fy, gradients2.fz, "gradients2.txt");
  return calculationResult{gradients1, gradients2, energy};
}

calculationResult testCoordinationNumberCUDA(const AtomGroupPositions& pos1, const AtomGroupPositions& pos2, double cutoffDistance) {
  cudaStream_t stream;
  checkGPUError(cudaStreamCreate(&stream));
  AtomGroupPositionsCUDA cudaPos1(pos1, stream);
  AtomGroupPositionsCUDA cudaPos2(pos2, stream);
  AtomGroupGradientsCUDA cudaGradient1;
  AtomGroupGradientsCUDA cudaGradient2;
  cudaGradient1.initialize(cudaPos1.getNumAtoms(), stream);
  cudaGradient2.initialize(cudaPos2.getNumAtoms(), stream);
  double* d_energy;
  checkGPUError(cudaMalloc(&d_energy, sizeof(double)));
  checkGPUError(cudaMemset(d_energy, 0, sizeof(double)));
  double* h_energy;
  checkGPUError(cudaMallocHost((void**)&h_energy, sizeof(double)));
  cudaGraph_t graph;
  checkGPUError(cudaGraphCreate(&graph, 0));
  computeCoordinationNumberTwoGroupsCUDA(
    cudaPos1, cudaPos2, cudaGradient1, cudaGradient2,
    1.0 / cutoffDistance, d_energy, graph, stream);
  cudaGraphExec_t graph_exec;
#if defined(USE_CUDA)
  checkGPUError(cudaGraphInstantiate(&graph_exec, graph));
#elif defined(USE_HIP)
  checkGPUError(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
#endif
  checkGPUError(cudaStreamSynchronize(stream));

  const auto start = std::chrono::high_resolution_clock::now();
  checkGPUError(cudaGraphLaunch(graph_exec, stream));
  checkGPUError(cudaStreamSynchronize(stream));
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;

  checkGPUError(cudaMemcpyAsync(h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost, stream));
  checkGPUError(cudaStreamSynchronize(stream));

  std::cout << fmt::format("Coordination number: {:15.7e}, time (GPU) = {:10.5f} ms\n", *h_energy, fp_ms.count());
  const double energy = *h_energy;
  const auto hostGradient1 = cudaGradient1.toHost();
  const auto hostGradient2 = cudaGradient2.toHost();
  // writeToFile(hostGradient1.fx,
  //             hostGradient1.fy,
  //             hostGradient1.fz,
  //             "gradients1_cuda.txt");
  // writeToFile(hostGradient2.fx,
  //             hostGradient2.fy,
  //             hostGradient2.fz,
  //             "gradients2_cuda.txt");

  checkGPUError(cudaFree(d_energy));
  checkGPUError(cudaFreeHost(h_energy));
  checkGPUError(cudaGraphExecDestroy(graph_exec));
  checkGPUError(cudaGraphDestroy(graph));
  checkGPUError(cudaStreamDestroy(stream));
  return calculationResult{hostGradient1, hostGradient2, energy};
}

calculationResult testCoordinationNumberSelf(const AtomGroupPositions& pos1, double cutoffDistance) {
  const size_t group1_size = pos1.x.size();

  AtomGroupGradients gradients1;
  gradients1.gx.resize(group1_size, 0.0);
  gradients1.gy.resize(group1_size, 0.0);
  gradients1.gz.resize(group1_size, 0.0);

  double energy = 0.0;

  const auto start = std::chrono::high_resolution_clock::now();
  computeCoordinationNumberSelfGroup(pos1, 1.0 / cutoffDistance, energy, gradients1);
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;

  std::cout << fmt::format("Coordination number: {:15.7e}, time (CPU) = {:10.5f} ms\n", energy, fp_ms.count());

  // Save positions and gradients to files for further analysis if needed
  // writeToFile(pos1.x, pos1.y, pos1.z, "positions1.txt");
  // writeToFile(pos2.x, pos2.y, pos2.z, "positions2.txt");
  // writeToFile(gradients1.fx, gradients1.fy, gradients1.fz, "gradients1.txt");
  // writeToFile(gradients2.fx, gradients2.fy, gradients2.fz, "gradients2.txt");
  return calculationResult{gradients1, AtomGroupGradients(), energy};
}

calculationResult testCoordinationNumberSelfPairlist(const AtomGroupPositions& pos1, double cutoffDistance) {
  const size_t group1_size = pos1.x.size();

  AtomGroupGradients gradients1;
  gradients1.gx.resize(group1_size, 0.0);
  gradients1.gy.resize(group1_size, 0.0);
  gradients1.gz.resize(group1_size, 0.0);
  double energy2 = 0;
  AtomGroupGradients gradients2;
  gradients2.gx.resize(group1_size, 0.0);
  gradients2.gy.resize(group1_size, 0.0);
  gradients2.gz.resize(group1_size, 0.0);
  const double pairlistTolerance = 0.1;
  std::vector<char> pairlist((group1_size - 1) * (group1_size - 1), 1);
  double energy = 0.0;

  const auto start = std::chrono::high_resolution_clock::now();
  computeCoordinationNumberSelfGroupWithPairlist(pos1, 1.0 / cutoffDistance, energy, gradients1, true, (bool*)pairlist.data(), pairlistTolerance);
#if 1
  computeCoordinationNumberSelfGroupWithPairlist(pos1, 1.0 / cutoffDistance, energy2, gradients2, false, (bool*)pairlist.data(), pairlistTolerance);
#endif
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;

  std::cout << fmt::format("Coordination number: {:15.7e}, time (CPU) = {:10.5f} ms\n", energy, fp_ms.count());
  // std::cout << fmt::format("Rrebuild pairlist = {}, no rebuild pairlist = {}\n", energy, energy2);

  // Save positions and gradients to files for further analysis if needed
  // writeToFile(pos1.x, pos1.y, pos1.z, "positions1.txt");
  // writeToFile(pos2.x, pos2.y, pos2.z, "positions2.txt");
  // writeToFile(gradients1.fx, gradients1.fy, gradients1.fz, "gradients1.txt");
  // writeToFile(gradients2.fx, gradients2.fy, gradients2.fz, "gradients2.txt");
  return calculationResult{gradients1, AtomGroupGradients(), energy, pairlist};
}

calculationResult testCoordinationNumberSelfCUDA(const AtomGroupPositions& pos1, double cutoffDistance, bool testPairlist = false) {
  cudaStream_t stream;
  checkGPUError(cudaStreamCreate(&stream));
  AtomGroupPositionsCUDA cudaPos1(pos1, stream);
  computeCoordinationNumberSelfGroupCUDAObject compute;
  if (testPairlist) {
    compute.initialize(pos1.x.size(), true, 0.1);
  } else {
    compute.initialize(pos1.x.size());
  }
  // AtomGroupPositionsCUDA cudaPos2(pos2, stream);
  AtomGroupGradientsCUDA cudaGradient1;
  AtomGroupGradientsCUDA cudaGradient1UsePairlist;
  cudaGradient1.initialize(cudaPos1.getNumAtoms(), stream);
  cudaGradient1UsePairlist.initialize(cudaPos1.getNumAtoms(), stream);
  double* d_energy;
  double* d_energy_pairlist;
  checkGPUError(cudaMalloc(&d_energy, sizeof(double)));
  checkGPUError(cudaMemset(d_energy, 0, sizeof(double)));
  checkGPUError(cudaMalloc(&d_energy_pairlist, sizeof(double)));
  checkGPUError(cudaMemset(d_energy_pairlist, 0, sizeof(double)));
  double* h_energy;
  checkGPUError(cudaMallocHost((void**)&h_energy, sizeof(double)));
  cudaGraph_t graph;
  checkGPUError(cudaGraphCreate(&graph, 0));

  compute.computeCoordinationNumberSelfGroupCUDA(
    cudaPos1, cudaGradient1, 1.0 / cutoffDistance,
    d_energy, true, graph, stream);
  compute.computeCoordinationNumberSelfGroupCUDA(
    cudaPos1, cudaGradient1UsePairlist, 1.0 / cutoffDistance,
    d_energy_pairlist, false, graph, stream);
  cudaGraphExec_t graph_exec;
#if defined(USE_CUDA)
  checkGPUError(cudaGraphInstantiate(&graph_exec, graph));
#elif defined(USE_HIP)
  checkGPUError(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
#endif
  checkGPUError(cudaStreamSynchronize(stream));

  const auto start = std::chrono::high_resolution_clock::now();
  checkGPUError(cudaGraphLaunch(graph_exec, stream));
  checkGPUError(cudaStreamSynchronize(stream));
  const auto end = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double, std::milli> fp_ms = end - start;

  checkGPUError(cudaMemcpyAsync(h_energy, d_energy_pairlist, sizeof(double), cudaMemcpyDeviceToHost, stream));
  checkGPUError(cudaStreamSynchronize(stream));

  std::cout << fmt::format("Coordination number: {:15.7e}, time (GPU) = {:10.5f} ms\n", *h_energy, fp_ms.count());
  const double energy = *h_energy;
  const auto hostGradient1 = cudaGradient1UsePairlist.toHost();
  auto pairlist = compute.pairlistToHost();
  checkGPUError(cudaFree(d_energy));
  checkGPUError(cudaFree(d_energy_pairlist));
  checkGPUError(cudaFreeHost(h_energy));
  checkGPUError(cudaGraphDestroy(graph));
  checkGPUError(cudaStreamDestroy(stream));
  return calculationResult{hostGradient1, AtomGroupGradients(), energy, pairlist};
}

void compareResults(const calculationResult& cpuResult,
                    const calculationResult& cudaResult,
                    bool twoGroups) {
  double maxRelErrorGradients1x = 0;
  double maxRelErrorGradients1y = 0;
  double maxRelErrorGradients1z = 0;
  const size_t numGrads1 = cpuResult.gradients1.gx.size();
  for (size_t i = 0; i < numGrads1; ++i) {
    const double diff_x = std::abs(cpuResult.gradients1.gx[i] - cudaResult.gradients1.gx[i]) / std::abs(cpuResult.gradients1.gx[i]);
    const double diff_y = std::abs(cpuResult.gradients1.gy[i] - cudaResult.gradients1.gy[i]) / std::abs(cpuResult.gradients1.gy[i]);
    const double diff_z = std::abs(cpuResult.gradients1.gz[i] - cudaResult.gradients1.gz[i]) / std::abs(cpuResult.gradients1.gz[i]);
    maxRelErrorGradients1x = std::max(diff_x, maxRelErrorGradients1x);
    maxRelErrorGradients1y = std::max(diff_y, maxRelErrorGradients1y);
    maxRelErrorGradients1z = std::max(diff_z, maxRelErrorGradients1z);
  }
  std::cout << fmt::format("Max relative error of gradients of group1: {:15.7e} {:15.7e} {:15.7e}\n",
                           maxRelErrorGradients1x, maxRelErrorGradients1y, maxRelErrorGradients1z);

  if (twoGroups) {
    double maxRelErrorGradients2x = 0;
    double maxRelErrorGradients2y = 0;
    double maxRelErrorGradients2z = 0;
    const size_t numGrads2 = cpuResult.gradients2.gx.size();
    for (size_t i = 0; i < numGrads2; ++i) {
      const double diff_x = std::abs(cpuResult.gradients2.gx[i] - cudaResult.gradients2.gx[i]) / std::abs(cpuResult.gradients2.gx[i]);
      const double diff_y = std::abs(cpuResult.gradients2.gy[i] - cudaResult.gradients2.gy[i]) / std::abs(cpuResult.gradients2.gy[i]);
      const double diff_z = std::abs(cpuResult.gradients2.gz[i] - cudaResult.gradients2.gz[i]) / std::abs(cpuResult.gradients2.gz[i]);
      maxRelErrorGradients2x = std::max(diff_x, maxRelErrorGradients2x);
      maxRelErrorGradients2y = std::max(diff_y, maxRelErrorGradients2y);
      maxRelErrorGradients2z = std::max(diff_z, maxRelErrorGradients2z);
    }
    std::cout << fmt::format("Max relative error of gradients of group2: {:15.7e} {:15.7e} {:15.7e}\n",
                            maxRelErrorGradients2x, maxRelErrorGradients2y, maxRelErrorGradients2z);
  }

  const double relErrorE = std::abs(cpuResult.energy - cudaResult.energy) / std::abs(cpuResult.energy);
  std::cout << fmt::format("Relative error of coordination number: {:15.7e}\n", relErrorE);

  std::cout << fmt::format("Pairlist size (CPU): {}\n", cpuResult.pairlist.size());
  std::cout << fmt::format("Pairlist size (GPU): {}\n", cudaResult.pairlist.size());
  bool pairlist_differ = false;
  if (cpuResult.pairlist.size() == cudaResult.pairlist.size()) {
    for (size_t i = 0; i < cpuResult.pairlist.size(); ++i) {
      if (cpuResult.pairlist[i] != cudaResult.pairlist[i]) {
        // std::cout << fmt::format("Pairlist differ at index {}: cpu = {}, gpu = {}\n", i, (int)cpuResult.pairlist[i], (int)cudaResult.pairlist[i]);
        pairlist_differ = true;
      }
    }
  } else {
    pairlist_differ = true;
  }
  if (pairlist_differ) {
    std::cout << "Pairlist differ!\n";
  }
}

int main(int argc, char* argv[]) {
  CLI::App app{"Test coordination number"};
  argv = app.ensure_utf8(argv);

  bool test_numerical_gradient = false;
  app.add_flag("--test_numerical_gradient", test_numerical_gradient, "Test the numerical gradient");

  bool test_two_groups = false;
  app.add_flag("--two_groups", test_two_groups, "Test the coordination number between two groups");

  bool test_self_group = false;
  app.add_flag("--self_group", test_self_group, "Test the coordination number between an atom group with itself");

  bool test_pairlist = false;
  app.add_flag("--pairlist", test_pairlist, "Use pairlist for testing");

  unsigned int group1_size = 10000;
  unsigned int group2_size = 2005;
  app.add_option("--group1", group1_size, "The number of atoms in group1");
  app.add_option("--group2", group2_size, "The number of atoms in group2");

  CLI11_PARSE(app, argc, argv);

  if (test_numerical_gradient) {
    testNumericalGradient();
  }

  if (test_two_groups) {
    AtomGroupPositions pos1 = generateRandomAtomGroupPositions(123, group1_size, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
    AtomGroupPositions pos2 = generateRandomAtomGroupPositions(456, group2_size, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
    double cutoffDistance = 6.0;
    const auto cpuResult = testCoordinationNumber(pos1, pos2, cutoffDistance);
    const auto gpuResult = testCoordinationNumberCUDA(pos1, pos2, cutoffDistance);
    compareResults(cpuResult, gpuResult, true);
  }

  if (test_self_group) {
    AtomGroupPositions pos1 = generateRandomAtomGroupPositions(123, group1_size, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0);
    double cutoffDistance = 6.0;
    if (!test_pairlist) {
      const auto cpuResult = testCoordinationNumberSelf(pos1, cutoffDistance);
      const auto gpuResult = testCoordinationNumberSelfCUDA(pos1, cutoffDistance);
      compareResults(cpuResult, gpuResult, false);
    } else {
      const auto cpuResult = testCoordinationNumberSelfPairlist(pos1, cutoffDistance);
      const auto gpuResult = testCoordinationNumberSelfCUDA(pos1, cutoffDistance, true);
      compareResults(cpuResult, gpuResult, false);
    }
  }

  return 0;
}
