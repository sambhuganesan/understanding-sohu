#pragma once

#include "matrix.h"

#include <vector>

struct SystolicResult {
  int total_cycles;
  std::vector<int> active_per_cycle;
};

struct BatchPoint {
  int batch;
  int cycles;
  double util;
};

SystolicResult simulate_systolic(int n, int m, int k);
Matrix systolic_tile_matmul(const Matrix& a, const Matrix& b);
std::vector<BatchPoint> batch_sweep(int n, int m, int max_batch);
