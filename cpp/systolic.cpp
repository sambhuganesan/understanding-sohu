#include "systolic.h"

#include <stdexcept>

using namespace std;

SystolicResult simulate_systolic(int n, int m, int k) {
  if (n <= 0 || m <= 0 || k <= 0) {
    throw logic_error("systolic dimensions must be positive");
  }

  int total_cycles = (n-1) + (m-1) + k;
  vector<int> active_per_cycle(total_cycles, 0);

  for (int t = 0; t < total_cycles; t++) {
    int active = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        int start = i + j;
        if (start <= t && t < start + k) {
          active++;
        }
      }
    }

    active_per_cycle[t] = active;
  }
  return {total_cycles, active_per_cycle};
}

Matrix systolic_tile_matmul(const Matrix& a, const Matrix& b) {
  // TODO:
  // Compute A(N,K) x B(K,N) using the same timing idea as simulate_systolic.
  // This should eventually match naive_matmul(a, b).output.
  if (a.cols() != b.rows()) {
    throw logic_error("wrong dimension");
  }
  int m = a.rows();
  int k_depth = a.cols();
  int n = b.cols();

  Matrix result(m, n);

  int total_cycles = (m-1) + (n-1) + k_depth;

  for (int t = 0; t < total_cycles; t++) {
    for (int i = 0; i < a.rows(); i++) {
      for (int j = 0; j < b.cols(); j++) {
        int dot_index = t - (i+j);

        if (0 <= dot_index && dot_index < k_depth) {
          result(i, j) += a(i, dot_index) * b(dot_index, j); 
        }
      }
    }
  }

  return result;
}

vector<BatchPoint> batch_sweep(int n, int m, int max_batch) {
  if (n <= 0 || m <= 0 || max_batch <= 0) {
    throw logic_error("batch sweep arguments must be positive");
  }

  vector<BatchPoint> points;

  for (int batch = 1; batch <= max_batch; batch *= 2) {
    int overhead = (n-1) + (m-1);
    int cycles = overhead + batch;
    double util = static_cast<double>(batch) / cycles;

    points.push_back({batch, cycles, util});
  }

  return points;
}
