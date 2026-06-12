#include "attention.h"
#include "matrix.h"
#include "naive_matmul.h"
#include "schedule.h"
#include "systolic.h"
#include "trace.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace std;

namespace {

Matrix make_demo_a(int rows, int cols) {
  Matrix a(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      a(r, c) = static_cast<float>((r + 1) * 0.5f + (c + 1));
    }
  }
  return a;
}

Matrix make_demo_b(int rows, int cols) {
  Matrix b(rows, cols);
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      b(r, c) = static_cast<float>((c + 1) * 0.25f - (r + 1) * 0.125f);
    }
  }
  return b;
}

long long toy_matmul_cycles(int d_model, int d_ff) {
  long long qkv = 3LL * d_model * d_model;
  long long out_proj = 1LL * d_model * d_model;
  long long ffn = 2LL * d_model * d_ff;
  return qkv + out_proj + ffn;
}

}  // namespace

int main() {
  try {
    ensure_trace_dir();

    int n = 8;
    int k = 16;
    SystolicResult timing = simulate_systolic(n, n, k);
    write_systolic_trace("traces/systolic_N8_K16.jsonl", n, timing);

    Matrix a = make_demo_a(n, k);
    Matrix b = make_demo_b(k, n);
    MatmulResult reference = naive_matmul(a, b);
    Matrix dut = systolic_tile_matmul(a, b);
    if (!almost_equal(reference.output, dut)) {
      throw logic_error("systolic DUT output did not match naive reference");
    }

    vector<BatchPoint> batches = batch_sweep(n, n, 256);
    write_batch_trace("traces/batch_sweep.jsonl", batches);

    int d_model = 32;
    int d_k = 8;
    int num_heads = 4;
    int d_ff = 128;
    int tokens = 128;
    long long matmul_cycles = toy_matmul_cycles(d_model, d_ff);
    vector<DecodePoint> contrast = contrast_decode(tokens, d_k, num_heads, matmul_cycles);
    write_contrast_trace("traces/contrast_decode.jsonl", contrast);
    write_schedule_trace("traces/schedule_serial.jsonl", serial_schedule(contrast));
    write_schedule_trace("traces/schedule_overlap.jsonl", overlap_schedule(contrast));

    int peak_active = *max_element(timing.active_per_cycle.begin(), timing.active_per_cycle.end());
    cout << "Dual-engine simulator traces written to traces/\n";
    cout << "Systolic N=" << n << " K=" << k << " total_cycles=" << timing.total_cycles
         << " peak_active_pes=" << peak_active << "/" << n * n << "\n";
    cout << "Reference check: naive_matmul == systolic_tile_matmul\n";
    cout << "Batch sweep: B=1 util=" << batches.front().util
         << ", B=" << batches.back().batch << " util=" << batches.back().util << "\n";
    cout << "Decode contrast: matmul cycles fixed at " << matmul_cycles
         << ", attention grows from " << contrast.front().attention_cycles << " to "
         << contrast.back().attention_cycles << "\n";
  } catch (const exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
