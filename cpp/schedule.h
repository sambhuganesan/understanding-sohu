#pragma once

#include "attention.h"

#include <string>
#include <vector>

struct DecodePoint {
  int token;
  int context_len;
  long long matmul_cycles;
  long long attention_cycles;
};

struct ScheduleOp {
  std::string sched;
  int token;
  std::string engine;
  std::string op;
  long long start;
  long long end;
};

std::vector<DecodePoint> contrast_decode(int tokens, int d_k, int num_heads, long long matmul_cycles);
std::vector<ScheduleOp> serial_schedule(const std::vector<DecodePoint>& points);
std::vector<ScheduleOp> overlap_schedule(const std::vector<DecodePoint>& points);
