#include "schedule.h"

#include <algorithm>
#include <stdexcept>

using namespace std;

vector<DecodePoint> contrast_decode(int tokens, int d_k, int num_heads, long long matmul_cycles) {
  if (tokens <= 0 || d_k <= 0 || num_heads <= 0 || matmul_cycles <= 0) {
    throw logic_error("decode arguments must be positive");
  }

  vector<DecodePoint> points;
  points.reserve(tokens);

  for (int token = 1; token <= tokens; token++) {
    int context_len = token;
    AttnCost per_head = attention_cost(context_len, d_k);
    long long attention_cycles = per_head.cycles * num_heads;
    points.push_back({token, context_len, matmul_cycles, attention_cycles});
  }

  return points;
}

vector<ScheduleOp> serial_schedule(const vector<DecodePoint>& points) {
  vector<ScheduleOp> ops;
  long long cursor = 0;

  for (const DecodePoint& point : points) {
    long long matmul_start = cursor;
    long long matmul_end = matmul_start + point.matmul_cycles;
    ops.push_back({"serial", point.token, "matmul", "qkv_ffn_proj", matmul_start, matmul_end});

    long long attention_start = matmul_end;
    long long attention_end = attention_start + point.attention_cycles;
    ops.push_back(
        {"serial", point.token, "attention", "kv_stream_softmax", attention_start, attention_end});

    cursor = attention_end;
  }

  return ops;
}

vector<ScheduleOp> overlap_schedule(const vector<DecodePoint>& points) {
  vector<ScheduleOp> ops;
  long long matmul_cursor = 0;
  long long attention_cursor = 0;

  for (const DecodePoint& point : points) {
    long long matmul_start = matmul_cursor;
    long long matmul_end = matmul_start + point.matmul_cycles;
    matmul_cursor = matmul_end;
    ops.push_back({"overlap", point.token, "matmul", "qkv_ffn_proj", matmul_start, matmul_end});

    long long attention_start = max(attention_cursor, matmul_end);
    long long attention_end = attention_start + point.attention_cycles;
    attention_cursor = attention_end;
    ops.push_back({"overlap", point.token, "attention", "kv_stream_softmax", attention_start,
                   attention_end});
  }

  return ops;
}
