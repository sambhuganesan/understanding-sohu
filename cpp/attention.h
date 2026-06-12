#pragma once

struct AttnCost {
  long long kv_reads;
  long long macs;
  long long softmax_ops;
  long long cycles;
};

AttnCost attention_cost(int context_len, int d_k, int lanes = 1);
