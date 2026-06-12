#include "attention.h"

#include <stdexcept>

using namespace std;

AttnCost attention_cost(int context_len, int d_k, int lanes) {
  if (context_len <= 0 || d_k <= 0 || lanes <= 0) {
    throw logic_error("attention arguments must be positive");
  }

  long long l = context_len;
  long long dk = d_k;

  long long kv_reads = 2 * l;
  long long macs = 2 * l * dk;
  long long softmax_ops = 3 * l;
  long long cycles = (macs + lanes - 1) / lanes + softmax_ops;
  
  return {kv_reads, macs, softmax_ops, cycles};
}
