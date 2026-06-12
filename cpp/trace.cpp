#include "trace.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

using namespace std;

namespace {

ofstream open_trace(const string& path) {
  ofstream out(path);
  if (!out) {
    throw logic_error("could not open trace file: " + path);
  }

  out << fixed << setprecision(6);
  return out;
}

}  // namespace

void ensure_trace_dir() {
  filesystem::create_directories("traces");
}

void write_systolic_trace(const string& path, int n, const SystolicResult& result) {
  ofstream out = open_trace(path);
  int total_pes = n * n;

  for (int cycle = 0; cycle < result.total_cycles; cycle++) {
    int active = result.active_per_cycle.at(cycle);
    double util = static_cast<double>(active) / total_pes;
    out << "{\"engine\":\"systolic\",\"cycle\":" << cycle << ",\"active_pes\":" << active
        << ",\"total_pes\":" << total_pes << ",\"util\":" << util << "}\n";
  }
}

void write_batch_trace(const string& path, const vector<BatchPoint>& points) {
  ofstream out = open_trace(path);

  for (const BatchPoint& point : points) {
    out << "{\"demo\":\"batch_sweep\",\"batch\":" << point.batch << ",\"cycles\":"
        << point.cycles << ",\"util\":" << point.util << "}\n";
  }
}

void write_contrast_trace(const string& path, const vector<DecodePoint>& points) {
  ofstream out = open_trace(path);

  for (const DecodePoint& point : points) {
    out << "{\"demo\":\"contrast_decode\",\"token\":" << point.token << ",\"context_len\":"
        << point.context_len << ",\"matmul_cycles\":" << point.matmul_cycles
        << ",\"attention_cycles\":" << point.attention_cycles << "}\n";
  }
}

void write_schedule_trace(const string& path, const vector<ScheduleOp>& ops) {
  ofstream out = open_trace(path);

  for (const ScheduleOp& op : ops) {
    out << "{\"sched\":\"" << op.sched << "\",\"token\":" << op.token << ",\"engine\":\""
        << op.engine << "\",\"op\":\"" << op.op << "\",\"start\":" << op.start
        << ",\"end\":" << op.end << "}\n";
  }
}
