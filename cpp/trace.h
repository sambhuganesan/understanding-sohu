#pragma once

#include "schedule.h"
#include "systolic.h"

#include <string>
#include <vector>

void ensure_trace_dir();
void write_systolic_trace(const std::string& path, int n, const SystolicResult& result);
void write_batch_trace(const std::string& path, const std::vector<BatchPoint>& points);
void write_contrast_trace(const std::string& path, const std::vector<DecodePoint>& points);
void write_schedule_trace(const std::string& path, const std::vector<ScheduleOp>& ops);
