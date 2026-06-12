#pragma once

#include "matrix.h"

struct MatmulResult {
  Matrix output;
  long long macs;
};

MatmulResult naive_matmul(const Matrix& a, const Matrix& b);
