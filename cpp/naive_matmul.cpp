#include "naive_matmul.h"

#include <stdexcept>

using namespace std;

MatmulResult naive_matmul(const Matrix& a, const Matrix& b) {
  if (a.cols() != b.rows()) {
    throw logic_error("wrong dimensions");
  }

  Matrix result(a.rows(), b.cols());
  long long number_of_MACs = 0;

  for (int r = 0; r < a.rows(); r++) {
    for (int c = 0; c < b.cols(); c++) {
      for (int k = 0; k < b.rows(); k++) {
        result(r, c) += a(r, k) * b(k, c);
        number_of_MACs++;
      }
    }
  }

  return {result, number_of_MACs};
}
