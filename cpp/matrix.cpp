#include "matrix.h"

#include <cmath>
#include <stdexcept>

using namespace std;

Matrix::Matrix(int rows, int cols) : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {
  if (rows <= 0 || cols <= 0) {
    throw invalid_argument("row or col is not positive");
  }
}

float& Matrix::operator()(int row, int col) {
  return data_.at(row * cols_ + col);
}

float Matrix::operator()(int row, int col) const {
  return data_.at(row * cols_ + col);
}

int Matrix::rows() const {
  return rows_;
}

int Matrix::cols() const {
  return cols_;
}

bool almost_equal(const Matrix& lhs, const Matrix& rhs, float eps) {
  if (lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()) {
    return false;
  }

  for (int r = 0; r < lhs.rows(); r++) {
    for (int c = 0; c < lhs.cols(); c++) {
      if (fabs(rhs(r, c) - lhs(r, c)) > eps) {
        return false;
      }
    }
  }

  return true;
}
