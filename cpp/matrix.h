#pragma once

#include <vector>

class Matrix {
 public:
  Matrix(int rows, int cols);

  float& operator()(int row, int col);
  float operator()(int row, int col) const;

  int rows() const;
  int cols() const;

 private:
  int rows_;
  int cols_;
  std::vector<float> data_;
};

bool almost_equal(const Matrix& lhs, const Matrix& rhs, float eps = 1e-4f);
