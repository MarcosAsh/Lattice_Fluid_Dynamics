#pragma once
#include <vector>
#include "allocator.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

class Tensor {
public:
    int rows, cols;  // backward compat: set for 2D, -1 otherwise
    std::vector<int> shape;
    std::vector<float, UnifiedMemoryAllocator<float>> data;

    // N-dimensional constructor
    Tensor(const std::vector<int>& shape);

    // 2D constructor (backward compatible)
    Tensor(int r, int c);

    // 1D constructor (backward compatible)
    Tensor(int size);

    int ndim() const { return static_cast<int>(shape.size()); }
    int numel() const { return static_cast<int>(data.size()); }

    void fill(float value);
    void print(const std::string& name = "") const;

    // 2D operations (assert ndim()==2)
    Tensor matmul(const Tensor& other) const;
    Tensor transpose() const;
    float dot(const Tensor& other) const;

    Tensor operator+(const Tensor& other) const;
    inline float& operator()(int i, int j) { return data[i * cols + j]; }
    inline const float& operator()(int i, int j) const { return data[i * cols + j]; }

    // N-dim operations
    Tensor reshape(const std::vector<int>& new_shape) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor permute(const std::vector<int>& order) const;
    Tensor flatten(int start_dim = 0, int end_dim = -1) const;

private:
    void sync_rows_cols();
    static int compute_numel(const std::vector<int>& shape);
};
