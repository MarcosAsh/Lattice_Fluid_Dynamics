#include "tensor.hpp"
#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#ifdef USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif
#include <algorithm>

int Tensor::compute_numel(const std::vector<int>& shape) {
    if (shape.empty()) return 0;
    int n = 1;
    for (int d : shape) n *= d;
    return n;
}

void Tensor::sync_rows_cols() {
    if (shape.size() == 2) {
        rows = shape[0];
        cols = shape[1];
    } else {
        rows = -1;
        cols = -1;
    }
}

Tensor::Tensor(const std::vector<int>& s)
    : rows(-1), cols(-1), shape(s), data(compute_numel(s), 0.0f) {
    sync_rows_cols();
}

Tensor::Tensor(int r, int c) : rows(r), cols(c), shape({r, c}), data(r * c, 0.0f) {}

Tensor::Tensor(int size) : Tensor(size, 1) {}

void Tensor::fill(float value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::print(const std::string& name) const {
    std::cout << name << " [";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) std::cout << "x";
        std::cout << shape[i];
    }
    std::cout << "]: ";
    int n = std::min(numel(), 20);
    for (int i = 0; i < n; ++i) std::cout << data[i] << " ";
    if (numel() > 20) std::cout << "...";
    std::cout << "\n";
}

Tensor Tensor::matmul(const Tensor& other) const {
    assert(ndim() == 2 && other.ndim() == 2);
    assert(cols == other.rows);
    Tensor result(rows, other.cols);
#ifdef USE_ACCELERATE
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, other.cols, cols,
                1.0f,
                data.data(), cols,
                other.data.data(), other.cols,
                0.0f,
                result.data.data(), other.cols);
    return result;
#else
    const int B = 32;
    for (int ii = 0; ii < rows; ii += B) {
        int iMax = std::min(ii + B, rows);
        for (int kk = 0; kk < cols; kk += B) {
            int kMax = std::min(kk + B, cols);
            for (int jj = 0; jj < other.cols; jj += B) {
                int jMax = std::min(jj + B, other.cols);
                for (int i = ii; i < iMax; ++i) {
                    for (int k = kk; k < kMax; ++k) {
                        const float a = (*this)(i, k);
                        float* res_ptr = &result.data[i * other.cols + jj];
                        const float* other_ptr = &other.data[k * other.cols + jj];
                        int len = jMax - jj;
                        int j = 0;
#if defined(__ARM_NEON__)
                        float32x4_t va = vdupq_n_f32(a);
                        for (; j + 4 <= len; j += 4) {
                            float32x4_t vb = vld1q_f32(other_ptr + j);
                            float32x4_t vr = vld1q_f32(res_ptr + j);
                            vr = vmlaq_f32(vr, vb, va);
                            vst1q_f32(res_ptr + j, vr);
                        }
#endif
                        for (; j < len; ++j) {
                            res_ptr[j] += a * other_ptr[j];
                        }
                    }
                }
            }
        }
    }
    return result;
#endif
}

float Tensor::dot(const Tensor& other) const {
    assert(data.size() == other.data.size());
    float sum = 0.0f;
    for (size_t i = 0; i < data.size(); ++i)
        sum += data[i] * other.data[i];
    return sum;
}

Tensor Tensor::operator+(const Tensor& other) const {
    assert(shape == other.shape && "Tensor dimension mismatch in addition");
    Tensor result(shape);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::transpose() const {
    assert(ndim() == 2);
    Tensor result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

// N-dim operations

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
    // Allow one -1 dimension to be inferred
    int total = numel();
    int neg_idx = -1;
    int product = 1;
    for (int i = 0; i < static_cast<int>(new_shape.size()); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx != -1) throw std::runtime_error("reshape: only one -1 allowed");
            neg_idx = i;
        } else {
            product *= new_shape[i];
        }
    }
    std::vector<int> resolved = new_shape;
    if (neg_idx != -1) {
        if (product == 0) throw std::runtime_error("reshape: cannot infer dim with 0 product");
        resolved[neg_idx] = total / product;
    }
    if (compute_numel(resolved) != total) {
        throw std::runtime_error("reshape: incompatible shapes");
    }
    Tensor result(resolved);
    std::copy(data.begin(), data.end(), result.data.begin());
    return result;
}

Tensor Tensor::squeeze(int dim) const {
    std::vector<int> new_shape;
    if (dim == -1) {
        for (int d : shape) {
            if (d != 1) new_shape.push_back(d);
        }
    } else {
        if (dim < 0) dim += ndim();
        for (int i = 0; i < ndim(); ++i) {
            if (i == dim && shape[i] == 1) continue;
            new_shape.push_back(shape[i]);
        }
    }
    if (new_shape.empty()) new_shape.push_back(1);
    Tensor result(new_shape);
    std::copy(data.begin(), data.end(), result.data.begin());
    return result;
}

Tensor Tensor::unsqueeze(int dim) const {
    if (dim < 0) dim += ndim() + 1;
    std::vector<int> new_shape = shape;
    new_shape.insert(new_shape.begin() + dim, 1);
    Tensor result(new_shape);
    std::copy(data.begin(), data.end(), result.data.begin());
    return result;
}

Tensor Tensor::permute(const std::vector<int>& order) const {
    assert(static_cast<int>(order.size()) == ndim());
    std::vector<int> new_shape(ndim());
    for (int i = 0; i < ndim(); ++i) {
        new_shape[i] = shape[order[i]];
    }

    // Compute strides for source
    std::vector<int> src_strides(ndim());
    src_strides[ndim() - 1] = 1;
    for (int i = ndim() - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for destination
    std::vector<int> dst_strides(ndim());
    dst_strides[ndim() - 1] = 1;
    for (int i = ndim() - 2; i >= 0; --i) {
        dst_strides[i] = dst_strides[i + 1] * new_shape[i + 1];
    }

    Tensor result(new_shape);
    int total = numel();
    for (int flat = 0; flat < total; ++flat) {
        // Convert flat index to multi-dim source index
        int tmp = flat;
        int src_offset = 0;
        for (int d = 0; d < ndim(); ++d) {
            int idx = tmp / dst_strides[d];
            tmp %= dst_strides[d];
            src_offset += idx * src_strides[order[d]];
        }
        result.data[flat] = data[src_offset];
    }
    return result;
}

Tensor Tensor::flatten(int start_dim, int end_dim) const {
    if (start_dim < 0) start_dim += ndim();
    if (end_dim < 0) end_dim += ndim();
    if (start_dim < 0) start_dim = 0;
    if (end_dim >= ndim()) end_dim = ndim() - 1;

    std::vector<int> new_shape;
    for (int i = 0; i < start_dim; ++i) new_shape.push_back(shape[i]);
    int flat = 1;
    for (int i = start_dim; i <= end_dim; ++i) flat *= shape[i];
    new_shape.push_back(flat);
    for (int i = end_dim + 1; i < ndim(); ++i) new_shape.push_back(shape[i]);

    Tensor result(new_shape);
    std::copy(data.begin(), data.end(), result.data.begin());
    return result;
}
