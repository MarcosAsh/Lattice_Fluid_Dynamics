#include "layers/ad_linear.hpp"
#include <random>
#include <cmath>

ADLinear::ADLinear(int input_dim, int output_dim) {
    Tensor tW(output_dim, input_dim);
    Tensor tb(output_dim, 1);
    std::mt19937 gen(std::random_device{}());
    float r = std::sqrt(6.0f / (input_dim + output_dim));
    std::uniform_real_distribution<float> dist(-r, r);
    for (auto &v : tW.data) v = dist(gen);
    tb.fill(0.0f);
    W = make_ad(tW);
    register_parameter(W);
    b = make_ad(tb);
    register_parameter(b);
}

std::shared_ptr<ADTensor> ADLinear::forward(
    const std::shared_ptr<ADTensor>& x) const {
    auto y = matmul(W, x);
    int seq_len = x->val.cols;
    if (seq_len != cached_seq_len) {
        cached_ones_row = Tensor(1, seq_len);
        cached_ones_row.data.assign(seq_len, 1.0f);
        cached_seq_len = seq_len;
    }
    auto ones_row = make_ad(cached_ones_row);
    auto b_mat = matmul(b, ones_row);
    return add(y, b_mat);
}